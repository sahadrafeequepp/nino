"""SQLite-persisted asynchronous job scheduler for reminders, briefings, and automations."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
import json
import logging
from typing import Any, Optional

from nino.automation.briefing import DailyBriefingGenerator
from nino.automation.notifications import NotificationDispatcher
from nino.automation.policy import ProactivePolicyEngine
from nino.config import Settings
from nino.memory.db import MemoryDB

logger = logging.getLogger("nino.automation.scheduler")


class SQLiteJobScheduler:
    """Manages scheduled jobs persisted in SQLite with quiet-hours and rate-limit enforcement."""

    def __init__(
        self,
        memory_db: MemoryDB,
        settings: Settings,
        policy_engine: ProactivePolicyEngine,
        dispatcher: NotificationDispatcher,
        briefing_gen: Optional[DailyBriefingGenerator] = None,
        clock_fn: Optional[Callable[[], datetime]] = None,
        poll_interval_seconds: float = 1.0,
    ):
        self.db = memory_db
        self.settings = settings
        self.policy = policy_engine
        self.dispatcher = dispatcher
        self.briefing_gen = briefing_gen
        self.clock = clock_fn or datetime.now
        self.poll_interval = poll_interval_seconds

        self._running = False
        self._task: Optional[asyncio.Task] = None

    def now(self) -> datetime:
        return self.clock()

    def add_job(
        self,
        job_id: str,
        name: str,
        job_type: str,
        next_run_time: datetime,
        cron_expr: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Add or update a scheduled job in SQLite."""
        kwargs_str = json.dumps(kwargs or {})
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO automations (
                    id, name, job_type, cron_expr, interval_seconds,
                    next_run_time, kwargs_json, enabled, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    job_type = excluded.job_type,
                    cron_expr = excluded.cron_expr,
                    interval_seconds = excluded.interval_seconds,
                    next_run_time = excluded.next_run_time,
                    kwargs_json = excluded.kwargs_json,
                    enabled = 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    job_id,
                    name,
                    job_type,
                    cron_expr,
                    interval_seconds,
                    next_run_time.isoformat(),
                    kwargs_str,
                ),
            )
        logger.info("Registered job '%s' (%s) next run: %s", job_id, job_type, next_run_time.isoformat())
        return self.get_job(job_id) or {}

    def remove_job(self, job_id: str) -> bool:
        """Cancel and delete a scheduled automation."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("DELETE FROM automations WHERE id = ?", (job_id,))
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Removed automation '%s'", job_id)
        return deleted

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single automation by ID."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM automations WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            try:
                res["kwargs"] = json.loads(res.get("kwargs_json", "{}"))
            except Exception:
                res["kwargs"] = {}
            return res

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all automations."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM automations ORDER BY next_run_time ASC")
            jobs = []
            for row in cursor.fetchall():
                d = dict(row)
                try:
                    d["kwargs"] = json.loads(d.get("kwargs_json", "{}"))
                except Exception:
                    d["kwargs"] = {}
                jobs.append(d)
            return jobs

    def snooze_job(self, job_id: str, until: datetime) -> bool:
        """Snooze a specific job until a future datetime."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE automations
                SET snoozed_until = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (until.isoformat(), job_id),
            )
            return cursor.rowcount > 0

    async def trigger_due_jobs(self, current_dt: Optional[datetime] = None) -> list[dict[str, Any]]:
        """Find and execute all due automations whose trigger time <= current_dt.

        Returns list of executed job execution summaries.
        """
        now_dt = current_dt or self.now()
        now_iso = now_dt.isoformat()

        # Query due jobs
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM automations
                WHERE enabled = 1
                  AND (snoozed_until IS NULL OR ? >= snoozed_until)
                  AND ? >= next_run_time
                ORDER BY next_run_time ASC
                """,
                (now_iso, now_iso),
            )
            due_rows = [dict(r) for r in cursor.fetchall()]

        executed = []

        for row in due_rows:
            job_id = row["id"]
            job_type = row["job_type"]
            try:
                kwargs = json.loads(row.get("kwargs_json", "{}"))
            except Exception:
                kwargs = {}

            is_urgent = bool(kwargs.get("urgent", False))

            # Policy check
            allowed, reason = self.policy.can_deliver_proactive_message(
                message_type=job_type,
                is_urgent=is_urgent,
                dt=now_dt,
            )

            if not allowed:
                logger.debug("Job '%s' (%s) suppressed by policy: %s", job_id, job_type, reason)
                # If suppressed by quiet hours / DND / rate limit, don't execute now, check again later
                continue

            # Execute job
            result = await self._execute_job(row, kwargs, now_dt)
            executed.append(result)

            # Advance next_run_time or disable
            interval_sec = row.get("interval_seconds")
            if interval_sec and interval_sec > 0:
                next_time = now_dt + timedelta(seconds=interval_sec)
                with self.db.get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE automations
                        SET next_run_time = ?, snoozed_until = NULL, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (next_time.isoformat(), job_id),
                    )
            elif row.get("cron_expr"):
                # Simple daily cron advance (e.g. daily briefing: +24 hours)
                next_time = now_dt + timedelta(days=1)
                with self.db.get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE automations
                        SET next_run_time = ?, snoozed_until = NULL, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (next_time.isoformat(), job_id),
                    )
            else:
                # One-shot job: disable after completion
                with self.db.get_connection() as conn:
                    conn.execute(
                        "UPDATE automations SET enabled = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (job_id,),
                    )

        return executed

    async def _execute_job(self, row: dict[str, Any], kwargs: dict[str, Any], now_dt: datetime) -> dict[str, Any]:
        """Dispatch job execution based on job_type."""
        job_id = row["id"]
        job_type = row["job_type"]
        is_urgent = bool(kwargs.get("urgent", False))

        message = ""
        title = row.get("name", "Nino Notification")

        if job_type == "reminder":
            reminder_id = kwargs.get("reminder_id")
            message = kwargs.get("message", "You have a scheduled reminder.")
            title = "Reminder"
            if reminder_id:
                with self.db.get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE reminders
                        SET status = 'delivered', delivered_at = ?
                        WHERE id = ?
                        """,
                        (now_dt.isoformat(), reminder_id),
                    )

        elif job_type == "daily_briefing":
            if self.briefing_gen:
                message = await self.briefing_gen.generate_briefing()
            else:
                message = "Good morning. Here is your daily briefing."
            title = "Daily Briefing"

        elif job_type == "focus_session":
            action = kwargs.get("action", "completed")
            task_title = kwargs.get("task", "Focus Session")
            if action == "completed":
                message = f"Focus session completed: '{task_title}'. Great job! Take a short break."
                title = "Focus Session Complete"
            else:
                message = f"Starting focus session: '{task_title}'."
                title = "Focus Session Started"

        else:
            message = kwargs.get("message", f"Automation '{title}' triggered.")

        # Tone sanitization
        message = self.policy.sanitize_message_content(message)

        # Broadcast via notification channels
        delivery_results = await self.dispatcher.broadcast(
            message=message,
            title=title,
            is_urgent=is_urgent,
        )

        # Record delivery
        self.policy.record_delivery(
            message_type=job_type,
            content=message,
            channel=",".join([k for k, v in delivery_results.items() if v]),
        )

        return {
            "job_id": job_id,
            "job_type": job_type,
            "title": title,
            "message": message,
            "delivery": delivery_results,
            "executed_at": now_dt.isoformat(),
        }

    async def start(self) -> None:
        """Start background polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("SQLiteJobScheduler started.")

    async def stop(self) -> None:
        """Stop background polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("SQLiteJobScheduler stopped.")

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self.trigger_due_jobs()
            except Exception as e:
                logger.error("Error in scheduler poll loop: %s", e)
            await asyncio.sleep(self.poll_interval)
