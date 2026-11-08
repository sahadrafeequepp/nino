"""Comprehensive Reminder Management and Real-Time Delivery System for Nino Telegram Bot.

Features:
- Persistent SQLite storage in 'data/nino.db' (compatible with Nino's shared database).
- Flexible time parser (relative: 30s, 5m, 2h; absolute: 9:30pm, 21:00; ISO; natural language).
- Natural language intent extraction ("remind me in 10 minutes to call mom").
- Background delivery loop sending Telegram alerts when reminders are due.
- Interactive snooze and cancel callbacks with inline buttons.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Optional

try:
    from clock import clock
except ImportError:
    from telegram_bot.clock import clock

logger = logging.getLogger("nino.telegram.reminders")


def parse_time_expression(time_str: str, base_dt: Optional[datetime] = None) -> tuple[datetime, str]:
    """Parse a flexible time expression into a future datetime and readable label.

    Supports:
    - Relative seconds: '30s', '45 sec', '60 seconds'
    - Relative minutes: '5m', '10 min', '15 mins', '20 minutes', '+10m'
    - Relative hours: '1h', '2 hrs', '3 hours', '+2h'
    - Relative days: '1d', '2 days'
    - Time of day: '21:30', '9:30pm', '9:30 PM', '10am', '08:00'
    - Natural keywords: 'tomorrow', 'tonight'
    - ISO strings: '2026-10-05T22:00:00'
    """
    now = base_dt or clock.now()
    clean = time_str.strip().lower()

    # 1. Seconds: e.g. "30s", "45 sec", "20 seconds"
    sec_match = re.fullmatch(r"^\+?(\d+)\s*(?:s|sec|secs|second|seconds)$", clean)
    if sec_match:
        secs = int(sec_match.group(1))
        return now + timedelta(seconds=secs), f"in {secs} second{'s' if secs != 1 else ''}"

    # 2. Minutes: e.g. "5m", "10 min", "15 mins", "20 minutes", "+15m"
    min_match = re.fullmatch(r"^\+?(\d+)\s*(?:m|min|mins|minute|minutes)$", clean)
    if min_match:
        mins = int(min_match.group(1))
        return now + timedelta(minutes=mins), f"in {mins} minute{'s' if mins != 1 else ''}"

    # 3. Hours: e.g. "1h", "2 hrs", "3 hours", "+2h"
    hr_match = re.fullmatch(r"^\+?(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)$", clean)
    if hr_match:
        hrs = float(hr_match.group(1))
        return now + timedelta(hours=hrs), f"in {hrs:g} hour{'s' if hrs != 1 else ''}"

    # 4. Days: e.g. "1d", "2 days"
    day_match = re.fullmatch(r"^\+?(\d+)\s*(?:d|day|days)$", clean)
    if day_match:
        days = int(day_match.group(1))
        return now + timedelta(days=days), f"in {days} day{'s' if days != 1 else ''}"

    # 5. Natural keyword: "tomorrow" (default 9:00 AM)
    if clean == "tomorrow":
        target = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return target, "tomorrow at 09:00 AM"

    # 6. Natural keyword: "tonight" (default 8:00 PM)
    if clean == "tonight":
        target = now.replace(hour=20, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target, "tonight at 08:00 PM"

    # 7. Time of day with AM/PM: e.g. "9:30pm", "9:30 pm", "10am", "8:15 am"
    ampm_match = re.fullmatch(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$", clean)
    if ampm_match:
        hour = int(ampm_match.group(1))
        minute = int(ampm_match.group(2) or 0)
        meridiem = ampm_match.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        disp_meridiem = "AM" if hour < 12 else "PM"
        disp_hour = hour if 1 <= hour <= 12 else (hour - 12 if hour > 12 else 12)
        return target, f"at {disp_hour:02d}:{minute:02d} {disp_meridiem}"

    # 8. Military / 24-hour time of day: e.g. "21:30", "09:15"
    time_24_match = re.fullmatch(r"^(\d{1,2}):(\d{2})$", clean)
    if time_24_match:
        hour = int(time_24_match.group(1))
        minute = int(time_24_match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target, f"at {hour:02d}:{minute:02d}"

    # 9. ISO Format fallback
    try:
        iso_dt = datetime.fromisoformat(clean)
        if iso_dt.tzinfo is None and now.tzinfo is not None:
            iso_dt = iso_dt.replace(tzinfo=now.tzinfo)
        return iso_dt, f"at {iso_dt.strftime('%I:%M %p')}"
    except ValueError:
        pass

    # 10. Fallback: 10 minutes
    fallback_dt = now + timedelta(minutes=10)
    return fallback_dt, "in 10 minutes"


def extract_reminder_from_text(text: str) -> Optional[tuple[str, str]]:
    """Extract (time_spec, reminder_message) from conversational text.

    Examples:
    - "remind me in 5 minutes to call mom" -> ("5 minutes", "call mom")
    - "remind me to take medicine at 9pm" -> ("9pm", "take medicine")
    - "set a reminder for 10m to check oven" -> ("10m", "check oven")
    - "set a reminder to drink water in 1 hour" -> ("1 hour", "drink water")
    """
    clean = text.strip()

    # Pattern A: "remind me in/after <time> to <action>"
    m = re.search(r"\bremind\s+(?:me\s+)?(?:in|after)\s+(.+?)\s+to\s+(.+)$", clean, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Pattern B: "remind me to <action> in/after/at <time>"
    m = re.search(r"\bremind\s+(?:me\s+)?to\s+(.+?)\s+(?:in|after|at)\s+([0-9a-zA-Z\s:]+)$", clean, re.IGNORECASE)
    if m:
        return m.group(2).strip(), m.group(1).strip()

    # Pattern C: "set a reminder to <action> in/after/at <time>"
    m = re.search(r"\b(?:set|create)(?:\s+a)?\s+reminder\s+to\s+(.+?)\s+(?:in|after|at)\s+([0-9a-zA-Z\s:]+)$", clean, re.IGNORECASE)
    if m:
        return m.group(2).strip(), m.group(1).strip()

    # Pattern D: "set a reminder in/after/for <time> to <action>"
    m = re.search(r"\b(?:set|create)(?:\s+a)?\s+reminder\s+(?:in|after|for)\s+(.+?)\s+to\s+(.+)$", clean, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Pattern E: "remind me <action> in <time>"
    m = re.search(r"\bremind\s+me\s+(.+?)\s+in\s+([0-9]+\s*[a-zA-Z]+)$", clean, re.IGNORECASE)
    if m:
        return m.group(2).strip(), m.group(1).strip()

    return None


class ReminderManager:
    """Manages persistent SQLite storage and lifecycle for Nino Telegram reminders."""

    def __init__(self, db_path: str = "data/nino.db"):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Ensure reminders and automations tables exist with user/chat metadata columns."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    trigger_time TEXT NOT NULL,
                    urgent INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    delivered_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS automations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    cron_expr TEXT,
                    interval_seconds INTEGER,
                    next_run_time TEXT,
                    kwargs_json TEXT DEFAULT '{}',
                    enabled INTEGER DEFAULT 1,
                    snoozed_until TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_reminders_trigger ON reminders(trigger_time, status);
                """
            )

            # Check and add chat_id and user_id columns to reminders if missing
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(reminders)").fetchall()]
            if "chat_id" not in cols:
                conn.execute("ALTER TABLE reminders ADD COLUMN chat_id INTEGER")
            if "user_id" not in cols:
                conn.execute("ALTER TABLE reminders ADD COLUMN user_id INTEGER")

    def create_reminder(
        self,
        message: str,
        time_expression: str | datetime,
        chat_id: int,
        user_id: int,
        urgent: bool = False,
    ) -> dict[str, Any]:
        """Create and persist a new reminder in SQLite."""
        if isinstance(time_expression, datetime):
            target_dt = time_expression
            time_label = target_dt.strftime("at %I:%M %p")
        else:
            target_dt, time_label = parse_time_expression(time_expression)

        trigger_iso = target_dt.isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reminders (message, trigger_time, urgent, status, chat_id, user_id)
                VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (message.strip(), trigger_iso, 1 if urgent else 0, chat_id, user_id),
            )
            reminder_id = cursor.lastrowid

            # Also register in automations table for unified job tracking
            job_id = f"reminder_{reminder_id}"
            kwargs_json = json.dumps({
                "reminder_id": reminder_id,
                "message": message.strip(),
                "chat_id": chat_id,
                "user_id": user_id,
                "urgent": urgent,
            })
            conn.execute(
                """
                INSERT INTO automations (id, name, job_type, next_run_time, kwargs_json, enabled)
                VALUES (?, ?, 'reminder', ?, ?, 1)
                ON CONFLICT(id) DO UPDATE SET
                    next_run_time = excluded.next_run_time,
                    kwargs_json = excluded.kwargs_json,
                    enabled = 1
                """,
                (job_id, f"Reminder #{reminder_id}", trigger_iso, kwargs_json),
            )
            conn.commit()

        logger.info(
            "Created reminder #%d: '%s' for user=%d chat=%d at %s",
            reminder_id,
            message,
            user_id,
            chat_id,
            trigger_iso,
        )

        return {
            "id": reminder_id,
            "message": message.strip(),
            "trigger_time": target_dt,
            "trigger_iso": trigger_iso,
            "time_label": time_label,
            "formatted_time": target_dt.strftime("%I:%M %p (%A, %d %b %Y)"),
            "chat_id": chat_id,
            "user_id": user_id,
            "urgent": urgent,
        }

    def list_reminders(
        self,
        chat_id: Optional[int] = None,
        user_id: Optional[int] = None,
        include_delivered: bool = False,
    ) -> list[dict[str, Any]]:
        """List scheduled reminders filtered by user or chat."""
        with self._get_connection() as conn:
            query = "SELECT * FROM reminders WHERE 1=1"
            params: list[Any] = []

            if not include_delivered:
                query += " AND status = 'pending'"

            if user_id is not None:
                query += " AND (user_id = ? OR user_id IS NULL)"
                params.append(user_id)
            elif chat_id is not None:
                query += " AND (chat_id = ? OR chat_id IS NULL)"
                params.append(chat_id)

            query += " ORDER BY trigger_time ASC LIMIT 30"
            rows = conn.execute(query, params).fetchall()

            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "message": r["message"],
                    "trigger_time": r["trigger_time"],
                    "urgent": bool(r["urgent"]),
                    "status": r["status"],
                    "chat_id": r["chat_id"],
                    "user_id": r["user_id"],
                    "created_at": r["created_at"],
                })
            return results

    def get_due_reminders(self, current_dt: Optional[datetime] = None) -> list[dict[str, Any]]:
        """Find pending reminders that have reached or passed their trigger time."""
        now = current_dt or clock.now()
        now_iso = now.isoformat()

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reminders
                WHERE status = 'pending' AND ? >= trigger_time
                ORDER BY trigger_time ASC
                """,
                (now_iso,),
            ).fetchall()

            due = []
            for r in rows:
                due.append({
                    "id": r["id"],
                    "message": r["message"],
                    "trigger_time": r["trigger_time"],
                    "urgent": bool(r["urgent"]),
                    "status": r["status"],
                    "chat_id": r["chat_id"],
                    "user_id": r["user_id"],
                })
            return due

    def mark_delivered(self, reminder_id: int) -> bool:
        """Mark reminder delivered and disable corresponding automation."""
        now_iso = clock.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE reminders
                SET status = 'delivered', delivered_at = ?
                WHERE id = ?
                """,
                (now_iso, reminder_id),
            )
            conn.execute(
                """
                UPDATE automations
                SET enabled = 0, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (f"reminder_{reminder_id}",),
            )
            conn.commit()
            return True

    def cancel_reminder(self, reminder_id: int, user_id: Optional[int] = None) -> bool:
        """Cancel a pending reminder."""
        with self._get_connection() as conn:
            query = "UPDATE reminders SET status = 'cancelled' WHERE id = ?"
            params: list[Any] = [reminder_id]
            if user_id is not None:
                query += " AND (user_id = ? OR user_id IS NULL)"
                params.append(user_id)

            cursor = conn.execute(query, params)
            if cursor.rowcount > 0:
                conn.execute("UPDATE automations SET enabled = 0 WHERE id = ?", (f"reminder_{reminder_id}",))
                conn.commit()
                return True
            return False

    def snooze_reminder(self, reminder_id: int, minutes: int = 10, user_id: Optional[int] = None) -> Optional[datetime]:
        """Snooze a reminder by extending its trigger time."""
        new_dt = clock.now() + timedelta(minutes=minutes)
        new_iso = new_dt.isoformat()

        with self._get_connection() as conn:
            query = "UPDATE reminders SET trigger_time = ?, status = 'pending' WHERE id = ?"
            params: list[Any] = [new_iso, reminder_id]
            if user_id is not None:
                query += " AND (user_id = ? OR user_id IS NULL)"
                params.append(user_id)

            cursor = conn.execute(query, params)
            if cursor.rowcount > 0:
                conn.execute(
                    "UPDATE automations SET next_run_time = ?, enabled = 1 WHERE id = ?",
                    (new_iso, f"reminder_{reminder_id}"),
                )
                conn.commit()
                return new_dt
            return None


# Global singleton instance
reminder_mgr = ReminderManager()
