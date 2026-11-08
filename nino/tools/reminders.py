"""Reminder tools: create, list, cancel, and snooze reminders with SQLite persistence."""

from datetime import datetime, timedelta
import logging
from typing import Any, Optional
from pydantic import BaseModel, Field

from nino.memory.db import MemoryDB
from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier

logger = logging.getLogger("nino.tools.reminders")


def parse_future_time(time_str: str) -> datetime:
    """Parse an ISO timestamp, relative minute offset (+15m), or HH:MM string into datetime."""
    clean = time_str.strip()
    now = datetime.now()

    # Relative offset: e.g. "+15m", "+2h", "15m", "10 min"
    if clean.startswith("+") or clean.endswith("m") or "min" in clean:
        digits = "".join(c for c in clean if c.isdigit())
        minutes = int(digits) if digits else 10
        if "h" in clean:
            minutes *= 60
        return now + timedelta(minutes=minutes)

    # Time of day: HH:MM
    if ":" in clean and len(clean) <= 5:
        parts = [int(p) for p in clean.split(":")]
        target = now.replace(hour=parts[0], minute=parts[1], second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    # Try standard ISO format
    try:
        return datetime.fromisoformat(clean)
    except ValueError:
        # Default fallback: 15 minutes from now
        return now + timedelta(minutes=15)


class ReminderCreateArgs(BaseModel):
    message: str = Field(..., description="What Nino should remind you about")
    time: str = Field(..., description="When to remind: ISO string, HH:MM, or relative offset like '+15m'")
    urgent: bool = Field(default=False, description="Whether this is urgent and should break through quiet hours / DND")


class ReminderCreateTool(BaseTool):
    """Create a persistent scheduled reminder."""

    name = "reminder_create"
    description = "Schedule a reminder for a specific time or relative offset."
    tier = PermissionTier.REVERSIBLE
    args_schema = ReminderCreateArgs

    def __init__(self, memory_db: MemoryDB, scheduler: Optional[Any] = None):
        self.db = memory_db
        self.scheduler = scheduler

    async def run(self, message: str, time: str, urgent: bool = False) -> dict[str, Any]:
        trigger_dt = parse_future_time(time)
        trigger_iso = trigger_dt.isoformat()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reminders (message, trigger_time, urgent, status)
                VALUES (?, ?, ?, 'pending')
                """,
                (message, trigger_iso, 1 if urgent else 0),
            )
            reminder_id = cursor.lastrowid

        # Schedule job in scheduler if present
        if self.scheduler:
            self.scheduler.add_job(
                job_id=f"reminder_{reminder_id}",
                name=f"Reminder #{reminder_id}",
                job_type="reminder",
                next_run_time=trigger_dt,
                kwargs={
                    "reminder_id": reminder_id,
                    "message": message,
                    "urgent": urgent,
                },
            )

        return {
            "status": "created",
            "reminder_id": reminder_id,
            "message": message,
            "trigger_time": trigger_iso,
            "urgent": urgent,
        }


class ReminderListArgs(BaseModel):
    status: Optional[str] = Field(default="pending", description="Filter by status: 'pending', 'all', 'delivered'")


class ReminderListTool(BaseTool):
    """List pending or past reminders."""

    name = "reminder_list"
    description = "List scheduled reminders."
    tier = PermissionTier.READ_ONLY
    args_schema = ReminderListArgs

    def __init__(self, memory_db: MemoryDB):
        self.db = memory_db

    async def run(self, status: Optional[str] = "pending") -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            if status and status != "all":
                cursor = conn.execute(
                    "SELECT id, message, trigger_time, urgent, status FROM reminders WHERE status = ? ORDER BY trigger_time ASC",
                    (status,),
                )
            else:
                cursor = conn.execute(
                    "SELECT id, message, trigger_time, urgent, status FROM reminders ORDER BY trigger_time ASC"
                )
            return [dict(r) for r in cursor.fetchall()]


class ReminderCancelArgs(BaseModel):
    reminder_id: int = Field(..., description="ID of the reminder to cancel")


class ReminderCancelTool(BaseTool):
    """Cancel a pending reminder."""

    name = "reminder_cancel"
    description = "Cancel a scheduled reminder."
    tier = PermissionTier.REVERSIBLE
    args_schema = ReminderCancelArgs

    def __init__(self, memory_db: MemoryDB, scheduler: Optional[Any] = None):
        self.db = memory_db
        self.scheduler = scheduler

    async def run(self, reminder_id: int) -> dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ?",
                (reminder_id,),
            )
            found = cursor.rowcount > 0

        if self.scheduler:
            self.scheduler.remove_job(f"reminder_{reminder_id}")

        if not found:
            return {"status": "not_found", "reminder_id": reminder_id}
        return {"status": "cancelled", "reminder_id": reminder_id}


class ReminderSnoozeArgs(BaseModel):
    reminder_id: int = Field(..., description="ID of the reminder to snooze")
    minutes: int = Field(default=15, description="Number of minutes to snooze")


class ReminderSnoozeTool(BaseTool):
    """Snooze an existing reminder."""

    name = "reminder_snooze"
    description = "Snooze a reminder by a number of minutes."
    tier = PermissionTier.REVERSIBLE
    args_schema = ReminderSnoozeArgs

    def __init__(self, memory_db: MemoryDB, scheduler: Optional[Any] = None):
        self.db = memory_db
        self.scheduler = scheduler

    async def run(self, reminder_id: int, minutes: int = 15) -> dict[str, Any]:
        new_time = datetime.now() + timedelta(minutes=minutes)
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET trigger_time = ?, status = 'pending' WHERE id = ?",
                (new_time.isoformat(), reminder_id),
            )
            found = cursor.rowcount > 0

        if self.scheduler:
            self.scheduler.snooze_job(f"reminder_{reminder_id}", new_time)

        if not found:
            return {"status": "not_found", "reminder_id": reminder_id}
        return {"status": "snoozed", "reminder_id": reminder_id, "new_trigger_time": new_time.isoformat()}
