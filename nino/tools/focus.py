"""Pomodoro-style focus timer tools: start, stop, status."""

from datetime import datetime, timedelta
import logging
from typing import Any, Optional
from pydantic import BaseModel, Field

from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier

logger = logging.getLogger("nino.tools.focus")


class FocusSessionManager:
    """Manages active focus timer session state."""

    def __init__(self, scheduler: Optional[Any] = None):
        self.scheduler = scheduler
        self.active_session: Optional[dict[str, Any]] = None

    def start(self, task: str, duration_minutes: int) -> dict[str, Any]:
        now = datetime.now()
        end_time = now + timedelta(minutes=duration_minutes)
        self.active_session = {
            "task": task,
            "duration_minutes": duration_minutes,
            "started_at": now.isoformat(),
            "ends_at": end_time.isoformat(),
        }

        if self.scheduler:
            self.scheduler.add_job(
                job_id="active_focus_session",
                name="Focus Session",
                job_type="focus_session",
                next_run_time=end_time,
                kwargs={
                    "task": task,
                    "action": "completed",
                    "urgent": False,
                },
            )
        return self.active_session

    def stop(self) -> Optional[dict[str, Any]]:
        if not self.active_session:
            return None
        session = self.active_session
        self.active_session = None
        if self.scheduler:
            self.scheduler.remove_job("active_focus_session")
        return session

    def status(self) -> dict[str, Any]:
        if not self.active_session:
            return {"active": False, "message": "No focus session currently active."}

        ends_at = datetime.fromisoformat(self.active_session["ends_at"])
        now = datetime.now()
        remaining_sec = max(0, int((ends_at - now).total_seconds()))
        remaining_min = remaining_sec // 60

        return {
            "active": True,
            "task": self.active_session["task"],
            "remaining_minutes": remaining_min,
            "remaining_seconds": remaining_sec,
            "ends_at": self.active_session["ends_at"],
        }


# Singleton manager instance
_GLOBAL_FOCUS_MGR: Optional[FocusSessionManager] = None


def get_focus_manager(scheduler: Optional[Any] = None) -> FocusSessionManager:
    global _GLOBAL_FOCUS_MGR
    if _GLOBAL_FOCUS_MGR is None:
        _GLOBAL_FOCUS_MGR = FocusSessionManager(scheduler)
    elif scheduler and not _GLOBAL_FOCUS_MGR.scheduler:
        _GLOBAL_FOCUS_MGR.scheduler = scheduler
    return _GLOBAL_FOCUS_MGR


class FocusStartArgs(BaseModel):
    duration_minutes: int = Field(default=25, description="Focus duration in minutes (default 25)")
    task: str = Field(default="Focus Session", description="Description of the task or project")


class FocusStartTool(BaseTool):
    """Start a Pomodoro-style focus session."""

    name = "focus_start"
    description = "Start a timed focus session with an announcement at start and finish."
    tier = PermissionTier.REVERSIBLE
    args_schema = FocusStartArgs

    def __init__(self, manager: Optional[FocusSessionManager] = None):
        self.manager = manager or get_focus_manager()

    async def run(self, duration_minutes: int = 25, task: str = "Focus Session") -> dict[str, Any]:
        session = self.manager.start(task, duration_minutes)
        return {
            "status": "started",
            "task": task,
            "duration_minutes": duration_minutes,
            "ends_at": session["ends_at"],
            "message": f"Focus session for '{task}' started ({duration_minutes} minutes). I'll alert you when it wraps up.",
        }


class FocusStopTool(BaseTool):
    """Stop the active focus session."""

    name = "focus_stop"
    description = "Stop or cancel the active focus session."
    tier = PermissionTier.REVERSIBLE

    def __init__(self, manager: Optional[FocusSessionManager] = None):
        self.manager = manager or get_focus_manager()

    async def run(self) -> dict[str, Any]:
        session = self.manager.stop()
        if not session:
            return {"status": "none_active", "message": "No active focus session to stop."}
        return {"status": "stopped", "task": session["task"], "message": f"Focus session for '{session['task']}' stopped."}


class FocusStatusTool(BaseTool):
    """Check remaining time on the current focus session."""

    name = "focus_status"
    description = "Check status and remaining minutes of the current focus session."
    tier = PermissionTier.READ_ONLY

    def __init__(self, manager: Optional[FocusSessionManager] = None):
        self.manager = manager or get_focus_manager()

    async def run(self) -> dict[str, Any]:
        return self.manager.status()
