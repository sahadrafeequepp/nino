"""API endpoints for managing automations, reminders, DND, and proactive messages."""

from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from nino.auth import verify_token
from nino.automation.briefing import DailyBriefingGenerator
from nino.automation.policy import ProactivePolicyEngine
from nino.automation.scheduler import SQLiteJobScheduler

router = APIRouter(prefix="", tags=["automations"], dependencies=[Depends(verify_token)])


class SnoozeRequest(BaseModel):
    job_id: Optional[str] = Field(default=None, description="Automation job ID to snooze")
    message_type: Optional[str] = Field(default=None, description="Proactive message category to snooze")
    duration_minutes: int = Field(default=60, description="Duration in minutes to snooze")


class DNDRequest(BaseModel):
    enabled: bool = Field(..., description="Whether Do Not Disturb mode is enabled")


def get_scheduler(request: Request) -> SQLiteJobScheduler:
    scheduler = getattr(request.app.state, "scheduler", None)
    if not scheduler:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler service is not initialized.",
        )
    return scheduler


def get_policy(request: Request) -> ProactivePolicyEngine:
    policy = getattr(request.app.state, "proactive_policy", None)
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Proactive policy engine is not initialized.",
        )
    return policy


@router.get("/automations", summary="List all active and scheduled automations")
async def list_automations(scheduler: SQLiteJobScheduler = Depends(get_scheduler)) -> list[dict[str, Any]]:
    """List all scheduled jobs, reminders, and recurring briefings."""
    return scheduler.list_jobs()


@router.delete("/automations/{automation_id}", summary="Remove a scheduled automation")
async def delete_automation(
    automation_id: str,
    scheduler: SQLiteJobScheduler = Depends(get_scheduler),
) -> dict[str, Any]:
    """Cancel and delete an automation by ID."""
    removed = scheduler.remove_job(automation_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Automation '{automation_id}' not found.")
    return {"status": "deleted", "id": automation_id}


@router.post("/automations/snooze", summary="Snooze an automation job or message category")
async def snooze_automation(
    payload: SnoozeRequest,
    scheduler: SQLiteJobScheduler = Depends(get_scheduler),
    policy: ProactivePolicyEngine = Depends(get_policy),
) -> dict[str, Any]:
    """Snooze a specific automation job or an entire proactive message type."""
    results = {}
    if payload.job_id:
        from datetime import datetime, timedelta
        until = policy.now() + timedelta(minutes=payload.duration_minutes)
        snoozed = scheduler.snooze_job(payload.job_id, until)
        results["job_snoozed"] = snoozed
        results["job_snoozed_until"] = until.isoformat()

    if payload.message_type:
        until_type = policy.snooze_type(payload.message_type, payload.duration_minutes)
        results["type_snoozed"] = payload.message_type
        results["type_snoozed_until"] = until_type.isoformat()

    return {"status": "ok", "results": results}


@router.post("/proactive/dnd", summary="Toggle Do Not Disturb mode")
async def set_dnd(
    payload: DNDRequest,
    policy: ProactivePolicyEngine = Depends(get_policy),
) -> dict[str, Any]:
    """Enable or disable Do Not Disturb mode. When active, only urgent reminders are announced."""
    policy.set_dnd(payload.enabled)
    return {"status": "ok", "dnd_enabled": payload.enabled}


@router.get("/proactive/status", summary="Check proactive messaging and quiet-hours status")
async def get_proactive_status(policy: ProactivePolicyEngine = Depends(get_policy)) -> dict[str, Any]:
    """Inspect current DND, quiet-hours state, and rate-limit budgets."""
    return {
        "proactive_enabled": policy.settings.proactive_enabled,
        "dnd_active": policy.is_dnd_active(),
        "quiet_hours_active": policy.is_in_quiet_hours(),
        "quiet_hours_window": f"{policy.settings.quiet_hours_start} - {policy.settings.quiet_hours_end}",
        "hourly_messages_delivered": policy.get_hourly_message_count(),
        "hourly_limit": policy.settings.max_proactive_per_hour,
        "daily_messages_delivered": policy.get_daily_message_count(),
        "daily_limit": policy.settings.max_proactive_per_day,
    }


@router.post("/briefing/trigger", summary="Generate and return the daily briefing immediately")
async def trigger_briefing(
    request: Request,
    policy: ProactivePolicyEngine = Depends(get_policy),
) -> dict[str, Any]:
    """Immediately compile today's agenda, calendar events, tasks, and weather for inspection."""
    briefing_gen = getattr(request.app.state, "briefing_generator", None) or DailyBriefingGenerator(
        memory_db=request.app.state.memory_db,
        settings=request.app.state.settings,
        adapter=getattr(request.app.state, "adapter", None),
    )
    briefing_text = await briefing_gen.generate_briefing()
    sanitized = policy.sanitize_message_content(briefing_text)
    return {
        "status": "generated",
        "briefing": sanitized,
    }
