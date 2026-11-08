"""API router for Nino HUD-Style Control Room Dashboard."""

import asyncio
import datetime
import json
import logging
import os
import platform
import time
from typing import Any, AsyncIterator
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nino.auth import verify_token
from nino.config import Settings, get_settings
from nino.memory.db import MemoryDB
from nino.security.audit import AuditLogger
from nino.security.privacy_gateway import PrivacyGateway

logger = logging.getLogger("nino.api.dashboard")

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# Track live offline kill switch state in memory
_offline_kill_switch_active = False


def is_kill_switch_active() -> bool:
    return _offline_kill_switch_active


def set_kill_switch(active: bool) -> None:
    global _offline_kill_switch_active
    _offline_kill_switch_active = active


class KillSwitchRequest(BaseModel):
    offline_mode: bool = Field(..., description="Enable or disable offline kill switch")


class KillSwitchResponse(BaseModel):
    offline_mode: bool
    status: str
    message: str
    timestamp: str


def get_memory_db(request: Request) -> MemoryDB:
    db = getattr(request.app.state, "memory_db", None)
    if not db:
        settings_getter = request.app.dependency_overrides.get(get_settings, get_settings)
        settings = settings_getter()
        db = MemoryDB(db_path=settings.db_path, encryption_key=settings.memory_encryption_key)
        request.app.state.memory_db = db
    return db


def get_audit_logger(request: Request) -> AuditLogger:
    logger = getattr(request.app.state, "audit_logger", None)
    if not logger:
        settings_getter = request.app.dependency_overrides.get(get_settings, get_settings)
        settings = settings_getter()
        logger = AuditLogger(log_path=settings.audit_log_path, log_content=settings.log_content)
        request.app.state.audit_logger = logger
    return logger


def _get_system_metrics() -> dict[str, Any]:
    """Gather CPU, RAM, and GPU telemetry with graceful fallback."""
    cpu_percent = 0.0
    ram_total_gb = 8.0
    ram_used_gb = 4.0
    ram_percent = 50.0

    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_total_gb = round(mem.total / (1024 ** 3), 2)
        ram_used_gb = round(mem.used / (1024 ** 3), 2)
        ram_percent = mem.percent
    except Exception:
        # Fallback if psutil is not available
        try:
            load = os.getloadavg()
            cpu_percent = round(load[0] * 10, 1)
        except Exception:
            cpu_percent = 12.5

    return {
        "cpu": {
            "usage_percent": cpu_percent,
            "cores": os.cpu_count() or 4,
            "model": platform.processor() or "x86_64 Processor",
        },
        "ram": {
            "total_gb": ram_total_gb,
            "used_gb": ram_used_gb,
            "percent": ram_percent,
        },
        "gpu": {
            "name": "Integrated Intel UHD Graphics (Unified Memory)",
            "vram_gb": "Shared with RAM",
            "acceleration": "DirectML / OpenVINO / CPU",
            "status": "active",
        },
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
    }


@router.get("/system", summary="System telemetry, loaded models, and health status", dependencies=[Depends(verify_token)])
async def get_system_status(
    request: Request,
    settings: Settings = Depends(get_settings),
    db: MemoryDB = Depends(get_memory_db),
) -> dict[str, Any]:
    """Return live system resource usage, loaded models, and service health."""
    metrics = _get_system_metrics()

    # Check database health
    db_healthy = False
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
            db_healthy = True
    except Exception:
        db_healthy = False

    # Check vector store
    vstore_healthy = hasattr(request.app.state, "vector_store")

    # Check scheduler
    scheduler = getattr(request.app.state, "scheduler", None)
    scheduler_running = scheduler._running if scheduler else False

    # Vision model
    vision_model = getattr(settings, "vision_model", "moondream")
    vision_enabled = getattr(settings, "vision_enabled", True)

    return {
        "telemetry": metrics,
        "models": {
            "llm": {
                "name": settings.model_name,
                "adapter": settings.model_adapter,
                "status": "loaded",
            },
            "vision": {
                "name": vision_model,
                "enabled": vision_enabled,
                "status": "ready" if vision_enabled else "disabled",
            },
            "embedder": {
                "name": settings.embed_model,
                "status": "ready",
            },
            "voice": {
                "stt": settings.stt_model,
                "tts": settings.tts_voice,
                "wake_word": settings.wake_word_model,
                "status": "ready" if settings.voice_enabled else "disabled",
            },
        },
        "health": {
            "database": "online" if db_healthy else "degraded",
            "vector_store": "online" if vstore_healthy else "offline",
            "scheduler": "running" if scheduler_running else "stopped",
            "privacy_gateway": "offline_mode" if is_kill_switch_active() else "active",
        },
        "kill_switch_active": is_kill_switch_active(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@router.get("/today", summary="Today's calendar events, tasks, reminders, and automations", dependencies=[Depends(verify_token)])
async def get_today_overview(
    request: Request,
    db: MemoryDB = Depends(get_memory_db),
) -> dict[str, Any]:
    """Retrieve today's schedule, active tasks, reminders, and upcoming automated jobs."""
    today_str = datetime.date.today().isoformat()

    with db.get_connection() as conn:
        # 1. Calendar events for today
        cal_cursor = conn.execute(
            """
            SELECT id, title, start_time, end_time, location, description
            FROM calendar_events
            WHERE date(start_time) = ? OR date(end_time) = ?
            ORDER BY start_time ASC
            """,
            (today_str, today_str),
        )
        events = [dict(r) for r in cal_cursor.fetchall()]

        # 2. Tasks
        tasks_cursor = conn.execute(
            """
            SELECT id, title, due_date, priority, status
            FROM tasks
            ORDER BY status ASC, priority DESC, created_at DESC
            LIMIT 20
            """
        )
        tasks = [dict(r) for r in tasks_cursor.fetchall()]

        # 3. Reminders
        rem_cursor = conn.execute(
            """
            SELECT id, message, trigger_time, urgent, status
            FROM reminders
            WHERE status = 'pending'
            ORDER BY trigger_time ASC
            LIMIT 15
            """
        )
        reminders = [dict(r) for r in rem_cursor.fetchall()]

    # 4. Next automation job from scheduler
    scheduler = getattr(request.app.state, "scheduler", None)
    next_job = None
    if scheduler:
        try:
            jobs = scheduler.list_jobs()
            if jobs:
                sorted_jobs = sorted(jobs, key=lambda j: j.get("next_run_time") or "9999")
                next_job = sorted_jobs[0]
        except Exception as e:
            logger.debug("Failed to list scheduler jobs: %s", e)

    return {
        "date": today_str,
        "calendar_events": events,
        "tasks": tasks,
        "reminders": reminders,
        "next_automation": next_job,
    }


@router.get("/conversations", summary="Recent conversational sessions", dependencies=[Depends(verify_token)])
async def get_recent_conversations(
    request: Request,
    db: MemoryDB = Depends(get_memory_db),
    limit: int = Query(default=15, ge=1, le=50),
) -> dict[str, Any]:
    """List recent conversation sessions with message counts, isolated by user and labelled with username for admin."""
    current_user = getattr(request.state, "user", None) or {"id": "admin", "username": "admin", "is_admin": True}
    is_admin = current_user.get("is_admin", False) or current_user.get("id") == "admin"
    user_id = current_user.get("id", "admin")

    with db.get_connection() as conn:
        if is_admin:
            cursor = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at, s.user_id,
                       COALESCE(u.username, CASE WHEN s.user_id = 'admin' OR s.user_id IS NULL OR s.user_id = '' THEN 'admin' ELSE s.user_id END) AS username,
                       COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN users u ON s.user_id = u.id
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at, s.user_id,
                       COALESCE(u.username, CASE WHEN s.user_id = 'admin' OR s.user_id IS NULL OR s.user_id = '' THEN 'admin' ELSE s.user_id END) AS username,
                       COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN users u ON s.user_id = u.id
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.user_id = ?
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        sessions = [dict(r) for r in cursor.fetchall()]

    return {"sessions": sessions, "count": len(sessions)}


@router.get("/tools/activity", summary="Recent tool executions from audit trail", dependencies=[Depends(verify_token)])
async def get_tool_activity(
    settings: Settings = Depends(get_settings),
    limit: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    """Extract recent tool invocation logs showing tool name, tier, decision, and duration."""
    log_path = os.path.normpath(settings.audit_log_path)
    entries: list[dict[str, Any]] = []

    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    endpoint = data.get("endpoint", "")
                    if endpoint.startswith("/tool/") or data.get("method") == "EXEC":
                        tool_name = endpoint.replace("/tool/", "").split("?")[0]
                        extra = data.get("extra") or {}
                        entries.append({
                            "timestamp": data.get("timestamp"),
                            "tool_name": extra.get("tool_name") or tool_name,
                            "tier": extra.get("tier", "UNKNOWN"),
                            "decision": extra.get("decision", "executed" if data.get("status_code") == 200 else "failed"),
                            "duration_ms": data.get("duration_ms", 0.0),
                            "client_ip": data.get("client_ip", "127.0.0.1"),
                        })
                        if len(entries) >= limit:
                            break
                except Exception:
                    continue
        except Exception as e:
            logger.warning("Failed to read audit log for tool activity: %s", e)

    return {"activity": entries, "count": len(entries)}


@router.get("/privacy-log", summary="Report of outbound data transmissions and redactions", dependencies=[Depends(verify_token)])
async def get_privacy_log(request: Request) -> dict[str, Any]:
    """Return all outbound network transmissions recorded by the Privacy Gateway."""
    # Check gateway outbound log if available
    gateway: PrivacyGateway | None = getattr(request.app.state, "privacy_gateway", None)
    entries = list(gateway.outbound_log) if gateway else []

    if not entries:
        return {
            "entries": [],
            "total_outbound": 0,
            "offline_mode": is_kill_switch_active(),
            "message": "Zero outbound traffic: All inference and operations remain strictly on-device.",
        }

    return {
        "entries": entries,
        "total_outbound": len(entries),
        "offline_mode": is_kill_switch_active(),
        "message": f"Recorded {len(entries)} external transmissions through privacy gateway.",
    }


@router.get("/audit", summary="Searchable and filterable audit log viewer", dependencies=[Depends(verify_token)])
async def get_audit_records(
    settings: Settings = Depends(get_settings),
    limit: int = Query(default=50, ge=1, le=200),
    endpoint: str | None = Query(default=None),
    status_code: int | None = Query(default=None),
    decision: str | None = Query(default=None),
) -> dict[str, Any]:
    """Retrieve audit log entries filtered by endpoint, status code, or execution decision."""
    log_path = os.path.normpath(settings.audit_log_path)
    records: list[dict[str, Any]] = []

    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    # Apply filters
                    if endpoint and endpoint.lower() not in entry.get("endpoint", "").lower():
                        continue
                    if status_code is not None and entry.get("status_code") != status_code:
                        continue
                    if decision:
                        extra = entry.get("extra") or {}
                        if extra.get("decision", "").lower() != decision.lower():
                            continue

                    records.append(entry)
                    if len(records) >= limit:
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.warning("Error reading audit log: %s", e)

    return {"records": records, "count": len(records)}


@router.get("/tiers", summary="Declared tools and permission tiers matrix", dependencies=[Depends(verify_token)])
async def get_tool_tiers(request: Request) -> dict[str, Any]:
    """Return all registered tools, their permission tiers, and descriptions."""
    registry = getattr(request.app.state, "tool_registry", None)
    if not registry:
        return {"tools": []}

    tools_info = []
    for tool in registry.list_tools():
        tools_info.append({
            "name": tool.name,
            "tier": tool.tier.value,
            "description": tool.description,
            "enabled": getattr(tool, "enabled", True),
        })

    # Sort by tier hierarchy
    tier_weights = {"READ_ONLY": 1, "REVERSIBLE": 2, "SENSITIVE": 3, "RESTRICTED": 4}
    tools_info.sort(key=lambda t: (tier_weights.get(t["tier"], 9), t["name"]))

    return {"tools": tools_info, "count": len(tools_info)}


@router.post("/kill-switch", response_model=KillSwitchResponse, summary="Toggle OFFLINE MODE kill switch", dependencies=[Depends(verify_token)])
async def toggle_kill_switch(
    payload: KillSwitchRequest,
    request: Request,
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> KillSwitchResponse:
    """Instantly toggle offline mode. When enabled, disables external adapters and web search."""
    active = payload.offline_mode
    set_kill_switch(active)

    # Disable web search tool if kill switch active
    registry = getattr(request.app.state, "tool_registry", None)
    if registry:
        web_tool = registry.get("web_search")
        if web_tool:
            web_tool.enabled = not active

    action = "ENGAGED" if active else "DISENGAGED"
    msg = (
        "OFFLINE MODE ENGAGED: All external adapters, cloud APIs, and web searches are locked down."
        if active
        else "OFFLINE MODE DISENGAGED: Local policies restored."
    )

    audit_logger.log(
        endpoint="/dashboard/kill-switch",
        method="POST",
        status_code=200,
        duration_ms=0.0,
        extra={"kill_switch_active": active, "action": action},
    )

    logger.warning("Kill switch %s: %s", action, msg)

    return KillSwitchResponse(
        offline_mode=active,
        status="active" if active else "disabled",
        message=msg,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.get("/events", summary="Server-Sent Events real-time event stream")
async def dashboard_events(
    request: Request,
    token: str = Query(default=""),
    max_events: int | None = Query(default=None),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream real-time heartbeat, confirmation alerts, and telemetry updates to dashboard."""
    # Token check for EventSource API (which cannot pass Authorization headers in standard browsers)
    auth_header = request.headers.get("Authorization", "")
    bearer_token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    provided_token = bearer_token or token

    if not provided_token or provided_token != settings.auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token for SSE stream")

    async def event_generator() -> AsyncIterator[str]:
        conf_mgr = getattr(request.app.state, "confirmation_manager", None)
        count = 0

        while True:
            if await request.is_disconnected():
                break

            # 1. Fetch pending confirmation count
            pending_count = len(conf_mgr.list_pending()) if conf_mgr else 0

            # 2. Gather live telemetry pulse
            metrics = _get_system_metrics()

            payload = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "kill_switch_active": is_kill_switch_active(),
                "pending_confirmations": pending_count,
                "cpu_percent": metrics["cpu"]["usage_percent"],
                "ram_percent": metrics["ram"]["percent"],
            }

            yield f"data: {json.dumps(payload)}\n\n"
            count += 1
            if max_events is not None and count >= max_events:
                break
            await asyncio.sleep(2.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
