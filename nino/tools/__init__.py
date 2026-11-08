"""Tooling and execution framework package for Nino."""

from typing import Any

from nino.config import Settings
from nino.memory.db import MemoryDB
from nino.tools.base import BaseTool
from nino.tools.calendar import (
    CalendarCreateEventTool,
    CalendarDeleteEventTool,
    CalendarExportICSTool,
    CalendarImportICSTool,
    CalendarListEventsTool,
    CalendarUpdateEventTool,
)
from nino.tools.code_execution import PythonCodeExecutionTool
from nino.tools.confirmation import Confirmation, ConfirmationManager
from nino.tools.executor import CentralExecutor, ToolExecutionResult
from nino.tools.files import FileReadTool, FileSearchTool
from nino.tools.notes import (
    NoteAppendTool,
    NoteCreateTool,
    NoteReadTool,
    NoteSearchTool,
)
from nino.tools.orchestrator import ToolOrchestrator
from nino.tools.permissions import PermissionTier
from nino.tools.registry import ToolRegistry
from nino.tools.taint import TaintTracker
from nino.tools.tasks import (
    TaskCompleteTool,
    TaskCreateTool,
    TaskDeleteTool,
    TaskListTool,
)
from nino.tools.utilities import CalculatorTool, CurrentTimeTool


def build_default_registry(
    memory_db: MemoryDB,
    settings: Settings,
    fact_manager: Any = None,
) -> ToolRegistry:
    """Instantiate and register all default tools with system configuration."""
    registry = ToolRegistry()

    # Memory Tools (Durable Facts)
    if fact_manager is not None:
        from nino.tools.memory_tools import (
            ForgetFactTool,
            RecallFactsTool,
            RememberFactTool,
        )
        registry.register(RememberFactTool(fact_manager))
        registry.register(RecallFactsTool(fact_manager))
        registry.register(ForgetFactTool(fact_manager))

    # Notes
    registry.register(NoteCreateTool(memory_db))
    registry.register(NoteSearchTool(memory_db))
    registry.register(NoteReadTool(memory_db))
    registry.register(NoteAppendTool(memory_db))

    # Tasks
    registry.register(TaskCreateTool(memory_db))
    registry.register(TaskListTool(memory_db))
    registry.register(TaskCompleteTool(memory_db))
    registry.register(TaskDeleteTool(memory_db))

    # Calendar
    registry.register(CalendarCreateEventTool(memory_db))
    registry.register(CalendarListEventsTool(memory_db))
    registry.register(CalendarUpdateEventTool(memory_db))
    registry.register(CalendarDeleteEventTool(memory_db))
    registry.register(CalendarExportICSTool(memory_db))
    registry.register(CalendarImportICSTool(memory_db))

    # Files
    registry.register(FileSearchTool(settings.allowed_dirs))
    registry.register(FileReadTool(settings.allowed_dirs))

    # Code execution
    registry.register(
        PythonCodeExecutionTool(
            enabled=settings.enable_code_execution,
            timeout_seconds=settings.code_execution_timeout,
        )
    )

    # Utilities
    registry.register(
        CurrentTimeTool(
            default_timezone=getattr(settings, "user_timezone", "Asia/Kolkata"),
            location=getattr(settings, "default_city", "Vadakara, Calicut, Kerala, India") or "Vadakara, Calicut, Kerala, India",
        )
    )
    registry.register(CalculatorTool())

    # GPS Location & Weather Forecast
    from nino.tools.weather import GetLocationTool, GetWeatherForecastTool, SetLocationTool
    registry.register(
        GetLocationTool(
            default_lat=getattr(settings, "default_latitude", None),
            default_lon=getattr(settings, "default_longitude", None),
            default_city=getattr(settings, "default_city", None),
            memory_db=memory_db,
        )
    )
    registry.register(
        SetLocationTool(
            memory_db=memory_db,
        )
    )
    registry.register(
        GetWeatherForecastTool(
            default_lat=getattr(settings, "default_latitude", None),
            default_lon=getattr(settings, "default_longitude", None),
            default_city=getattr(settings, "default_city", None),
            memory_db=memory_db,
        )
    )


    # Reminders (Phase 5)
    from nino.tools.reminders import (
        ReminderCancelTool,
        ReminderCreateTool,
        ReminderListTool,
        ReminderSnoozeTool,
    )
    registry.register(ReminderCreateTool(memory_db, scheduler=getattr(settings, "_scheduler", None)))
    registry.register(ReminderListTool(memory_db))
    registry.register(ReminderCancelTool(memory_db, scheduler=getattr(settings, "_scheduler", None)))
    registry.register(ReminderSnoozeTool(memory_db, scheduler=getattr(settings, "_scheduler", None)))

    # Focus Sessions (Phase 5)
    from nino.tools.focus import FocusStartTool, FocusStatusTool, FocusStopTool, get_focus_manager
    focus_mgr = get_focus_manager(scheduler=getattr(settings, "_scheduler", None))
    registry.register(FocusStartTool(focus_mgr))
    registry.register(FocusStopTool(focus_mgr))
    registry.register(FocusStatusTool(focus_mgr))

    # Home Assistant Smart Home (Phase 5)
    from nino.integrations.homeassistant import (
        HAGetStateTool,
        HAListEntitiesTool,
        HASecurityControlTool,
        HASetValueTool,
        HATurnOffTool,
        HATurnOnTool,
        HomeAssistantClient,
    )
    ha_client = HomeAssistantClient(settings)
    registry.register(HAListEntitiesTool(ha_client))
    registry.register(HAGetStateTool(ha_client))
    registry.register(HATurnOnTool(ha_client))
    registry.register(HATurnOffTool(ha_client))
    registry.register(HASetValueTool(ha_client))
    registry.register(HASecurityControlTool(ha_client, enabled=settings.ha_enable_restricted))

    # Vision & Camera (Phase 6A)
    if getattr(settings, "vision_enabled", True):
        from nino.vision.ollama import OllamaVisionAdapter
        from nino.tools.vision import (
            ScreenshotCaptureTool,
            CameraSnapshotTool,
            ImageDescribeTool,
        )
        vision_adapter = getattr(settings, "_vision_adapter", None) or OllamaVisionAdapter(
            base_url=settings.ollama_base_url,
            model=settings.vision_model,
        )
        audit_logger = getattr(settings, "_audit_logger", None)
        registry.register(ScreenshotCaptureTool(vision_adapter, audit_logger=audit_logger))
        registry.register(CameraSnapshotTool(vision_adapter, audit_logger=audit_logger))
        registry.register(ImageDescribeTool(vision_adapter, allowed_dirs=settings.allowed_dirs, audit_logger=audit_logger))

    return registry


__all__ = [
    "PermissionTier",
    "BaseTool",
    "ToolRegistry",
    "ConfirmationManager",
    "Confirmation",
    "TaintTracker",
    "CentralExecutor",
    "ToolExecutionResult",
    "ToolOrchestrator",
    "build_default_registry",
]
