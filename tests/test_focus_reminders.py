"""Unit tests for Reminders and Focus Session tools."""

import pytest

from nino.memory.db import MemoryDB
from nino.tools.focus import FocusSessionManager, FocusStartTool, FocusStatusTool, FocusStopTool
from nino.tools.reminders import (
    ReminderCancelTool,
    ReminderCreateTool,
    ReminderListTool,
    ReminderSnoozeTool,
    parse_future_time,
)


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_reminders.db"
    return MemoryDB(str(db_file))


def test_time_parsing():
    """Verify parsing relative offsets and time strings."""
    t_15m = parse_future_time("+15m")
    assert t_15m is not None

    t_1h = parse_future_time("1h")
    assert t_1h is not None

    t_hhmm = parse_future_time("14:30")
    assert t_hhmm is not None


@pytest.mark.asyncio
async def test_reminder_tools_lifecycle(test_db):
    create_tool = ReminderCreateTool(test_db)
    list_tool = ReminderListTool(test_db)
    cancel_tool = ReminderCancelTool(test_db)
    snooze_tool = ReminderSnoozeTool(test_db)

    # 1. Create reminder
    created = await create_tool.run(message="Take vitamins", time="+30m", urgent=True)
    assert created["status"] == "created"
    assert created["urgent"] is True
    reminder_id = created["reminder_id"]

    # 2. List reminders
    items = await list_tool.run(status="pending")
    assert len(items) == 1
    assert items[0]["id"] == reminder_id
    assert items[0]["message"] == "Take vitamins"

    # 3. Snooze reminder
    snoozed = await snooze_tool.run(reminder_id=reminder_id, minutes=20)
    assert snoozed["status"] == "snoozed"

    # 4. Cancel reminder
    cancelled = await cancel_tool.run(reminder_id=reminder_id)
    assert cancelled["status"] == "cancelled"

    # 5. List pending should now be empty
    pending = await list_tool.run(status="pending")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_focus_session_tools():
    manager = FocusSessionManager()
    start_tool = FocusStartTool(manager)
    status_tool = FocusStatusTool(manager)
    stop_tool = FocusStopTool(manager)

    # Initial status
    stat_init = await status_tool.run()
    assert stat_init["active"] is False

    # Start 25-minute Pomodoro
    started = await start_tool.run(duration_minutes=25, task="Refactor Database")
    assert started["status"] == "started"
    assert started["duration_minutes"] == 25

    # Check active status
    stat_active = await status_tool.run()
    assert stat_active["active"] is True
    assert stat_active["task"] == "Refactor Database"
    assert stat_active["remaining_minutes"] >= 24

    # Stop session
    stopped = await stop_tool.run()
    assert stopped["status"] == "stopped"

    stat_after = await status_tool.run()
    assert stat_after["active"] is False
