"""Unit tests for initial tools: notes, tasks, calendar, and utilities."""

import pytest

from nino.memory.db import MemoryDB
from nino.tools.calendar import (
    CalendarCreateEventTool,
    CalendarDeleteEventTool,
    CalendarExportICSTool,
    CalendarImportICSTool,
    CalendarListEventsTool,
    CalendarUpdateEventTool,
)
from nino.tools.notes import (
    NoteAppendTool,
    NoteCreateTool,
    NoteReadTool,
    NoteSearchTool,
)
from nino.tools.tasks import (
    TaskCompleteTool,
    TaskCreateTool,
    TaskDeleteTool,
    TaskListTool,
)
from nino.tools.utilities import CalculatorTool, CurrentTimeTool


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_initial_tools.db"
    return MemoryDB(str(db_file))


@pytest.mark.asyncio
async def test_notes_tools(test_db):
    create_tool = NoteCreateTool(test_db)
    search_tool = NoteSearchTool(test_db)
    read_tool = NoteReadTool(test_db)
    append_tool = NoteAppendTool(test_db)

    # 1. Create
    res_create = await create_tool.run(
        title="Project Apollo",
        content="Flight plans to the Moon.",
        tags=["space", "apollo"],
    )
    assert res_create["status"] == "created"
    note_id = res_create["note_id"]

    # 2. Search
    res_search = await search_tool.run(query="Apollo")
    assert "Project Apollo" in res_search
    assert 'taint="UNTRUSTED"' in res_search

    # 3. Read
    res_read = await read_tool.run(note_id=note_id)
    assert "Flight plans to the Moon." in res_read
    assert 'taint="UNTRUSTED"' in res_read

    # 4. Append
    res_append = await append_tool.run(note_id=note_id, content="Updated with lunar module details.")
    assert res_append["status"] == "updated"

    # Verify append
    res_read_updated = await read_tool.run(note_id=note_id)
    assert "lunar module details" in res_read_updated


@pytest.mark.asyncio
async def test_tasks_tools(test_db):
    create_tool = TaskCreateTool(test_db)
    list_tool = TaskListTool(test_db)
    comp_tool = TaskCompleteTool(test_db)
    del_tool = TaskDeleteTool(test_db)

    # 1. Create tasks
    res1 = await create_tool.run(title="Review security architecture", priority="high")
    res2 = await create_tool.run(title="Buy coffee beans", due_date="2026-10-01")
    task1_id = res1["task_id"]
    task2_id = res2["task_id"]

    # 2. List pending
    list_res = await list_tool.run(status="pending")
    assert "Review security architecture" in list_res
    assert "Buy coffee beans" in list_res

    # 3. Complete task 1
    res_comp = await comp_tool.run(task_id=task1_id)
    assert res_comp["status"] == "completed"

    # Verify list pending shows only task 2
    list_after = await list_tool.run(status="pending")
    assert "Buy coffee beans" in list_after
    assert "Review security architecture" not in list_after

    # 4. Delete task 2
    res_del = await del_tool.run(task_id=task2_id)
    assert res_del["status"] == "deleted"

    # List all
    list_all = await list_tool.run(status="all")
    assert "Buy coffee beans" not in list_all


@pytest.mark.asyncio
async def test_calendar_tools_and_ics(test_db):
    create_tool = CalendarCreateEventTool(test_db)
    list_tool = CalendarListEventsTool(test_db)
    update_tool = CalendarUpdateEventTool(test_db)
    del_tool = CalendarDeleteEventTool(test_db)
    export_tool = CalendarExportICSTool(test_db)
    import_tool = CalendarImportICSTool(test_db)

    # 1. Create event
    res_create = await create_tool.run(
        title="Nino Architecture Review",
        start_time="2026-10-05 10:00",
        end_time="2026-10-05 11:00",
        location="Conference Room A",
        description="Review Phase 3 tools framework.",
    )
    assert res_create["status"] == "created"
    event_id = res_create["event_id"]

    # 2. List
    list_res = await list_tool.run()
    assert "Nino Architecture Review" in list_res
    assert "Conference Room A" in list_res

    # 3. Update
    res_update = await update_tool.run(event_id=event_id, location="Virtual Meeting")
    assert res_update["status"] == "updated"

    # 4. Export ICS
    ics_str = await export_tool.run()
    assert "BEGIN:VCALENDAR" in ics_str
    assert "BEGIN:VEVENT" in ics_str
    assert "SUMMARY:Nino Architecture Review" in ics_str
    assert "LOCATION:Virtual Meeting" in ics_str
    assert "END:VCALENDAR" in ics_str

    # 5. Import ICS
    sample_ics = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "BEGIN:VEVENT\r\n"
        "SUMMARY:Imported Standup\r\n"
        "DTSTART:20261006T090000Z\r\n"
        "DTEND:20261006T093000Z\r\n"
        "DESCRIPTION:Daily sync\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR"
    )
    import_res = await import_tool.run(ics_content=sample_ics)
    assert import_res["status"] == "imported"
    assert import_res["events_imported"] == 1

    # Verify imported event in list
    list_after_import = await list_tool.run()
    assert "Imported Standup" in list_after_import

    # 6. Delete
    res_del = await del_tool.run(event_id=event_id)
    assert res_del["status"] == "deleted"


@pytest.mark.asyncio
async def test_utilities_tools():
    time_tool = CurrentTimeTool()
    calc_tool = CalculatorTool()

    # Time tool
    time_res = await time_tool.run()
    assert "Current UTC Time:" in time_res
    assert "Current Local Time:" in time_res

    # Safe calculator: valid arithmetic
    calc_1 = await calc_tool.run("((10 + 5) * 3) / 5")
    assert calc_1 == "Result: 9.0"

    # Math functions
    calc_2 = await calc_tool.run("sqrt(144) + pow(2, 3)")
    assert calc_2 == "Result: 20.0"

    # Safe calculator: blocked unsafe operations
    calc_hack = await calc_tool.run("__import__('os').system('ls')")
    assert "Calculation error" in calc_hack

    # Blocked variable assignment
    calc_assign = await calc_tool.run("x = 10")
    assert "Calculation error" in calc_assign

    # Blocked excessively large exponent
    calc_big = await calc_tool.run("9999 ** 9999")
    assert "Calculation error" in calc_big
