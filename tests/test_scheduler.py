"""Unit tests for SQLiteJobScheduler using controllable clock and database persistence."""

from datetime import datetime, timedelta
import pytest

from nino.automation.notifications import NotificationChannel, NotificationDispatcher
from nino.automation.policy import ProactivePolicyEngine
from nino.automation.scheduler import SQLiteJobScheduler
from nino.config import Settings
from nino.memory.db import MemoryDB


class ControllableClock:
    def __init__(self, start_dt: datetime):
        self.current_dt = start_dt

    def __call__(self) -> datetime:
        return self.current_dt

    def advance(self, minutes: int):
        self.current_dt += timedelta(minutes=minutes)


class MockChannel(NotificationChannel):
    def __init__(self):
        self.delivered_messages = []

    @property
    def name(self) -> str:
        return "mock"

    async def send(self, message: str, title: str = "Nino Notification", is_urgent: bool = False) -> bool:
        self.delivered_messages.append({"title": title, "message": message, "urgent": is_urgent})
        return True


@pytest.fixture
def scheduler_fixture(tmp_path):
    db_file = tmp_path / "test_scheduler.db"
    db = MemoryDB(str(db_file))
    settings = Settings(proactive_enabled=True, quiet_hours_start="23:00", quiet_hours_end="07:00")
    clock = ControllableClock(datetime(2026, 9, 21, 9, 0, 0))
    policy = ProactivePolicyEngine(db, settings, clock_fn=clock)
    mock_channel = MockChannel()
    dispatcher = NotificationDispatcher([mock_channel])

    scheduler = SQLiteJobScheduler(
        memory_db=db,
        settings=settings,
        policy_engine=policy,
        dispatcher=dispatcher,
        clock_fn=clock,
    )
    return scheduler, clock, mock_channel, db, settings, policy


@pytest.mark.asyncio
async def test_scheduler_persistence_and_job_execution(scheduler_fixture):
    scheduler, clock, mock_channel, db, settings, policy = scheduler_fixture

    # Schedule a reminder for 15 minutes in the future (9:15)
    run_time = clock() + timedelta(minutes=15)
    scheduler.add_job(
        job_id="test_reminder_1",
        name="Team Sync Reminder",
        job_type="reminder",
        next_run_time=run_time,
        kwargs={"message": "Team sync starts now.", "urgent": False},
    )

    # 1. At 9:00: nothing is due
    executed_early = await scheduler.trigger_due_jobs()
    assert len(executed_early) == 0
    assert len(mock_channel.delivered_messages) == 0

    # 2. Advance clock to 9:10: still not due
    clock.advance(10)
    executed_mid = await scheduler.trigger_due_jobs()
    assert len(executed_mid) == 0

    # 3. Simulate process restart by creating a new scheduler instance on the same SQLite DB
    new_scheduler = SQLiteJobScheduler(
        memory_db=db,
        settings=settings,
        policy_engine=policy,
        dispatcher=NotificationDispatcher([mock_channel]),
        clock_fn=clock,
    )
    jobs = new_scheduler.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["id"] == "test_reminder_1"

    # 4. Advance clock to 9:15: job triggers!
    clock.advance(5)
    executed = await new_scheduler.trigger_due_jobs()
    assert len(executed) == 1
    assert executed[0]["job_id"] == "test_reminder_1"
    assert len(mock_channel.delivered_messages) == 1
    assert "Team sync starts now." in mock_channel.delivered_messages[0]["message"]

    # 5. Check one-shot job is now disabled in DB
    reloaded_job = new_scheduler.get_job("test_reminder_1")
    assert reloaded_job["enabled"] == 0


@pytest.mark.asyncio
async def test_scheduler_snooze_and_remove(scheduler_fixture):
    scheduler, clock, mock_channel, _, _, _ = scheduler_fixture

    run_time = clock() + timedelta(minutes=10)
    scheduler.add_job(
        job_id="test_snooze_job",
        name="Task Reminder",
        job_type="reminder",
        next_run_time=run_time,
        kwargs={"message": "Review PR.", "urgent": False},
    )

    # Snooze until 9:30
    snooze_until = clock() + timedelta(minutes=30)
    scheduler.snooze_job("test_snooze_job", snooze_until)

    # At 9:15: trigger_due_jobs should skip snoozed job
    clock.advance(15)
    executed = await scheduler.trigger_due_jobs()
    assert len(executed) == 0

    # At 9:30: snoozed job fires
    clock.advance(15)
    executed_snoozed = await scheduler.trigger_due_jobs()
    assert len(executed_snoozed) == 1

    # Remove job
    removed = scheduler.remove_job("test_snooze_job")
    assert removed is True
    assert scheduler.get_job("test_snooze_job") is None
