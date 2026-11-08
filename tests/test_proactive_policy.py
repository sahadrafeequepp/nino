"""Unit tests for ProactivePolicyEngine: rate limiting, quiet hours, DND, snooze, and tone safety."""

from datetime import datetime
import pytest

from nino.automation.policy import ProactivePolicyEngine
from nino.config import Settings
from nino.memory.db import MemoryDB


class ControllableClock:
    def __init__(self, start_dt: datetime):
        self.current_dt = start_dt

    def __call__(self) -> datetime:
        return self.current_dt

    def set(self, dt: datetime):
        self.current_dt = dt


@pytest.fixture
def policy_fixture(tmp_path):
    db_file = tmp_path / "test_policy.db"
    db = MemoryDB(str(db_file))
    settings = Settings(
        proactive_enabled=True,
        max_proactive_per_hour=2,
        max_proactive_per_day=5,
        quiet_hours_start="22:00",
        quiet_hours_end="08:00",
    )
    # Start at 10:00 AM on a weekday
    clock = ControllableClock(datetime(2026, 9, 21, 10, 0, 0))
    engine = ProactivePolicyEngine(memory_db=db, settings=settings, clock_fn=clock)
    return engine, clock, db


def test_quiet_hours_suppression_and_urgent_bypass(policy_fixture):
    engine, clock, _ = policy_fixture

    # 10:00 AM is NOT in quiet hours (22:00 to 08:00)
    assert engine.is_in_quiet_hours() is False
    allowed, _ = engine.can_deliver_proactive_message(message_type="reminder", is_urgent=False)
    assert allowed is True

    # Move time to 23:30 (Quiet hours!)
    clock.set(datetime(2026, 9, 21, 23, 30, 0))
    assert engine.is_in_quiet_hours() is True

    # Non-urgent message must be suppressed
    allowed_norm, reason_norm = engine.can_deliver_proactive_message(message_type="reminder", is_urgent=False)
    assert allowed_norm is False
    assert "Quiet hours" in reason_norm

    # Urgent reminder created by user MUST break through quiet hours
    allowed_urg, reason_urg = engine.can_deliver_proactive_message(message_type="reminder", is_urgent=True)
    assert allowed_urg is True
    assert "Urgent reminder breaking through" in reason_urg


def test_dnd_mode(policy_fixture):
    engine, clock, _ = policy_fixture
    assert engine.is_dnd_active() is False

    engine.set_dnd(True)
    assert engine.is_dnd_active() is True

    # Non-urgent blocked
    allowed, reason = engine.can_deliver_proactive_message("daily_briefing", is_urgent=False)
    assert allowed is False
    assert "Do Not Disturb" in reason

    # Urgent allowed
    allowed_urg, _ = engine.can_deliver_proactive_message("reminder", is_urgent=True)
    assert allowed_urg is True

    engine.set_dnd(False)
    assert engine.is_dnd_active() is False


def test_rate_limiting_hourly_and_daily(policy_fixture):
    engine, clock, _ = policy_fixture
    # Max per hour is 2
    engine.record_delivery("reminder", "Test 1", "console")
    engine.record_delivery("reminder", "Test 2", "console")

    assert engine.get_hourly_message_count() == 2

    # Third message within the hour should be blocked by rate limit
    allowed, reason = engine.can_deliver_proactive_message("reminder", is_urgent=False)
    assert allowed is False
    assert "Hourly proactive rate limit reached" in reason

    # But an urgent reminder bypasses rate limits
    allowed_urg, _ = engine.can_deliver_proactive_message("reminder", is_urgent=True)
    assert allowed_urg is True


def test_snooze_and_disable_message_type(policy_fixture):
    engine, clock, _ = policy_fixture

    # Snooze daily briefing for 30 minutes
    engine.snooze_type("daily_briefing", duration_minutes=30)
    assert engine.is_type_snoozed("daily_briefing") is True

    allowed, reason = engine.can_deliver_proactive_message("daily_briefing")
    assert allowed is False
    assert "snoozed" in reason

    # Fast forward clock 35 minutes
    clock.set(datetime(2026, 9, 21, 10, 35, 0))
    assert engine.is_type_snoozed("daily_briefing") is False

    # Permanently disable type
    engine.disable_type("daily_briefing")
    assert engine.is_type_disabled("daily_briefing") is True
    allowed_dis, reason_dis = engine.can_deliver_proactive_message("daily_briefing")
    assert allowed_dis is False
    assert "disabled" in reason_dis


def test_tone_safety_and_no_engagement_bait(policy_fixture):
    engine, _, _ = policy_fixture

    # Clean factual message remains unchanged
    clean_msg = "Reminder: Project review starts in 10 minutes."
    assert engine.sanitize_message_content(clean_msg) == clean_msg

    # Engagement bait / urgency manipulation stripped
    manipulative_msg = "Hey! Check this out hurry up before it's too late!"
    sanitized = engine.sanitize_message_content(manipulative_msg)
    assert "hurry up" not in sanitized.lower()
    assert "check this out" not in sanitized.lower()
