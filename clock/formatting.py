"""Date and time formatting utilities for Nino AI Assistant."""

from datetime import datetime
from clock.location_clock import clock


def get_current_time_str() -> str:
    """Fetch live host timestamp formatted with day of week and timezone.

    Example output: "Monday, October 05, 2026, 08:35 PM IST"
    """
    now = clock.now()
    return now.strftime("%A, %B %d, %Y, %I:%M %p %Z").strip()


def format_short_time(dt: datetime | None = None) -> str:
    """Return 12-hour formatted time with AM/PM (e.g. '08:35 PM')."""
    target = dt or clock.now()
    return target.strftime("%I:%M %p").strip()


def get_live_time() -> str:
    """Always pull fresh hardware clock formatted with seconds, day, and date.

    Example output: "08:35:12 PM (Monday, 05 Oct 2026)"
    """
    return clock.now().strftime("%I:%M:%S %p (%A, %d %b %Y)")
