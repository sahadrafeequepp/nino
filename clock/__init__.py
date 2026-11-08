"""Clock Package: Unified Real-Time Location Clock System for Nino.

Provides:
- Location-aware real-time clock tracking (LocationClock, clock)
- Time formatting utilities (get_current_time_str, format_short_time)
- Fast natural language time intent detection (is_time_query, get_time_conversational_reply)
- Agent & Orchestrator tool integration (ClockTool)
"""

from clock.formatting import format_short_time, get_current_time_str, get_live_time
from clock.intent import get_time_conversational_reply, is_time_query
from clock.location_clock import (
    LocationClock,
    STANDARD_TIMEZONE_MAP,
    clock,
)
from clock.tool import ClockTool

__all__ = [
    "ClockTool",
    "LocationClock",
    "STANDARD_TIMEZONE_MAP",
    "clock",
    "format_short_time",
    "get_current_time_str",
    "get_live_time",
    "get_time_conversational_reply",
    "is_time_query",
]
