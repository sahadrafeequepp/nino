"""Tool integration for Nino's Central Tool Framework and Agent execution."""

from typing import Optional
from pydantic import BaseModel, Field

from clock.location_clock import LocationClock, clock


class ClockToolInput(BaseModel):
    timezone_name: Optional[str] = Field(default=None, description="Optional timezone name (e.g. 'Asia/Kolkata', 'UTC+05:30')")


class ClockTool:
    """Read-only tool providing verified current local time and geographical context."""

    name = "get_current_time"
    description = "Get the current accurate system date, day of week, and local time. Read-only."

    def __init__(self, location_clock: Optional[LocationClock] = None):
        self.clock = location_clock or clock

    async def run(self, timezone_name: Optional[str] = None) -> str:
        dt = self.clock.now(timezone_name)
        time_12 = dt.strftime("%I:%M:%S %p")
        time_24 = dt.strftime("%H:%M:%S")
        date_str = dt.strftime("%A, %B %d, %Y")
        tz_str = dt.strftime("%Z (%z)").strip() or "IST (+0530)"

        return (
            f"Current Local Time: {date_str}, {time_12} ({tz_str})\n"
            f"24-Hour Format: {time_24}\n"
            f"Location: {self.clock.city}, {self.clock.region}, {self.clock.country}\n"
            f"Timezone: {timezone_name or self.clock.timezone_name} ({tz_str})"
        )
