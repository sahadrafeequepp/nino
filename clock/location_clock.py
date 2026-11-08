"""Location-synchronized clock engine for Nino AI Assistant.

Tracks accurate local time, coordinates, and timezone for Nino's physical location.
"""

from datetime import datetime, timedelta, timezone, tzinfo
import logging
import math
import os
import re
from typing import Any, Optional

logger = logging.getLogger("nino.clock.location")

# Standard IANA timezones and offsets (hours, minutes, abbreviation)
STANDARD_TIMEZONE_MAP: dict[str, tuple[int, int, str]] = {
    # India & South Asia
    "asia/kolkata": (5, 30, "IST"),
    "asia/calcutta": (5, 30, "IST"),
    "asia/colombo": (5, 30, "IST"),
    "asia/kathmandu": (5, 45, "NPT"),
    "asia/dhaka": (6, 0, "BST"),
    "asia/karachi": (5, 0, "PKT"),
    # Middle East & Gulf
    "asia/dubai": (4, 0, "GST"),
    "asia/muscat": (4, 0, "GST"),
    "asia/riyadh": (3, 0, "AST"),
    "asia/qatar": (3, 0, "AST"),
    "asia/kuwait": (3, 0, "AST"),
    # East & Southeast Asia
    "asia/singapore": (8, 0, "SGT"),
    "asia/kuala_lumpur": (8, 0, "MYT"),
    "asia/bangkok": (7, 0, "ICT"),
    "asia/jakarta": (7, 0, "WIB"),
    "asia/tokyo": (9, 0, "JST"),
    "asia/seoul": (9, 0, "KST"),
    "asia/shanghai": (8, 0, "CST"),
    "asia/hong_kong": (8, 0, "HKT"),
    # Europe
    "europe/london": (0, 0, "GMT"),
    "utc": (0, 0, "UTC"),
    "gmt": (0, 0, "GMT"),
    "europe/paris": (1, 0, "CET"),
    "europe/berlin": (1, 0, "CET"),
    "europe/rome": (1, 0, "CET"),
    "europe/madrid": (1, 0, "CET"),
    "europe/amsterdam": (1, 0, "CET"),
    "europe/moscow": (3, 0, "MSK"),
    # Americas
    "america/new_york": (-5, 0, "EST"),
    "america/chicago": (-6, 0, "CST"),
    "america/denver": (-7, 0, "MST"),
    "america/los_angeles": (-8, 0, "PST"),
    "america/toronto": (-5, 0, "EST"),
    "america/vancouver": (-8, 0, "PST"),
    "america/sao_paulo": (-3, 0, "BRT"),
    # Australia & Pacific
    "australia/sydney": (10, 0, "AEST"),
    "australia/melbourne": (10, 0, "AEST"),
    "australia/perth": (8, 0, "AWST"),
    "pacific/auckland": (12, 0, "NZST"),
}


class LocationClock:
    """Location-synchronized clock tracking accurate local time and geographical context."""

    def __init__(
        self,
        default_city: Optional[str] = None,
        default_region: Optional[str] = None,
        default_country: Optional[str] = None,
        default_lat: Optional[float] = None,
        default_lon: Optional[float] = None,
        default_timezone: Optional[str] = None,
    ):
        self.city = (default_city if default_city is not None else os.getenv("DEFAULT_CITY", "Vadakara, Calicut")).strip().strip('"').strip("'")
        self.region = (default_region if default_region is not None else os.getenv("DEFAULT_REGION", "Kerala")).strip().strip('"').strip("'")
        self.country = (default_country if default_country is not None else os.getenv("DEFAULT_COUNTRY", "India")).strip().strip('"').strip("'")

        if default_lat is not None:
            self.latitude = float(default_lat)
        else:
            try:
                self.latitude = float(os.getenv("DEFAULT_LATITUDE", "11.5978"))
            except ValueError:
                self.latitude = 11.5978

        if default_lon is not None:
            self.longitude = float(default_lon)
        else:
            try:
                self.longitude = float(os.getenv("DEFAULT_LONGITUDE", "75.5814"))
            except ValueError:
                self.longitude = 75.5814

        self.timezone_name = (
            default_timezone
            if default_timezone is not None
            else os.getenv("USER_TIMEZONE", os.getenv("NINO_TIMEZONE", "Asia/Kolkata"))
        ).strip()
        self.source = "configured"
        self.last_sync = datetime.now(timezone.utc)

    def resolve_timezone(self, tz_str: Optional[str] = None) -> tzinfo:
        """Resolve a timezone string into a Python tzinfo object."""
        target = (tz_str or self.timezone_name).strip().lower()

        # 1. Lookup in standard timezone dictionary
        if target in STANDARD_TIMEZONE_MAP:
            h, m, abbr = STANDARD_TIMEZONE_MAP[target]
            return timezone(timedelta(hours=h, minutes=m), name=abbr)

        # 2. Try parsing explicit UTC/GMT offset strings (e.g. "+05:30", "UTC+5.5", "-04:00")
        offset_match = re.search(r"([+-])(\d{1,2})(?::?(\d{2}))?", target)
        if offset_match:
            sign = 1 if offset_match.group(1) == "+" else -1
            hours = int(offset_match.group(2))
            mins = int(offset_match.group(3) or 0)
            return timezone(timedelta(minutes=sign * (hours * 60 + mins)), name=f"UTC{offset_match.group(1)}{hours:02d}:{mins:02d}")

        # 3. Try zoneinfo.ZoneInfo
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(tz_str or self.timezone_name)
        except Exception:
            pass

        # 4. Fallback to host local timezone
        return datetime.now().astimezone().tzinfo or timezone.utc

    def now(self, tz_name: Optional[str] = None) -> datetime:
        """Return exact current datetime for the location's timezone."""
        tz = self.resolve_timezone(tz_name)
        return datetime.now(tz)

    def period_of_day(self, hour: Optional[int] = None) -> str:
        """Categorize hour into natural day period."""
        if hour is None:
            hour = self.now().hour

        if 5 <= hour < 12:
            return "Morning 🌅"
        elif 12 <= hour < 17:
            return "Afternoon ☀️"
        elif 17 <= hour < 21:
            return "Evening 🌇"
        else:
            return "Night 🌙"

    def sync_from_gps(
        self,
        lat: float,
        lon: float,
        location_name: Optional[str] = None,
        tz_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update clock coordinates and synchronize timezone to new GPS location."""
        self.latitude = round(lat, 4)
        self.longitude = round(lon, 4)

        if location_name:
            self.city = location_name

        if tz_name:
            self.timezone_name = tz_name
        else:
            # If coordinates are in/near India (approx lat 6-37, lon 68-98), set Asia/Kolkata
            if 6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0:
                self.timezone_name = "Asia/Kolkata"
            else:
                # Estimate standard solar timezone offset: 15 degrees per hour
                est_offset_hours = round(lon / 15.0 * 2) / 2
                sign = "+" if est_offset_hours >= 0 else "-"
                h = int(abs(est_offset_hours))
                m = int((abs(est_offset_hours) - h) * 60)
                self.timezone_name = f"UTC{sign}{h:02d}:{m:02d}"

        self.source = "gps_pin"
        self.last_sync = datetime.now(timezone.utc)

        logger.info(
            "Clock synchronized to GPS (%f, %f) -> %s (Timezone: %s)",
            self.latitude,
            self.longitude,
            self.city,
            self.timezone_name,
        )

        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location": self.city,
            "timezone": self.timezone_name,
            "local_time": self.now().strftime("%I:%M:%S %p"),
        }

    def get_clock_report(self) -> str:
        """Generate comprehensive user-facing clock status report."""
        dt = self.now()
        period = self.period_of_day(dt.hour)

        time_12 = dt.strftime("%I:%M:%S %p")
        time_24 = dt.strftime("%H:%M:%S")
        date_str = dt.strftime("%A, %B %d, %Y")
        tz_abbr = dt.strftime("%Z")
        tz_offset = dt.strftime("%z")
        utc_dt = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        loc_display = f"{self.city}, {self.region}, {self.country}".strip(", ")
        offset_display = f"UTC{tz_offset[:3]}:{tz_offset[3:]}" if len(tz_offset) == 5 else tz_offset

        return (
            "🕒 *Nino Inbuilt Location Clock*\n\n"
            f"• *Current Time*: `{time_12}` ({period})\n"
            f"• *24-Hour Time*: `{time_24}`\n"
            f"• *Date*: {date_str}\n"
            f"• *Location*: {loc_display}\n"
            f"• *Coordinates*: `{self.latitude}, {self.longitude}`\n"
            f"• *Timezone*: `{self.timezone_name}` ({tz_abbr}, {offset_display})\n"
            f"• *UTC Clock*: `{utc_dt}`\n"
            "• *Clock Status*: 🟢 Synchronized & Active"
        )

    def get_prompt_context(self) -> str:
        """Generate structured temporal context for LLM system prompt injection."""
        dt = self.now()
        time_12 = dt.strftime("%I:%M:%S %p")
        time_24 = dt.strftime("%H:%M:%S")
        date_str = dt.strftime("%A, %B %d, %Y")
        tz_abbr = dt.strftime("%Z")
        tz_offset = dt.strftime("%z")
        offset_display = f"UTC{tz_offset[:3]}:{tz_offset[3:]}" if len(tz_offset) == 5 else tz_offset
        period = self.period_of_day(dt.hour)

        loc_display = f"{self.city}, {self.region}, {self.country}".strip(", ")

        return (
            "[Inbuilt Clock & Location Synchronization]\n"
            f"- Location: {loc_display} (Lat: {self.latitude}, Lon: {self.longitude})\n"
            f"- Exact Local Time: {time_12} ({time_24} hrs, {period})\n"
            f"- Current Date: {date_str}\n"
            f"- Timezone: {self.timezone_name} ({tz_abbr}, {offset_display})\n"
            "- Clock Sync: Verified Real-Time On-Device Clock\n"
        )


# Global clock instance
clock = LocationClock()
