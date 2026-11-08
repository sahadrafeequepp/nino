"""Proactive messaging policy engine: rate limiting, quiet hours, DND, snooze, and tone safety."""

from collections.abc import Callable
from datetime import datetime, time, timedelta
import logging
import re
from typing import Optional

from nino.config import Settings
from nino.memory.db import MemoryDB

logger = logging.getLogger("nino.automation.policy")

# Engagement bait and emotional manipulation pattern check
ENGAGEMENT_BAIT_PATTERNS = re.compile(
    r"\b(?:don't\s+forget\s+about\s+me|miss\s+you|where\s+have\s+you\s+been|haven't\s+talked|check\s+this\s+out|hurry\s+up|act\s+fast|before\s+it's\s+too\s+late)\b",
    re.IGNORECASE,
)


class ProactivePolicyEngine:
    """Evaluates and enforces quiet hours, rate limits, DND, and snooze rules for proactive messages."""

    def __init__(
        self,
        memory_db: MemoryDB,
        settings: Settings,
        clock_fn: Optional[Callable[[], datetime]] = None,
    ):
        self.db = memory_db
        self.settings = settings
        self.clock = clock_fn or datetime.now

    def now(self) -> datetime:
        """Return current datetime via controllable clock."""
        return self.clock()

    def is_in_quiet_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check whether the given datetime falls within configured quiet hours."""
        current_dt = dt or self.now()
        current_time = current_dt.time()

        try:
            start_parts = [int(p) for p in self.settings.quiet_hours_start.split(":")]
            end_parts = [int(p) for p in self.settings.quiet_hours_end.split(":")]
            start_time = time(start_parts[0], start_parts[1])
            end_time = time(end_parts[0], end_parts[1])
        except Exception:
            start_time = time(22, 0)
            end_time = time(8, 0)

        if start_time <= end_time:
            # Simple interval within same day (e.g. 13:00 to 15:00)
            return start_time <= current_time <= end_time
        else:
            # Interval wrapping past midnight (e.g. 22:00 to 08:00)
            return current_time >= start_time or current_time <= end_time

    def is_dnd_active(self) -> bool:
        """Check if Do Not Disturb mode is enabled."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM proactive_preferences WHERE key = 'dnd_enabled'"
            )
            row = cursor.fetchone()
            if row and row["value"].lower() == "true":
                return True
        return False

    def set_dnd(self, enabled: bool) -> None:
        """Set Do Not Disturb state."""
        val = "true" if enabled else "false"
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO proactive_preferences (key, value, updated_at)
                VALUES ('dnd_enabled', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (val,),
            )
        logger.info("Set Do Not Disturb (DND) to %s", enabled)

    def is_type_disabled(self, message_type: str) -> bool:
        """Check if a specific proactive message type has been turned off."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM proactive_preferences WHERE key = ?",
                (f"disabled:{message_type.lower()}",),
            )
            row = cursor.fetchone()
            return bool(row and row["value"].lower() == "true")

    def disable_type(self, message_type: str) -> None:
        """Permanently turn off a proactive message type."""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO proactive_preferences (key, value, updated_at)
                VALUES (?, 'true', CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value = 'true', updated_at = CURRENT_TIMESTAMP
                """,
                (f"disabled:{message_type.lower()}",),
            )
        logger.info("Disabled proactive message type '%s'", message_type)

    def enable_type(self, message_type: str) -> None:
        """Re-enable a proactive message type."""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO proactive_preferences (key, value, updated_at)
                VALUES (?, 'false', CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value = 'false', updated_at = CURRENT_TIMESTAMP
                """,
                (f"disabled:{message_type.lower()}",),
            )
        logger.info("Enabled proactive message type '%s'", message_type)

    def is_type_snoozed(self, message_type: str, dt: Optional[datetime] = None) -> bool:
        """Check if a message type is currently snoozed."""
        current_dt = dt or self.now()
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM proactive_preferences WHERE key = ?",
                (f"snooze:{message_type.lower()}",),
            )
            row = cursor.fetchone()
            if row and row["value"]:
                try:
                    snooze_until = datetime.fromisoformat(row["value"])
                    if current_dt < snooze_until:
                        return True
                except ValueError:
                    pass
        return False

    def snooze_type(self, message_type: str, duration_minutes: int) -> datetime:
        """Snooze a proactive message type for a duration in minutes."""
        until = self.now() + timedelta(minutes=duration_minutes)
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO proactive_preferences (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (f"snooze:{message_type.lower()}", until.isoformat()),
            )
        logger.info("Snoozed proactive type '%s' until %s", message_type, until.isoformat())
        return until

    def get_hourly_message_count(self, dt: Optional[datetime] = None) -> int:
        """Count proactive messages delivered in the last 1 hour."""
        current_dt = dt or self.now()
        one_hour_ago = (current_dt - timedelta(hours=1)).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM proactive_log WHERE delivered_at >= ?",
                (one_hour_ago,),
            )
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def get_daily_message_count(self, dt: Optional[datetime] = None) -> int:
        """Count proactive messages delivered in the last 24 hours."""
        current_dt = dt or self.now()
        one_day_ago = (current_dt - timedelta(days=1)).isoformat()
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM proactive_log WHERE delivered_at >= ?",
                (one_day_ago,),
            )
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def can_deliver_proactive_message(
        self,
        message_type: str,
        is_urgent: bool = False,
        dt: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        """Evaluate whether a proactive message can be announced right now.

        Returns:
            (allowed: bool, reason: str)
        """
        if not self.settings.proactive_enabled:
            return False, "Proactive messaging is globally disabled in configuration."

        current_dt = dt or self.now()

        # 1. Check explicit type disabled
        if self.is_type_disabled(message_type):
            return False, f"Message type '{message_type}' is disabled by user."

        # 2. Check snooze
        if self.is_type_snoozed(message_type, current_dt):
            return False, f"Message type '{message_type}' is currently snoozed."

        # 3. Check DND
        if self.is_dnd_active():
            if is_urgent:
                return True, "Allowed: Urgent reminder breaking through Do Not Disturb."
            return False, "Do Not Disturb (DND) is active."

        # 4. Check Quiet Hours
        if self.is_in_quiet_hours(current_dt):
            if is_urgent:
                return True, "Allowed: Urgent reminder breaking through quiet hours."
            return False, "Quiet hours in effect; non-urgent message suppressed."

        # 5. Check Rate Limits (Urgent messages bypass rate limits)
        if not is_urgent:
            hourly_count = self.get_hourly_message_count(current_dt)
            if hourly_count >= self.settings.max_proactive_per_hour:
                return False, f"Hourly proactive rate limit reached ({hourly_count}/{self.settings.max_proactive_per_hour})."

            daily_count = self.get_daily_message_count(current_dt)
            if daily_count >= self.settings.max_proactive_per_day:
                return False, f"Daily proactive rate limit reached ({daily_count}/{self.settings.max_proactive_per_day})."

        return True, "Allowed by policy."

    def record_delivery(self, message_type: str, content: str, channel: str) -> None:
        """Record that a proactive message was delivered to an output channel."""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO proactive_log (message_type, content, channel, delivered_at)
                VALUES (?, ?, ?, ?)
                """,
                (message_type, content, channel, self.now().isoformat()),
            )

    def sanitize_message_content(self, text: str) -> str:
        """Ensure message adheres to zero-pressure, no engagement-bait guidelines."""
        # Replace engagement bait phrases if present
        sanitized = ENGAGEMENT_BAIT_PATTERNS.sub("", text)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized
