"""Client IP-based rate limiting and temporary lockout on repeated failed authentication attempts."""

from collections.abc import Callable
from datetime import datetime, timedelta
import logging
from typing import Optional

logger = logging.getLogger("nino.security.lockout")


class AuthLockoutManager:
    """Tracks failed authentication attempts per client IP and enforces a lockout period."""

    def __init__(
        self,
        max_failed_attempts: int = 5,
        lockout_seconds: int = 300,
        clock_fn: Optional[Callable[[], datetime]] = None,
    ):
        self.max_failed = max_failed_attempts
        self.lockout_duration = lockout_seconds
        self.clock = clock_fn or datetime.now

        # IP -> {"failed_count": int, "locked_until": Optional[datetime]}
        self._ip_states: dict[str, dict] = {}

    def now(self) -> datetime:
        return self.clock()

    def is_locked_out(self, ip: str, current_dt: Optional[datetime] = None) -> tuple[bool, int]:
        """Check if an IP address is currently locked out.

        Returns:
            (is_locked: bool, remaining_seconds: int)
        """
        now_dt = current_dt or self.now()
        state = self._ip_states.get(ip)
        if not state:
            return False, 0

        locked_until: Optional[datetime] = state.get("locked_until")
        if locked_until:
            if now_dt < locked_until:
                remaining = int((locked_until - now_dt).total_seconds())
                return True, max(1, remaining)
            else:
                # Lockout expired, reset counters
                del self._ip_states[ip]
                return False, 0

        return False, 0

    def record_failure(self, ip: str, current_dt: Optional[datetime] = None) -> bool:
        """Record a failed authentication attempt. Returns True if IP became locked out."""
        now_dt = current_dt or self.now()
        state = self._ip_states.setdefault(ip, {"failed_count": 0, "locked_until": None})

        # If already locked, keep lock
        if state["locked_until"] and now_dt < state["locked_until"]:
            return True

        state["failed_count"] += 1
        logger.warning("Auth failure #%d for IP: %s", state["failed_count"], ip)

        if state["failed_count"] >= self.max_failed:
            state["locked_until"] = now_dt + timedelta(seconds=self.lockout_duration)
            logger.warning("IP %s locked out for %d seconds due to repeated failed auth.", ip, self.lockout_duration)
            return True

        return False

    def record_success(self, ip: str) -> None:
        """Clear failed attempts for IP upon successful authentication."""
        if ip in self._ip_states:
            # If not actively locked, reset counter
            state = self._ip_states[ip]
            if not state.get("locked_until") or self.now() >= state["locked_until"]:
                del self._ip_states[ip]

    def reset(self) -> None:
        """Clear all lockout tracking data."""
        self._ip_states.clear()


# Global singleton instance
_GLOBAL_LOCKOUT_MGR: Optional[AuthLockoutManager] = None


def get_lockout_manager() -> AuthLockoutManager:
    global _GLOBAL_LOCKOUT_MGR
    if _GLOBAL_LOCKOUT_MGR is None:
        _GLOBAL_LOCKOUT_MGR = AuthLockoutManager()
    return _GLOBAL_LOCKOUT_MGR
