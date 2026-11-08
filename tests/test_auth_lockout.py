"""Unit tests for authentication rate-limiting and temporary IP lockout on repeated failures."""

from datetime import datetime, timedelta
import pytest
from starlette.testclient import TestClient

from nino.main import app
from nino.security.lockout import AuthLockoutManager


class ControllableClock:
    def __init__(self, start_dt: datetime):
        self.current_dt = start_dt

    def __call__(self) -> datetime:
        return self.current_dt

    def advance(self, seconds: int):
        self.current_dt += timedelta(seconds=seconds)


def test_lockout_manager_unit():
    """Verify failed attempts tracking, threshold lockout, and expiry."""
    clock = ControllableClock(datetime(2026, 9, 21, 12, 0, 0))
    manager = AuthLockoutManager(max_failed_attempts=3, lockout_seconds=120, clock_fn=clock)
    ip = "192.168.1.100"

    # Attempt 1: failure
    is_locked = manager.record_failure(ip)
    assert is_locked is False
    locked, _ = manager.is_locked_out(ip)
    assert locked is False

    # Attempt 2: failure
    is_locked = manager.record_failure(ip)
    assert is_locked is False

    # Attempt 3: threshold reached -> locked out!
    is_locked = manager.record_failure(ip)
    assert is_locked is True
    locked, remaining = manager.is_locked_out(ip)
    assert locked is True
    assert remaining == 120

    # Advance clock by 60 seconds -> still locked
    clock.advance(60)
    locked, remaining = manager.is_locked_out(ip)
    assert locked is True
    assert remaining <= 60

    # Advance clock past lockout period (70 more seconds -> total 130s)
    clock.advance(70)
    locked, _ = manager.is_locked_out(ip)
    assert locked is False


def test_auth_endpoint_lockout_integration():
    """Verify HTTP 429 is returned after repeated bad token attempts."""
    clock = ControllableClock(datetime(2026, 9, 21, 12, 0, 0))
    test_manager = AuthLockoutManager(max_failed_attempts=3, lockout_seconds=300, clock_fn=clock)

    with TestClient(app) as client:
        app.state.lockout_manager = test_manager

        # Send 2 bad token requests -> 401 Unauthorized
        for _ in range(2):
            res = client.get("/automations", headers={"Authorization": "Bearer bad-token"})
            assert res.status_code == 401

        # 3rd bad token request triggers lockout threshold
        res3 = client.get("/automations", headers={"Authorization": "Bearer bad-token"})
        # The 3rd failed request records failure #3 and locks out subsequent requests
        assert res3.status_code in [401, 429]

        # 4th request from same client is rejected with 429 Too Many Requests
        res_locked = client.get("/automations", headers={"Authorization": "Bearer bad-token"})
        assert res_locked.status_code == 429
        assert "Too many failed" in res_locked.json()["detail"]

        # Even with the correct token, requests are blocked while locked out
        from nino.config import get_settings
        valid_token = get_settings().auth_token
        res_valid_while_locked = client.get("/automations", headers={"Authorization": f"Bearer {valid_token}"})
        assert res_valid_while_locked.status_code == 429

        # Fast forward past lockout period
        clock.advance(310)
        res_after_cooldown = client.get("/automations", headers={"Authorization": f"Bearer {valid_token}"})
        assert res_after_cooldown.status_code == 200
