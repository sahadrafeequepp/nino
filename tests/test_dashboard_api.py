"""Integration tests for Nino HUD-Style Control Room Dashboard API endpoints."""

import json
import pytest
from starlette.testclient import TestClient

from nino.config import get_settings
from nino.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_dashboard_system_unauthorized(client):
    response = client.get("/dashboard/system")
    assert response.status_code == 401


def test_dashboard_system_authorized(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    response = client.get("/dashboard/system", headers=headers)
    assert response.status_code == 200
    data = response.json()

    # Telemetry
    assert "telemetry" in data
    assert "cpu" in data["telemetry"]
    assert "ram" in data["telemetry"]
    assert "gpu" in data["telemetry"]

    # Models & Health
    assert "models" in data
    assert data["models"]["llm"]["name"] == settings.model_name
    assert "health" in data
    assert data["health"]["database"] in ("online", "degraded")
    assert "kill_switch_active" in data


def test_dashboard_today_overview(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    response = client.get("/dashboard/today", headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert "date" in data
    assert "calendar_events" in data
    assert "tasks" in data
    assert "reminders" in data


def test_dashboard_conversations(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    response = client.get("/dashboard/conversations", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert "count" in data


def test_dashboard_tools_activity(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    response = client.get("/dashboard/tools/activity", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "activity" in data
    assert isinstance(data["activity"], list)


def test_dashboard_privacy_log_zero_outbound(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    response = client.get("/dashboard/privacy-log", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert "total_outbound" in data
    assert "Zero outbound traffic" in data["message"]


def test_dashboard_audit_filterable(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    response = client.get("/dashboard/audit?limit=10", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "records" in data


def test_dashboard_tiers_matrix(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    response = client.get("/dashboard/tiers", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert len(data["tools"]) > 0

    tool_names = [t["name"] for t in data["tools"]]
    assert "note_create" in tool_names or "file_read" in tool_names


def test_dashboard_kill_switch_toggle(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    # 1. Enable offline mode kill switch
    res_on = client.post(
        "/dashboard/kill-switch",
        headers=headers,
        json={"offline_mode": True},
    )
    assert res_on.status_code == 200
    data_on = res_on.json()
    assert data_on["offline_mode"] is True
    assert "ENGAGED" in data_on["message"]

    # Verify system telemetry shows kill switch is active
    sys_res = client.get("/dashboard/system", headers=headers)
    assert sys_res.json()["kill_switch_active"] is True

    # 2. Disable offline mode kill switch
    res_off = client.post(
        "/dashboard/kill-switch",
        headers=headers,
        json={"offline_mode": False},
    )
    assert res_off.status_code == 200
    data_off = res_off.json()
    assert data_off["offline_mode"] is False
    assert "DISENGAGED" in data_off["message"]


def test_dashboard_events_sse_auth(client):
    # Unauthenticated should fail with 401
    res_unauth = client.get("/dashboard/events")
    assert res_unauth.status_code == 401

    # Authenticated via query token parameter with max_events=1
    settings = get_settings()
    with client.stream("GET", f"/dashboard/events?token={settings.auth_token}&max_events=1") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        found_data = False
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                assert "cpu_percent" in data
                assert "pending_confirmations" in data
                found_data = True
                break
        assert found_data
