"""Integration tests for /automations, /proactive, and /briefing API endpoints."""

from datetime import datetime, timedelta
import pytest
from starlette.testclient import TestClient

from nino.config import get_settings
from nino.main import app


@pytest.fixture
def api_client():
    with TestClient(app) as client:
        if hasattr(app.state, "briefing_generator"):
            app.state.briefing_generator.adapter = None
        yield client


def test_automations_api_flow(api_client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    # 1. Inspect proactive status
    status_res = api_client.get("/proactive/status", headers=headers)
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert "dnd_active" in status_data
    assert "quiet_hours_active" in status_data
    assert "hourly_limit" in status_data

    # 2. Toggle DND mode
    dnd_res = api_client.post("/proactive/dnd", headers=headers, json={"enabled": True})
    assert dnd_res.status_code == 200
    assert dnd_res.json()["dnd_enabled"] is True

    dnd_check = api_client.get("/proactive/status", headers=headers).json()
    assert dnd_check["dnd_active"] is True

    # Disable DND
    api_client.post("/proactive/dnd", headers=headers, json={"enabled": False})

    # 3. Add a test job directly to scheduler and inspect via GET /automations
    scheduler = app.state.scheduler
    run_time = datetime.now() + timedelta(hours=2)
    scheduler.add_job(
        job_id="test_api_job_1",
        name="API Test Job",
        job_type="reminder",
        next_run_time=run_time,
        kwargs={"message": "Meeting soon"},
    )

    list_res = api_client.get("/automations", headers=headers)
    assert list_res.status_code == 200
    jobs = list_res.json()
    assert any(j["id"] == "test_api_job_1" for j in jobs)

    # 4. Snooze job
    snooze_res = api_client.post(
        "/automations/snooze",
        headers=headers,
        json={"job_id": "test_api_job_1", "duration_minutes": 45},
    )
    assert snooze_res.status_code == 200
    assert snooze_res.json()["status"] == "ok"

    # 5. Delete job
    del_res = api_client.delete("/automations/test_api_job_1", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # 6. Trigger morning briefing
    briefing_res = api_client.post("/briefing/trigger", headers=headers)
    assert briefing_res.status_code == 200
    briefing_data = briefing_res.json()
    assert briefing_data["status"] == "generated"
    assert "briefing" in briefing_data
    assert "Daily Briefing" in briefing_data["briefing"] or len(briefing_data["briefing"]) > 10
