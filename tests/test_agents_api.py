"""Tests for Sub-Agents API endpoints (/agents/definitions and /agents/runs)."""

import pytest
from starlette.testclient import TestClient

from nino.agents.models import AgentBudget, AgentRunRecord
from nino.agents.store import get_run_store
from nino.config import get_settings
from nino.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_list_agent_definitions_unauthorized(client):
    response = client.get("/agents/definitions")
    assert response.status_code == 401


def test_list_agent_definitions_authorized(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    response = client.get(
        "/agents/definitions",
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3

    agent_names = {item["name"] for item in data}
    assert {"researcher", "coder", "planner"}.issubset(agent_names)

    # Check structure
    for item in data:
        assert "name" in item
        assert "description" in item
        assert "allowed_tools" in item
        assert "default_budget" in item


def test_list_and_get_agent_runs(client):
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.auth_token}"}

    store = get_run_store()
    # Seed a run record
    sample_run = AgentRunRecord(
        run_id="run_test_12345",
        parent_run_id="parent_999",
        agent_name="researcher",
        goal="Find documentation on SQLite WAL",
        status="success",
        tools_used=["file_read"],
        budget=AgentBudget(max_steps=5, timeout_seconds=60.0),
        steps_taken=2,
        tokens_used=120,
        duration_ms=450.0,
        tainted=False,
        result="WAL provides concurrency for readers.",
    )
    store.record_run(sample_run)

    # Test listing runs
    response = client.get(
        "/agents/runs",
        headers=headers,
    )
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) >= 1

    matched = [r for r in runs if r["run_id"] == "run_test_12345"]
    assert len(matched) == 1
    assert matched[0]["agent_name"] == "researcher"
    assert matched[0]["status"] == "success"
    assert matched[0]["tools_used"] == ["file_read"]

    # Test getting single run
    detail_resp = client.get(
        "/agents/runs/run_test_12345",
        headers=headers,
    )
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["run_id"] == "run_test_12345"
    assert detail["goal"] == "Find documentation on SQLite WAL"

    # Test 404 for unknown run
    not_found = client.get(
        "/agents/runs/unknown_run_xyz",
        headers=headers,
    )
    assert not_found.status_code == 404
