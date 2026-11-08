"""Tests for authentication and constant-time token verification."""

import pytest
from httpx import ASGITransport, AsyncClient

from nino.config import Settings, get_settings
from nino.main import app


@pytest.fixture
def test_settings(monkeypatch):
    """Provide deterministic test settings."""
    settings = Settings(
        auth_token="test-secret-token-xyz-123",
        model_adapter="ollama",
        model_name="gemma2:2b",
        audit_log_path="tests/test_audit.jsonl",
        log_content=False,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    yield settings
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_endpoint_public_without_auth(test_settings):
    """Verify /health is accessible without any Authorization header."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "gemma2:2b"


@pytest.mark.asyncio
async def test_chat_missing_auth_header(test_settings):
    """Verify /chat rejects requests missing the Authorization header."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 401
        assert "Authorization header" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_wrong_auth_token(test_settings):
    """Verify /chat rejects invalid/wrong tokens."""
    headers = {"Authorization": "Bearer wrong-token-value"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat",
            headers=headers,
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 401
        assert "Invalid authentication token" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_valid_auth_token_accepted(test_settings, monkeypatch):
    """Verify valid token passes authentication dependency."""
    # Mock adapter to avoid actual Ollama call in auth test
    class MockAdapter:
        async def chat(self, messages, **kwargs):
            return "Hello from mock model"

    from nino.main import get_adapter
    app.dependency_overrides[get_adapter] = lambda: MockAdapter()

    headers = {"Authorization": f"Bearer {test_settings.auth_token}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat",
            headers=headers,
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        assert response.json()["reply"] == "Hello from mock model"
