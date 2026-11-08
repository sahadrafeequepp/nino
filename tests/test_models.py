"""Tests for LLM adapter implementations and factory resolution."""

import json
import pytest
import httpx
from nino.config import Settings
from nino.models.ollama import OllamaAdapter
from nino.models.external import ExternalAdapter
from nino.models.factory import get_llm_adapter
from nino.security.privacy_gateway import SecurityException


@pytest.mark.asyncio
async def test_ollama_adapter_chat_mocked(monkeypatch):
    """Test OllamaAdapter.chat() with a mocked httpx.AsyncClient response."""
    expected_reply = "I am a local model running offline."

    async def mock_post(self, url, json=None, **kwargs):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json["model"] == "gemma2:2b"
        assert json["messages"] == [{"role": "user", "content": "Hi"}]
        assert json["stream"] is False

        mock_resp = httpx.Response(
            status_code=200,
            json={
                "model": "gemma2:2b",
                "message": {"role": "assistant", "content": expected_reply},
                "done": True,
            },
            request=httpx.Request("POST", url),
        )
        return mock_resp

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    adapter = OllamaAdapter(base_url="http://127.0.0.1:11434", model_name="gemma2:2b")
    reply = await adapter.chat([{"role": "user", "content": "Hi"}])
    assert reply == expected_reply


@pytest.mark.asyncio
async def test_ollama_adapter_stream_chat_mocked(monkeypatch):
    """Test OllamaAdapter.stream_chat() yields token chunks from mocked stream."""
    chunks = ["Hello", " world", "!"]

    class MockStreamResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for c in chunks:
                yield json.dumps({"message": {"content": c}, "done": False})
            yield json.dumps({"message": {"content": ""}, "done": True})

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_stream(self, method, url, json=None, **kwargs):
        return MockStreamResponse()

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    adapter = OllamaAdapter(base_url="http://127.0.0.1:11434", model_name="gemma2:2b")
    received = []
    async for chunk in adapter.stream_chat([{"role": "user", "content": "Hi"}]):
        received.append(chunk)

    assert "".join(received) == "Hello world!"


@pytest.mark.asyncio
async def test_ollama_adapter_connection_error(monkeypatch):
    """Test OllamaAdapter handles unreachable connection gracefully."""
    async def mock_post(self, url, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    adapter = OllamaAdapter(base_url="http://127.0.0.1:9999", model_name="gemma2:2b")
    with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
        await adapter.chat([{"role": "user", "content": "Hi"}])


def test_factory_returns_ollama():
    settings = Settings(model_adapter="ollama", model_name="gemma2:2b")
    adapter = get_llm_adapter(settings)
    assert isinstance(adapter, OllamaAdapter)
    assert adapter.model_name == "gemma2:2b"


def test_factory_returns_external():
    settings = Settings(model_adapter="external", allow_external_providers=False)
    adapter = get_llm_adapter(settings)
    assert isinstance(adapter, ExternalAdapter)


def test_factory_unsupported():
    settings = Settings(model_adapter="unsupported_model")
    with pytest.raises(ValueError, match="Unsupported MODEL_ADAPTER"):
        get_llm_adapter(settings)


@pytest.mark.asyncio
async def test_external_adapter_disabled_by_default():
    adapter = ExternalAdapter()
    with pytest.raises(SecurityException, match="External AI providers are disabled"):
        await adapter.chat([{"role": "user", "content": "test"}])
