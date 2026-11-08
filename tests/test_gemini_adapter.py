"""Unit tests for GeminiAdapter with Privacy Gateway enforcement and SSRF protection."""

import json
import pytest
import httpx

from nino.models.gemini import GeminiAdapter
from nino.security.privacy_gateway import (
    PrivacyGateway,
    SecurityException,
    BlockedSecretException,
)


@pytest.mark.asyncio
async def test_gemini_adapter_missing_api_key():
    gateway = PrivacyGateway(allow_external=True)
    adapter = GeminiAdapter(api_key=None, privacy_gateway=gateway)
    with pytest.raises(SecurityException, match="Gemini API key is missing"):
        await adapter.chat([{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_gemini_adapter_blocked_by_default_policy():
    # Privacy Gateway defaults to allow_external=False
    gateway = PrivacyGateway(allow_external=False)
    adapter = GeminiAdapter(api_key="AIzaSyTestFakeApiKey123456789012345", privacy_gateway=gateway)
    with pytest.raises(SecurityException, match="External AI providers are disabled by policy"):
        await adapter.chat([{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_gemini_adapter_blocks_secrets_in_prompt():
    gateway = PrivacyGateway(allow_external=True)
    adapter = GeminiAdapter(api_key="AIzaSyTestFakeApiKey123456789012345", privacy_gateway=gateway)
    with pytest.raises(BlockedSecretException):
        await adapter.chat([{"role": "user", "content": "My AWS key is AKIAIOSFODNN7EXAMPLE"}])


@pytest.mark.asyncio
async def test_gemini_adapter_redacts_pii_and_dispatches_mocked(monkeypatch):
    gateway = PrivacyGateway(allow_external=True)
    adapter = GeminiAdapter(
        api_key="AIzaSyTestFakeApiKey123456789012345",
        model_name="gemini-1.5-flash",
        privacy_gateway=gateway,
    )

    captured_payload = {}

    async def mock_post(self, url, json=None, headers=None, **kwargs):
        nonlocal captured_payload
        captured_payload = json
        # Assert key is sent in header, not in URL
        assert headers.get("x-goog-api-key") == "AIzaSyTestFakeApiKey123456789012345"
        mock_resp = httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello! I am Gemini, protected by Privacy Gateway."}],
                            "role": "model",
                        }
                    }
                ]
            },
            request=httpx.Request("POST", url),
        )
        return mock_resp

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    reply = await adapter.chat([
        {"role": "user", "content": "Hi, contact me at bob@example.com"}
    ])

    assert reply == "Hello! I am Gemini, protected by Privacy Gateway."
    # Verify PII was redacted before payload dispatch
    user_part_text = captured_payload["contents"][0]["parts"][0]["text"]
    assert "bob@example.com" not in user_part_text
    assert "[REDACTED_EMAIL]" in user_part_text


@pytest.mark.asyncio
async def test_gemini_adapter_stream_chat_mocked(monkeypatch):
    gateway = PrivacyGateway(allow_external=True)
    adapter = GeminiAdapter(
        api_key="AIzaSyTestFakeApiKey123456789012345",
        model_name="gemini-1.5-flash",
        privacy_gateway=gateway,
    )

    chunks = ["Hello ", "from ", "Gemini!"]

    class MockStreamResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for c in chunks:
                event_data = {
                    "candidates": [
                        {"content": {"parts": [{"text": c}], "role": "model"}}
                    ]
                }
                yield f"data: {json.dumps(event_data)}"
            yield "data: [DONE]"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_stream(self, method, url, **kwargs):
        return MockStreamResponse()

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    streamed_tokens = []
    async for token in adapter.stream_chat([{"role": "user", "content": "Stream me a greeting"}]):
        streamed_tokens.append(token)

    assert "".join(streamed_tokens) == "Hello from Gemini!"
