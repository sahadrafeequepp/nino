"""Google Gemini LLM adapter strictly gated behind the Privacy Gateway."""

import json
import logging
from typing import Any, AsyncIterator
import httpx

from nino.models.base import LLMAdapter
from nino.security.privacy_gateway import PrivacyGateway, SecurityException

logger = logging.getLogger("nino.models.gemini")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiAdapter(LLMAdapter):
    """Google Gemini API adapter strictly routed through Privacy Gateway.

    Enforces:
    1. Outbound PII redaction and secret blocking.
    2. Local media quarantine (zero unapproved audio/image forwarding).
    3. Network SSRF Firewall validation of the destination endpoint.
    4. Safe metadata-only logging (never logging raw prompts or returned secrets).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-1.5-flash",
        privacy_gateway: PrivacyGateway | None = None,
    ):
        self.api_key = api_key or ""
        self.model_name = model_name
        self.privacy_gateway = privacy_gateway or PrivacyGateway(allow_external=False)

    def _format_messages_for_gemini(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Format OpenAI-style messages into Google Gemini API contents payload."""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Gemini expects 'user' or 'model' roles
            gemini_role = "model" if role in ("assistant", "model") else "user"
            content_text = msg.get("content", "")
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content_text}],
            })
        return contents

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Sanitize prompt via Privacy Gateway and request Gemini response."""
        if not self.api_key:
            raise SecurityException(
                "Gemini API key is missing. Set GEMINI_API_KEY in your environment or .env."
            )

        model = kwargs.get("model", self.model_name)
        endpoint = f"{GEMINI_BASE_URL}/{model}:generateContent"

        # 1. Gate request through Privacy Gateway (secret checks, PII redaction, SSRF check)
        sanitized_messages = await self.privacy_gateway.dispatch_external(
            provider=f"google-gemini-{model}",
            messages=messages,
            target_endpoint_url=endpoint,
            **kwargs,
        )

        contents = self._format_messages_for_gemini(sanitized_messages)
        payload = {"contents": contents}

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

                candidates = data.get("candidates", [])
                if not candidates:
                    return ""
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(part.get("text", "") for part in parts)

        except httpx.HTTPStatusError as e:
            # Mask API key from any error representations
            status = e.response.status_code
            clean_err = e.response.text.replace(self.api_key, "[REDACTED_API_KEY]")
            raise RuntimeError(f"Gemini API returned error {status}: {clean_err}") from None
        except Exception as e:
            err_msg = str(e).replace(self.api_key, "[REDACTED_API_KEY]")
            raise RuntimeError(f"Failed to communicate with Gemini API: {err_msg}") from None

    async def stream_chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream response chunks from Gemini API strictly gated by Privacy Gateway."""
        if not self.api_key:
            raise SecurityException(
                "Gemini API key is missing. Set GEMINI_API_KEY in your environment or .env."
            )

        model = kwargs.get("model", self.model_name)
        endpoint = f"{GEMINI_BASE_URL}/{model}:streamGenerateContent?alt=sse"

        # 1. Gate prompt via Privacy Gateway
        sanitized_messages = await self.privacy_gateway.dispatch_external(
            provider=f"google-gemini-{model}",
            messages=messages,
            target_endpoint_url=endpoint,
            **kwargs,
        )

        contents = self._format_messages_for_gemini(sanitized_messages)
        payload = {"contents": contents}

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        json_str = line[5:].strip()
                        if not json_str:
                            continue
                        try:
                            chunk_data = json.loads(json_str)
                            candidates = chunk_data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    text = p.get("text", "")
                                    if text:
                                        yield text
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            clean_err = e.response.text.replace(self.api_key, "[REDACTED_API_KEY]")
            raise RuntimeError(f"Gemini API stream returned error {status}: {clean_err}") from None
        except Exception as e:
            err_msg = str(e).replace(self.api_key, "[REDACTED_API_KEY]")
            raise RuntimeError(f"Failed to stream from Gemini API: {err_msg}") from None
