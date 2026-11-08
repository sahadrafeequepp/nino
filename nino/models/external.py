"""External LLM adapter stub strictly routed through Privacy Gateway."""

from typing import AsyncIterator
from nino.models.base import LLMAdapter
from nino.security.privacy_gateway import PrivacyGateway


class ExternalAdapter(LLMAdapter):
    """Stub adapter for external/third-party AI providers.

    All outbound calls must flow through the PrivacyGateway for validation,
    secret blocking, and PII redaction. Disabled by default.
    """

    def __init__(self, privacy_gateway: PrivacyGateway | None = None, provider_name: str = "external-stub"):
        self.privacy_gateway = privacy_gateway or PrivacyGateway(allow_external=False)
        self.provider_name = provider_name

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Sanitize prompt and dispatch to external provider."""
        sanitized_messages = await self.privacy_gateway.dispatch_external(
            provider=self.provider_name,
            messages=messages,
            **kwargs,
        )
        return f"[{self.provider_name} response to {len(sanitized_messages)} sanitized message(s)]"

    async def stream_chat(
        self, messages: list[dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        """Sanitize prompt and stream from external provider."""
        sanitized_messages = await self.privacy_gateway.dispatch_external(
            provider=self.provider_name,
            messages=messages,
            **kwargs,
        )
        chunks = ["[", self.provider_name, " streamed response]"]
        for chunk in chunks:
            yield chunk
