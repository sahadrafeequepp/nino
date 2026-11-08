"""Factory for instantiating the configured LLM adapter."""

from nino.config import Settings
from nino.models.base import LLMAdapter
from nino.models.ollama import OllamaAdapter
from nino.models.external import ExternalAdapter
from nino.models.gemini import GeminiAdapter
from nino.security.privacy_gateway import PrivacyGateway


def get_llm_adapter(settings: Settings) -> LLMAdapter:
    """Instantiate and return the configured LLM adapter based on settings."""
    adapter_type = settings.model_adapter.lower().strip()

    if adapter_type == "ollama":
        return OllamaAdapter(
            base_url=settings.ollama_base_url,
            model_name=settings.model_name,
            num_ctx=getattr(settings, "ollama_num_ctx", 8192),
        )
    elif adapter_type == "gemini":
        privacy_gateway = PrivacyGateway(
            allow_external=settings.allow_external_providers
        )
        return GeminiAdapter(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            privacy_gateway=privacy_gateway,
        )
    elif adapter_type == "external":
        privacy_gateway = PrivacyGateway(
            allow_external=settings.allow_external_providers
        )
        return ExternalAdapter(
            privacy_gateway=privacy_gateway,
            provider_name="external-stub",
        )
    else:
        raise ValueError(
            f"Unsupported MODEL_ADAPTER '{adapter_type}'. Supported adapters: 'ollama', 'gemini', 'external'."
        )
