"""Abstract LLM adapter interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMAdapter(ABC):
    """Abstract interface for language model backends."""

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send a list of messages to the model and return the complete response."""
        pass

    @abstractmethod
    async def stream_chat(
        self, messages: list[dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        """Stream chunks of the model response."""
        pass
