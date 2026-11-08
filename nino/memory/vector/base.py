"""Abstract interfaces for embeddings and vector search."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Represents a matched vector search result."""
    id: str
    entity_type: str
    entity_id: str
    score: float
    metadata: dict[str, Any]


class Embedder(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate vector embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for multiple texts."""
        pass


class VectorStore(ABC):
    """Abstract interface for local vector stores."""

    @abstractmethod
    async def upsert(
        self,
        id: str,
        vector: list[float],
        entity_type: str,
        entity_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Insert or update a vector entry."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        entity_type: str | None = None,
    ) -> list[SearchResult]:
        """Search top-k most similar items to query_vector."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a vector entry by ID."""
        pass

    @abstractmethod
    async def delete_by_prefix(self, prefix: str) -> int:
        """Delete all vector entries matching prefix."""
        pass

    @abstractmethod
    async def delete_all(self) -> int:
        """Purge all vector entries."""
        pass
