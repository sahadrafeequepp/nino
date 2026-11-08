"""Vector search and embedding abstractions for Nino."""

from nino.memory.vector.base import Embedder, VectorStore, SearchResult
from nino.memory.vector.ollama_embedder import OllamaEmbedder
from nino.memory.vector.sqlite_store import SQLiteVectorStore

__all__ = [
    "Embedder",
    "VectorStore",
    "SearchResult",
    "OllamaEmbedder",
    "SQLiteVectorStore",
]
