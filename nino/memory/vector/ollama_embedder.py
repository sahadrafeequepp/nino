"""Ollama embeddings client implementing Embedder interface."""

from typing import Any
import httpx
from nino.memory.vector.base import Embedder


class OllamaEmbedder(Embedder):
    """Embedder using local Ollama instance."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model_name: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        results = await self.embed_batch([text])
        if not results:
            raise RuntimeError(f"Ollama returned empty embedding for text.")
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch of texts."""
        if not texts:
            return []

        # Try /api/embed (Ollama >= 0.3.0)
        embed_url = f"{self.base_url}/api/embed"
        payload = {"model": self.model_name, "input": texts}

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                res = await client.post(embed_url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings:
                        return embeddings

                # Fallback to legacy /api/embeddings endpoint (per text)
                fallback_url = f"{self.base_url}/api/embeddings"
                results = []
                for text in texts:
                    f_res = await client.post(fallback_url, json={"model": self.model_name, "prompt": text})
                    f_res.raise_for_status()
                    f_data = f_res.json()
                    emb = f_data.get("embedding")
                    if not emb:
                        raise RuntimeError("Legacy Ollama embeddings response missing 'embedding' field.")
                    results.append(emb)
                return results

            except httpx.ConnectError as e:
                raise RuntimeError(
                    f"Could not connect to Ollama at {self.base_url} for embeddings. Ensure Ollama is running."
                ) from e
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Ollama embeddings call failed with status {e.response.status_code}: {e.response.text}"
                ) from e
