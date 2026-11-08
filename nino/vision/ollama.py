"""Ollama VisionAdapter implementation running local vision-language models."""

import base64
import logging
from typing import Any
import httpx

from nino.vision.base import VisionAdapter

logger = logging.getLogger("nino.vision.ollama")


class OllamaVisionAdapter(VisionAdapter):
    """Local on-device vision model adapter communicating with Ollama."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "moondream:latest",
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=self.timeout)

    async def describe_image(
        self,
        image_data: bytes,
        prompt: str = "Describe what you see in this image in detail.",
    ) -> str:
        """Query local Ollama vision model with base64 encoded image."""
        if not image_data:
            raise ValueError("Cannot analyze empty image data.")

        b64_image = base64.b64encode(image_data).decode("utf-8")
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
        }

        url = f"{self.base_url}/api/generate"
        client = await self._get_client()
        should_close = self._client is None

        try:
            logger.info("Dispatching image to local Ollama vision model: %s", self.model)
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except httpx.HTTPStatusError as e:
            logger.error("Ollama vision API error (%s): %s", e.response.status_code, e.response.text)
            raise RuntimeError(f"Ollama vision request failed: HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error("Failed to reach Ollama vision service at %s: %s", url, e)
            raise RuntimeError(f"Could not connect to Ollama at {self.base_url}. Ensure Ollama is running.") from e
        finally:
            if should_close:
                await client.aclose()

    async def extract_text(self, image_data: bytes) -> str:
        """Extract and transcribe all text visible within the image."""
        prompt = (
            "Read and transcribe all visible text in this image verbatim. "
            "Do not add interpretation or commentary. If there is no text in the image, reply with ''."
        )
        raw_text = await self.describe_image(image_data, prompt=prompt)
        # Normalize empty responses
        if raw_text.strip().lower() in ("", "''", "none", "no text", "no text found", "no text visible"):
            return ""
        return raw_text
