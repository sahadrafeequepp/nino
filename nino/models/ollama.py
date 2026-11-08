"""Ollama model adapter implementation."""

import json
from typing import AsyncIterator
import httpx

from nino.models.base import LLMAdapter


class OllamaAdapter(LLMAdapter):
    """Adapter for communicating with a local Ollama instance."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model_name: str = "gemma2:2b"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send chat messages to Ollama and return the full reply."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "stream": False,
        }
        if "tools" in kwargs and kwargs["tools"]:
            payload["tools"] = kwargs["tools"]

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                msg = data.get("message", {})
                if kwargs.get("return_raw_message"):
                    return msg
                return msg.get("content", "")

        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Could not connect to Ollama service at {self.base_url}. Ensure Ollama is running."
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama request timed out at {self.base_url}. Model generation on CPU took longer than 120s."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama returned error status {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            err_msg = str(e).strip() or repr(e)
            raise RuntimeError(f"Ollama request error: {err_msg}") from e

    async def stream_chat(
        self, messages: list[dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        """Stream chat chunks from Ollama."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Could not connect to Ollama service at {self.base_url}. Ensure Ollama is running."
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama streaming timed out at {self.base_url}. Model generation on CPU took longer than 120s."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama returned error status {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            err_msg = str(e).strip() or repr(e)
            raise RuntimeError(f"Ollama streaming error: {err_msg}") from e

