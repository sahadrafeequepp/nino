"""Tests for VisionAdapter interface and OllamaVisionAdapter."""

import io
import pytest
from PIL import Image
import httpx

from nino.vision.ollama import OllamaVisionAdapter


def create_fake_image_bytes(color: str = "red", size: tuple[int, int] = (64, 64)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_ollama_vision_describe_image_success():
    fake_img = create_fake_image_bytes()

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = httpx.ByteStream(request.read())
        return httpx.Response(
            200,
            json={"response": "A solid red square icon with smooth edges."},
            request=request,
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = OllamaVisionAdapter(
            base_url="http://127.0.0.1:11434",
            model="moondream:latest",
            http_client=client,
        )
        desc = await adapter.describe_image(fake_img, prompt="What is this?")
        assert desc == "A solid red square icon with smooth edges."


@pytest.mark.asyncio
async def test_ollama_vision_extract_text():
    fake_img = create_fake_image_bytes()

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"response": "Store Open 9AM - 5PM"},
            request=request,
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = OllamaVisionAdapter(
            base_url="http://127.0.0.1:11434",
            model="moondream:latest",
            http_client=client,
        )
        text = await adapter.extract_text(fake_img)
        assert text == "Store Open 9AM - 5PM"


@pytest.mark.asyncio
async def test_ollama_vision_empty_image_rejected():
    adapter = OllamaVisionAdapter()
    with pytest.raises(ValueError, match="Cannot analyze empty image data"):
        await adapter.describe_image(b"")


@pytest.mark.asyncio
async def test_ollama_vision_error_handling():
    fake_img = create_fake_image_bytes()

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Ollama Error", request=request)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = OllamaVisionAdapter(
            base_url="http://127.0.0.1:11434",
            model="moondream:latest",
            http_client=client,
        )
        with pytest.raises(RuntimeError, match="Ollama vision request failed: HTTP 500"):
            await adapter.describe_image(fake_img)
