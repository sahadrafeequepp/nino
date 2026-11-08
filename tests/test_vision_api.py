"""Integration tests for Vision REST API endpoints."""

import io
import pytest
from PIL import Image
from starlette.testclient import TestClient

from nino.config import get_settings
from nino.main import app
from nino.vision.base import VisionAdapter


class MockVisionAdapter(VisionAdapter):
    async def describe_image(self, image_data: bytes, prompt: str = "") -> str:
        return "A test photo of a coffee mug on a desk."

    async def extract_text(self, image_data: bytes) -> str:
        return "CAFE LATTE"


@pytest.fixture
def client(tmp_path):
    settings = get_settings()
    settings.auth_token = "test-vision-token"
    settings.vision_enabled = True
    settings.temp_image_dir = str(tmp_path / "tmp_vision")

    with TestClient(app) as test_client:
        app.state.vision_adapter = MockVisionAdapter()
        yield test_client


def test_vision_upload_success(client):
    img = Image.new("RGB", (64, 64), color="orange")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    headers = {"Authorization": "Bearer test-vision-token"}
    files = {"file": ("mug.png", buf.getvalue(), "image/png")}
    data = {"prompt": "What is in this picture?"}

    response = client.post("/vision/upload", headers=headers, files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["filename"] == "mug.png"
    assert res_data["format"] == "PNG"
    assert "coffee mug" in res_data["description"]
    assert "CAFE LATTE" in res_data["extracted_text"]


def test_vision_upload_unauthorized(client):
    response = client.post("/vision/upload")
    assert response.status_code == 401


def test_vision_screenshot_triggers_confirmation(client):
    headers = {"Authorization": "Bearer test-vision-token"}
    response = client.post(
        "/vision/screenshot",
        headers=headers,
        json={"prompt": "Describe screen"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "confirmation_required"
    assert data["tool_name"] == "screenshot_capture"
    assert data["confirmation"]["id"] is not None


def test_vision_camera_triggers_confirmation(client):
    headers = {"Authorization": "Bearer test-vision-token"}
    response = client.post(
        "/vision/camera",
        headers=headers,
        json={"prompt": "Describe camera view", "device_index": 0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "confirmation_required"
    assert data["tool_name"] == "camera_snapshot"
    assert data["confirmation"]["id"] is not None
