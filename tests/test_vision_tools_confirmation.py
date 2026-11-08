"""Tests verifying SENSITIVE tier confirmation enforcement for screenshot and camera tools."""

import io
import pytest
from PIL import Image

from nino.memory.db import MemoryDB
from nino.security.audit import AuditLogger
from nino.tools.confirmation import ConfirmationManager
from nino.tools.executor import CentralExecutor
from nino.tools.permissions import PermissionTier
from nino.tools.registry import ToolRegistry
from nino.tools.vision import ScreenshotCaptureTool, CameraSnapshotTool
from nino.vision.base import VisionAdapter


class FakeVisionAdapter(VisionAdapter):
    async def describe_image(self, image_data: bytes, prompt: str = "") -> str:
        return "A desktop screen showing code editor and browser."

    async def extract_text(self, image_data: bytes) -> str:
        return "System Status: Online"


class FakeCaptureEngine:
    @staticmethod
    def capture(*args, **kwargs) -> bytes:
        img = Image.new("RGB", (64, 64), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


@pytest.fixture
def vision_executor(tmp_path):
    registry = ToolRegistry()
    adapter = FakeVisionAdapter()
    audit_log_path = tmp_path / "audit.jsonl"
    audit_logger = AuditLogger(log_path=str(audit_log_path))
    db_path = str(tmp_path / "test.db")
    memory_db = MemoryDB(db_path=db_path)
    conf_mgr = ConfirmationManager(memory_db=memory_db)

    screenshot_tool = ScreenshotCaptureTool(
        vision_adapter=adapter,
        audit_logger=audit_logger,
        capture_engine=FakeCaptureEngine(),
    )
    camera_tool = CameraSnapshotTool(
        vision_adapter=adapter,
        audit_logger=audit_logger,
        capture_engine=FakeCaptureEngine(),
    )

    registry.register(screenshot_tool)
    registry.register(camera_tool)

    executor = CentralExecutor(
        registry=registry,
        confirmation_manager=conf_mgr,
        audit_logger=audit_logger,
    )
    return executor, conf_mgr


@pytest.mark.asyncio
async def test_screenshot_requires_confirmation(vision_executor):
    executor, conf_mgr = vision_executor

    # 1. Verify tier is SENSITIVE
    tool = executor.registry.get("screenshot_capture")
    assert tool.tier == PermissionTier.SENSITIVE

    # 2. First call without confirmation token
    result = await executor.execute(
        tool_name="screenshot_capture",
        arguments={"prompt": "What is on my screen?"},
    )

    assert result.success is False
    assert result.requires_confirmation is True
    assert result.decision == "confirmation_required"
    assert result.confirmation is not None

    cid = result.confirmation.id
    conf_mgr.approve(cid)

    # 3. Second call with the approved confirmation ID
    confirmed_result = await executor.execute(
        tool_name="screenshot_capture",
        arguments={"prompt": "What is on my screen?"},
        confirmation_id=cid,
    )

    assert confirmed_result.success is True
    assert confirmed_result.decision == "executed"
    assert confirmed_result.output["status"] == "success"
    assert "desktop screen" in confirmed_result.output["description"]
    assert "System Status: Online" in confirmed_result.output["extracted_text"]


@pytest.mark.asyncio
async def test_camera_requires_confirmation(vision_executor):
    executor, conf_mgr = vision_executor

    # 1. Verify tier is SENSITIVE
    tool = executor.registry.get("camera_snapshot")
    assert tool.tier == PermissionTier.SENSITIVE

    # 2. Call without confirmation token
    result = await executor.execute(
        tool_name="camera_snapshot",
        arguments={"prompt": "What is in front of the camera?", "device_index": 0},
    )

    assert result.success is False
    assert result.requires_confirmation is True
    assert result.decision == "confirmation_required"
    assert result.confirmation is not None

    cid = result.confirmation.id
    conf_mgr.approve(cid)

    # 3. Call with approved confirmation ID
    confirmed_result = await executor.execute(
        tool_name="camera_snapshot",
        arguments={"prompt": "What is in front of the camera?", "device_index": 0},
        confirmation_id=cid,
    )

    assert confirmed_result.success is True
    assert confirmed_result.decision == "executed"
    assert confirmed_result.output["capture_type"] == "camera"
