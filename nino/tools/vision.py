"""Vision tools: screenshot_capture, camera_snapshot, and image_describe."""

import logging
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

from nino.security.audit import AuditLogger
from nino.tools.base import BaseTool
from nino.tools.permissions import PermissionTier
from nino.tools.taint import TaintTracker
from nino.vision.base import VisionAdapter
from nino.vision.capture import ScreenCaptureEngine, CameraCaptureEngine
from nino.vision.storage import TempImageManager

logger = logging.getLogger("nino.tools.vision")


class ScreenshotCaptureArgs(BaseModel):
    prompt: str = Field(
        default="Describe what is visible on my screen in detail.",
        description="Question or focus area for analyzing the screenshot",
    )


class ScreenshotCaptureTool(BaseTool):
    """Capture desktop screenshot on demand and analyze with local vision model."""

    name = "screenshot_capture"
    description = "Capture the user's active screen on demand to analyze content. SENSITIVE: requires human confirmation."
    tier = PermissionTier.SENSITIVE
    args_schema = ScreenshotCaptureArgs

    def __init__(
        self,
        vision_adapter: VisionAdapter,
        temp_manager: Optional[TempImageManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        capture_engine: Optional[Any] = None,
    ):
        self.vision_adapter = vision_adapter
        self.temp_manager = temp_manager or TempImageManager()
        self.audit_logger = audit_logger
        self.capture_engine = capture_engine or ScreenCaptureEngine

    async def run(self, prompt: str = "Describe what is visible on my screen in detail.", **kwargs: Any) -> dict[str, Any]:
        """Capture screen, analyze image, and return description with tainted extracted text."""
        raw_bytes = self.capture_engine.capture()

        if self.audit_logger:
            self.audit_logger.log_image_capture(capture_type="screenshot", status="success")

        with self.temp_manager.temp_image_file(raw_bytes, suffix=".png"):
            # Image is verified and held in temp storage during analysis, then automatically deleted
            description = await self.vision_adapter.describe_image(raw_bytes, prompt=prompt)
            extracted_text = await self.vision_adapter.extract_text(raw_bytes)

        result: dict[str, Any] = {
            "status": "success",
            "capture_type": "screenshot",
            "description": description,
        }

        if extracted_text:
            result["extracted_text"] = TaintTracker.wrap_untrusted(extracted_text, source="image:screenshot")

        return result


class CameraSnapshotArgs(BaseModel):
    prompt: str = Field(
        default="Describe what is visible in front of the camera.",
        description="Instruction or query for analyzing the camera frame",
    )
    device_index: int = Field(
        default=0,
        description="Webcam hardware device index (default: 0)",
    )


class CameraSnapshotTool(BaseTool):
    """Capture single webcam frame on demand with a visible indicator and analyze with local vision model."""

    name = "camera_snapshot"
    description = "Capture a single frame from the camera with a visible indicator. SENSITIVE: requires human confirmation."
    tier = PermissionTier.SENSITIVE
    args_schema = CameraSnapshotArgs

    def __init__(
        self,
        vision_adapter: VisionAdapter,
        temp_manager: Optional[TempImageManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        capture_engine: Optional[Any] = None,
    ):
        self.vision_adapter = vision_adapter
        self.temp_manager = temp_manager or TempImageManager()
        self.audit_logger = audit_logger
        self.capture_engine = capture_engine or CameraCaptureEngine()

    async def run(
        self,
        prompt: str = "Describe what is visible in front of the camera.",
        device_index: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Capture camera frame, analyze image, and return description with tainted extracted text."""
        raw_bytes = self.capture_engine.capture(device_index=device_index)

        if self.audit_logger:
            self.audit_logger.log_image_capture(capture_type="camera", status="success")

        with self.temp_manager.temp_image_file(raw_bytes, suffix=".jpg"):
            description = await self.vision_adapter.describe_image(raw_bytes, prompt=prompt)
            extracted_text = await self.vision_adapter.extract_text(raw_bytes)

        result: dict[str, Any] = {
            "status": "success",
            "capture_type": "camera",
            "description": description,
        }

        if extracted_text:
            result["extracted_text"] = TaintTracker.wrap_untrusted(extracted_text, source="image:camera")

        return result


class ImageDescribeArgs(BaseModel):
    file_path: str = Field(..., description="Path to an existing image file (e.g. in notes or data)")
    prompt: str = Field(
        default="Describe what you see in this image in detail.",
        description="Question or instruction for analyzing the image",
    )


class ImageDescribeTool(BaseTool):
    """Analyze and describe an existing local image file. READ_ONLY tier."""

    name = "image_describe"
    description = "Analyze an existing image file from disk. READ_ONLY tier."
    tier = PermissionTier.READ_ONLY
    args_schema = ImageDescribeArgs

    def __init__(
        self,
        vision_adapter: VisionAdapter,
        allowed_dirs: str = ".,data,notes",
        temp_manager: Optional[TempImageManager] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.vision_adapter = vision_adapter
        self.allowed_dirs = [d.strip() for d in allowed_dirs.split(",") if d.strip()]
        self.temp_manager = temp_manager or TempImageManager()
        self.audit_logger = audit_logger

    def _is_path_allowed(self, target: Path) -> bool:
        resolved = target.resolve()
        for d in self.allowed_dirs:
            allowed_root = Path(d).resolve()
            if resolved == allowed_root or allowed_root in resolved.parents:
                return True
        return False

    async def run(
        self,
        file_path: str,
        prompt: str = "Describe what you see in this image in detail.",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load image file, validate against allowed directories, and analyze."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image file '{file_path}' does not exist.")

        if not self._is_path_allowed(path):
            raise PermissionError(f"Access to path '{file_path}' is restricted outside allowed directories.")

        raw_bytes = path.read_bytes()
        self.temp_manager.validate_image(raw_bytes)

        if self.audit_logger:
            self.audit_logger.log_image_capture(capture_type="file", status="success")

        description = await self.vision_adapter.describe_image(raw_bytes, prompt=prompt)
        extracted_text = await self.vision_adapter.extract_text(raw_bytes)

        result: dict[str, Any] = {
            "status": "success",
            "capture_type": "file",
            "file_path": str(file_path),
            "description": description,
        }

        if extracted_text:
            result["extracted_text"] = TaintTracker.wrap_untrusted(extracted_text, source=f"image:{path.name}")

        return result
