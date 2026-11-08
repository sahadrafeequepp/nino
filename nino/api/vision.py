"""Vision API router for image upload, on-demand screenshots, and camera snapshots."""

import logging
from typing import Any, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from nino.auth import verify_token
from nino.config import Settings, get_settings
from nino.security.audit import AuditLogger
from nino.tools.executor import CentralExecutor
from nino.tools.taint import TaintTracker
from nino.vision.base import VisionAdapter
from nino.vision.ollama import OllamaVisionAdapter
from nino.vision.storage import TempImageManager

logger = logging.getLogger("nino.api.vision")
vision_router = APIRouter(prefix="/vision", tags=["vision"])


class VisionActionRequest(BaseModel):
    prompt: Optional[str] = None
    confirmation_id: Optional[str] = None
    device_index: Optional[int] = 0


@vision_router.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form("Describe what you see in this image in detail."),
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_token),
) -> dict[str, Any]:
    """Upload an image for local vision analysis and text transcription."""
    if not settings.vision_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision subsystem is disabled by configuration.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    temp_mgr: TempImageManager = getattr(request.app.state, "temp_image_manager", None) or TempImageManager(
        settings.temp_image_dir
    )
    try:
        fmt, w, h = temp_mgr.validate_image(content)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {e}",
        )

    vision_adapter: VisionAdapter = getattr(request.app.state, "vision_adapter", None) or OllamaVisionAdapter(
        base_url=settings.ollama_base_url,
        model=settings.vision_model,
    )
    audit_logger: AuditLogger = getattr(request.app.state, "audit_logger", None)

    try:
        # Zero-residual disk guarantee: image is held in temporary file only during analysis
        with temp_mgr.temp_image_file(content, suffix=f".{fmt.lower()}"):
            description = await vision_adapter.describe_image(content, prompt=prompt)
            extracted_text = await vision_adapter.extract_text(content)

        if audit_logger:
            audit_logger.log_image_capture(capture_type="upload", status="success")

        response_data: dict[str, Any] = {
            "status": "success",
            "filename": file.filename,
            "format": fmt,
            "width": w,
            "height": h,
            "description": description,
        }

        if extracted_text:
            response_data["extracted_text"] = TaintTracker.wrap_untrusted(
                extracted_text,
                source=f"image:upload:{file.filename or 'image'}",
            )

        return response_data
    except Exception as e:
        logger.error("Vision upload analysis failed: %s", e)
        if audit_logger:
            audit_logger.log_image_capture(capture_type="upload", status="failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision analysis failed: {e}",
        )


@vision_router.post("/screenshot")
async def capture_screenshot_api(
    request: Request,
    payload: VisionActionRequest = VisionActionRequest(),
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_token),
) -> dict[str, Any]:
    """Trigger desktop screenshot through CentralExecutor with confirmation checking."""
    executor: CentralExecutor = getattr(request.app.state, "executor", None)
    if not executor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tool execution service unavailable.",
        )

    prompt = payload.prompt or "Describe what is visible on my screen in detail."
    args = {"prompt": prompt}

    result = await executor.execute(
        tool_name="screenshot_capture",
        arguments=args,
        confirmation_id=payload.confirmation_id,
    )

    if result.requires_confirmation:
        return {
            "success": False,
            "decision": "confirmation_required",
            "tool_name": "screenshot_capture",
            "tier": result.tier,
            "confirmation": result.confirmation.model_dump() if result.confirmation else None,
            "message": "Desktop screenshot requires interactive confirmation because screens may contain private data.",
        }

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST if result.error else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "Failed to capture screenshot.",
        )

    return {
        "success": True,
        "decision": result.decision,
        "output": result.output,
    }


@vision_router.post("/camera")
async def capture_camera_api(
    request: Request,
    payload: VisionActionRequest = VisionActionRequest(),
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_token),
) -> dict[str, Any]:
    """Trigger single camera snapshot through CentralExecutor with confirmation checking."""
    executor: CentralExecutor = getattr(request.app.state, "executor", None)
    if not executor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tool execution service unavailable.",
        )

    prompt = payload.prompt or "Describe what is visible in front of the camera."
    dev = payload.device_index if payload.device_index is not None else 0
    args = {"prompt": prompt, "device_index": dev}

    result = await executor.execute(
        tool_name="camera_snapshot",
        arguments=args,
        confirmation_id=payload.confirmation_id,
    )

    if result.requires_confirmation:
        return {
            "success": False,
            "decision": "confirmation_required",
            "tool_name": "camera_snapshot",
            "tier": result.tier,
            "confirmation": result.confirmation.model_dump() if result.confirmation else None,
            "message": "Webcam snapshot requires interactive confirmation.",
        }

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST if result.error else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "Failed to capture camera snapshot.",
        )

    return {
        "success": True,
        "decision": result.decision,
        "output": result.output,
    }
