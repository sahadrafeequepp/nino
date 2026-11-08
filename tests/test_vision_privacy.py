"""Tests for Privacy Gateway image quarantine and audit log content omission."""

import json
import pytest

from nino.security.audit import AuditLogger
from nino.security.privacy_gateway import MediaQuarantineException, PrivacyGateway


@pytest.mark.asyncio
async def test_privacy_gateway_refuses_image_data_uri_by_default():
    gateway = PrivacyGateway(allow_external=True, allow_external_images=False)
    image_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # Content check directly
    with pytest.raises(MediaQuarantineException, match="Image data URI detected"):
        gateway.check_media_quarantine(image_uri)

    # In dispatch_external messages
    messages = [{"role": "user", "content": f"Look at this: {image_uri}"}]
    with pytest.raises(MediaQuarantineException, match="Image data URI detected"):
        await gateway.dispatch_external(provider="test_external", messages=messages)


@pytest.mark.asyncio
async def test_privacy_gateway_refuses_image_keyword_parameters():
    gateway = PrivacyGateway(allow_external=True, allow_external_images=False)
    messages = [{"role": "user", "content": "Analyze image"}]

    # In dispatch_external kwargs
    with pytest.raises(MediaQuarantineException, match="Refusing external dispatch with image parameter 'images'"):
        await gateway.dispatch_external(
            provider="test_external",
            messages=messages,
            images=["base64_data_here"],
        )


@pytest.mark.asyncio
async def test_privacy_gateway_refuses_binary_image_bytes():
    gateway = PrivacyGateway(allow_external=True, allow_external_images=False)
    raw_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

    with pytest.raises(MediaQuarantineException, match="Binary audio/media data transmission"):
        gateway.check_media_quarantine(raw_bytes)


def test_audit_logger_records_capture_metadata_never_content(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=str(log_file))

    entry = logger.log_image_capture(capture_type="screenshot", status="success")
    assert entry["event_type"] == "image_capture"
    assert entry["capture_type"] == "screenshot"
    assert entry["status"] == "success"
    assert entry["content"] == "[IMAGE_CONTENT_NEVER_LOGGED]"

    # Verify JSONL on disk
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["capture_type"] == "screenshot"
    assert logged["content"] == "[IMAGE_CONTENT_NEVER_LOGGED]"
    assert "data:image" not in lines[0]
