"""Screen and camera capture engine with visible usage indicators and immediate resource release."""

import io
import logging
from typing import Callable
from PIL import ImageGrab
import cv2

logger = logging.getLogger("nino.vision.capture")


class ScreenCaptureEngine:
    """Captures single-frame desktop screenshots on demand."""

    @staticmethod
    def capture() -> bytes:
        """Capture the primary display screen and return PNG bytes."""
        logger.info("[🖥️ SCREEN CAPTURE] Capturing desktop screen on demand...")
        screenshot = ImageGrab.grab()
        buf = io.BytesIO()
        screenshot.save(buf, format="PNG")
        raw_bytes = buf.getvalue()
        logger.info("[🖥️ SCREEN CAPTURE] Successfully captured desktop (%d bytes)", len(raw_bytes))
        return raw_bytes


class CameraCaptureEngine:
    """Captures single-frame webcam snapshots on demand with visible status indication."""

    def __init__(self, default_device: int = 0, indicator_callback: Callable[[str], None] | None = None):
        self.default_device = default_device
        self.indicator_callback = indicator_callback

    def _notify_indicator(self, message: str) -> None:
        logger.info(message)
        if self.indicator_callback:
            try:
                self.indicator_callback(message)
            except Exception as e:
                logger.debug("Indicator callback failed: %s", e)

    def capture(self, device_index: int | None = None) -> bytes:
        """Capture exactly 1 frame from the specified camera and immediately release hardware."""
        dev = self.default_device if device_index is None else device_index

        self._notify_indicator(f"[📷 CAMERA ACTIVE] Activating camera device {dev} for snapshot...")

        cap = cv2.VideoCapture(dev)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera device index {dev}. Verify webcam is connected and accessible.")

            # Read a single snapshot frame
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Failed to capture frame from camera device {dev}.")

            # Encode frame to JPEG in memory
            success, encoded_image = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError("Failed to encode camera frame to JPEG.")

            raw_bytes = bytes(encoded_image)
            self._notify_indicator(f"[📷 CAMERA DEACTIVATED] Captured single frame ({len(raw_bytes)} bytes). Camera hardware released.")
            return raw_bytes
        finally:
            cap.release()
            logger.debug("Camera hardware resource released.")
