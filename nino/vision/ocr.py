"""Optional Tesseract OCR fallback interface for high-density document transcription."""

import io
import logging
from PIL import Image

logger = logging.getLogger("nino.vision.ocr")


class TesseractOCR:
    """Optional OCR processor using Tesseract for verbatim text extraction."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def is_available(self) -> bool:
        """Check if Tesseract OCR is enabled and the underlying package is installed."""
        if not self.enabled:
            return False
        try:
            import pytesseract  # type: ignore
            # Test getting version to check if binary is accessible
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_text(self, image_data: bytes) -> str:
        """Extract text using Tesseract OCR if enabled and available."""
        if not self.enabled:
            logger.debug("Tesseract OCR is disabled by policy.")
            return ""

        try:
            import pytesseract  # type: ignore
            with Image.open(io.BytesIO(image_data)) as img:
                text = pytesseract.image_to_string(img)
                return text.strip()
        except ImportError:
            logger.warning("pytesseract package is not installed. Falling back to VLM text extraction.")
            return ""
        except Exception as e:
            logger.warning("Tesseract OCR extraction failed: %s. Falling back to VLM.", e)
            return ""
