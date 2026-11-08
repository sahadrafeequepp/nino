"""Temporary image storage manager guaranteeing zero residual image files on disk."""

from contextlib import contextmanager
import io
import logging
import os
from pathlib import Path
import tempfile
from typing import Generator
from PIL import Image

logger = logging.getLogger("nino.vision.storage")


class TempImageManager:
    """Manages ephemeral image buffers and temp disk storage with guaranteed cleanup."""

    def __init__(self, temp_dir: str = "data/tmp_vision"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_image(image_data: bytes) -> tuple[str, int, int]:
        """Validate raw image bytes and return format, width, and height.

        Raises:
            ValueError: If the data is empty or does not represent a valid image.
        """
        if not image_data:
            raise ValueError("Image data is empty.")

        try:
            with Image.open(io.BytesIO(image_data)) as img:
                img.verify()
                fmt = img.format or "UNKNOWN"
                w, h = img.size
                return fmt, w, h
        except Exception as e:
            raise ValueError(f"Invalid image content or unsupported format: {e}") from e

    @contextmanager
    def temp_image_file(
        self,
        image_data: bytes,
        suffix: str = ".png",
    ) -> Generator[Path, None, None]:
        """Write image bytes to a secure temporary file and guarantee deletion on exit.

        Yields:
            Path to the temporary file.
        """
        # Ensure target temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        fd, raw_path = tempfile.mkstemp(suffix=suffix, dir=str(self.temp_dir))
        temp_path = Path(raw_path)

        try:
            with os.fdopen(fd, "wb") as f:
                f.write(image_data)
            logger.debug("Created temporary image file: %s (%d bytes)", temp_path, len(image_data))
            yield temp_path
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug("Successfully deleted temporary image file: %s", temp_path)
                except Exception as e:
                    logger.warning("Failed to delete temporary image %s: %e", temp_path, e)

    def cleanup_all(self) -> int:
        """Purge all files remaining in the temp directory. Returns count of deleted files."""
        if not self.temp_dir.exists():
            return 0

        deleted = 0
        for item in self.temp_dir.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning("Could not unlink residual temp image %s: %s", item, e)
        if deleted > 0:
            logger.info("Cleaned up %d residual temporary image files from %s", deleted, self.temp_dir)
        return deleted
