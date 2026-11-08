"""Tests verifying zero residual image files on disk during vision processing."""

import io
from pathlib import Path
import pytest
from PIL import Image

from nino.vision.storage import TempImageManager


def create_sample_image_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_temp_image_cleaned_up_on_normal_exit(tmp_path):
    temp_dir = tmp_path / "tmp_vision"
    manager = TempImageManager(str(temp_dir))
    data = create_sample_image_bytes()

    temp_file_path: Path | None = None
    with manager.temp_image_file(data, suffix=".png") as path:
        temp_file_path = path
        assert temp_file_path.exists()
        assert temp_file_path.is_file()
        assert temp_file_path.stat().st_size == len(data)

    # Immediately upon exit, the file must be deleted
    assert not temp_file_path.exists()
    assert list(temp_dir.iterdir()) == []


def test_temp_image_cleaned_up_on_exception(tmp_path):
    temp_dir = tmp_path / "tmp_vision"
    manager = TempImageManager(str(temp_dir))
    data = create_sample_image_bytes()

    temp_file_path: Path | None = None
    with pytest.raises(RuntimeError, match="Simulated vision processing failure"):
        with manager.temp_image_file(data, suffix=".png") as path:
            temp_file_path = path
            assert temp_file_path.exists()
            raise RuntimeError("Simulated vision processing failure")

    # Even with an exception, the file must be cleaned up
    assert temp_file_path is not None
    assert not temp_file_path.exists()
    assert list(temp_dir.iterdir()) == []


def test_cleanup_all_clears_orphan_files(tmp_path):
    temp_dir = tmp_path / "tmp_vision"
    manager = TempImageManager(str(temp_dir))

    # Create dummy residual files
    (temp_dir / "residual_1.png").write_bytes(b"residual 1")
    (temp_dir / "residual_2.jpg").write_bytes(b"residual 2")
    assert len(list(temp_dir.iterdir())) == 2

    cleaned = manager.cleanup_all()
    assert cleaned == 2
    assert list(temp_dir.iterdir()) == []
