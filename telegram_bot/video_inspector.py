"""Video Inspection and Keyframe Extraction for Nino Telegram Bot.

Uses OpenCV to inspect video metadata, extract keyframe snapshots, and generate
composite frame grids for multimodal AI reasoning.
"""

import io
import logging
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np

logger = logging.getLogger("nino.telegram.video_inspector")


class VideoMetadata:
    """Stores video dimensions, fps, duration, and frame count."""

    def __init__(
        self,
        duration_sec: float,
        width: int,
        height: int,
        fps: float,
        total_frames: int,
    ):
        self.duration_sec = duration_sec
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames

    def summary(self) -> str:
        return f"Duration: {self.duration_sec:.1f}s | Resolution: {self.width}x{self.height} | FPS: {self.fps:.1f}"


def inspect_video_file(video_path: str | Path) -> Optional[VideoMetadata]:
    """Read metadata from a video file on disk using OpenCV."""
    path_str = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        logger.warning("Could not open video file: %s", path_str)
        return None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (total_frames / fps) if fps > 0 else 0.0
        return VideoMetadata(
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
        )
    finally:
        cap.release()


def extract_keyframes(
    video_path: str | Path,
    num_frames: int = 4,
) -> list[bytes]:
    """Extract evenly spaced keyframes across the video as JPEG byte arrays."""
    path_str = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    # Calculate target frame indices evenly spaced (e.g. 10%, 35%, 65%, 90%)
    indices = [
        int(total_frames * (i + 1) / (num_frames + 1))
        for i in range(num_frames)
    ]

    frames_bytes: list[bytes] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Resize if frame is excessively large (max 1080p for efficiency)
            h, w = frame.shape[:2]
            if w > 1280 or h > 720:
                scale = min(1280 / w, 720 / h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if success:
                frames_bytes.append(encoded.tobytes())
    finally:
        cap.release()

    return frames_bytes


def generate_video_storyboard(
    video_path: str | Path,
    num_frames: int = 4,
) -> Optional[bytes]:
    """Generate a single stitched 2x2 storyboard image of video keyframes."""
    path_str = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return None

    indices = [
        int(total_frames * (i + 1) / (num_frames + 1))
        for i in range(num_frames)
    ]

    images: list[np.ndarray] = []
    target_w, target_h = 480, 270

    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                resized = cv2.resize(frame, (target_w, target_h))
                images.append(resized)
    finally:
        cap.release()

    if not images:
        return None

    # Pad if fewer than 4 frames
    while len(images) < 4:
        images.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    # Stitch into 2x2 grid
    top_row = np.hstack([images[0], images[1]])
    bottom_row = np.hstack([images[2], images[3]])
    grid = np.vstack([top_row, bottom_row])

    success, encoded = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if success:
        return encoded.tobytes()

    return None
