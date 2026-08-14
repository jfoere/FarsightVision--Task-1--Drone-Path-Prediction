"""Shared video input helpers used by production and debug applications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


class VideoOpenError(RuntimeError):
    """Raised when an input video cannot be opened or decoded."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Metadata read from an input video."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float | None


def open_video(video_path: str | Path) -> tuple[Path, cv2.VideoCapture]:
    """Open a local video and return its resolved path and capture object."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise VideoOpenError(f"Video file does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoOpenError(f"OpenCV could not open the video: {path}")

    return path, capture


def read_video_info(video_path: str | Path) -> VideoInfo:
    """Read metadata and decode the first frame to validate the video."""
    path, capture = open_video(video_path)
    try:
        decoded, frame = capture.read()
        if not decoded or frame is None:
            raise VideoOpenError(f"OpenCV could not decode the first frame: {path}")

        height, width = frame.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 0.0

        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        duration = frame_count / fps if frame_count > 0 and fps > 0 else None

        return VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )
    finally:
        capture.release()
