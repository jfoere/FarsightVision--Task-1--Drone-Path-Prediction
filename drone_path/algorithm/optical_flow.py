"""Sparse optical-flow measurements with no UI dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class OpticalFlowConfig:
    """Parameters for feature detection and Lucas-Kanade tracking."""

    max_corners: int = 300
    quality_level: float = 0.01
    min_distance: float = 12.0
    block_size: int = 7
    window_size: tuple[int, int] = (21, 21)
    max_pyramid_level: int = 3
    max_tracking_error: float = 30.0
    max_forward_backward_error: float = 1.5


@dataclass(frozen=True, slots=True)
class OpticalFlowResult:
    """Valid point correspondences measured between two frames."""

    previous_points: np.ndarray
    current_points: np.ndarray
    tracking_errors: np.ndarray
    forward_backward_errors: np.ndarray
    detected_count: int

    @property
    def tracked_count(self) -> int:
        return int(self.current_points.shape[0])

    @classmethod
    def empty(cls, detected_count: int = 0) -> "OpticalFlowResult":
        points = np.empty((0, 2), dtype=np.float32)
        errors = np.empty((0,), dtype=np.float32)
        return cls(points, points.copy(), errors, errors.copy(), detected_count)


class OpticalFlowEstimator:
    """Track newly detected feature points across one pair of frames."""

    def __init__(self, config: OpticalFlowConfig | None = None) -> None:
        self.config = config or OpticalFlowConfig()

    def estimate(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
    ) -> OpticalFlowResult:
        """Return reliable sparse point tracks from previous to current frame."""
        previous_gray = _to_grayscale(previous_frame)
        current_gray = _to_grayscale(current_frame)
        if previous_gray.shape != current_gray.shape:
            raise ValueError(
                "Optical-flow frames must have identical dimensions: "
                f"{previous_gray.shape} != {current_gray.shape}"
            )

        config = self.config
        detected = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=config.max_corners,
            qualityLevel=config.quality_level,
            minDistance=config.min_distance,
            blockSize=config.block_size,
        )
        if detected is None:
            return OpticalFlowResult.empty()

        detected_count = int(detected.shape[0])
        tracked, forward_status, tracking_errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            detected,
            None,
            winSize=config.window_size,
            maxLevel=config.max_pyramid_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if tracked is None or forward_status is None or tracking_errors is None:
            return OpticalFlowResult.empty(detected_count)

        previous_points = detected.reshape(-1, 2)
        current_points = tracked.reshape(-1, 2)
        tracking_errors = tracking_errors.reshape(-1)
        forward_status = forward_status.reshape(-1).astype(bool)

        height, width = current_gray.shape
        valid = (
            forward_status
            & np.isfinite(previous_points).all(axis=1)
            & np.isfinite(current_points).all(axis=1)
            & np.isfinite(tracking_errors)
            & (tracking_errors <= config.max_tracking_error)
            & (current_points[:, 0] >= 0)
            & (current_points[:, 0] < width)
            & (current_points[:, 1] >= 0)
            & (current_points[:, 1] < height)
        )
        if not np.any(valid):
            return OpticalFlowResult.empty(detected_count)

        previous_points = previous_points[valid]
        current_points = current_points[valid]
        tracking_errors = tracking_errors[valid]

        tracked_back, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            current_points.reshape(-1, 1, 2),
            None,
            winSize=config.window_size,
            maxLevel=config.max_pyramid_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if tracked_back is None or backward_status is None:
            return OpticalFlowResult.empty(detected_count)

        tracked_back = tracked_back.reshape(-1, 2)
        backward_status = backward_status.reshape(-1).astype(bool)
        forward_backward_errors = np.linalg.norm(
            previous_points - tracked_back,
            axis=1,
        )
        reliable = (
            backward_status
            & np.isfinite(tracked_back).all(axis=1)
            & np.isfinite(forward_backward_errors)
            & (forward_backward_errors <= config.max_forward_backward_error)
        )

        return OpticalFlowResult(
            previous_points=previous_points[reliable],
            current_points=current_points[reliable],
            tracking_errors=tracking_errors[reliable],
            forward_backward_errors=forward_backward_errors[reliable],
            detected_count=detected_count,
        )


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Validate a frame and return an 8-bit grayscale image."""
    if frame is None or frame.size == 0:
        raise ValueError("Optical-flow frames cannot be empty")
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")
    if frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")
