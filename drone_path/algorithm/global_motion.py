"""Robust global-motion measurements derived from sparse optical flow."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from drone_path.algorithm.optical_flow import OpticalFlowResult


@dataclass(frozen=True, slots=True)
class GlobalMotionConfig:
    """Parameters for robust similarity-transform estimation."""

    minimum_points: int = 6
    ransac_reprojection_threshold: float = 2.0
    ransac_max_iterations: int = 2_000
    ransac_confidence: float = 0.99
    refine_iterations: int = 10


@dataclass(frozen=True, slots=True)
class GlobalMotionMeasurement:
    """Raw global motion between two consecutive frames."""

    valid: bool
    median_flow_pixels: float
    translation_x_pixels: float
    translation_y_pixels: float
    rotation_degrees: float
    scale: float
    inlier_ratio: float
    inlier_count: int
    tracked_count: int

    @classmethod
    def unavailable(
        cls,
        *,
        median_flow_pixels: float = 0.0,
        tracked_count: int = 0,
    ) -> "GlobalMotionMeasurement":
        return cls(
            valid=False,
            median_flow_pixels=median_flow_pixels,
            translation_x_pixels=0.0,
            translation_y_pixels=0.0,
            rotation_degrees=0.0,
            scale=1.0,
            inlier_ratio=0.0,
            inlier_count=0,
            tracked_count=tracked_count,
        )


class GlobalMotionEstimator:
    """Fit a robust translation/rotation/scale model to tracked points."""

    def __init__(self, config: GlobalMotionConfig | None = None) -> None:
        self.config = config or GlobalMotionConfig()

    def measure(
        self,
        flow: OpticalFlowResult,
        frame_size: tuple[int, int],
    ) -> GlobalMotionMeasurement:
        """Measure motion for a `(width, height)` analysis-frame size."""
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame_size dimensions must be greater than zero")

        tracked_count = flow.tracked_count
        median_flow = _median_flow_magnitude(flow)
        if tracked_count < self.config.minimum_points:
            return GlobalMotionMeasurement.unavailable(
                median_flow_pixels=median_flow,
                tracked_count=tracked_count,
            )

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            flow.previous_points,
            flow.current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reprojection_threshold,
            maxIters=self.config.ransac_max_iterations,
            confidence=self.config.ransac_confidence,
            refineIters=self.config.refine_iterations,
        )
        if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
            return GlobalMotionMeasurement.unavailable(
                median_flow_pixels=median_flow,
                tracked_count=tracked_count,
            )

        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < self.config.minimum_points:
            return GlobalMotionMeasurement.unavailable(
                median_flow_pixels=median_flow,
                tracked_count=tracked_count,
            )

        linear = matrix[:, :2]
        scale = math.hypot(float(linear[0, 0]), float(linear[1, 0]))
        rotation_degrees = math.degrees(
            math.atan2(float(linear[1, 0]), float(linear[0, 0]))
        )

        center = np.array([(width - 1) / 2, (height - 1) / 2, 1.0])
        transformed_center = matrix @ center
        translation = transformed_center - center[:2]

        return GlobalMotionMeasurement(
            valid=True,
            median_flow_pixels=median_flow,
            translation_x_pixels=float(translation[0]),
            translation_y_pixels=float(translation[1]),
            rotation_degrees=rotation_degrees,
            scale=scale,
            inlier_ratio=inlier_count / tracked_count,
            inlier_count=inlier_count,
            tracked_count=tracked_count,
        )


def _median_flow_magnitude(flow: OpticalFlowResult) -> float:
    if flow.tracked_count == 0:
        return 0.0
    displacement = flow.current_points - flow.previous_points
    magnitudes = np.linalg.norm(displacement, axis=1)
    return float(np.median(magnitudes))
