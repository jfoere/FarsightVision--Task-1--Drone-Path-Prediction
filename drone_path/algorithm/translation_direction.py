"""Camera-relative translation direction from sparse point correspondences."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from drone_path.algorithm.optical_flow import OpticalFlowResult


@dataclass(frozen=True, slots=True)
class TranslationDirectionConfig:
    """Robust-pose and temporal-smoothing parameters."""

    minimum_points: int = 20
    ransac_probability: float = 0.999
    ransac_threshold_pixels: float = 1.5
    minimum_pose_inlier_ratio: float = 0.20
    minimum_horizontal_component: float = 0.10
    smoothing_alpha: float = 0.35

    def __post_init__(self) -> None:
        if self.minimum_points < 5:
            raise ValueError("minimum_points must be at least 5")
        if not 0 < self.ransac_probability <= 1:
            raise ValueError("ransac_probability must be between 0 and 1")
        if self.ransac_threshold_pixels <= 0:
            raise ValueError("ransac_threshold_pixels must be greater than zero")
        if not 0 <= self.minimum_pose_inlier_ratio <= 1:
            raise ValueError("minimum_pose_inlier_ratio must be between 0 and 1")
        if not 0 <= self.minimum_horizontal_component <= 1:
            raise ValueError("minimum_horizontal_component must be between 0 and 1")
        if not 0 < self.smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TranslationDirectionMeasurement:
    """Unit camera-motion vector in OpenCV camera coordinates."""

    valid: bool
    right: float
    down: float
    forward: float
    horizontal_angle_degrees: float
    inlier_ratio: float
    inlier_count: int
    tracked_count: int

    @classmethod
    def unavailable(
        cls,
        *,
        tracked_count: int = 0,
        inlier_count: int = 0,
    ) -> "TranslationDirectionMeasurement":
        ratio = inlier_count / tracked_count if tracked_count else 0.0
        return cls(
            valid=False,
            right=0.0,
            down=0.0,
            forward=0.0,
            horizontal_angle_degrees=0.0,
            inlier_ratio=ratio,
            inlier_count=inlier_count,
            tracked_count=tracked_count,
        )


class TranslationDirectionEstimator:
    """Recover and smooth translation direction up to unknown scale."""

    def __init__(self, config: TranslationDirectionConfig | None = None) -> None:
        self.config = config or TranslationDirectionConfig()
        self.reset()

    def reset(self) -> None:
        self._smoothed_direction: np.ndarray | None = None

    def estimate(
        self,
        flow: OpticalFlowResult,
        frame_size: tuple[int, int],
        *,
        horizontal_fov_degrees: float,
    ) -> TranslationDirectionMeasurement:
        """Estimate camera movement from an earlier frame to a later frame."""
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame_size dimensions must be greater than zero")
        if not 0 < horizontal_fov_degrees < 180:
            raise ValueError("horizontal_fov_degrees must be between 0 and 180")

        tracked_count = flow.tracked_count
        if tracked_count < self.config.minimum_points:
            return TranslationDirectionMeasurement.unavailable(
                tracked_count=tracked_count
            )

        camera_matrix = _camera_matrix(
            width,
            height,
            horizontal_fov_degrees,
        )
        try:
            essential_matrix, essential_mask = cv2.findEssentialMat(
                flow.previous_points,
                flow.current_points,
                camera_matrix,
                method=cv2.RANSAC,
                prob=self.config.ransac_probability,
                threshold=self.config.ransac_threshold_pixels,
            )
            if essential_matrix is None or essential_mask is None:
                return TranslationDirectionMeasurement.unavailable(
                    tracked_count=tracked_count
                )

            pose_inliers, rotation, translation, _ = cv2.recoverPose(
                essential_matrix[:3, :3],
                flow.previous_points,
                flow.current_points,
                camera_matrix,
                mask=essential_mask,
            )
        except cv2.error:
            return TranslationDirectionMeasurement.unavailable(
                tracked_count=tracked_count
            )

        inlier_count = int(pose_inliers)
        inlier_ratio = inlier_count / tracked_count
        if (
            inlier_count < self.config.minimum_points
            or inlier_ratio < self.config.minimum_pose_inlier_ratio
        ):
            return TranslationDirectionMeasurement.unavailable(
                tracked_count=tracked_count,
                inlier_count=inlier_count,
            )

        # recoverPose returns P2 = R * P1 + t. The second camera center in the
        # first camera's coordinates is therefore C2 = -R.T * t.
        direction = (-rotation.T @ translation).reshape(3).astype(np.float64)
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= 0 or not np.isfinite(direction).all():
            return TranslationDirectionMeasurement.unavailable(
                tracked_count=tracked_count,
                inlier_count=inlier_count,
            )
        direction /= magnitude

        if self._smoothed_direction is not None:
            alpha = self.config.smoothing_alpha
            direction = alpha * direction + (1 - alpha) * self._smoothed_direction
            magnitude = float(np.linalg.norm(direction))
            if magnitude <= 0:
                return TranslationDirectionMeasurement.unavailable(
                    tracked_count=tracked_count,
                    inlier_count=inlier_count,
                )
            direction /= magnitude

        horizontal_component = math.hypot(float(direction[0]), float(direction[2]))
        if horizontal_component < self.config.minimum_horizontal_component:
            return TranslationDirectionMeasurement.unavailable(
                tracked_count=tracked_count,
                inlier_count=inlier_count,
            )

        self._smoothed_direction = direction
        horizontal_angle = math.degrees(
            math.atan2(float(direction[0]), float(direction[2]))
        )
        return TranslationDirectionMeasurement(
            valid=True,
            right=float(direction[0]),
            down=float(direction[1]),
            forward=float(direction[2]),
            horizontal_angle_degrees=horizontal_angle,
            inlier_ratio=inlier_ratio,
            inlier_count=inlier_count,
            tracked_count=tracked_count,
        )


def _camera_matrix(
    width: int,
    height: int,
    horizontal_fov_degrees: float,
) -> np.ndarray:
    half_fov = math.radians(horizontal_fov_degrees / 2)
    focal_length = width / (2 * math.tan(half_fov))
    return np.array(
        [
            [focal_length, 0.0, (width - 1) / 2],
            [0.0, focal_length, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
