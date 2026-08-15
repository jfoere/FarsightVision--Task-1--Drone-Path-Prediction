"""Pure-camera-rotation estimation and accumulation from sparse optical flow."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from drone_path.algorithm.optical_flow import OpticalFlowResult


@dataclass(frozen=True, slots=True)
class CameraRotationConfig:
    minimum_points: int = 20
    ransac_reprojection_threshold_pixels: float = 2.0
    ransac_confidence: float = 0.995
    minimum_inlier_ratio: float = 0.50
    maximum_rotation_reprojection_error_pixels: float = 8.0
    maximum_delta_rotation_degrees: float = 15.0

    def __post_init__(self) -> None:
        if self.minimum_points < 4:
            raise ValueError("minimum_points must be at least 4")
        if self.ransac_reprojection_threshold_pixels <= 0:
            raise ValueError(
                "ransac_reprojection_threshold_pixels must be greater than zero"
            )
        if not 0 < self.ransac_confidence <= 1:
            raise ValueError("ransac_confidence must be between 0 and 1")
        if not 0 <= self.minimum_inlier_ratio <= 1:
            raise ValueError("minimum_inlier_ratio must be between 0 and 1")
        if self.maximum_rotation_reprojection_error_pixels <= 0:
            raise ValueError(
                "maximum_rotation_reprojection_error_pixels must be greater than zero"
            )
        if not 0 < self.maximum_delta_rotation_degrees <= 180:
            raise ValueError(
                "maximum_delta_rotation_degrees must be between 0 and 180"
            )


@dataclass(frozen=True, slots=True)
class CameraRotationMeasurement:
    """Accumulated physical camera rotation for one non-translation section."""

    valid: bool
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float
    inlier_ratio: float
    median_reprojection_error_pixels: float
    sample_count: int
    tracked_count: int
    # Axis-angle rotation vector in camera coordinates. X is the gimbal-pitch
    # axis; a level drone's yaw axis appears in the camera Y/Z plane when the
    # gimbal is pitched.
    rotation_vector_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def unavailable(cls) -> "CameraRotationMeasurement":
        return cls(
            valid=False,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            inlier_ratio=0.0,
            median_reprojection_error_pixels=0.0,
            sample_count=0,
            tracked_count=0,
            rotation_vector_degrees=(0.0, 0.0, 0.0),
        )


class CameraRotationHandler:
    """Estimate per-frame pure rotation and accumulate it across a section."""

    def __init__(self, config: CameraRotationConfig | None = None) -> None:
        self.config = config or CameraRotationConfig()
        self.reset()

    @property
    def measurement(self) -> CameraRotationMeasurement:
        return self._measurement

    def reset(self) -> None:
        self._camera_orientation = np.eye(3, dtype=np.float64)
        self._measurement = CameraRotationMeasurement.unavailable()

    def update(
        self,
        flow: OpticalFlowResult,
        frame_size: tuple[int, int],
        *,
        horizontal_fov_degrees: float,
    ) -> CameraRotationMeasurement:
        """Accumulate one frame-pair rotation if it fits a pure-rotation model."""
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame_size dimensions must be greater than zero")
        if not 0 < horizontal_fov_degrees < 180:
            raise ValueError("horizontal_fov_degrees must be between 0 and 180")
        if flow.tracked_count < self.config.minimum_points:
            return self._measurement

        camera_matrix = _camera_matrix(width, height, horizontal_fov_degrees)
        try:
            homography, inlier_mask = cv2.findHomography(
                flow.previous_points,
                flow.current_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=(
                    self.config.ransac_reprojection_threshold_pixels
                ),
                confidence=self.config.ransac_confidence,
            )
        except cv2.error:
            return self._measurement
        if homography is None or inlier_mask is None or not np.isfinite(homography).all():
            return self._measurement

        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / flow.tracked_count
        if (
            inlier_count < self.config.minimum_points
            or inlier_ratio < self.config.minimum_inlier_ratio
        ):
            return self._measurement

        coordinate_rotation = _rotation_from_homography(
            homography,
            camera_matrix,
        )
        if coordinate_rotation is None:
            return self._measurement

        delta_rotation_vector, _ = cv2.Rodrigues(coordinate_rotation)
        delta_degrees = math.degrees(float(np.linalg.norm(delta_rotation_vector)))
        if delta_degrees > self.config.maximum_delta_rotation_degrees:
            return self._measurement

        rotation_homography = (
            camera_matrix @ coordinate_rotation @ np.linalg.inv(camera_matrix)
        )
        projected = cv2.perspectiveTransform(
            flow.previous_points.reshape(-1, 1, 2),
            rotation_homography,
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected[inliers] - flow.current_points[inliers], axis=1)
        finite_errors = errors[np.isfinite(errors)]
        if finite_errors.size < self.config.minimum_points:
            return self._measurement
        median_error = float(np.median(finite_errors))
        if median_error > self.config.maximum_rotation_reprojection_error_pixels:
            return self._measurement

        # The homography contains the coordinate transform X2 = R * X1.
        # Physical camera orientation therefore changes by the transpose.
        physical_delta = coordinate_rotation.T
        self._camera_orientation = self._camera_orientation @ physical_delta
        self._camera_orientation = _nearest_rotation(self._camera_orientation)
        yaw, pitch, roll = _camera_euler_degrees(self._camera_orientation)
        accumulated_vector, _ = cv2.Rodrigues(self._camera_orientation)
        rotation_vector_degrees = tuple(
            math.degrees(float(value)) for value in accumulated_vector.reshape(3)
        )
        self._measurement = CameraRotationMeasurement(
            valid=True,
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            roll_degrees=roll,
            inlier_ratio=inlier_ratio,
            median_reprojection_error_pixels=median_error,
            sample_count=self._measurement.sample_count + 1,
            tracked_count=flow.tracked_count,
            rotation_vector_degrees=rotation_vector_degrees,
        )
        return self._measurement


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


def _rotation_from_homography(
    homography: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray | None:
    normalized = np.linalg.inv(camera_matrix) @ homography @ camera_matrix
    determinant = float(np.linalg.det(normalized))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        return None
    if determinant < 0:
        normalized = -normalized
        determinant = -determinant
    normalized /= math.cbrt(determinant)
    return _nearest_rotation(normalized)


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right_transposed = np.linalg.svd(matrix)
    rotation = left @ right_transposed
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transposed
    return rotation


def _camera_euler_degrees(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return yaw-right, pitch-down, and roll-clockwise Euler angles."""
    pitch = math.asin(float(np.clip(rotation[1, 2], -1.0, 1.0)))
    yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
    roll = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
    return tuple(math.degrees(value) for value in (yaw, pitch, roll))
