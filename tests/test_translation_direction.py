"""Tests for camera-relative translation-direction recovery."""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from drone_path.algorithm import (
    OpticalFlowResult,
    TranslationDirectionConfig,
    TranslationDirectionEstimator,
)


def _flow(previous: np.ndarray, current: np.ndarray) -> OpticalFlowResult:
    count = previous.shape[0]
    zeros = np.zeros((count,), dtype=np.float32)
    return OpticalFlowResult(
        previous_points=previous.astype(np.float32),
        current_points=current.astype(np.float32),
        tracking_errors=zeros,
        forward_backward_errors=zeros.copy(),
        detected_count=count,
    )


def _project(points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    image = points[:, :2] / points[:, 2, np.newaxis]
    return image * np.array([camera_matrix[0, 0], camera_matrix[1, 1]]) + np.array(
        [camera_matrix[0, 2], camera_matrix[1, 2]]
    )


class TranslationDirectionEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        cv2.setRNGSeed(7)
        self.frame_size = (960, 540)
        focal_length = 960 / (2 * math.tan(math.radians(90 / 2)))
        self.camera_matrix = np.array(
            [[focal_length, 0, 479.5], [0, focal_length, 269.5], [0, 0, 1]],
            dtype=np.float64,
        )
        random = np.random.default_rng(7)
        self.world_points = np.column_stack(
            (
                random.uniform(-3.0, 3.0, 250),
                random.uniform(-1.8, 1.8, 250),
                random.uniform(5.0, 12.0, 250),
            )
        )
        self.estimator = TranslationDirectionEstimator(
            TranslationDirectionConfig(smoothing_alpha=1.0)
        )

    def test_recovers_diagonal_camera_movement_direction(self) -> None:
        camera_movement = np.array([0.4, 0.0, 0.4])
        previous = _project(self.world_points, self.camera_matrix)
        current = _project(
            self.world_points - camera_movement,
            self.camera_matrix,
        )

        measurement = self.estimator.estimate(
            _flow(previous, current),
            self.frame_size,
            horizontal_fov_degrees=90.0,
        )

        self.assertTrue(measurement.valid)
        self.assertAlmostEqual(measurement.horizontal_angle_degrees, 45.0, delta=1.0)
        self.assertGreater(measurement.inlier_ratio, 0.9)

    def test_reports_unavailable_with_too_few_tracks(self) -> None:
        points = np.zeros((5, 2), dtype=np.float32)

        measurement = self.estimator.estimate(
            _flow(points, points),
            self.frame_size,
            horizontal_fov_degrees=90.0,
        )

        self.assertFalse(measurement.valid)
        self.assertEqual(measurement.tracked_count, 5)

    def test_rejects_invalid_field_of_view(self) -> None:
        points = np.zeros((25, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "between 0 and 180"):
            self.estimator.estimate(
                _flow(points, points),
                self.frame_size,
                horizontal_fov_degrees=180.0,
            )


if __name__ == "__main__":
    unittest.main()
