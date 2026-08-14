"""Tests for robust global-motion measurement."""

from __future__ import annotations

import math
import unittest

import numpy as np

from drone_path.algorithm import GlobalMotionEstimator, OpticalFlowResult


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


class GlobalMotionEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        random = np.random.default_rng(11)
        self.points = random.uniform(
            low=(20.0, 20.0),
            high=(620.0, 340.0),
            size=(120, 2),
        ).astype(np.float32)
        self.estimator = GlobalMotionEstimator()

    def test_measures_translation(self) -> None:
        translation = np.array([5.0, -3.0], dtype=np.float32)

        measurement = self.estimator.measure(
            _flow(self.points, self.points + translation),
            (640, 360),
        )

        self.assertTrue(measurement.valid)
        self.assertAlmostEqual(measurement.translation_x_pixels, 5.0, delta=0.01)
        self.assertAlmostEqual(measurement.translation_y_pixels, -3.0, delta=0.01)
        self.assertAlmostEqual(measurement.rotation_degrees, 0.0, delta=0.01)
        self.assertAlmostEqual(measurement.scale, 1.0, delta=0.0001)
        self.assertAlmostEqual(measurement.inlier_ratio, 1.0, delta=0.001)
        self.assertAlmostEqual(
            measurement.median_flow_pixels,
            math.hypot(5.0, -3.0),
            delta=0.01,
        )

    def test_measures_rotation_around_frame_center_with_outliers(self) -> None:
        angle_degrees = 3.0
        angle = math.radians(angle_degrees)
        linear = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=np.float32,
        )
        center = np.array([319.5, 179.5], dtype=np.float32)
        current = (self.points - center) @ linear.T + center
        current[:12] += np.array([80.0, -60.0], dtype=np.float32)

        measurement = self.estimator.measure(
            _flow(self.points, current),
            (640, 360),
        )

        self.assertTrue(measurement.valid)
        self.assertAlmostEqual(measurement.rotation_degrees, angle_degrees, delta=0.05)
        self.assertAlmostEqual(measurement.translation_x_pixels, 0.0, delta=0.1)
        self.assertAlmostEqual(measurement.translation_y_pixels, 0.0, delta=0.1)
        self.assertAlmostEqual(measurement.scale, 1.0, delta=0.001)
        self.assertLess(measurement.inlier_ratio, 0.95)
        self.assertGreater(measurement.inlier_ratio, 0.85)

    def test_reports_unavailable_when_too_few_points_exist(self) -> None:
        points = self.points[:3]

        measurement = self.estimator.measure(_flow(points, points), (640, 360))

        self.assertFalse(measurement.valid)
        self.assertEqual(measurement.tracked_count, 3)


if __name__ == "__main__":
    unittest.main()
