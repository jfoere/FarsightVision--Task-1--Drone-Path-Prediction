"""Tests for raw sparse optical-flow measurement."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from drone_path.algorithm import OpticalFlowEstimator


class OpticalFlowEstimatorTests(unittest.TestCase):
    def test_tracks_known_translation(self) -> None:
        random = np.random.default_rng(7)
        texture = random.integers(0, 256, size=(240, 320), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (5, 5), 0)
        previous = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

        expected_dx = 4.0
        expected_dy = -3.0
        transform = np.float32([[1, 0, expected_dx], [0, 1, expected_dy]])
        current = cv2.warpAffine(previous, transform, (320, 240))

        result = OpticalFlowEstimator().estimate(previous, current)
        displacement = result.current_points - result.previous_points
        median_dx, median_dy = np.median(displacement, axis=0)

        self.assertGreater(result.tracked_count, 150)
        self.assertAlmostEqual(float(median_dx), expected_dx, delta=0.25)
        self.assertAlmostEqual(float(median_dy), expected_dy, delta=0.25)

    def test_returns_empty_result_when_no_features_exist(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        result = OpticalFlowEstimator().estimate(frame, frame.copy())

        self.assertEqual(result.detected_count, 0)
        self.assertEqual(result.tracked_count, 0)


if __name__ == "__main__":
    unittest.main()
