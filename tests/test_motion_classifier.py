"""Tests for temporal translation/rotation classification."""

from __future__ import annotations

import unittest

from drone_path.algorithm import (
    GlobalMotionMeasurement,
    MotionClassifier,
    MotionState,
)


def _measurement(
    *,
    flow: float,
    rotation: float,
    inliers: float,
    valid: bool = True,
) -> GlobalMotionMeasurement:
    return GlobalMotionMeasurement(
        valid=valid,
        median_flow_pixels=flow,
        translation_x_pixels=0.0,
        translation_y_pixels=0.0,
        rotation_degrees=rotation,
        scale=1.0,
        inlier_ratio=inliers,
        inlier_count=round(inliers * 300),
        tracked_count=300,
    )


TRANSLATION = _measurement(flow=0.24, rotation=0.005, inliers=1.0)
ROTATION = _measurement(flow=21.7, rotation=0.30, inliers=0.59)
FPS = 30.0
FRAME_WIDTH = 960


class MotionClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = MotionClassifier()

    def update(self, measurement: GlobalMotionMeasurement) -> MotionState:
        return self.classifier.update(
            measurement,
            fps=FPS,
            frame_width=FRAME_WIDTH,
        )

    def establish_translation(self) -> None:
        for _ in range(5):
            state = self.update(TRANSLATION)
        self.assertEqual(state, MotionState.TRANSLATION)

    def test_requires_consecutive_translation_evidence(self) -> None:
        for _ in range(4):
            self.assertEqual(self.update(TRANSLATION), MotionState.UNCERTAIN)

        self.assertEqual(self.update(TRANSLATION), MotionState.TRANSLATION)

    def test_requires_consecutive_rotation_evidence(self) -> None:
        self.establish_translation()

        self.assertEqual(self.update(ROTATION), MotionState.TRANSLATION)
        self.assertEqual(self.update(ROTATION), MotionState.TRANSLATION)
        self.assertEqual(self.update(ROTATION), MotionState.ROTATION)

    def test_rejects_a_single_rotation_spike(self) -> None:
        self.establish_translation()

        self.assertEqual(self.update(ROTATION), MotionState.TRANSLATION)
        self.assertEqual(self.update(TRANSLATION), MotionState.TRANSLATION)

    def test_returns_to_translation_after_five_frames(self) -> None:
        self.establish_translation()
        for _ in range(3):
            self.update(ROTATION)
        self.assertEqual(self.classifier.state, MotionState.ROTATION)

        for _ in range(4):
            self.assertEqual(self.update(TRANSLATION), MotionState.ROTATION)
        self.assertEqual(self.update(TRANSLATION), MotionState.TRANSLATION)

    def test_high_flow_and_low_inliers_can_detect_rotation(self) -> None:
        high_flow = _measurement(flow=21.7, rotation=0.0, inliers=0.59)

        for _ in range(3):
            state = self.update(high_flow)

        self.assertEqual(state, MotionState.ROTATION)

    def test_invalid_measurements_become_uncertain(self) -> None:
        self.establish_translation()
        invalid = _measurement(flow=0.0, rotation=0.0, inliers=0.0, valid=False)

        for _ in range(3):
            state = self.update(invalid)

        self.assertEqual(state, MotionState.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
