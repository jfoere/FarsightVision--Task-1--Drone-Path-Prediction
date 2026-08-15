"""Tests for constraint-based optical-rotation section classification."""

from __future__ import annotations

import unittest

from drone_path.algorithm import (
    CameraRotationMeasurement,
    RotationSectionClassifier,
    RotationSectionConfig,
    RotationSectionKind,
)


def _measurement(
    rotation_vector: tuple[float, float, float],
    *,
    sample_count: int = 20,
) -> CameraRotationMeasurement:
    return CameraRotationMeasurement(
        valid=True,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        roll_degrees=0.0,
        inlier_ratio=0.9,
        median_reprojection_error_pixels=0.5,
        sample_count=sample_count,
        tracked_count=100,
        rotation_vector_degrees=rotation_vector,
    )


class RotationSectionClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RotationSectionClassifier()

    def test_classifies_rotation_around_camera_x_as_gimbal_pitch(self) -> None:
        result = self.classifier.classify(_measurement((-10.0, 0.2, 0.1)))

        self.assertEqual(result.kind, RotationSectionKind.GIMBAL_PITCH)
        self.assertAlmostEqual(result.pitch_component_degrees, -10.0)

    def test_classifies_camera_yz_mixture_as_drone_yaw(self) -> None:
        result = self.classifier.classify(_measurement((0.2, 37.0, 10.0)))

        self.assertEqual(result.kind, RotationSectionKind.DRONE_YAW)
        self.assertAlmostEqual(result.total_rotation_degrees, 38.33, delta=0.01)
        self.assertAlmostEqual(result.heading_change_degrees or 0.0, 38.33, delta=0.01)

    def test_preserves_left_turn_sign(self) -> None:
        result = self.classifier.classify(_measurement((0.2, -37.0, -10.0)))

        self.assertEqual(result.kind, RotationSectionKind.DRONE_YAW)
        self.assertAlmostEqual(result.heading_change_degrees or 0.0, -38.33, delta=0.01)

    def test_rejects_disagreeing_yaw_sign_components(self) -> None:
        result = self.classifier.classify(_measurement((0.2, 10.0, -10.0)))

        self.assertEqual(result.kind, RotationSectionKind.UNCERTAIN)
        self.assertIsNone(result.heading_change_degrees)
        self.assertTrue(result.significant)

    def test_marks_mixed_axes_as_uncertain(self) -> None:
        result = self.classifier.classify(_measurement((8.0, 7.0, 0.0)))

        self.assertEqual(result.kind, RotationSectionKind.UNCERTAIN)
        self.assertTrue(result.significant)

    def test_marks_short_sections_as_uncertain(self) -> None:
        result = self.classifier.classify(
            _measurement((0.0, 15.0, 0.0), sample_count=3)
        )

        self.assertEqual(result.kind, RotationSectionKind.UNCERTAIN)
        self.assertFalse(result.significant)

    def test_marks_small_rotation_as_uncertain(self) -> None:
        classifier = RotationSectionClassifier(
            RotationSectionConfig(minimum_rotation_degrees=3.0)
        )

        result = classifier.classify(_measurement((0.0, 2.0, 0.0)))

        self.assertEqual(result.kind, RotationSectionKind.UNCERTAIN)
        self.assertFalse(result.significant)


if __name__ == "__main__":
    unittest.main()
