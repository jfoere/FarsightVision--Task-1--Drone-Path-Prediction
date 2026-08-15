"""Tests for safe relative-path accumulation."""

from __future__ import annotations

import unittest

from drone_path.algorithm import (
    RelativePathStatus,
    RelativePathTracker,
    TranslationDirectionMeasurement,
)


def _direction(
    angle: float,
    *,
    valid: bool = True,
    confidence: float = 1.0,
) -> TranslationDirectionMeasurement:
    return TranslationDirectionMeasurement(
        valid=valid,
        right=0.0,
        down=0.0,
        forward=1.0,
        horizontal_angle_degrees=angle,
        inlier_ratio=confidence if valid else 0.0,
        inlier_count=round(confidence * 100) if valid else 0,
        tracked_count=100,
    )


class RelativePathTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = RelativePathTracker()

    def test_starts_waiting_at_origin(self) -> None:
        self.assertEqual(self.path.status, RelativePathStatus.WAITING)
        self.assertEqual(list(self.path.points), [(0.0, 0.0)])

    def test_combines_samples_into_one_straight_section(self) -> None:
        self.assertTrue(
            self.path.add_direction_sample(_direction(0.0), distance=1.0)
        )
        self.assertTrue(
            self.path.add_direction_sample(_direction(90.0), distance=1.0)
        )

        self.assertEqual(self.path.status, RelativePathStatus.ACTIVE)
        self.assertEqual(len(self.path.points), 2)
        self.assertEqual(self.path.section_count, 1)
        self.assertEqual(self.path.sample_count, 2)
        self.assertAlmostEqual(self.path.average_direction_degrees, 45.0)
        x, y = self.path.current_position
        self.assertAlmostEqual(x, 2**0.5)
        self.assertAlmostEqual(y, 2**0.5)

    def test_weights_direction_samples_by_pose_confidence(self) -> None:
        self.path.add_direction_sample(
            _direction(0.0, confidence=0.9),
            distance=1.0,
        )
        self.path.add_direction_sample(
            _direction(90.0, confidence=0.1),
            distance=1.0,
        )

        self.assertAlmostEqual(
            self.path.average_direction_degrees,
            6.34,
            delta=0.01,
        )

    def test_initial_uncertainty_does_not_prevent_first_segment(self) -> None:
        self.path.mark_orientation_unknown()

        added = self.path.add_direction_sample(_direction(-45.0), distance=1.0)

        self.assertTrue(added)
        self.assertEqual(self.path.status, RelativePathStatus.ACTIVE)

    def test_orientation_loss_freezes_existing_path(self) -> None:
        self.path.add_direction_sample(_direction(0.0), distance=1.0)
        points_before = list(self.path.points)

        self.path.mark_orientation_unknown()
        added = self.path.add_direction_sample(_direction(90.0), distance=1.0)

        self.assertFalse(added)
        self.assertEqual(
            self.path.status,
            RelativePathStatus.ORIENTATION_UNKNOWN,
        )
        self.assertEqual(list(self.path.points), points_before)

    def test_invalid_direction_does_not_add_a_step(self) -> None:
        added = self.path.add_direction_sample(
            _direction(0.0, valid=False),
            distance=1.0,
        )

        self.assertFalse(added)
        self.assertEqual(self.path.status, RelativePathStatus.WAITING)

    def test_reset_restores_initial_state(self) -> None:
        self.path.add_direction_sample(_direction(0.0), distance=1.0)
        self.path.mark_orientation_unknown()

        self.path.reset()

        self.assertEqual(self.path.status, RelativePathStatus.WAITING)
        self.assertEqual(list(self.path.points), [(0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
