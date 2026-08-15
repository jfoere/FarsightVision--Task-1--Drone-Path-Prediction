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

    def test_initial_pause_does_not_prevent_first_segment(self) -> None:
        self.path.end_translation_section()

        added = self.path.add_direction_sample(_direction(-45.0), distance=1.0)

        self.assertTrue(added)
        self.assertEqual(self.path.status, RelativePathStatus.ACTIVE)

    def test_rotation_pause_starts_a_connected_new_section(self) -> None:
        self.path.add_direction_sample(_direction(0.0), distance=1.0)

        self.path.end_translation_section()
        added = self.path.add_direction_sample(
            _direction(0.0),
            distance=1.0,
            map_heading_degrees=90.0,
        )

        self.assertTrue(added)
        self.assertEqual(self.path.status, RelativePathStatus.ACTIVE)
        self.assertEqual(self.path.section_count, 2)
        self.assertEqual(len(self.path.points), 3)
        first_x, first_y = self.path.points[1]
        final_x, final_y = self.path.points[2]
        self.assertAlmostEqual(first_x, 0.0)
        self.assertAlmostEqual(first_y, 1.0)
        self.assertAlmostEqual(final_x, 1.0)
        self.assertAlmostEqual(final_y, 1.0)

    def test_marks_uncertain_position_without_stopping_path(self) -> None:
        self.path.add_direction_sample(_direction(0.0), distance=1.0)
        self.path.end_translation_section()
        self.path.mark_uncertainty()

        added = self.path.add_direction_sample(_direction(0.0), distance=1.0)

        self.assertTrue(added)
        self.assertEqual(list(self.path.uncertainty_markers), [(0.0, 1.0)])
        self.assertEqual(len(self.path.points), 3)

    def test_adds_map_heading_to_camera_relative_direction(self) -> None:
        self.path.add_direction_sample(
            _direction(-5.0),
            distance=1.0,
            map_heading_degrees=38.0,
        )

        self.assertAlmostEqual(self.path.average_direction_degrees, 33.0)

    def test_invalid_direction_does_not_add_a_step(self) -> None:
        added = self.path.add_direction_sample(
            _direction(0.0, valid=False),
            distance=1.0,
        )

        self.assertFalse(added)
        self.assertEqual(self.path.status, RelativePathStatus.WAITING)

    def test_reset_restores_initial_state(self) -> None:
        self.path.add_direction_sample(_direction(0.0), distance=1.0)
        self.path.mark_uncertainty()

        self.path.reset()

        self.assertEqual(self.path.status, RelativePathStatus.WAITING)
        self.assertEqual(list(self.path.points), [(0.0, 0.0)])
        self.assertEqual(list(self.path.uncertainty_markers), [])


if __name__ == "__main__":
    unittest.main()
