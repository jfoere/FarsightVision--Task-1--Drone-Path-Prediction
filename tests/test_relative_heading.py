"""Tests for relative map-heading accumulation."""

from __future__ import annotations

import unittest

from drone_path.algorithm import (
    RelativeHeadingTracker,
    RotationSectionClassification,
    RotationSectionKind,
)


def _section(
    kind: RotationSectionKind,
    *,
    heading_change: float | None,
    significant: bool = True,
) -> RotationSectionClassification:
    return RotationSectionClassification(
        kind=kind,
        total_rotation_degrees=abs(heading_change or 10.0),
        pitch_component_degrees=0.0,
        yaw_plane_component_degrees=abs(heading_change or 0.0),
        sample_count=20,
        heading_change_degrees=heading_change,
        significant=significant,
    )


class RelativeHeadingTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.heading = RelativeHeadingTracker()

    def test_starts_at_relative_zero(self) -> None:
        self.assertTrue(self.heading.known)
        self.assertEqual(self.heading.heading_degrees, 0.0)

    def test_accumulates_signed_drone_yaw(self) -> None:
        self.heading.apply(
            _section(RotationSectionKind.DRONE_YAW, heading_change=38.0)
        )
        self.heading.apply(
            _section(RotationSectionKind.DRONE_YAW, heading_change=-10.0)
        )

        self.assertEqual(self.heading.heading_degrees, 28.0)
        self.assertEqual(self.heading.last_change_degrees, -10.0)

    def test_gimbal_pitch_does_not_change_heading(self) -> None:
        applied = self.heading.apply(
            _section(RotationSectionKind.GIMBAL_PITCH, heading_change=0.0)
        )

        self.assertTrue(applied)
        self.assertEqual(self.heading.heading_degrees, 0.0)
        self.assertEqual(self.heading.last_change_degrees, 0.0)

    def test_significant_uncertainty_makes_heading_unknown(self) -> None:
        self.heading.apply(
            _section(RotationSectionKind.UNCERTAIN, heading_change=None)
        )

        self.assertFalse(self.heading.known)

    def test_insignificant_warmup_is_ignored(self) -> None:
        applied = self.heading.apply(
            _section(
                RotationSectionKind.UNCERTAIN,
                heading_change=None,
                significant=False,
            )
        )

        self.assertFalse(applied)
        self.assertTrue(self.heading.known)

    def test_normalizes_accumulated_heading(self) -> None:
        self.heading.apply(
            _section(RotationSectionKind.DRONE_YAW, heading_change=170.0)
        )
        self.heading.apply(
            _section(RotationSectionKind.DRONE_YAW, heading_change=30.0)
        )

        self.assertEqual(self.heading.heading_degrees, -160.0)


if __name__ == "__main__":
    unittest.main()
