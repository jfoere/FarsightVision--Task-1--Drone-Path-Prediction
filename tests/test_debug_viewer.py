"""Unit tests for non-window debug-viewer controls."""

from __future__ import annotations

import unittest

import numpy as np

from debug_ui.viewer import (
    _change_playback_speed,
    _draw_speed_controls,
    _draw_timeline,
    _frame_from_timeline,
    _point_inside,
)


class PlaybackSpeedTests(unittest.TestCase):
    def test_changes_speed_in_point_one_steps(self) -> None:
        self.assertEqual(_change_playback_speed(1.0, 1), 1.1)
        self.assertEqual(_change_playback_speed(1.0, -1), 0.9)

    def test_clamps_speed_to_supported_range(self) -> None:
        self.assertEqual(_change_playback_speed(4.0, 1), 4.0)
        self.assertEqual(_change_playback_speed(0.1, -1), 0.1)

    def test_draws_clickable_controls_inside_frame(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        display, decrease, increase = _draw_speed_controls(frame, 1.0)

        self.assertTrue(np.any(display != frame))
        self.assertTrue(_point_inside(decrease[0], decrease[1], decrease))
        self.assertTrue(_point_inside(increase[2], increase[3], increase))
        self.assertLessEqual(increase[2], frame.shape[1])
        self.assertLessEqual(increase[3], frame.shape[0])

    def test_draws_one_timeline_and_maps_its_position_to_a_frame(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        display, timeline = _draw_timeline(frame, 500, 1001, 250)

        self.assertIsNotNone(timeline)
        assert timeline is not None
        self.assertTrue(np.any(display != frame))
        self.assertEqual(_frame_from_timeline(timeline[0], timeline, 1001), 0)
        self.assertEqual(_frame_from_timeline(timeline[2], timeline, 1001), 1000)
        midpoint = (timeline[0] + timeline[2]) // 2
        self.assertAlmostEqual(
            _frame_from_timeline(midpoint, timeline, 1001),
            500,
            delta=1,
        )


if __name__ == "__main__":
    unittest.main()
