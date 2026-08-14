"""Unit tests for non-window debug-viewer controls."""

from __future__ import annotations

import unittest

import numpy as np

from drone_path.algorithm import GlobalMotionMeasurement, MotionState
from debug_ui.viewer import (
    MOTION_STATE_CODES,
    MOTION_STATE_COLORS,
    StateTimelineViewport,
    _change_playback_speed,
    _draw_motion_metrics,
    _draw_speed_controls,
    _draw_state_timeline,
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

    def test_timeline_colors_processed_motion_states(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        states = np.array(
            [
                MOTION_STATE_CODES[MotionState.TRANSLATION],
                MOTION_STATE_CODES[MotionState.TRANSLATION],
                MOTION_STATE_CODES[MotionState.ROTATION],
                MOTION_STATE_CODES[MotionState.ROTATION],
            ],
            dtype=np.int8,
        )

        display = _draw_state_timeline(frame, 0, states, 0, 4, 250)

        y = 674
        right = frame.shape[1] - 14
        one_quarter = 250 + (right - 250) // 4
        three_quarters = 250 + 3 * (right - 250) // 4
        self.assertTupleEqual(
            tuple(int(value) for value in display[y, one_quarter]),
            MOTION_STATE_COLORS[MotionState.TRANSLATION],
        )
        self.assertTupleEqual(
            tuple(int(value) for value in display[y, three_quarters]),
            MOTION_STATE_COLORS[MotionState.ROTATION],
        )

    def test_state_viewport_advances_from_ninety_to_ten_percent(self) -> None:
        viewport = StateTimelineViewport()
        total_frames = 18_888

        initial_start, initial_end = viewport.bounds(240, total_frames)
        window_frames = initial_end - initial_start
        boundary_frame = round(window_frames * 0.90)
        shifted_start, shifted_end = viewport.bounds(boundary_frame, total_frames)

        self.assertEqual(initial_start, 0)
        self.assertGreater(shifted_start, initial_start)
        shifted_position = (boundary_frame - shifted_start) / (
            shifted_end - shifted_start
        )
        self.assertAlmostEqual(shifted_position, 0.10, delta=0.002)
        overlap = initial_end - shifted_start
        self.assertAlmostEqual(overlap / window_frames, 0.20, delta=0.002)

    def test_state_viewport_repositions_after_a_distant_seek(self) -> None:
        viewport = StateTimelineViewport()
        total_frames = 18_888
        viewport.bounds(240, total_frames)

        start, end = viewport.bounds(9_000, total_frames)

        position = (9_000 - start) / (end - start)
        self.assertAlmostEqual(position, 0.10, delta=0.002)

    def test_draws_valid_motion_metrics(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        measurement = GlobalMotionMeasurement(
            valid=True,
            median_flow_pixels=1.25,
            translation_x_pixels=0.5,
            translation_y_pixels=-0.25,
            rotation_degrees=0.1,
            scale=1.0,
            inlier_ratio=0.95,
            inlier_count=95,
            tracked_count=100,
        )

        display = _draw_motion_metrics(
            frame,
            measurement,
            MotionState.TRANSLATION,
        )

        self.assertTrue(np.any(display != frame))


if __name__ == "__main__":
    unittest.main()
