"""Tests for headless path-pipeline values and validation."""

from __future__ import annotations

from pathlib import Path
import unittest

from drone_path.pipeline import PathPredictionResult, _minimum_known, predict_path


class PathPredictionPipelineTests(unittest.TestCase):
    def test_result_converts_points_to_json_values(self) -> None:
        result = PathPredictionResult(
            video_path=Path("video.mp4"),
            start_seconds=0.0,
            processed_duration_seconds=1.0,
            processed_frames=30,
            source_fps=30.0,
            points=((0.0, 0.0), (1.0, 2.0)),
            uncertainty_markers=(),
            section_count=1,
            final_heading_degrees=0.0,
            heading_assumed=False,
        )

        data = result.as_dict()

        self.assertEqual(data["points"][1], {"x": 1.0, "y": 2.0})
        self.assertTrue(data["relative_units"])

    def test_rejects_invalid_interval_before_opening_video(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero or greater"):
            predict_path("missing.mp4", start_seconds=-1.0)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            predict_path("missing.mp4", duration_seconds=0.0)

    def test_uses_smallest_known_frame_limit(self) -> None:
        self.assertEqual(_minimum_known(100, 80), 80)
        self.assertEqual(_minimum_known(None, 80), 80)
        self.assertIsNone(_minimum_known(None, None))


if __name__ == "__main__":
    unittest.main()
