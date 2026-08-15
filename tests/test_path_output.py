"""Tests for headless path serialization and image rendering."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from drone_path.path_output import PATH_COLOR, render_path_image, save_prediction_json
from drone_path.pipeline import PathPredictionResult


def _result() -> PathPredictionResult:
    return PathPredictionResult(
        video_path=Path("video.mp4"),
        start_seconds=8.0,
        processed_duration_seconds=12.0,
        processed_frames=360,
        source_fps=30.0,
        points=((0.0, 0.0), (0.0, 5.0), (-1.0, 9.0)),
        uncertainty_markers=((0.0, 5.0),),
        section_count=2,
        final_heading_degrees=38.0,
        heading_assumed=True,
    )


class PathOutputTests(unittest.TestCase):
    def test_saves_complete_json_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "path.json"

            saved = save_prediction_json(_result(), output)
            data = json.loads(saved.read_text(encoding="utf-8"))

        self.assertEqual(data["section_count"], 2)
        self.assertEqual(len(data["points"]), 3)
        self.assertEqual(len(data["uncertainty_markers"]), 1)
        self.assertTrue(data["heading_assumed"])

    def test_renders_requested_square_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "path.png"

            saved = render_path_image(_result(), output, image_size=800)
            image = cv2.imread(str(saved))

        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (800, 800, 3))
        path_pixels = np.all(image == np.asarray(PATH_COLOR), axis=2)
        self.assertTrue(np.any(path_pixels))

    def test_rejects_tiny_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 400"):
            render_path_image(_result(), "unused.png", image_size=399)


if __name__ == "__main__":
    unittest.main()
