"""Tests for independent planar visual odometry and its diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from reference_path.__main__ import build_parser
from reference_path.output import (
    ROTATION_COLOR,
    VO_PATH_COLOR,
    _align_interval_up,
    render_visual_odometry_diagnostic,
    save_visual_odometry_json,
)
from reference_path.visual_odometry import (
    ROTATION_ONLY,
    TRANSLATION,
    VisualMotionEstimate,
    VisualOdometryConfig,
    VisualOdometryResult,
    VisualOdometryProgress,
    _camera_matrix,
    _decompose_planar_motion,
)


def _motion(start: float, end: float, state: str, rotation: float):
    direction = (0.0, 0.0, 1.0) if state == TRANSLATION else None
    return VisualMotionEstimate(
        start_seconds=start,
        end_seconds=end,
        state=state,
        matched_features=200,
        inlier_count=190,
        homography_inlier_ratio=0.95,
        median_reprojection_error_pixels=0.2,
        ground_normal_alignment=0.98,
        translation_ratio=0.02,
        rotation_degrees=rotation,
        ground_yaw_degrees=rotation if state == ROTATION_ONLY else 0.0,
        translation_direction_camera=direction,
        translation_direction_world=direction,
    )


def _result() -> VisualOdometryResult:
    return VisualOdometryResult(
        video_path=Path("video.mp4"),
        start_seconds=8.0,
        processed_duration_seconds=0.6,
        source_fps=30.0,
        sample_interval_seconds=0.2,
        analysis_size=(960, 540),
        positions=(
            (8.0, 0.0, 0.0, 0.0),
            (8.2, 0.0, 0.0, 1.0),
            (8.4, 0.0, 0.0, 1.0),
            (8.6, 0.5, 0.0, 1.8),
        ),
        motions=(
            _motion(8.0, 8.2, TRANSLATION, 0.1),
            _motion(8.2, 8.4, ROTATION_ONLY, 10.0),
            _motion(8.4, 8.6, TRANSLATION, 0.2),
        ),
    )


class ReferenceVisualOdometryTests(unittest.TestCase):
    def test_cli_defaults_to_processing_the_full_video(self) -> None:
        arguments = build_parser().parse_args(["video.mp4"])

        self.assertIsNone(arguments.duration)

    def test_selects_ground_plane_translation_solution(self) -> None:
        camera_matrix = _camera_matrix(960, 540, 84.0)
        rotation = np.eye(3, dtype=np.float64)
        translation = np.array((0.0, 0.006, -0.017), dtype=np.float64)
        normal = np.array((0.0, 0.965, 0.262), dtype=np.float64)
        homography = camera_matrix @ (
            rotation + np.outer(translation, normal)
        ) @ np.linalg.inv(camera_matrix)

        geometry = _decompose_planar_motion(
            homography,
            camera_matrix,
            np.array((0.0, 1.0, 0.0)),
            VisualOdometryConfig(),
        )

        self.assertEqual(geometry.state, TRANSLATION)
        self.assertGreater(geometry.ground_normal_alignment, 0.9)
        self.assertGreater(geometry.translation_camera[2], 0.0)

    def test_treats_pure_rotation_as_rotation_only(self) -> None:
        camera_matrix = _camera_matrix(960, 540, 84.0)
        rotation_vector = np.array((0.0, math.radians(10.0), 0.0))
        rotation, _ = cv2.Rodrigues(rotation_vector)
        homography = camera_matrix @ rotation @ np.linalg.inv(camera_matrix)

        geometry = _decompose_planar_motion(
            homography,
            camera_matrix,
            np.array((0.0, 1.0, 0.0)),
            VisualOdometryConfig(),
        )

        self.assertEqual(geometry.state, ROTATION_ONLY)
        self.assertAlmostEqual(geometry.rotation_degrees, 10.0, delta=0.01)
        self.assertAlmostEqual(abs(geometry.ground_yaw_degrees), 10.0, delta=0.01)

    def test_renders_and_serializes_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "vo.png"
            json_path = Path(directory) / "vo.json"

            rendered = render_visual_odometry_diagnostic(
                _result(), image_path, image_size=800
            )
            saved = save_visual_odometry_json(_result(), json_path)
            image = cv2.imread(str(rendered))
            text = saved.read_text(encoding="utf-8")

        self.assertEqual(image.shape, (800, 800, 3))
        self.assertTrue(np.any(np.all(image == np.asarray(VO_PATH_COLOR), axis=2)))
        self.assertTrue(np.any(np.all(image == np.asarray(ROTATION_COLOR), axis=2)))
        self.assertIn('"state": "ROTATION_ONLY"', text)

    def test_aligns_selected_interval_up(self) -> None:
        result = _result()
        points = np.asarray([position[1:] for position in result.positions])[:, (0, 2)]

        aligned, _ = _align_interval_up(
            points,
            result,
            start_seconds=8.0,
            end_seconds=8.6,
        )

        displacement = aligned[-1] - aligned[0]
        self.assertAlmostEqual(displacement[0], 0.0, delta=1e-9)
        self.assertGreater(displacement[1], 0.0)

    def test_reports_rounded_progress_percentage(self) -> None:
        progress = VisualOdometryProgress(processed_pairs=252, total_pairs=600)

        self.assertEqual(progress.percentage, 42)

    def test_rejects_invalid_sampling_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            VisualOdometryConfig(sample_interval_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
