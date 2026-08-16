"""Tests for metric GPS/video reference trajectory fusion."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from gps_reference.extractor import GpsSample
from reference_path.fused_output import (
    FUSED_PATH_COLOR,
    render_fused_reference,
    save_fused_reference_json,
)
from reference_path.fusion import ReferenceFusionError, fuse_reference_trajectory
from reference_path.visual_odometry import (
    ROTATION_ONLY,
    TRANSLATION,
    VisualMotionEstimate,
    VisualOdometryResult,
)


def _gps_samples(duration: int = 20) -> list[GpsSample]:
    metres_per_latitude_degree = 111_194.9
    return [
        GpsSample(
            time_seconds=float(second),
            longitude=25.0,
            latitude=48.0 + (second * 4.0) / metres_per_latitude_degree,
            gps_status=25,
            distance_home_m=second * 4.0,
            height_m=50.0,
            horizontal_speed_m_s=4.0,
            vertical_speed_m_s=0.0,
        )
        for second in range(duration + 1)
    ]


def _visual_odometry(duration: float = 20.0) -> VisualOdometryResult:
    timestamps = np.arange(0.0, duration + 0.01, 0.2)
    positions = tuple((float(time), float(time), 0.0, 0.0) for time in timestamps)
    motions = []
    for start, end in zip(timestamps[:-1], timestamps[1:], strict=True):
        state = ROTATION_ONLY if 9.0 <= end <= 10.0 else TRANSLATION
        direction = (1.0, 0.0, 0.0) if state == TRANSLATION else None
        motions.append(
            VisualMotionEstimate(
                start_seconds=float(start),
                end_seconds=float(end),
                state=state,
                matched_features=1_000,
                inlier_count=990,
                homography_inlier_ratio=0.99,
                median_reprojection_error_pixels=0.2,
                ground_normal_alignment=0.98,
                translation_ratio=0.02,
                rotation_degrees=8.0 if state == ROTATION_ONLY else 0.05,
                ground_yaw_degrees=8.0 if state == ROTATION_ONLY else 0.0,
                translation_direction_camera=direction,
                translation_direction_world=direction,
            )
        )
    return VisualOdometryResult(
        video_path=Path("video.mp4"),
        start_seconds=0.0,
        processed_duration_seconds=duration,
        source_fps=30.0,
        sample_interval_seconds=0.2,
        analysis_size=(960, 540),
        positions=positions,
        motions=tuple(motions),
    )


class ReferenceFusionTests(unittest.TestCase):
    def test_recovers_metric_northward_path_and_bridges_rotation_gap(self) -> None:
        result = fuse_reference_trajectory(
            _gps_samples(),
            _visual_odometry(),
            alignment_interval=(0.0, 5.0),
        )
        points = np.asarray(result.positions_m)

        self.assertGreaterEqual(result.alignment_window_count, 3)
        self.assertAlmostEqual(
            result.visual_to_gps_rotation_degrees,
            90.0,
            delta=3.0,
        )
        self.assertAlmostEqual(points[-1, 0], 0.0, delta=3.0)
        self.assertAlmostEqual(points[-1, 1], 80.0, delta=6.0)
        self.assertGreater(points[50, 1] - points[45, 1], 0.5)
        self.assertLess(result.endpoint_gps_residual_m, 6.0)

    def test_rejects_reversed_or_out_of_range_display_alignment(self) -> None:
        for interval in ((10.0, 5.0), (0.0, 21.0)):
            with self.subTest(interval=interval):
                with self.assertRaises(ReferenceFusionError):
                    fuse_reference_trajectory(
                        _gps_samples(),
                        _visual_odometry(),
                        alignment_interval=interval,
                    )

    def test_does_not_force_the_endpoint_onto_a_noisy_last_gps_anchor(self) -> None:
        samples = _gps_samples()
        last = samples[-1]
        samples[-1] = GpsSample(
            time_seconds=last.time_seconds,
            longitude=last.longitude,
            latitude=last.latitude + 30.0 / 111_194.9,
            gps_status=last.gps_status,
            distance_home_m=last.distance_home_m,
            height_m=last.height_m,
            horizontal_speed_m_s=last.horizontal_speed_m_s,
            vertical_speed_m_s=last.vertical_speed_m_s,
        )

        result = fuse_reference_trajectory(
            samples,
            _visual_odometry(),
            alignment_interval=(0.0, 5.0),
        )

        self.assertGreater(result.endpoint_gps_residual_m, 1.0)

    def test_renders_and_serializes_fused_reference(self) -> None:
        result = fuse_reference_trajectory(
            _gps_samples(),
            _visual_odometry(),
            alignment_interval=(0.0, 5.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "reference.png"
            json_path = Path(directory) / "reference.json"

            rendered = render_fused_reference(result, image_path, image_size=800)
            saved = save_fused_reference_json(result, json_path)
            image = cv2.imread(str(rendered))
            text = saved.read_text(encoding="utf-8")

        self.assertEqual(image.shape, (800, 800, 3))
        self.assertTrue(
            np.any(np.all(image == np.asarray(FUSED_PATH_COLOR), axis=2))
        )
        self.assertIn('"gps_rms_residual_m"', text)


if __name__ == "__main__":
    unittest.main()
