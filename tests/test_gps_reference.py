"""Tests for independent DJI GPS extraction and path rendering."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from gps_reference.extractor import GpsSample, parse_dji_subtitle, to_local_metres
from gps_reference.renderer import (
    GPS_PATH_COLOR,
    align_points_to_interval,
    render_gps_path,
)


def _sample(time: float, longitude: float, latitude: float) -> GpsSample:
    return GpsSample(
        time_seconds=time,
        longitude=longitude,
        latitude=latitude,
        gps_status=26,
        distance_home_m=0.0,
        height_m=50.0,
        horizontal_speed_m_s=5.0,
        vertical_speed_m_s=0.0,
    )


class GpsReferenceTests(unittest.TestCase):
    def test_parses_dji_subtitle_telemetry(self) -> None:
        text = (
            "F/2.8, SS 724.08, ISO 100, EV 0, DZOOM 1.000, "
            "GPS (25.9185, 48.2658, 26), D 52.48m, H 51.30m, "
            "H.S 7.57m/s, V.S -0.20m/s"
        )

        sample = parse_dji_subtitle(text, time_seconds=8.0)

        self.assertEqual(sample.time_seconds, 8.0)
        self.assertEqual(sample.longitude, 25.9185)
        self.assertEqual(sample.latitude, 48.2658)
        self.assertEqual(sample.gps_status, 26)
        self.assertEqual(sample.vertical_speed_m_s, -0.2)

    def test_converts_coordinates_to_east_and_north_metres(self) -> None:
        points = to_local_metres(
            [
                _sample(0.0, 25.0, 48.0),
                _sample(1.0, 25.001, 48.001),
            ]
        )

        self.assertEqual(points[0], (0.0, 0.0))
        self.assertAlmostEqual(points[1][0], 74.4, delta=0.3)
        self.assertAlmostEqual(points[1][1], 111.2, delta=0.3)

    def test_renders_requested_square_image(self) -> None:
        samples = [
            _sample(0.0, 25.0, 48.0),
            _sample(1.0, 25.001, 48.001),
            _sample(2.0, 25.002, 48.0005),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gps.png"

            saved = render_gps_path(samples, output, image_size=800)
            image = cv2.imread(str(saved))

        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (800, 800, 3))
        path_pixels = np.all(image == np.asarray(GPS_PATH_COLOR), axis=2)
        self.assertTrue(np.any(path_pixels))

    def test_aligns_selected_movement_interval_upward(self) -> None:
        samples = [
            _sample(0.0, 25.0, 48.0),
            _sample(1.0, 25.001, 48.0),
            _sample(2.0, 25.002, 48.0),
        ]
        points = np.asarray(to_local_metres(samples), dtype=np.float64)

        aligned, rotation_degrees = align_points_to_interval(
            samples,
            points,
            start_seconds=0.0,
            end_seconds=2.0,
        )

        displacement = aligned[-1] - aligned[0]
        self.assertAlmostEqual(displacement[0], 0.0, delta=1e-8)
        self.assertGreater(displacement[1], 0.0)
        self.assertAlmostEqual(rotation_degrees, 90.0)

    def test_renders_an_aligned_image(self) -> None:
        samples = [
            _sample(0.0, 25.0, 48.0),
            _sample(1.0, 25.001, 48.0),
            _sample(2.0, 25.002, 48.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "aligned.png"

            saved = render_gps_path(
                samples,
                output,
                image_size=800,
                alignment_interval=(0.0, 2.0),
            )

        self.assertEqual(saved.name, "aligned.png")

    def test_rejects_empty_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            render_gps_path([], "unused.png")


if __name__ == "__main__":
    unittest.main()
