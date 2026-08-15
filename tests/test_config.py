"""Tests for shared project configuration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from drone_path.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_loads_camera_field_of_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                "[camera]\nhorizontal_fov_degrees = 76.5\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.camera.horizontal_fov_degrees, 76.5)

    def test_loads_movement_direction_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                "[movement_direction]\n"
                "window_seconds = 0.75\n"
                "smoothing_alpha = 0.5\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.movement_direction.window_seconds, 0.75)
        self.assertEqual(config.movement_direction.smoothing_alpha, 0.5)

    def test_loads_camera_rotation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                "[camera_rotation]\n"
                "maximum_reprojection_error_pixels = 6.0\n"
                "minimum_inlier_ratio = 0.6\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(
            config.camera_rotation.maximum_reprojection_error_pixels,
            6.0,
        )
        self.assertEqual(config.camera_rotation.minimum_inlier_ratio, 0.6)

    def test_loads_rotation_section_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                "[rotation_section]\n"
                "minimum_rotation_degrees = 4.0\n"
                "minimum_samples = 8\n"
                "axis_dominance_ratio = 2.5\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.rotation_section.minimum_rotation_degrees, 4.0)
        self.assertEqual(config.rotation_section.minimum_samples, 8)
        self.assertEqual(config.rotation_section.axis_dominance_ratio, 2.5)

    def test_rejects_field_of_view_outside_valid_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                "[camera]\nhorizontal_fov_degrees = 180\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "between 0 and 180"):
                load_config(config_path)

    def test_reports_missing_configuration_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.toml"

            with self.assertRaisesRegex(ConfigError, "not found"):
                load_config(missing_path)


if __name__ == "__main__":
    unittest.main()
