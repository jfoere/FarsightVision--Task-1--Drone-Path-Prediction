"""Validated project configuration shared by production and debugging code."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


class ConfigError(ValueError):
    """Raised when project configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class CameraConfig:
    horizontal_fov_degrees: float = 84.0

    def __post_init__(self) -> None:
        value = self.horizontal_fov_degrees
        if not math.isfinite(value) or not 0 < value < 180:
            raise ConfigError(
                "camera.horizontal_fov_degrees must be between 0 and 180"
            )


@dataclass(frozen=True, slots=True)
class MovementDirectionConfig:
    window_seconds: float = 1.0
    smoothing_alpha: float = 0.35

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ConfigError(
                "movement_direction.window_seconds must be greater than zero"
            )
        if (
            not math.isfinite(self.smoothing_alpha)
            or not 0 < self.smoothing_alpha <= 1
        ):
            raise ConfigError(
                "movement_direction.smoothing_alpha must be between 0 and 1"
            )


@dataclass(frozen=True, slots=True)
class CameraRotationConfig:
    maximum_reprojection_error_pixels: float = 8.0
    minimum_inlier_ratio: float = 0.50

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.maximum_reprojection_error_pixels)
            or self.maximum_reprojection_error_pixels <= 0
        ):
            raise ConfigError(
                "camera_rotation.maximum_reprojection_error_pixels "
                "must be greater than zero"
            )
        if (
            not math.isfinite(self.minimum_inlier_ratio)
            or not 0 <= self.minimum_inlier_ratio <= 1
        ):
            raise ConfigError(
                "camera_rotation.minimum_inlier_ratio must be between 0 and 1"
            )


@dataclass(frozen=True, slots=True)
class DronePathConfig:
    camera: CameraConfig = CameraConfig()
    movement_direction: MovementDirectionConfig = MovementDirectionConfig()
    camera_rotation: CameraRotationConfig = CameraRotationConfig()


def load_config(path: str | Path | None = None) -> DronePathConfig:
    """Load and validate a TOML configuration file."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        with config_path.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Invalid TOML in {config_path}: {error}") from error

    camera_section = raw_config.get("camera", {})
    if not isinstance(camera_section, dict):
        raise ConfigError("camera must be a TOML table")

    raw_fov = camera_section.get(
        "horizontal_fov_degrees",
        CameraConfig().horizontal_fov_degrees,
    )
    if isinstance(raw_fov, bool) or not isinstance(raw_fov, (int, float)):
        raise ConfigError("camera.horizontal_fov_degrees must be a number")

    direction_section = raw_config.get("movement_direction", {})
    if not isinstance(direction_section, dict):
        raise ConfigError("movement_direction must be a TOML table")
    raw_window_seconds = direction_section.get(
        "window_seconds",
        MovementDirectionConfig().window_seconds,
    )
    raw_smoothing_alpha = direction_section.get(
        "smoothing_alpha",
        MovementDirectionConfig().smoothing_alpha,
    )
    for key, value in (
        ("window_seconds", raw_window_seconds),
        ("smoothing_alpha", raw_smoothing_alpha),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"movement_direction.{key} must be a number")

    rotation_section = raw_config.get("camera_rotation", {})
    if not isinstance(rotation_section, dict):
        raise ConfigError("camera_rotation must be a TOML table")
    raw_rotation_error = rotation_section.get(
        "maximum_reprojection_error_pixels",
        CameraRotationConfig().maximum_reprojection_error_pixels,
    )
    raw_rotation_inliers = rotation_section.get(
        "minimum_inlier_ratio",
        CameraRotationConfig().minimum_inlier_ratio,
    )
    for key, value in (
        ("maximum_reprojection_error_pixels", raw_rotation_error),
        ("minimum_inlier_ratio", raw_rotation_inliers),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"camera_rotation.{key} must be a number")

    return DronePathConfig(
        camera=CameraConfig(horizontal_fov_degrees=float(raw_fov)),
        movement_direction=MovementDirectionConfig(
            window_seconds=float(raw_window_seconds),
            smoothing_alpha=float(raw_smoothing_alpha),
        ),
        camera_rotation=CameraRotationConfig(
            maximum_reprojection_error_pixels=float(raw_rotation_error),
            minimum_inlier_ratio=float(raw_rotation_inliers),
        ),
    )
