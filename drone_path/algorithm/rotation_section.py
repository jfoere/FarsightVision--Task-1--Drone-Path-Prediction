"""Interpret an optical-rotation section using the known drone constraints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from drone_path.algorithm.camera_rotation import CameraRotationMeasurement


class RotationSectionKind(str, Enum):
    """Physical interpretation of one completed non-translation section."""

    GIMBAL_PITCH = "GIMBAL PITCH"
    DRONE_YAW = "DRONE YAW"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class RotationSectionConfig:
    """Thresholds for separating gimbal pitch from drone yaw."""

    minimum_rotation_degrees: float = 3.0
    minimum_samples: int = 5
    axis_dominance_ratio: float = 2.0
    minimum_yaw_sign_consistency: float = 0.50

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_rotation_degrees)
            or self.minimum_rotation_degrees <= 0
        ):
            raise ValueError("minimum_rotation_degrees must be greater than zero")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be at least 1")
        if (
            not math.isfinite(self.axis_dominance_ratio)
            or self.axis_dominance_ratio <= 1
        ):
            raise ValueError("axis_dominance_ratio must be greater than one")
        if (
            not math.isfinite(self.minimum_yaw_sign_consistency)
            or not 0 <= self.minimum_yaw_sign_consistency <= 1
        ):
            raise ValueError(
                "minimum_yaw_sign_consistency must be between zero and one"
            )


@dataclass(frozen=True, slots=True)
class RotationSectionClassification:
    """Classification plus the components used to make the decision."""

    kind: RotationSectionKind
    total_rotation_degrees: float
    pitch_component_degrees: float
    yaw_plane_component_degrees: float
    sample_count: int
    heading_change_degrees: float | None = None
    significant: bool = False

    @classmethod
    def unavailable(cls) -> "RotationSectionClassification":
        return cls(
            kind=RotationSectionKind.UNCERTAIN,
            total_rotation_degrees=0.0,
            pitch_component_degrees=0.0,
            yaw_plane_component_degrees=0.0,
            sample_count=0,
            heading_change_degrees=None,
            significant=False,
        )


class RotationSectionClassifier:
    """Classify rotation under a pitch-only gimbal and level-drone model.

    The camera's X axis is the mechanical gimbal-pitch axis. Drone yaw is not
    generally camera-axis yaw: once the gimbal is pitched, the drone's physical
    yaw axis is represented by a mixture of camera Y and Z. This classifier
    therefore compares X against the combined Y/Z rotation-vector magnitude.
    """

    def __init__(self, config: RotationSectionConfig | None = None) -> None:
        self.config = config or RotationSectionConfig()

    def classify(
        self,
        measurement: CameraRotationMeasurement,
    ) -> RotationSectionClassification:
        x_degrees, y_degrees, z_degrees = measurement.rotation_vector_degrees
        pitch_component = abs(x_degrees)
        yaw_plane_component = math.hypot(y_degrees, z_degrees)
        total_rotation = math.hypot(pitch_component, yaw_plane_component)
        result = RotationSectionClassification(
            kind=RotationSectionKind.UNCERTAIN,
            total_rotation_degrees=total_rotation,
            pitch_component_degrees=x_degrees,
            yaw_plane_component_degrees=yaw_plane_component,
            sample_count=measurement.sample_count,
        )
        if (
            not measurement.valid
            or measurement.sample_count < self.config.minimum_samples
            or total_rotation < self.config.minimum_rotation_degrees
        ):
            return result

        dominance = self.config.axis_dominance_ratio
        if pitch_component > yaw_plane_component * dominance:
            kind = RotationSectionKind.GIMBAL_PITCH
            heading_change = 0.0
        elif yaw_plane_component > pitch_component * dominance:
            kind = RotationSectionKind.DRONE_YAW
            # For a camera pitched between horizontal and downward, the
            # drone's vertical yaw axis has camera-Y and camera-Z components
            # with the same sign. Their signed sum therefore distinguishes a
            # right turn from a left turn without confusing pitch with yaw.
            sign_source = y_degrees + z_degrees
            component_sum = abs(y_degrees) + abs(z_degrees)
            sign_consistency = (
                abs(sign_source) / component_sum if component_sum > 1e-9 else 0.0
            )
            if sign_consistency < self.config.minimum_yaw_sign_consistency:
                kind = RotationSectionKind.UNCERTAIN
                heading_change = None
            else:
                heading_change = math.copysign(yaw_plane_component, sign_source)
        else:
            kind = RotationSectionKind.UNCERTAIN
            heading_change = None
        return RotationSectionClassification(
            kind=kind,
            total_rotation_degrees=total_rotation,
            pitch_component_degrees=x_degrees,
            yaw_plane_component_degrees=yaw_plane_component,
            sample_count=measurement.sample_count,
            heading_change_degrees=heading_change,
            significant=True,
        )
