"""Safe relative-path accumulation from validated movement directions."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
import math

from drone_path.algorithm.translation_direction import (
    TranslationDirectionMeasurement,
)


PathPoint = tuple[float, float]


class RelativePathStatus(str, Enum):
    WAITING = "WAITING"
    ACTIVE = "ACTIVE"


class RelativePathTracker:
    """Build connected straight translation sections in a relative map."""

    def __init__(self) -> None:
        self.reset()

    @property
    def points(self) -> Sequence[PathPoint]:
        return self._points

    @property
    def status(self) -> RelativePathStatus:
        if self._started:
            return RelativePathStatus.ACTIVE
        return RelativePathStatus.WAITING

    @property
    def uncertainty_markers(self) -> Sequence[PathPoint]:
        return self._uncertainty_markers

    @property
    def current_position(self) -> PathPoint:
        return self._points[-1]

    @property
    def average_direction_degrees(self) -> float | None:
        return self._average_direction_degrees

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def section_count(self) -> int:
        return self._section_count

    def reset(self) -> None:
        self._points: list[PathPoint] = [(0.0, 0.0)]
        self._uncertainty_markers: list[PathPoint] = []
        self._started = False
        self._section_active = False
        self._section_start: PathPoint = (0.0, 0.0)
        self._section_distance = 0.0
        self._weighted_right = 0.0
        self._weighted_forward = 0.0
        self._average_direction_degrees: float | None = None
        self._sample_count = 0
        self._section_count = 0

    def end_translation_section(self) -> None:
        """Finish the current straight section without moving the endpoint."""
        self._section_active = False

    def mark_uncertainty(self) -> None:
        """Record where a zero-heading-change assumption was made."""
        marker = self.current_position
        if not self._uncertainty_markers or self._uncertainty_markers[-1] != marker:
            self._uncertainty_markers.append(marker)

    def add_direction_sample(
        self,
        direction: TranslationDirectionMeasurement,
        *,
        distance: float,
        map_heading_degrees: float = 0.0,
    ) -> bool:
        """Update one straight section from another direction observation."""
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("distance must be greater than zero")
        if not direction.valid:
            return False

        camera_angle_degrees = direction.horizontal_angle_degrees
        if not math.isfinite(camera_angle_degrees):
            return False
        if not math.isfinite(map_heading_degrees):
            raise ValueError("map_heading_degrees must be finite")
        angle_degrees = camera_angle_degrees + map_heading_degrees
        angle = math.radians(angle_degrees)
        confidence = direction.inlier_ratio
        if not math.isfinite(confidence) or confidence < 0:
            return False
        weight = max(0.001, confidence)

        if not self._section_active:
            self._section_active = True
            self._section_start = self._points[-1]
            self._section_distance = 0.0
            self._weighted_right = 0.0
            self._weighted_forward = 0.0
            self._sample_count = 0
            self._section_count += 1

        weighted_right = self._weighted_right + math.sin(angle) * weight
        weighted_forward = self._weighted_forward + math.cos(angle) * weight
        weighted_magnitude = math.hypot(weighted_right, weighted_forward)
        if weighted_magnitude <= 1e-9:
            return False

        self._weighted_right = weighted_right
        self._weighted_forward = weighted_forward
        self._section_distance += distance
        self._sample_count += 1

        average_angle = math.atan2(weighted_right, weighted_forward)
        self._average_direction_degrees = math.degrees(average_angle)
        section_start_x, section_start_y = self._section_start
        endpoint = (
            section_start_x + math.sin(average_angle) * self._section_distance,
            section_start_y + math.cos(average_angle) * self._section_distance,
        )
        if self._sample_count == 1:
            self._points.append(endpoint)
        else:
            self._points[-1] = endpoint
        self._started = True
        return True
