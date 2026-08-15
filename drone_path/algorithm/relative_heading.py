"""Relative map-heading updates from completed rotation sections."""

from __future__ import annotations

import math

from drone_path.algorithm.rotation_section import (
    RotationSectionClassification,
    RotationSectionKind,
)


class RelativeHeadingTracker:
    """Accumulate signed drone yaw while ignoring gimbal-only pitch."""

    def __init__(self) -> None:
        self.reset()

    @property
    def heading_degrees(self) -> float:
        return self._heading_degrees

    @property
    def last_change_degrees(self) -> float | None:
        return self._last_change_degrees

    @property
    def reliable(self) -> bool:
        return not self._assumed

    @property
    def assumed(self) -> bool:
        return self._assumed

    def reset(self) -> None:
        self._heading_degrees = 0.0
        self._last_change_degrees: float | None = None
        self._assumed = False

    def apply(self, section: RotationSectionClassification) -> bool:
        """Apply one completed section without changing path position."""
        if not section.significant:
            return False
        if section.kind == RotationSectionKind.UNCERTAIN:
            self._last_change_degrees = None
            self._assumed = True
            return True
        if section.kind == RotationSectionKind.GIMBAL_PITCH:
            self._last_change_degrees = 0.0
            return True

        change = section.heading_change_degrees
        if change is None or not math.isfinite(change):
            self._last_change_degrees = None
            return False
        self._heading_degrees = _normalize_degrees(self._heading_degrees + change)
        self._last_change_degrees = change
        return True


def _normalize_degrees(angle_degrees: float) -> float:
    normalized = (angle_degrees + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized
