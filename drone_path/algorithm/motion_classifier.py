"""Temporal classification of global motion into translation or rotation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from drone_path.algorithm.global_motion import GlobalMotionMeasurement


class MotionState(str, Enum):
    TRANSLATION = "TRANSLATION"
    ROTATION = "ROTATION"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class MotionClassifierConfig:
    """Normalized thresholds and temporal confirmation lengths."""

    rotation_rate_enter_degrees_per_second: float = 1.5
    rotation_rate_hold_degrees_per_second: float = 0.75
    flow_rate_enter_frame_widths_per_second: float = 0.08
    flow_rate_hold_frame_widths_per_second: float = 0.04
    high_flow_enter_frame_widths_per_second: float = 0.20
    high_flow_hold_frame_widths_per_second: float = 0.10
    maximum_rotation_inlier_ratio: float = 0.85
    maximum_rotation_hold_inlier_ratio: float = 0.90
    minimum_valid_inlier_ratio: float = 0.25
    rotation_confirmation_frames: int = 3
    translation_confirmation_frames: int = 5
    uncertain_confirmation_frames: int = 3


class MotionClassifier:
    """Classify consecutive measurements while suppressing one-frame flicker."""

    def __init__(self, config: MotionClassifierConfig | None = None) -> None:
        self.config = config or MotionClassifierConfig()
        self.reset()

    @property
    def state(self) -> MotionState:
        return self._state

    def reset(self) -> None:
        self._state = MotionState.UNCERTAIN
        self._pending_state: MotionState | None = None
        self._pending_count = 0

    def update(
        self,
        measurement: GlobalMotionMeasurement,
        *,
        fps: float,
        frame_width: int,
    ) -> MotionState:
        """Update temporal state using one frame-pair measurement."""
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        if frame_width <= 0:
            raise ValueError("frame_width must be greater than zero")

        candidate = self._candidate_state(
            measurement,
            fps=fps,
            frame_width=frame_width,
        )
        if candidate == self._state:
            self._pending_state = None
            self._pending_count = 0
            return self._state

        if candidate != self._pending_state:
            self._pending_state = candidate
            self._pending_count = 1
        else:
            self._pending_count += 1

        if self._pending_count >= self._confirmation_frames(candidate):
            self._state = candidate
            self._pending_state = None
            self._pending_count = 0

        return self._state

    def _candidate_state(
        self,
        measurement: GlobalMotionMeasurement,
        *,
        fps: float,
        frame_width: int,
    ) -> MotionState:
        config = self.config
        if (
            not measurement.valid
            or measurement.inlier_ratio < config.minimum_valid_inlier_ratio
        ):
            return MotionState.UNCERTAIN

        rotation_rate = abs(measurement.rotation_degrees) * fps
        normalized_flow_rate = (
            measurement.median_flow_pixels * fps / frame_width
        )

        if self._state == MotionState.ROTATION:
            rotation_rate_threshold = (
                config.rotation_rate_hold_degrees_per_second
            )
            flow_rate_threshold = config.flow_rate_hold_frame_widths_per_second
            high_flow_threshold = config.high_flow_hold_frame_widths_per_second
            maximum_inlier_ratio = config.maximum_rotation_hold_inlier_ratio
        else:
            rotation_rate_threshold = (
                config.rotation_rate_enter_degrees_per_second
            )
            flow_rate_threshold = config.flow_rate_enter_frame_widths_per_second
            high_flow_threshold = config.high_flow_enter_frame_widths_per_second
            maximum_inlier_ratio = config.maximum_rotation_inlier_ratio

        angular_rotation = (
            rotation_rate >= rotation_rate_threshold
            and normalized_flow_rate >= flow_rate_threshold
        )
        high_flow_with_model_disagreement = (
            normalized_flow_rate >= high_flow_threshold
            and measurement.inlier_ratio <= maximum_inlier_ratio
        )
        if angular_rotation or high_flow_with_model_disagreement:
            return MotionState.ROTATION
        return MotionState.TRANSLATION

    def _confirmation_frames(self, state: MotionState) -> int:
        if state == MotionState.ROTATION:
            return self.config.rotation_confirmation_frames
        if state == MotionState.TRANSLATION:
            return self.config.translation_confirmation_frames
        return self.config.uncertain_confirmation_frames
