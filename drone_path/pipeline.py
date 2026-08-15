"""Headless drone-path prediction pipeline with no UI dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path

import cv2

from drone_path.algorithm import (
    CameraRotationConfig,
    CameraRotationHandler,
    GlobalMotionEstimator,
    MotionClassifier,
    MotionState,
    OpticalFlowEstimator,
    OpticalFlowResult,
    RelativeHeadingTracker,
    RelativePathTracker,
    RotationSectionClassifier,
    RotationSectionConfig,
    RotationSectionKind,
    TranslationDirectionConfig,
    TranslationDirectionEstimator,
)
from drone_path.algorithm.relative_path import PathPoint
from drone_path.config import DronePathConfig, load_config
from drone_path.video import open_video


ANALYSIS_MAX_WIDTH = 960


class PathPredictionError(RuntimeError):
    """Raised when a video cannot be processed into a relative path."""


@dataclass(frozen=True, slots=True)
class PathPredictionProgress:
    processed_frames: int
    total_frames: int | None
    current_seconds: float


@dataclass(frozen=True, slots=True)
class PathPredictionResult:
    video_path: Path
    start_seconds: float
    processed_duration_seconds: float
    processed_frames: int
    source_fps: float
    points: tuple[PathPoint, ...]
    uncertainty_markers: tuple[PathPoint, ...]
    section_count: int
    final_heading_degrees: float
    heading_assumed: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the prediction."""
        return {
            "video": str(self.video_path),
            "start_seconds": self.start_seconds,
            "processed_duration_seconds": self.processed_duration_seconds,
            "processed_frames": self.processed_frames,
            "source_fps": self.source_fps,
            "relative_units": True,
            "points": [
                {"x": float(x), "y": float(y)} for x, y in self.points
            ],
            "uncertainty_markers": [
                {"x": float(x), "y": float(y)}
                for x, y in self.uncertainty_markers
            ],
            "section_count": self.section_count,
            "final_heading_degrees": self.final_heading_degrees,
            "heading_assumed": self.heading_assumed,
        }


ProgressCallback = Callable[[PathPredictionProgress], None]


def predict_path(
    video_path: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    config: DronePathConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PathPredictionResult:
    """Process a video as fast as possible and return its relative path."""
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be zero or greater")
    if duration_seconds is not None and (
        not math.isfinite(duration_seconds) or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be greater than zero")

    project_config = config or load_config()
    resolved_path, capture = open_video(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise PathPredictionError(
                "The video must report a positive frame rate for motion analysis"
            )

        source_frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        start_frame = round(start_seconds * fps)
        if source_frame_count > 0:
            if start_frame >= source_frame_count:
                raise PathPredictionError(
                    f"Start time {start_seconds:.2f}s is beyond the video duration"
                )
            start_frame = min(start_frame, source_frame_count - 1)
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        requested_frames = (
            max(1, math.ceil(duration_seconds * fps))
            if duration_seconds is not None
            else None
        )
        available_frames = (
            max(0, source_frame_count - start_frame)
            if source_frame_count > 0
            else None
        )
        total_frames = _minimum_known(requested_frames, available_frames)

        flow_estimator = OpticalFlowEstimator()
        motion_estimator = GlobalMotionEstimator()
        motion_classifier = MotionClassifier()
        direction_estimator = TranslationDirectionEstimator(
            TranslationDirectionConfig(
                smoothing_alpha=(
                    project_config.movement_direction.smoothing_alpha
                )
            )
        )
        rotation_handler = CameraRotationHandler(
            CameraRotationConfig(
                minimum_inlier_ratio=(
                    project_config.camera_rotation.minimum_inlier_ratio
                ),
                maximum_rotation_reprojection_error_pixels=(
                    project_config.camera_rotation
                    .maximum_reprojection_error_pixels
                ),
            )
        )
        rotation_classifier = RotationSectionClassifier(
            RotationSectionConfig(
                minimum_rotation_degrees=(
                    project_config.rotation_section.minimum_rotation_degrees
                ),
                minimum_samples=(
                    project_config.rotation_section.minimum_samples
                ),
                axis_dominance_ratio=(
                    project_config.rotation_section.axis_dominance_ratio
                ),
                minimum_yaw_sign_consistency=(
                    project_config.rotation_section
                    .minimum_yaw_sign_consistency
                ),
            )
        )
        relative_heading = RelativeHeadingTracker()
        relative_path = RelativePathTracker()
        direction_window_frames = max(
            1,
            round(project_config.movement_direction.window_seconds * fps),
        )

        previous_frame = None
        direction_anchor = None
        direction_anchor_index: int | None = None
        rotation_section_active = False
        processed_frames = 0

        def complete_rotation_section() -> None:
            nonlocal rotation_section_active
            classification = rotation_classifier.classify(
                rotation_handler.measurement
            )
            relative_heading.apply(classification)
            if (
                classification.significant
                and classification.kind == RotationSectionKind.UNCERTAIN
            ):
                relative_path.mark_uncertainty()
            rotation_section_active = False

        while total_frames is None or processed_frames < total_frames:
            decoded, source_frame = capture.read()
            if not decoded or source_frame is None:
                break

            frame_index = start_frame + processed_frames
            processed_frames += 1
            current_frame = _resize_for_analysis(source_frame)
            flow = (
                OpticalFlowResult.empty()
                if previous_frame is None
                else flow_estimator.estimate(previous_frame, current_frame)
            )
            motion = motion_estimator.measure(
                flow,
                (current_frame.shape[1], current_frame.shape[0]),
            )
            motion_state = motion_classifier.update(
                motion,
                fps=fps,
                frame_width=current_frame.shape[1],
            )

            if motion_state == MotionState.TRANSLATION:
                if rotation_section_active:
                    complete_rotation_section()
                rotation_handler.reset()
                if direction_anchor is None:
                    direction_anchor = current_frame.copy()
                    direction_anchor_index = frame_index
                elif (
                    direction_anchor_index is not None
                    and frame_index - direction_anchor_index
                    >= direction_window_frames
                ):
                    direction_flow = flow_estimator.estimate(
                        direction_anchor,
                        current_frame,
                    )
                    direction = direction_estimator.estimate(
                        direction_flow,
                        (current_frame.shape[1], current_frame.shape[0]),
                        horizontal_fov_degrees=(
                            project_config.camera.horizontal_fov_degrees
                        ),
                    )
                    relative_path.add_direction_sample(
                        direction,
                        distance=(
                            project_config.movement_direction.window_seconds
                        ),
                        map_heading_degrees=relative_heading.heading_degrees,
                    )
                    direction_anchor = current_frame.copy()
                    direction_anchor_index = frame_index
            else:
                rotation_section_active = True
                relative_path.end_translation_section()
                rotation_handler.update(
                    flow,
                    (current_frame.shape[1], current_frame.shape[0]),
                    horizontal_fov_degrees=(
                        project_config.camera.horizontal_fov_degrees
                    ),
                )
                direction_estimator.reset()
                direction_anchor = None
                direction_anchor_index = None

            previous_frame = current_frame
            if progress_callback is not None:
                progress_callback(
                    PathPredictionProgress(
                        processed_frames=processed_frames,
                        total_frames=total_frames,
                        current_seconds=start_seconds + processed_frames / fps,
                    )
                )

        if rotation_section_active:
            complete_rotation_section()
        if processed_frames == 0:
            raise PathPredictionError(
                f"OpenCV could not decode frames from the video: {resolved_path}"
            )

        return PathPredictionResult(
            video_path=resolved_path,
            start_seconds=start_seconds,
            processed_duration_seconds=processed_frames / fps,
            processed_frames=processed_frames,
            source_fps=fps,
            points=tuple(relative_path.points),
            uncertainty_markers=tuple(relative_path.uncertainty_markers),
            section_count=relative_path.section_count,
            final_heading_degrees=relative_heading.heading_degrees,
            heading_assumed=relative_heading.assumed,
        )
    finally:
        capture.release()


def _resize_for_analysis(frame):
    height, width = frame.shape[:2]
    if width <= ANALYSIS_MAX_WIDTH:
        return frame
    output_height = round(height * ANALYSIS_MAX_WIDTH / width)
    return cv2.resize(
        frame,
        (ANALYSIS_MAX_WIDTH, output_height),
        interpolation=cv2.INTER_AREA,
    )


def _minimum_known(
    first: int | None,
    second: int | None,
) -> int | None:
    known = [value for value in (first, second) if value is not None]
    return min(known) if known else None
