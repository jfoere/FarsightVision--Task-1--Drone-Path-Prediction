"""Diagnostic JSON and image output for independent visual odometry."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from reference_path.visual_odometry import (
    ROTATION_ONLY,
    TRANSLATION,
    UNRELIABLE,
    VisualOdometryResult,
)


BACKGROUND = (27, 31, 36)
FOREGROUND = (230, 230, 230)
MUTED = (145, 150, 158)
GRID = (62, 68, 76)
VO_PATH_COLOR = (210, 105, 240)
ROTATION_COLOR = (0, 155, 255)
UNRELIABLE_COLOR = (70, 70, 220)
END_COLOR = (0, 220, 255)


def save_visual_odometry_json(
    result: VisualOdometryResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def render_visual_odometry_diagnostic(
    result: VisualOdometryResult,
    output_path: str | Path,
    *,
    image_size: int = 1600,
    alignment_interval: tuple[float, float] | None = None,
) -> Path:
    """Render an unscaled path projection and a per-pair rotation chart."""
    if image_size < 600:
        raise ValueError("image_size must be at least 600 pixels")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((image_size, image_size, 3), BACKGROUND, dtype=np.uint8)
    margin = round(image_size * 0.08)
    title_bottom = round(image_size * 0.09)
    path_bottom = round(image_size * 0.84)
    footer_y = round(image_size * 0.91)
    plot_left, plot_right = margin, image_size - margin
    plot_top = title_bottom

    positions_3d = np.asarray([position[1:] for position in result.positions])
    points = positions_3d[:, (0, 2)]
    if alignment_interval is None:
        points, alignment_degrees = _align_first_translation_up(points, result)
        alignment_note = "first reliable movement is aligned up"
    else:
        points, alignment_degrees = _align_interval_up(
            points,
            result,
            start_seconds=alignment_interval[0],
            end_seconds=alignment_interval[1],
        )
        alignment_note = (
            f"movement at {alignment_interval[0]:g}-{alignment_interval[1]:g} s "
            f"is aligned up ({alignment_degrees:+.1f} deg)"
        )
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2
    span = np.maximum(maximum - minimum, 1.0)
    plot_width = plot_right - plot_left
    plot_height = path_bottom - plot_top
    scale = min(plot_width / span[0], plot_height / span[1]) * 0.78
    pixel_center = np.array(
        ((plot_left + plot_right) / 2, (plot_top + path_bottom) / 2),
        dtype=np.float64,
    )

    def to_pixel(point: np.ndarray) -> tuple[int, int]:
        relative = point - center
        return (
            round(pixel_center[0] + relative[0] * scale),
            round(pixel_center[1] - relative[1] * scale),
        )

    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, path_bottom), GRID, 2)
    pixels = np.asarray([to_pixel(point) for point in points], dtype=np.int32)
    path_thickness = max(3, round(image_size / 350))
    rotation_dot_radius = max(2, math.ceil(path_thickness / 2))
    for index, motion in enumerate(result.motions, start=1):
        start_pixel = tuple(int(value) for value in pixels[index - 1])
        end_pixel = tuple(int(value) for value in pixels[index])
        segment_color = (
            ROTATION_COLOR if motion.state == ROTATION_ONLY else VO_PATH_COLOR
        )
        cv2.line(
            canvas,
            start_pixel,
            end_pixel,
            segment_color,
            path_thickness,
            cv2.LINE_AA,
        )
    for index, motion in enumerate(result.motions, start=1):
        if motion.state == ROTATION_ONLY:
            point = tuple(int(value) for value in pixels[index])
            cv2.circle(
                canvas,
                point,
                rotation_dot_radius,
                ROTATION_COLOR,
                -1,
                cv2.LINE_AA,
            )
    marker_radius = max(7, round(image_size / 120))
    for index, motion in enumerate(result.motions, start=1):
        point = tuple(int(value) for value in pixels[index])
        if motion.state == UNRELIABLE:
            cv2.drawMarker(
                canvas,
                point,
                UNRELIABLE_COLOR,
                cv2.MARKER_TILTED_CROSS,
                marker_radius * 2,
                3,
                cv2.LINE_AA,
            )
    start = tuple(int(value) for value in pixels[0])
    end = tuple(int(value) for value in pixels[-1])
    cv2.circle(canvas, start, marker_radius, FOREGROUND, -1, cv2.LINE_AA)
    cv2.circle(canvas, end, marker_radius, END_COLOR, -1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "VISUAL ODOMETRY CHECK - UNSCALED",
        (margin, round(title_bottom * 0.56)),
        font,
        image_size / 1400,
        FOREGROUND,
        max(2, round(image_size / 750)),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"initial-camera X/Z projection; {alignment_note}",
        (plot_left + 12, plot_top + 32),
        font,
        image_size / 2800,
        MUTED,
        max(1, round(image_size / 1100)),
        cv2.LINE_AA,
    )
    summary = (
        f"{result.start_seconds:.1f}-{result.start_seconds + result.processed_duration_seconds:.1f} s"
        f"  |  {result.sample_interval_seconds:.2f} s pairs"
        f"  |  translation {result.translation_count}"
        f"  |  rotation-only {result.rotation_only_count}"
        f" ({result.rotation_only_ground_yaw_degrees:+.1f} deg ground yaw)"
        f"  |  unreliable {result.unreliable_count}"
    )
    cv2.putText(
        canvas,
        summary,
        (margin, footer_y),
        font,
        image_size / 2450,
        MUTED,
        max(1, round(image_size / 1000)),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "purple: unit translation   orange: rotation-only   red: unreliable   white/yellow: start/end",
        (margin, footer_y + round(image_size * 0.040)),
        font,
        image_size / 2800,
        MUTED,
        max(1, round(image_size / 1100)),
        cv2.LINE_AA,
    )

    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"OpenCV could not write visual-odometry image: {output}")
    return output.resolve()


def _align_first_translation_up(
    points: np.ndarray,
    result: VisualOdometryResult,
) -> tuple[np.ndarray, float]:
    deltas = []
    for index, motion in enumerate(result.motions, start=1):
        if motion.state == TRANSLATION:
            delta = points[index] - points[index - 1]
            if np.linalg.norm(delta) > 1e-9:
                deltas.append(delta)
        if len(deltas) == 5:
            break
    if not deltas:
        return points.copy(), 0.0
    direction = np.sum(deltas, axis=0)
    if np.linalg.norm(direction) <= 1e-9:
        return points.copy(), 0.0
    return _rotate_direction_up(points, direction)


def _align_interval_up(
    points: np.ndarray,
    result: VisualOdometryResult,
    *,
    start_seconds: float,
    end_seconds: float,
) -> tuple[np.ndarray, float]:
    first_time = result.positions[0][0]
    last_time = result.positions[-1][0]
    frame_rounding_tolerance = max(
        result.sample_interval_seconds / 2,
        1 / result.source_fps,
    )
    if (
        start_seconds < first_time - frame_rounding_tolerance
        or end_seconds > last_time + frame_rounding_tolerance
    ):
        raise ValueError(
            f"alignment interval must be within "
            f"{first_time:.1f}-{last_time:.1f} seconds"
        )
    if end_seconds <= start_seconds:
        raise ValueError("alignment end must be greater than its start")
    start_seconds = max(start_seconds, first_time)
    end_seconds = min(end_seconds, last_time)
    times = np.asarray([position[0] for position in result.positions], dtype=np.float64)
    start = np.array(
        (
            np.interp(start_seconds, times, points[:, 0]),
            np.interp(start_seconds, times, points[:, 1]),
        )
    )
    end = np.array(
        (
            np.interp(end_seconds, times, points[:, 0]),
            np.interp(end_seconds, times, points[:, 1]),
        )
    )
    direction = end - start
    if np.linalg.norm(direction) <= 1e-9:
        raise ValueError("alignment interval has no visual-odometry displacement")
    return _rotate_direction_up(points, direction)


def _rotate_direction_up(
    points: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, float]:
    angle = math.pi / 2 - math.atan2(direction[1], direction[0])
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    degrees = (math.degrees(angle) + 180.0) % 360.0 - 180.0
    return points @ rotation.T, degrees
