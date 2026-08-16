"""JSON and map rendering for the GPS/video fused reference trajectory."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from reference_gps_and_video.fusion import FusedReferenceResult
from reference_gps_and_video.visual_odometry import ROTATION_ONLY, UNRELIABLE


BACKGROUND = (27, 31, 36)
FOREGROUND = (230, 230, 230)
MUTED = (145, 150, 158)
GRID = (62, 68, 76)
RAW_GPS_COLOR = (78, 91, 105)
FUSED_PATH_COLOR = (255, 165, 55)
ROTATION_COLOR = (0, 155, 255)
UNRELIABLE_COLOR = (70, 70, 220)
END_COLOR = (0, 220, 255)


def save_fused_reference_json(
    result: FusedReferenceResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def render_fused_reference(
    result: FusedReferenceResult,
    output_path: str | Path,
    *,
    image_size: int = 1600,
) -> Path:
    """Render a metric path with GPS/video states colored along one line."""
    if image_size < 600:
        raise ValueError("image_size must be at least 600 pixels")
    if len(result.positions_m) != len(result.motion_states) + 1:
        raise ValueError("fused positions and motion states are inconsistent")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((image_size, image_size, 3), BACKGROUND, dtype=np.uint8)
    margin = round(image_size * 0.08)
    title_bottom = round(image_size * 0.09)
    plot_bottom = round(image_size * 0.84)
    footer_y = round(image_size * 0.91)
    plot_left, plot_right = margin, image_size - margin
    plot_top = title_bottom

    points = np.asarray(result.positions_m, dtype=np.float64)
    raw_gps = np.asarray(
        [(point[1], point[2]) for point in result.raw_gps_points_m],
        dtype=np.float64,
    ).reshape(-1, 2)
    points, raw_gps, display_rotation = _align_for_display(
        points,
        raw_gps,
        np.asarray(result.timestamps, dtype=np.float64),
        result.alignment_interval,
    )
    all_points = points if raw_gps.size == 0 else np.vstack((points, raw_gps))
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    center = (minimum + maximum) / 2
    data_span = np.maximum(maximum - minimum, 1.0)
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    scale = min(plot_width / data_span[0], plot_height / data_span[1]) * 0.82
    visible_half_span = np.array((plot_width, plot_height)) / (2 * scale)
    visible_minimum = center - visible_half_span
    visible_maximum = center + visible_half_span
    pixel_center = np.array(
        ((plot_left + plot_right) / 2, (plot_top + plot_bottom) / 2),
        dtype=np.float64,
    )

    def to_pixel(point: np.ndarray) -> tuple[int, int]:
        relative = point - center
        return (
            round(pixel_center[0] + relative[0] * scale),
            round(pixel_center[1] - relative[1] * scale),
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    grid_step = _nice_grid_step(float(max(data_span)))
    grid_thickness = max(1, round(image_size / 1200))
    x_value = math.ceil(visible_minimum[0] / grid_step) * grid_step
    while x_value <= visible_maximum[0]:
        x, _ = to_pixel(np.array((x_value, 0.0)))
        cv2.line(canvas, (x, plot_top), (x, plot_bottom), GRID, grid_thickness)
        cv2.putText(
            canvas,
            f"{x_value:+.0f} m",
            (x + 5, plot_bottom - 10),
            font,
            image_size / 3100,
            MUTED,
            grid_thickness,
            cv2.LINE_AA,
        )
        x_value += grid_step
    y_value = math.ceil(visible_minimum[1] / grid_step) * grid_step
    while y_value <= visible_maximum[1]:
        _, y = to_pixel(np.array((0.0, y_value)))
        cv2.line(canvas, (plot_left, y), (plot_right, y), GRID, grid_thickness)
        cv2.putText(
            canvas,
            f"{y_value:+.0f} m",
            (plot_left + 8, y - 7),
            font,
            image_size / 3100,
            MUTED,
            grid_thickness,
            cv2.LINE_AA,
        )
        y_value += grid_step
    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), GRID, 2)

    if len(raw_gps) > 1:
        raw_pixels = np.asarray([to_pixel(point) for point in raw_gps], np.int32)
        cv2.polylines(
            canvas,
            [raw_pixels.reshape(-1, 1, 2)],
            False,
            RAW_GPS_COLOR,
            max(2, round(image_size / 650)),
            cv2.LINE_AA,
        )

    pixels = np.asarray([to_pixel(point) for point in points], np.int32)
    path_thickness = max(3, round(image_size / 350))
    for index, state in enumerate(result.motion_states, start=1):
        color = (
            ROTATION_COLOR
            if state == ROTATION_ONLY
            else UNRELIABLE_COLOR
            if state == UNRELIABLE
            else FUSED_PATH_COLOR
        )
        cv2.line(
            canvas,
            tuple(int(value) for value in pixels[index - 1]),
            tuple(int(value) for value in pixels[index]),
            color,
            path_thickness,
            cv2.LINE_AA,
        )

    marker_radius = max(7, round(image_size / 120))
    start = tuple(int(value) for value in pixels[0])
    end = tuple(int(value) for value in pixels[-1])
    cv2.circle(canvas, start, marker_radius, FOREGROUND, -1, cv2.LINE_AA)
    cv2.circle(canvas, end, marker_radius, END_COLOR, -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "GPS + VIDEO REFERENCE (ESTIMATED)",
        (margin, round(title_bottom * 0.56)),
        font,
        image_size / 1400,
        FOREGROUND,
        max(2, round(image_size / 750)),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"metric fused trajectory; movement at "
            f"{result.alignment_interval[0]:g}-{result.alignment_interval[1]:g} s "
            f"is aligned up ({display_rotation:+.1f} deg)"
        ),
        (plot_left + 12, plot_top + 32),
        font,
        image_size / 2800,
        MUTED,
        max(1, round(image_size / 1100)),
        cv2.LINE_AA,
    )
    summary = (
        f"{result.start_seconds:.1f}-{result.start_seconds + result.processed_duration_seconds:.1f} s"
        f"  |  GPS anchors {result.gps_anchor_count}"
        f"  |  visual velocity {result.visual_velocity_count}"
        f"  |  start-to-end {result.start_to_end_distance_m:.1f} m"
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
        (
            f"GPS fit RMS {result.gps_rms_residual_m:.1f} m; last-GPS fit residual "
            f"{result.endpoint_gps_residual_m:.1f} m   |   blue: fused translation   "
            "orange: rotation interval   gray: raw GPS"
        ),
        (margin, footer_y + round(image_size * 0.040)),
        font,
        image_size / 2800,
        MUTED,
        max(1, round(image_size / 1100)),
        cv2.LINE_AA,
    )

    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"OpenCV could not write fused reference image: {output}")
    return output.resolve()


def _align_for_display(
    points: np.ndarray,
    raw_gps: np.ndarray,
    timestamps: np.ndarray,
    interval: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    start, end = interval
    start_point = np.array(
        (
            np.interp(start, timestamps, points[:, 0]),
            np.interp(start, timestamps, points[:, 1]),
        )
    )
    end_point = np.array(
        (
            np.interp(end, timestamps, points[:, 0]),
            np.interp(end, timestamps, points[:, 1]),
        )
    )
    direction = end_point - start_point
    if np.linalg.norm(direction) <= 1e-9:
        raise ValueError("display alignment interval has no fused displacement")
    angle = math.pi / 2 - math.atan2(direction[1], direction[0])
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    degrees = (math.degrees(angle) + 180.0) % 360.0 - 180.0
    return points @ rotation.T, raw_gps @ rotation.T, degrees


def _nice_grid_step(span: float) -> float:
    rough_step = max(span / 6.0, 1.0)
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    factor = 1.0 if normalized <= 1 else 2.0 if normalized <= 2 else 5.0
    return factor * magnitude
