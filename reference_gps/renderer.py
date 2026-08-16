"""Render embedded GPS samples as a north-up reference path image."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from reference_gps.extractor import GpsSample, to_local_metres


BACKGROUND = (27, 31, 36)
FOREGROUND = (230, 230, 230)
MUTED = (145, 150, 158)
GRID = (62, 68, 76)
GPS_PATH_COLOR = (255, 165, 55)
START_COLOR = (245, 245, 245)
END_COLOR = (0, 220, 255)


def render_gps_path(
    samples: list[GpsSample],
    output_path: str | Path,
    *,
    image_size: int = 1600,
    alignment_interval: tuple[float, float] | None = None,
) -> Path:
    """Save an equal-scale GPS map, optionally aligned to a flight interval."""
    if image_size < 400:
        raise ValueError("image_size must be at least 400 pixels")
    if not samples:
        raise ValueError("at least one GPS sample is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((image_size, image_size, 3), BACKGROUND, dtype=np.uint8)
    title_height = round(image_size * 0.09)
    footer_height = round(image_size * 0.11)
    side_margin = round(image_size * 0.09)
    plot_left = side_margin
    plot_right = image_size - side_margin
    plot_top = title_height
    plot_bottom = image_size - footer_height
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    points = np.asarray(to_local_metres(samples), dtype=np.float64)
    rotation_degrees = 0.0
    if alignment_interval is not None:
        points, rotation_degrees = align_points_to_interval(
            samples,
            points,
            start_seconds=alignment_interval[0],
            end_seconds=alignment_interval[1],
        )
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    data_span = np.maximum(maximum - minimum, 1.0)
    scale = min(plot_width / data_span[0], plot_height / data_span[1]) * 0.84
    visible_half_span = np.array((plot_width, plot_height), dtype=np.float64) / (2 * scale)
    visible_minimum = center - visible_half_span
    visible_maximum = center + visible_half_span
    plot_center = np.array(
        ((plot_left + plot_right) / 2, (plot_top + plot_bottom) / 2),
        dtype=np.float64,
    )

    def to_pixel(position: np.ndarray) -> tuple[int, int]:
        relative = position - center
        return (
            round(plot_center[0] + relative[0] * scale),
            round(plot_center[1] - relative[1] * scale),
        )

    grid_step = _nice_grid_step(float(max(data_span)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    grid_font_scale = image_size / 3000
    grid_thickness = max(1, round(image_size / 1200))
    east = math.ceil(visible_minimum[0] / grid_step) * grid_step
    while east <= visible_maximum[0]:
        x, _ = to_pixel(np.array((east, 0.0)))
        cv2.line(canvas, (x, plot_top), (x, plot_bottom), GRID, grid_thickness, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{east:+.0f} m",
            (x + 5, plot_bottom - 10),
            font,
            grid_font_scale,
            MUTED,
            grid_thickness,
            cv2.LINE_AA,
        )
        east += grid_step
    north = math.ceil(visible_minimum[1] / grid_step) * grid_step
    while north <= visible_maximum[1]:
        _, y = to_pixel(np.array((0.0, north)))
        cv2.line(canvas, (plot_left, y), (plot_right, y), GRID, grid_thickness, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{north:+.0f} m",
            (plot_left + 8, y - 7),
            font,
            grid_font_scale,
            MUTED,
            grid_thickness,
            cv2.LINE_AA,
        )
        north += grid_step

    cv2.rectangle(
        canvas,
        (plot_left, plot_top),
        (plot_right, plot_bottom),
        GRID,
        max(2, round(image_size / 800)),
        cv2.LINE_AA,
    )
    pixel_points = np.asarray([to_pixel(point) for point in points], np.int32)
    if len(pixel_points) > 1:
        cv2.polylines(
            canvas,
            [pixel_points.reshape(-1, 1, 2)],
            False,
            GPS_PATH_COLOR,
            max(3, round(image_size / 330)),
            cv2.LINE_AA,
        )

    marker_radius = max(8, round(image_size / 100))
    start = tuple(int(value) for value in pixel_points[0])
    end = tuple(int(value) for value in pixel_points[-1])
    cv2.circle(canvas, start, marker_radius, START_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, end, marker_radius, END_COLOR, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "START",
        (start[0] + marker_radius, start[1] - marker_radius),
        font,
        image_size / 2600,
        START_COLOR,
        max(1, round(image_size / 1000)),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "END",
        (end[0] + marker_radius, end[1] - marker_radius),
        font,
        image_size / 2600,
        END_COLOR,
        max(1, round(image_size / 1000)),
        cv2.LINE_AA,
    )

    compass_center = np.array(
        (
            plot_right - round(image_size * 0.055),
            plot_top + round(image_size * 0.060),
        ),
        dtype=np.float64,
    )
    compass_length = image_size * 0.040
    rotation_radians = math.radians(rotation_degrees)
    rotated_north = np.array(
        (-math.sin(rotation_radians), math.cos(rotation_radians)),
        dtype=np.float64,
    )
    screen_north = np.array((rotated_north[0], -rotated_north[1]))
    compass_tip = compass_center + screen_north * compass_length
    cv2.arrowedLine(
        canvas,
        tuple(np.rint(compass_center).astype(int)),
        tuple(np.rint(compass_tip).astype(int)),
        FOREGROUND,
        max(2, round(image_size / 600)),
        cv2.LINE_AA,
        tipLength=0.3,
    )
    cv2.putText(
        canvas,
        "N",
        tuple(np.rint(compass_tip + screen_north * 18).astype(int)),
        font,
        image_size / 1900,
        FOREGROUND,
        max(1, round(image_size / 900)),
        cv2.LINE_AA,
    )

    title = "GPS ALIGNED PATH" if alignment_interval is not None else "GPS REFERENCE PATH"
    cv2.putText(
        canvas,
        title,
        (side_margin, round(title_height * 0.55)),
        font,
        image_size / 1250,
        FOREGROUND,
        max(2, round(image_size / 700)),
        cv2.LINE_AA,
    )
    displacement = float(np.linalg.norm(points[-1] - points[0]))
    duration = samples[-1].time_seconds - samples[0].time_seconds
    if alignment_interval is None:
        orientation_summary = "East right, North up"
    else:
        orientation_summary = (
            f"aligned {alignment_interval[0]:g}-{alignment_interval[1]:g} s  |  "
            f"rotation {rotation_degrees:+.1f} deg"
        )
    summary = (
        f"{len(samples)} metadata samples  |  {duration:.0f} s  |  "
        f"start-to-end {displacement:.1f} m  |  {orientation_summary}"
    )
    cv2.putText(
        canvas,
        summary,
        (side_margin, image_size - round(footer_height * 0.58)),
        font,
        image_size / 2350,
        MUTED,
        max(1, round(image_size / 950)),
        cv2.LINE_AA,
    )
    legend = "blue: raw embedded GPS path   white: start   yellow: end   coordinates are coarse"
    cv2.putText(
        canvas,
        legend,
        (side_margin, image_size - round(footer_height * 0.23)),
        font,
        image_size / 2500,
        MUTED,
        max(1, round(image_size / 1000)),
        cv2.LINE_AA,
    )

    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"OpenCV could not write the GPS image: {output}")
    return output.resolve()


def align_points_to_interval(
    samples: list[GpsSample],
    points: np.ndarray,
    *,
    start_seconds: float,
    end_seconds: float,
) -> tuple[np.ndarray, float]:
    """Rotate points until the selected interval's displacement points upward."""
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("alignment end must be greater than its non-negative start")
    if len(samples) != len(points):
        raise ValueError("GPS samples and map points must have the same length")
    if start_seconds < samples[0].time_seconds or end_seconds > samples[-1].time_seconds:
        raise ValueError(
            f"alignment interval must be within "
            f"{samples[0].time_seconds:g}-{samples[-1].time_seconds:g} seconds"
        )

    times = np.asarray([sample.time_seconds for sample in samples], dtype=np.float64)
    start_index = int(np.argmin(np.abs(times - start_seconds)))
    end_index = int(np.argmin(np.abs(times - end_seconds)))
    displacement = points[end_index] - points[start_index]
    if float(np.linalg.norm(displacement)) < 1.0:
        raise ValueError("alignment interval has less than 1 metre of GPS displacement")

    direction_radians = math.atan2(displacement[1], displacement[0])
    rotation_radians = math.pi / 2 - direction_radians
    rotation_degrees = math.degrees(rotation_radians)
    rotation_degrees = (rotation_degrees + 180.0) % 360.0 - 180.0
    rotation_radians = math.radians(rotation_degrees)
    cosine = math.cos(rotation_radians)
    sine = math.sin(rotation_radians)
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return points @ rotation.T, rotation_degrees


def _nice_grid_step(span: float) -> float:
    rough_step = max(span / 6.0, 1.0)
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    factor = 1.0 if normalized <= 1 else 2.0 if normalized <= 2 else 5.0
    return factor * magnitude
