"""JSON and large PNG outputs for relative drone paths."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from drone_path.pipeline import PathPredictionResult


BACKGROUND = (27, 31, 36)
FOREGROUND = (230, 230, 230)
MUTED = (145, 150, 158)
GRID = (62, 68, 76)
PATH_COLOR = (70, 210, 70)
END_COLOR = (255, 210, 50)
HEADING_COLOR = (0, 220, 255)
UNCERTAIN_COLOR = (0, 150, 255)


def save_prediction_json(
    result: PathPredictionResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def render_path_image(
    result: PathPredictionResult,
    output_path: str | Path,
    *,
    image_size: int = 1600,
) -> Path:
    """Render a square, equal-scale top view and save it as an image."""
    if image_size < 400:
        raise ValueError("image_size must be at least 400 pixels")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((image_size, image_size, 3), BACKGROUND, dtype=np.uint8)
    title_height = round(image_size * 0.09)
    footer_height = round(image_size * 0.10)
    side_margin = round(image_size * 0.08)
    plot_left = side_margin
    plot_right = image_size - side_margin
    plot_top = title_height
    plot_bottom = image_size - footer_height
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    points = np.asarray(result.points, dtype=np.float64)
    if points.size == 0:
        points = np.zeros((1, 2), dtype=np.float64)
    markers = np.asarray(result.uncertainty_markers, dtype=np.float64)
    all_positions = points if markers.size == 0 else np.vstack((points, markers))

    minimum = np.min(all_positions, axis=0)
    maximum = np.max(all_positions, axis=0)
    center = (minimum + maximum) / 2
    span = np.maximum(maximum - minimum, 1.0)
    scale = min(plot_width / span[0], plot_height / span[1]) * 0.78
    plot_center = np.array(
        [(plot_left + plot_right) / 2, (plot_top + plot_bottom) / 2],
        dtype=np.float64,
    )

    def to_pixel(position: np.ndarray) -> tuple[int, int]:
        relative = position - center
        return (
            round(plot_center[0] + relative[0] * scale),
            round(plot_center[1] - relative[1] * scale),
        )

    origin = to_pixel(np.array((0.0, 0.0), dtype=np.float64))
    if plot_left <= origin[0] <= plot_right:
        cv2.line(
            canvas,
            (origin[0], plot_top),
            (origin[0], plot_bottom),
            GRID,
            2,
            cv2.LINE_AA,
        )
    if plot_top <= origin[1] <= plot_bottom:
        cv2.line(
            canvas,
            (plot_left, origin[1]),
            (plot_right, origin[1]),
            GRID,
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(
        canvas,
        (plot_left, plot_top),
        (plot_right, plot_bottom),
        GRID,
        2,
        cv2.LINE_AA,
    )

    pixel_points = np.asarray([to_pixel(point) for point in points], np.int32)
    if len(pixel_points) > 1:
        cv2.polylines(
            canvas,
            [pixel_points.reshape(-1, 1, 2)],
            False,
            PATH_COLOR,
            max(3, round(image_size / 320)),
            cv2.LINE_AA,
        )

    marker_radius = max(8, round(image_size / 100))
    for marker in markers.reshape(-1, 2) if markers.size else ():
        marker_pixel = to_pixel(marker)
        cv2.circle(
            canvas,
            marker_pixel,
            marker_radius,
            UNCERTAIN_COLOR,
            max(2, round(image_size / 600)),
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "?",
            (marker_pixel[0] + marker_radius, marker_pixel[1] - marker_radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            image_size / 1800,
            UNCERTAIN_COLOR,
            max(1, round(image_size / 800)),
            cv2.LINE_AA,
        )

    start = tuple(int(value) for value in pixel_points[0])
    end = tuple(int(value) for value in pixel_points[-1])
    cv2.circle(canvas, start, marker_radius, FOREGROUND, -1, cv2.LINE_AA)
    cv2.circle(canvas, end, marker_radius, END_COLOR, -1, cv2.LINE_AA)

    heading_radians = math.radians(result.final_heading_degrees)
    arrow_length = max(45, round(image_size * 0.055))
    heading_end = (
        end[0] + round(math.sin(heading_radians) * arrow_length),
        end[1] - round(math.cos(heading_radians) * arrow_length),
    )
    heading_color = UNCERTAIN_COLOR if result.heading_assumed else HEADING_COLOR
    cv2.arrowedLine(
        canvas,
        end,
        heading_end,
        heading_color,
        max(3, round(image_size / 400)),
        cv2.LINE_AA,
        tipLength=0.25,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "RELATIVE DRONE PATH",
        (side_margin, round(title_height * 0.55)),
        font,
        image_size / 1250,
        FOREGROUND,
        max(2, round(image_size / 700)),
        cv2.LINE_AA,
    )
    reliability = "ASSUMED AFTER UNCERTAINTY" if result.heading_assumed else "RELIABLE"
    summary = (
        f"{result.section_count} sections  |  {result.processed_frames} frames  |  "
        f"heading {result.final_heading_degrees:+.1f} deg  |  {reliability}"
    )
    cv2.putText(
        canvas,
        summary,
        (side_margin, image_size - round(footer_height * 0.55)),
        font,
        image_size / 2300,
        MUTED,
        max(1, round(image_size / 900)),
        cv2.LINE_AA,
    )
    legend = "green: connected path   white: start   blue: end   orange ?: uncertainty"
    cv2.putText(
        canvas,
        legend,
        (side_margin, image_size - round(footer_height * 0.22)),
        font,
        image_size / 2500,
        MUTED,
        max(1, round(image_size / 1000)),
        cv2.LINE_AA,
    )

    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"OpenCV could not write the path image: {output}")
    return output.resolve()
