"""Minimal OpenCV video player used only during debugging."""

from __future__ import annotations

import math
from pathlib import Path
import time

import cv2
import numpy as np

from drone_path.algorithm import OpticalFlowEstimator, OpticalFlowResult
from drone_path.video import open_video


WINDOW_NAME = "Drone Path Debug Viewer"
INITIAL_MAX_WIDTH = 1280
INITIAL_MAX_HEIGHT = 720
MAX_FLOW_VECTORS = 120
FLOW_PROCESSING_MAX_WIDTH = 960
FLOW_VECTOR_DISPLAY_SCALE = 10.0


def _draw_status(
    frame,
    frame_index: int,
    fps: float,
    paused: bool,
    details: str | None = None,
):
    """Draw status after the frame has been resized for display."""
    display = frame.copy()
    overlay = display.copy()
    display_height = display.shape[0]
    font_scale = max(0.42, min(0.58, display_height / 1500))

    elapsed = frame_index / fps if fps > 0 else 0.0
    state = "PAUSED" if paused else "PLAYING"
    status_text = f"{state}  |  #{frame_index}  |  {elapsed:.2f} s"
    if details:
        status_text = f"{status_text}  |  {details}"
    controls_text = "Space: play/pause  |  N: next  |  Q: quit"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        status_text,
        font,
        font_scale,
        1,
    )
    badge_x = 10
    badge_y = 10
    padding_x = 8
    padding_y = 6
    badge_width = text_width + (padding_x * 2)
    badge_height = text_height + baseline + (padding_y * 2)

    (controls_width, controls_height), controls_baseline = cv2.getTextSize(
        controls_text,
        font,
        font_scale,
        1,
    )
    controls_badge_width = controls_width + (padding_x * 2)
    controls_badge_height = controls_height + controls_baseline + (padding_y * 2)
    controls_x = display.shape[1] - controls_badge_width - 10
    controls_y = 10
    if controls_x <= badge_x + badge_width + 10:
        controls_x = badge_x
        controls_y = badge_y + badge_height + 6

    cv2.rectangle(
        overlay,
        (badge_x, badge_y),
        (badge_x + badge_width, badge_y + badge_height),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        overlay,
        (controls_x, controls_y),
        (
            controls_x + controls_badge_width,
            controls_y + controls_badge_height,
        ),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.62, display, 0.38, 0, display)

    cv2.putText(
        display,
        status_text,
        (badge_x + padding_x, badge_y + padding_y + text_height),
        font,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        controls_text,
        (controls_x + padding_x, controls_y + padding_y + controls_height),
        font,
        font_scale,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return display


def _draw_optical_flow(
    frame,
    flow: OpticalFlowResult,
    source_shape: tuple[int, ...],
):
    """Draw a representative subset of raw flow vectors at display scale."""
    display = frame.copy()
    if flow.tracked_count == 0:
        return display

    source_height, source_width = source_shape[:2]
    scale_x = display.shape[1] / source_width
    scale_y = display.shape[0] / source_height
    sample_count = min(flow.tracked_count, MAX_FLOW_VECTORS)
    sample_indices = np.linspace(
        0,
        flow.tracked_count - 1,
        sample_count,
        dtype=int,
    )

    for index in sample_indices:
        previous_x, previous_y = flow.previous_points[index]
        current_x, current_y = flow.current_points[index]
        previous = (
            round(float(previous_x) * scale_x),
            round(float(previous_y) * scale_y),
        )
        current = (
            round(float(current_x) * scale_x),
            round(float(current_y) * scale_y),
        )
        vector_end = (
            round(
                previous[0]
                + float(current_x - previous_x)
                * scale_x
                * FLOW_VECTOR_DISPLAY_SCALE
            ),
            round(
                previous[1]
                + float(current_y - previous_y)
                * scale_y
                * FLOW_VECTOR_DISPLAY_SCALE
            ),
        )
        cv2.arrowedLine(
            display,
            previous,
            vector_end,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
            tipLength=0.3,
        )
        cv2.circle(display, current, 2, (70, 255, 70), -1, cv2.LINE_AA)

    return display


def _fit_frame(frame, target_width: int, target_height: int):
    """Resize a frame to fit inside a target area without changing its aspect ratio."""
    frame_height, frame_width = frame.shape[:2]
    if target_width <= 0 or target_height <= 0:
        return frame

    scale = min(target_width / frame_width, target_height / frame_height)
    output_width = max(1, round(frame_width * scale))
    output_height = max(1, round(frame_height * scale))

    if output_width == frame_width and output_height == frame_height:
        return frame

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, (output_width, output_height), interpolation=interpolation)


def _resize_for_flow(frame):
    """Use a smaller analysis frame while keeping the displayed video sharp."""
    height, width = frame.shape[:2]
    if width <= FLOW_PROCESSING_MAX_WIDTH:
        return frame
    output_height = round(height * FLOW_PROCESSING_MAX_WIDTH / width)
    return cv2.resize(
        frame,
        (FLOW_PROCESSING_MAX_WIDTH, output_height),
        interpolation=cv2.INTER_AREA,
    )


def _current_display_size(frame) -> tuple[int, int]:
    """Return the current image area, with a safe size before the first render."""
    try:
        _, _, width, height = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        width = height = 0

    if width > 1 and height > 1:
        return width, height

    frame_height, frame_width = frame.shape[:2]
    scale = min(
        1.0,
        INITIAL_MAX_WIDTH / frame_width,
        INITIAL_MAX_HEIGHT / frame_height,
    )
    return round(frame_width * scale), round(frame_height * scale)


def run_debug_viewer(
    video_path: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> None:
    """Play a video with pause and single-frame stepping controls."""
    if start_seconds < 0:
        raise ValueError("start_seconds must be zero or greater")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")

    _, capture = open_video(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_delay_ms = max(1, round(1000 / fps)) if fps > 0 else 33
    start_frame = round(start_seconds * fps) if fps > 0 else 0
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    elif start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    max_frames = (
        max(1, math.ceil(duration_seconds * fps))
        if duration_seconds is not None and fps > 0
        else None
    )
    estimator = OpticalFlowEstimator()

    paused = False
    advance_one_frame = True
    frame_index = start_frame - 1
    frames_read = 0
    previous_flow_frame = None
    current_frame = None
    current_flow_shape = None
    flow_result = OpticalFlowResult.empty()
    frame_started_at = time.perf_counter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    source_width = max(1, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = max(1, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    initial_scale = min(
        1.0,
        INITIAL_MAX_WIDTH / source_width,
        INITIAL_MAX_HEIGHT / source_height,
    )
    cv2.resizeWindow(
        WINDOW_NAME,
        round(source_width * initial_scale),
        round(source_height * initial_scale),
    )
    try:
        while True:
            if not paused or advance_one_frame:
                frame_started_at = time.perf_counter()
                if max_frames is not None and frames_read >= max_frames:
                    print("Reached the end of the selected interval.")
                    break
                if (
                    duration_seconds is not None
                    and fps <= 0
                    and capture.get(cv2.CAP_PROP_POS_MSEC)
                    >= (start_seconds + duration_seconds) * 1000
                ):
                    print("Reached the end of the selected interval.")
                    break

                decoded, frame = capture.read()
                if not decoded or frame is None:
                    print("Reached the end of the video.")
                    break

                reported_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                frame_index = max(frame_index + 1, reported_frame)
                frames_read += 1
                current_frame = frame
                current_flow_frame = _resize_for_flow(current_frame)
                current_flow_shape = current_flow_frame.shape
                if previous_flow_frame is None:
                    flow_result = OpticalFlowResult.empty()
                else:
                    flow_result = estimator.estimate(
                        previous_flow_frame,
                        current_flow_frame,
                    )
                previous_flow_frame = current_flow_frame
                advance_one_frame = False

            if current_frame is None:
                break

            target_width, target_height = _current_display_size(current_frame)
            fitted_frame = _fit_frame(current_frame, target_width, target_height)
            flow_display = _draw_optical_flow(
                fitted_frame,
                flow_result,
                current_flow_shape or current_frame.shape,
            )
            display = _draw_status(
                flow_display,
                frame_index,
                fps,
                paused,
                details=(
                    f"tracks {flow_result.tracked_count}/{flow_result.detected_count}"
                    f"  |  arrows x{FLOW_VECTOR_DISPLAY_SCALE:g}"
                ),
            )
            cv2.imshow(WINDOW_NAME, display)

            processing_ms = round((time.perf_counter() - frame_started_at) * 1000)
            delay = 30 if paused else max(1, frame_delay_ms - processing_ms)
            key = cv2.waitKey(delay) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("n"):
                paused = True
                advance_one_frame = True

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyWindow(WINDOW_NAME)
