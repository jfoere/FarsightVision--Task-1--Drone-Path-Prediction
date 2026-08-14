"""Minimal OpenCV video player used only during debugging."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np

from drone_path.algorithm import (
    GlobalMotionEstimator,
    GlobalMotionMeasurement,
    MotionClassifier,
    MotionState,
    OpticalFlowEstimator,
    OpticalFlowResult,
)
from drone_path.video import open_video


WINDOW_NAME = "Drone Path Debug Viewer"
INITIAL_MAX_WIDTH = 1280
INITIAL_MAX_HEIGHT = 720
MAX_FLOW_VECTORS = 120
FLOW_PROCESSING_MAX_WIDTH = 960
FLOW_VECTOR_DISPLAY_SCALE = 10.0
MIN_PLAYBACK_SPEED = 0.1
MAX_PLAYBACK_SPEED = 4.0
PLAYBACK_SPEED_STEP = 0.1

ButtonRectangle = tuple[int, int, int, int]

MOTION_STATE_CODES = {
    MotionState.UNCERTAIN: 0,
    MotionState.TRANSLATION: 1,
    MotionState.ROTATION: 2,
}
MOTION_STATE_COLORS = {
    MotionState.UNCERTAIN: (120, 120, 120),
    MotionState.TRANSLATION: (70, 210, 70),
    MotionState.ROTATION: (0, 150, 255),
}
UNKNOWN_TIMELINE_COLOR = (80, 80, 80)


@dataclass(slots=True)
class StateTimelineViewport:
    """Keep a stable 10x state window and advance it at the 90% boundary."""

    zoom: float = 10.0
    left_trigger: float = 0.10
    right_trigger: float = 0.90
    start_frame: int = 0
    initialized: bool = False

    def bounds(self, current_frame: int, total_frames: int) -> tuple[int, int]:
        if total_frames <= 0:
            return 0, 0

        window_frames = max(1, math.ceil(total_frames / self.zoom))
        max_start = max(0, total_frames - window_frames)
        current_frame = min(total_frames - 1, max(0, current_frame))

        if not self.initialized:
            if current_frame <= round(window_frames * self.right_trigger):
                self.start_frame = 0
            else:
                self.start_frame = current_frame - round(
                    window_frames * self.left_trigger
                )
            self.initialized = True

        visible_position = current_frame - self.start_frame
        right_boundary = round(window_frames * self.right_trigger)
        if visible_position >= right_boundary or visible_position < 0:
            self.start_frame = current_frame - round(
                window_frames * self.left_trigger
            )

        self.start_frame = min(max_start, max(0, self.start_frame))
        return self.start_frame, min(total_frames, self.start_frame + window_frames)


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


def _draw_motion_metrics(
    frame,
    motion: GlobalMotionMeasurement,
    state: MotionState,
):
    """Draw the current temporal state followed by its raw measurements."""
    display = frame.copy()
    overlay = display.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, min(0.56, display.shape[0] / 1500))
    if motion.valid:
        text = (
            f"{state.value}"
            f"  |  flow {motion.median_flow_pixels:.2f} px"
            f"  |  shift ({motion.translation_x_pixels:+.2f}, "
            f"{motion.translation_y_pixels:+.2f}) px"
            f"  |  rot {motion.rotation_degrees:+.3f} deg"
            f"  |  scale {motion.scale:.4f}"
            f"  |  inliers {motion.inlier_ratio:.0%}"
        )
    else:
        text = f"{state.value}  |  motion unavailable"
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        1,
    )
    badge_x = 10
    badge_y = 45
    padding_x = 8
    padding_y = 6
    badge_width = text_width + (padding_x * 2)
    badge_height = text_height + baseline + (padding_y * 2)
    cv2.rectangle(
        overlay,
        (badge_x, badge_y),
        (badge_x + badge_width, badge_y + badge_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.62, display, 0.38, 0, display)
    cv2.putText(
        display,
        text,
        (badge_x + padding_x, badge_y + padding_y + text_height),
        font,
        font_scale,
        MOTION_STATE_COLORS[state],
        1,
        cv2.LINE_AA,
    )
    return display


def _change_playback_speed(current_speed: float, steps: int) -> float:
    """Adjust speed in exact 0.1x steps and clamp it to the supported range."""
    changed = current_speed + (steps * PLAYBACK_SPEED_STEP)
    return min(MAX_PLAYBACK_SPEED, max(MIN_PLAYBACK_SPEED, round(changed, 1)))


def _draw_speed_controls(
    frame,
    playback_speed: float,
) -> tuple[np.ndarray, ButtonRectangle, ButtonRectangle]:
    """Draw compact clickable speed controls and return their hit areas."""
    display = frame.copy()
    overlay = display.copy()
    display_height = display.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, min(0.56, display_height / 1500))
    button_height = max(28, min(36, round(display_height * 0.05)))
    button_width = 58
    speed_width = 106
    gap = 4
    left = 10
    top = display_height - button_height - 10

    decrease = (left, top, left + button_width, top + button_height)
    speed_box_left = decrease[2] + gap
    speed_box = (
        speed_box_left,
        top,
        speed_box_left + speed_width,
        top + button_height,
    )
    increase_left = speed_box[2] + gap
    increase = (
        increase_left,
        top,
        increase_left + button_width,
        top + button_height,
    )

    for rectangle in (decrease, speed_box, increase):
        cv2.rectangle(
            overlay,
            (rectangle[0], rectangle[1]),
            (rectangle[2], rectangle[3]),
            (0, 0, 0),
            -1,
        )
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

    labels = (
        (decrease, "-0.1"),
        (speed_box, f"Speed {playback_speed:.1f}x"),
        (increase, "+0.1"),
    )
    for rectangle, label in labels:
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            1,
        )
        text_x = rectangle[0] + (rectangle[2] - rectangle[0] - text_width) // 2
        text_y = rectangle[1] + (
            rectangle[3] - rectangle[1] + text_height
        ) // 2
        cv2.putText(
            display,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    return display, decrease, increase


def _draw_timeline(
    frame,
    frame_index: int,
    total_frames: int,
    left: int,
) -> tuple[np.ndarray, ButtonRectangle | None]:
    """Draw the fixed full-video position slider."""
    display = frame.copy()
    if total_frames <= 1:
        return display, None

    right = display.shape[1] - 14
    control_center_y = display.shape[0] - 10 - max(
        28,
        min(36, round(display.shape[0] * 0.05)),
    ) // 2
    if right - left < 80:
        left = 14
        control_center_y -= 28

    progress = min(1.0, max(0.0, frame_index / (total_frames - 1)))
    thumb_x = round(left + progress * (right - left))

    cv2.line(
        display,
        (left, control_center_y),
        (right, control_center_y),
        (110, 110, 110),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        display,
        (left, control_center_y),
        (thumb_x, control_center_y),
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.circle(
        display,
        (thumb_x, control_center_y),
        7,
        (25, 25, 25),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        display,
        (thumb_x, control_center_y),
        5,
        (245, 245, 245),
        -1,
        cv2.LINE_AA,
    )
    return display, (left, control_center_y - 12, right, control_center_y + 12)


def _draw_state_timeline(
    frame,
    frame_index: int,
    state_codes: np.ndarray | None,
    view_start: int,
    view_end: int,
    left: int,
) -> np.ndarray:
    """Draw a non-interactive 10x state line above the full-video slider."""
    display = frame.copy()
    if state_codes is None or view_end <= view_start:
        return display

    right = display.shape[1] - 14
    if right - left < 80:
        left = 76
    if right <= left:
        return display
    position_line_y = display.shape[0] - 10 - max(
        28,
        min(36, round(display.shape[0] * 0.05)),
    ) // 2
    state_line_y = position_line_y - 18
    line_width = right - left + 1

    sampled_frames = np.rint(
        np.linspace(view_start, view_end - 1, line_width)
    ).astype(int)
    sampled_codes = state_codes[sampled_frames]
    colors = np.full(
        (line_width, 3),
        UNKNOWN_TIMELINE_COLOR,
        dtype=np.uint8,
    )
    for state, code in MOTION_STATE_CODES.items():
        colors[sampled_codes == code] = MOTION_STATE_COLORS[state]

    for offset in (-2, -1, 0, 1, 2):
        row = state_line_y + offset
        if 0 <= row < display.shape[0]:
            display[row, left : right + 1] = colors

    visible_frames = max(1, view_end - view_start)
    position = (frame_index - view_start) / visible_frames
    position = min(1.0, max(0.0, position))
    pointer_x = round(left + position * (right - left))
    cv2.circle(
        display,
        (pointer_x, state_line_y),
        6,
        (25, 25, 25),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        display,
        (pointer_x, state_line_y),
        4,
        (245, 245, 245),
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "STATE 10x",
        (max(4, left - 68), state_line_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return display


def _frame_from_timeline(x: int, rectangle: ButtonRectangle, total_frames: int) -> int:
    """Convert a timeline mouse position to a clamped zero-based frame index."""
    width = max(1, rectangle[2] - rectangle[0])
    progress = min(1.0, max(0.0, (x - rectangle[0]) / width))
    return round(progress * max(0, total_frames - 1))


def _point_inside(x: int, y: int, rectangle: ButtonRectangle | None) -> bool:
    return bool(
        rectangle
        and rectangle[0] <= x <= rectangle[2]
        and rectangle[1] <= y <= rectangle[3]
    )


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
    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    start_frame = round(start_seconds * fps) if fps > 0 else 0
    if total_frames > 0:
        start_frame = min(start_frame, total_frames - 1)
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
    motion_estimator = GlobalMotionEstimator()
    motion_classifier = MotionClassifier()

    paused = False
    advance_one_frame = True
    frame_index = start_frame - 1
    frames_read = 0
    previous_flow_frame = None
    current_frame = None
    current_flow_shape = None
    flow_result = OpticalFlowResult.empty()
    motion_measurement = GlobalMotionMeasurement.unavailable()
    motion_state = MotionState.UNCERTAIN
    classified_states = (
        np.full((total_frames,), -1, dtype=np.int8)
        if total_frames > 0
        else None
    )
    state_viewport = StateTimelineViewport()
    frame_started_at = time.perf_counter()
    speed_state = {"value": 1.0}
    button_state: dict[str, ButtonRectangle | None] = {
        "decrease": None,
        "increase": None,
        "timeline": None,
    }
    seek_state: dict[str, int | bool | None] = {
        "requested": None,
        "dragging": False,
    }

    def request_timeline_frame(x: int) -> None:
        timeline = button_state["timeline"]
        if timeline is not None:
            seek_state["requested"] = _frame_from_timeline(
                x,
                timeline,
                total_frames,
            )

    def on_mouse(event: int, x: int, y: int, flags: int, _data) -> None:
        timeline = button_state["timeline"]
        if event == cv2.EVENT_LBUTTONDOWN and _point_inside(x, y, timeline):
            seek_state["dragging"] = True
            request_timeline_frame(x)
            return

        if (
            event == cv2.EVENT_MOUSEMOVE
            and seek_state["dragging"]
            and flags & cv2.EVENT_FLAG_LBUTTON
        ):
            request_timeline_frame(x)
            return

        if event != cv2.EVENT_LBUTTONUP:
            return
        if seek_state["dragging"]:
            request_timeline_frame(x)
            seek_state["dragging"] = False
        elif _point_inside(x, y, button_state["decrease"]):
            speed_state["value"] = _change_playback_speed(
                float(speed_state["value"]),
                -1,
            )
        elif _point_inside(x, y, button_state["increase"]):
            speed_state["value"] = _change_playback_speed(
                float(speed_state["value"]),
                1,
            )

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
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            requested_frame = seek_state["requested"]
            if requested_frame is not None:
                seek_state["requested"] = None
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(requested_frame))
                frame_index = int(requested_frame) - 1
                frames_read = 0
                previous_flow_frame = None
                current_frame = None
                current_flow_shape = None
                flow_result = OpticalFlowResult.empty()
                motion_measurement = GlobalMotionMeasurement.unavailable()
                motion_classifier.reset()
                motion_state = MotionState.UNCERTAIN
                paused = True
                advance_one_frame = True

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
                motion_measurement = motion_estimator.measure(
                    flow_result,
                    (current_flow_frame.shape[1], current_flow_frame.shape[0]),
                )
                if fps > 0:
                    motion_state = motion_classifier.update(
                        motion_measurement,
                        fps=fps,
                        frame_width=current_flow_frame.shape[1],
                    )
                else:
                    motion_state = MotionState.UNCERTAIN
                if (
                    classified_states is not None
                    and 0 <= frame_index < classified_states.size
                ):
                    classified_states[frame_index] = MOTION_STATE_CODES[motion_state]
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
            display = _draw_motion_metrics(
                display,
                motion_measurement,
                motion_state,
            )
            display, decrease_button, increase_button = _draw_speed_controls(
                display,
                float(speed_state["value"]),
            )
            button_state["decrease"] = decrease_button
            button_state["increase"] = increase_button
            timeline_left = increase_button[2] + 72
            view_start, view_end = state_viewport.bounds(
                frame_index,
                total_frames,
            )
            display = _draw_state_timeline(
                display,
                frame_index,
                classified_states,
                view_start,
                view_end,
                timeline_left,
            )
            display, timeline = _draw_timeline(
                display,
                frame_index,
                total_frames,
                timeline_left,
            )
            button_state["timeline"] = timeline
            cv2.imshow(WINDOW_NAME, display)

            processing_ms = round((time.perf_counter() - frame_started_at) * 1000)
            requested_delay = frame_delay_ms / float(speed_state["value"])
            delay = 30 if paused else max(1, round(requested_delay) - processing_ms)
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
