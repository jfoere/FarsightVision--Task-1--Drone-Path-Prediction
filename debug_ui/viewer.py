"""Minimal OpenCV video player used only during debugging."""

from __future__ import annotations

from pathlib import Path

import cv2

from drone_path.video import open_video


WINDOW_NAME = "Drone Path Debug Viewer"
INITIAL_MAX_WIDTH = 1280
INITIAL_MAX_HEIGHT = 720


def _draw_status(frame, frame_index: int, fps: float, paused: bool):
    """Draw status after the frame has been resized for display."""
    display = frame.copy()
    overlay = display.copy()
    display_height = display.shape[0]
    font_scale = max(0.42, min(0.58, display_height / 1500))

    elapsed = frame_index / fps if fps > 0 else 0.0
    state = "PAUSED" if paused else "PLAYING"
    status_text = f"{state}  |  #{frame_index}  |  {elapsed:.2f} s"
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


def run_debug_viewer(video_path: str | Path) -> None:
    """Play a video with pause and single-frame stepping controls."""
    _, capture = open_video(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_delay_ms = max(1, round(1000 / fps)) if fps > 0 else 33

    paused = False
    advance_one_frame = True
    frame_index = -1
    current_frame = None

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
                decoded, frame = capture.read()
                if not decoded or frame is None:
                    print("Reached the end of the video.")
                    break
                frame_index += 1
                current_frame = frame
                advance_one_frame = False

            if current_frame is None:
                break

            target_width, target_height = _current_display_size(current_frame)
            fitted_frame = _fit_frame(current_frame, target_width, target_height)
            display = _draw_status(fitted_frame, frame_index, fps, paused)
            cv2.imshow(WINDOW_NAME, display)

            delay = 30 if paused else frame_delay_ms
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
