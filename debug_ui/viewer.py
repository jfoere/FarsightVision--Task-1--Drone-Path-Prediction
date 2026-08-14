"""Minimal OpenCV video player used only during debugging."""

from __future__ import annotations

from pathlib import Path

import cv2

from drone_path.video import open_video


WINDOW_NAME = "Drone Path Debug Viewer"


def _draw_status(frame, frame_index: int, fps: float, paused: bool):
    """Return a copy of a frame with playback status and controls."""
    display = frame.copy()
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, display, 0.38, 0, display)

    elapsed = frame_index / fps if fps > 0 else 0.0
    state = "PAUSED" if paused else "PLAYING"
    cv2.putText(
        display,
        f"{state}  |  frame {frame_index}  |  {elapsed:.2f} s",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "Space: pause/resume   N: next frame   Q or Esc: quit",
        (16, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return display


def run_debug_viewer(video_path: str | Path) -> None:
    """Play a video with pause and single-frame stepping controls."""
    _, capture = open_video(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_delay_ms = max(1, round(1000 / fps)) if fps > 0 else 33

    paused = False
    advance_one_frame = True
    frame_index = -1
    current_frame = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
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

            display = _draw_status(current_frame, frame_index, fps, paused)
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
