"""Command-line launcher for the optional debugging UI."""

from __future__ import annotations

import argparse
from pathlib import Path

from drone_path.video import VideoOpenError

from debug_ui.viewer import run_debug_viewer


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="Play a video in the debugging UI.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument(
        "--start",
        type=_non_negative_float,
        default=0.0,
        metavar="SECONDS",
        help="start playback at this timestamp (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        metavar="SECONDS",
        help="stop after this many seconds",
    )
    args = parser.parse_args()

    try:
        run_debug_viewer(
            args.video,
            start_seconds=args.start,
            duration_seconds=args.duration,
        )
    except (VideoOpenError, ValueError) as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
