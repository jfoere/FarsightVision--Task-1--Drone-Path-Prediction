"""Command-line launcher for the optional debugging UI."""

from __future__ import annotations

import argparse
from pathlib import Path

from drone_path.video import VideoOpenError

from debug_ui.viewer import run_debug_viewer


def main() -> int:
    parser = argparse.ArgumentParser(description="Play a video in the debugging UI.")
    parser.add_argument("video", type=Path, help="Path to the input video")
    args = parser.parse_args()

    try:
        run_debug_viewer(args.video)
    except VideoOpenError as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
