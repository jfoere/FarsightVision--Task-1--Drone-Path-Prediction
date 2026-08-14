"""Production entry point for drone video processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from drone_path.video import VideoOpenError, read_video_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process a drone video without opening a UI.",
    )
    parser.add_argument("video", type=Path, help="Path to the input video")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        info = read_video_info(args.video)
    except VideoOpenError as error:
        parser.error(str(error))

    duration = (
        f"{info.duration_seconds:.2f} seconds"
        if info.duration_seconds is not None
        else "unknown"
    )

    print(f"Video: {info.path}")
    print(f"Resolution: {info.width}x{info.height}")
    print(f"Frame rate: {info.fps:.3f} FPS")
    print(f"Reported frames: {info.frame_count}")
    print(f"Reported duration: {duration}")
    print("Video opened successfully. Motion processing is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
