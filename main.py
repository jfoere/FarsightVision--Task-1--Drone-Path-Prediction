"""Production entry point for headless drone-path prediction."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import time

from drone_path.config import ConfigError
from drone_path.path_output import render_path_image, save_prediction_json
from drone_path.pipeline import (
    PathPredictionError,
    PathPredictionProgress,
    predict_path,
)
from drone_path.video import VideoOpenError


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


def _image_size(value: str) -> int:
    number = int(value)
    if number < 400:
        raise argparse.ArgumentTypeError("must be at least 400")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a relative drone path and save PNG/JSON outputs.",
    )
    parser.add_argument("video", type=Path, help="path to the input video")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("path.png"),
        metavar="PNG",
        help="path image to create (default: path.png)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        metavar="JSON",
        help="path data to create (default: same name as PNG with .json)",
    )
    parser.add_argument(
        "--start",
        type=_non_negative_float,
        default=0.0,
        metavar="SECONDS",
        help="start processing at this timestamp (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        metavar="SECONDS",
        help="process only this many seconds",
    )
    parser.add_argument(
        "--image-size",
        type=_image_size,
        default=1600,
        metavar="PIXELS",
        help="square image size, at least 400 (default: 1600)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    json_output = args.json_output or args.output.with_suffix(".json")
    last_progress = {"percent": -1, "frames": 0}

    def report_progress(progress: PathPredictionProgress) -> None:
        if progress.total_frames:
            exact_percent = min(
                100,
                round(progress.processed_frames * 100 / progress.total_frames),
            )
            percent = 100 if exact_percent == 100 else (exact_percent // 5) * 5
            if percent == last_progress["percent"]:
                return
            last_progress["percent"] = percent
            print(
                f"\rProcessing: {percent:3d}% "
                f"({progress.processed_frames}/{progress.total_frames} frames)",
                end="",
                flush=True,
            )
        elif progress.processed_frames - last_progress["frames"] >= 100:
            last_progress["frames"] = progress.processed_frames
            print(
                f"\rProcessing: {progress.processed_frames} frames",
                end="",
                flush=True,
            )

    started_at = time.perf_counter()
    try:
        result = predict_path(
            args.video,
            start_seconds=args.start,
            duration_seconds=args.duration,
            progress_callback=report_progress,
        )
        print()
        image_path = render_path_image(
            result,
            args.output,
            image_size=args.image_size,
        )
        json_path = save_prediction_json(result, json_output)
    except (ConfigError, PathPredictionError, VideoOpenError, OSError, ValueError) as error:
        print()
        parser.error(str(error))

    elapsed = time.perf_counter() - started_at
    print(f"Processed {result.processed_frames} frames in {elapsed:.1f} seconds.")
    print(
        f"Path: {result.section_count} sections, "
        f"{len(result.uncertainty_markers)} uncertainty markers."
    )
    print(f"Image: {image_path}")
    print(f"Data:  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
