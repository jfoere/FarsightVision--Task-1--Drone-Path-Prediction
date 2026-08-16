"""Run the independent visual-odometry checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import time

from reference_path.output import (
    render_visual_odometry_diagnostic,
    save_visual_odometry_json,
)
from reference_path.visual_odometry import (
    VisualOdometryConfig,
    VisualOdometryError,
    VisualOdometryProgress,
    estimate_visual_odometry,
)


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
    if number < 600:
        raise argparse.ArgumentTypeError("must be at least 600")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check independent SIFT/homography visual odometry.",
    )
    parser.add_argument("video", type=Path, help="path to the input video")
    parser.add_argument(
        "--start",
        type=_non_negative_float,
        default=0.0,
        metavar="SECONDS",
        help="start timestamp (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        metavar="SECONDS",
        help="process only this many seconds (default: process to the end)",
    )
    parser.add_argument(
        "--interval",
        type=_positive_float,
        default=0.2,
        metavar="SECONDS",
        help="time between analyzed keyframes (default: 0.2)",
    )
    parser.add_argument(
        "--align",
        type=_non_negative_float,
        nargs=2,
        metavar=("START", "END"),
        help="rotate the path so movement from START to END seconds points up",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vo-diagnostic.png"),
        metavar="PNG",
        help="diagnostic image (default: vo-diagnostic.png)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        metavar="JSON",
        help="motion data (default: same name as PNG with .json)",
    )
    parser.add_argument(
        "--image-size",
        type=_image_size,
        default=1600,
        metavar="PIXELS",
        help="square image size, at least 600 (default: 1600)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    json_output = args.json_output or args.output.with_suffix(".json")
    started_at = time.perf_counter()
    last_percentage = {"value": -1}

    def report_progress(progress: VisualOdometryProgress) -> None:
        if progress.percentage == last_percentage["value"]:
            return
        last_percentage["value"] = progress.percentage
        print(
            f"\rProcessing: {progress.percentage:3d}% "
            f"({progress.processed_pairs}/{progress.total_pairs} keyframe pairs)",
            end="",
            flush=True,
        )

    try:
        result = estimate_visual_odometry(
            args.video,
            start_seconds=args.start,
            duration_seconds=args.duration,
            config=VisualOdometryConfig(sample_interval_seconds=args.interval),
            progress_callback=report_progress,
        )
        if last_percentage["value"] >= 0:
            print()
        image_path = render_visual_odometry_diagnostic(
            result,
            args.output,
            image_size=args.image_size,
            alignment_interval=(tuple(args.align) if args.align is not None else None),
        )
        json_path = save_visual_odometry_json(result, json_output)
    except (VisualOdometryError, OSError, ValueError) as error:
        if last_percentage["value"] >= 0:
            print()
        parser.error(str(error))

    elapsed = time.perf_counter() - started_at
    print(f"Analyzed {len(result.motions)} frame pairs in {elapsed:.2f} seconds.")
    print(
        f"Translation: {result.translation_count}; "
        f"rotation-only: {result.rotation_only_count}; "
        f"unreliable: {result.unreliable_count}."
    )
    print(
        f"Rotation-only signed ground yaw: "
        f"{result.rotation_only_ground_yaw_degrees:+.1f} degrees."
    )
    print("This trajectory is unscaled and is not GPS-fused yet.")
    print(f"Image: {image_path}")
    print(f"Data:  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
