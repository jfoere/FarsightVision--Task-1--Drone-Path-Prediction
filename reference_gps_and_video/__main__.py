"""Build an estimated metric reference path from GPS and video motion."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import time

from reference_gps.extractor import GpsMetadataError, extract_gps_samples
from reference_gps_and_video.fused_output import (
    render_fused_reference,
    save_fused_reference_json,
)
from reference_gps_and_video.fusion import (
    ReferenceFusionError,
    fuse_reference_trajectory,
)
from reference_gps_and_video.output import (
    render_visual_odometry_diagnostic,
    save_visual_odometry_json,
)
from reference_gps_and_video.visual_odometry import (
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
        description="Build a GPS + video estimated reference path.",
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
        help=(
            "rotate only the rendered map so movement from START to END seconds "
            "points up (default: first stable movement)"
        ),
    )
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help="write the unscaled visual-odometry diagnostic instead of GPS fusion",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PNG",
        help=(
            "output image (default: reference-path.png, or vo-diagnostic.png "
            "with --visual-only)"
        ),
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
    output = args.output or Path(
        "vo-diagnostic.png" if args.visual_only else "reference-path.png"
    )
    json_output = args.json_output or output.with_suffix(".json")
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
        alignment_interval = (
            tuple(args.align) if args.align is not None else None
        )
        if args.visual_only:
            image_path = render_visual_odometry_diagnostic(
                result,
                output,
                image_size=args.image_size,
                alignment_interval=alignment_interval,
            )
            json_path = save_visual_odometry_json(result, json_output)
            fused_result = None
        else:
            gps_samples = extract_gps_samples(args.video)
            fused_result = fuse_reference_trajectory(
                gps_samples,
                result,
                alignment_interval=alignment_interval,
            )
            image_path = render_fused_reference(
                fused_result,
                output,
                image_size=args.image_size,
            )
            json_path = save_fused_reference_json(fused_result, json_output)
    except (
        GpsMetadataError,
        ReferenceFusionError,
        VisualOdometryError,
        OSError,
        ValueError,
    ) as error:
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
    if fused_result is None:
        print(
            f"Rotation-only signed ground yaw: "
            f"{result.rotation_only_ground_yaw_degrees:+.1f} degrees."
        )
        print("This visual-only trajectory is unscaled and is not GPS-fused.")
    else:
        print(
            f"Estimated metric reference: {fused_result.start_to_end_distance_m:.1f} m "
            f"start-to-end; GPS fit RMS "
            f"{fused_result.gps_rms_residual_m:.1f} m; last-GPS fit residual "
            f"{fused_result.endpoint_gps_residual_m:.1f} m."
        )
        print(
            f"Direction calibration: {fused_result.alignment_window_count} windows; "
            f"VO-to-GPS alignment correction changed "
            f"{fused_result.visual_alignment_drift_degrees:+.1f} degrees."
        )
    print(f"Image: {image_path}")
    print(f"Data:  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
