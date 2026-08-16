"""Command-line entry point for the independent GPS reference image."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import time

from reference_gps.extractor import GpsMetadataError, extract_gps_samples, to_local_metres
from reference_gps.renderer import render_gps_path


def _image_size(value: str) -> int:
    number = int(value)
    if number < 400:
        raise argparse.ArgumentTypeError("must be at least 400")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reference path image from embedded DJI GPS metadata.",
    )
    parser.add_argument("video", type=Path, help="path to the DJI MP4 video")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gps-path.png"),
        metavar="PNG",
        help="GPS path image to create (default: gps-path.png)",
    )
    parser.add_argument(
        "--image-size",
        type=_image_size,
        default=1600,
        metavar="PIXELS",
        help="square image size, at least 400 (default: 1600)",
    )
    parser.add_argument(
        "--align",
        type=_non_negative_float,
        nargs=2,
        metavar=("START", "END"),
        help="rotate the path so movement from START to END seconds points up",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started_at = time.perf_counter()
    try:
        samples = extract_gps_samples(args.video)
        alignment_interval = tuple(args.align) if args.align is not None else None
        image_path = render_gps_path(
            samples,
            args.output,
            image_size=args.image_size,
            alignment_interval=alignment_interval,
        )
    except (GpsMetadataError, OSError, ValueError) as error:
        parser.error(str(error))

    points = to_local_metres(samples)
    east = points[-1][0] - points[0][0]
    north = points[-1][1] - points[0][1]
    displacement = (east * east + north * north) ** 0.5
    elapsed = time.perf_counter() - started_at
    print(f"Read {len(samples)} GPS samples in {elapsed:.2f} seconds.")
    print(
        f"Start-to-end: {displacement:.1f} m "
        f"(East {east:+.1f} m, North {north:+.1f} m)."
    )
    if args.align is not None:
        print(f"Alignment: movement at {args.align[0]:g}-{args.align[1]:g} s points up.")
    print(f"Image: {image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
