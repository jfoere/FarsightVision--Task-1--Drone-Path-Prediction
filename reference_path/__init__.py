"""Independent video constraints and GPS/video reference trajectory fusion."""

from reference_path.fusion import (
    FusedReferenceResult,
    ReferenceFusionConfig,
    ReferenceFusionError,
    fuse_reference_trajectory,
)

from reference_path.visual_odometry import (
    VisualMotionEstimate,
    VisualOdometryConfig,
    VisualOdometryError,
    VisualOdometryProgress,
    VisualOdometryResult,
    estimate_visual_odometry,
)

__all__ = [
    "FusedReferenceResult",
    "ReferenceFusionConfig",
    "ReferenceFusionError",
    "VisualMotionEstimate",
    "VisualOdometryConfig",
    "VisualOdometryError",
    "VisualOdometryProgress",
    "VisualOdometryResult",
    "estimate_visual_odometry",
    "fuse_reference_trajectory",
]
