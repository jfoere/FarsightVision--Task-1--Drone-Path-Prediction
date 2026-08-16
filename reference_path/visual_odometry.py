"""Independent SIFT/homography visual odometry for mostly planar drone footage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np


TRANSLATION = "TRANSLATION"
ROTATION_ONLY = "ROTATION_ONLY"
UNRELIABLE = "UNRELIABLE"


class VisualOdometryError(RuntimeError):
    """Raised when independent visual odometry cannot process a video."""


@dataclass(frozen=True, slots=True)
class VisualOdometryConfig:
    analysis_width: int = 960
    sample_interval_seconds: float = 0.2
    horizontal_fov_degrees: float = 84.0
    maximum_features: int = 1_800
    sift_contrast_threshold: float = 0.02
    descriptor_ratio_threshold: float = 0.75
    minimum_matches: int = 80
    homography_reprojection_threshold_pixels: float = 2.0
    minimum_homography_inlier_ratio: float = 0.70
    minimum_ground_normal_alignment: float = 0.90
    minimum_translation_ratio: float = 0.002
    maximum_translation_rotation_degrees: float = 2.0
    normal_tracking_alpha: float = 0.20
    initial_ground_normal_camera: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        if self.analysis_width < 320:
            raise ValueError("analysis_width must be at least 320")
        if self.sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be greater than zero")
        if not 0 < self.horizontal_fov_degrees < 180:
            raise ValueError("horizontal_fov_degrees must be between 0 and 180")
        if self.maximum_features < 100:
            raise ValueError("maximum_features must be at least 100")
        if self.sift_contrast_threshold <= 0:
            raise ValueError("sift_contrast_threshold must be greater than zero")
        if not 0 < self.descriptor_ratio_threshold < 1:
            raise ValueError("descriptor_ratio_threshold must be between 0 and 1")
        if self.minimum_matches < 4:
            raise ValueError("minimum_matches must be at least 4")
        if self.homography_reprojection_threshold_pixels <= 0:
            raise ValueError(
                "homography_reprojection_threshold_pixels must be greater than zero"
            )
        if not 0 <= self.minimum_homography_inlier_ratio <= 1:
            raise ValueError(
                "minimum_homography_inlier_ratio must be between 0 and 1"
            )
        if not 0 <= self.minimum_ground_normal_alignment <= 1:
            raise ValueError(
                "minimum_ground_normal_alignment must be between 0 and 1"
            )
        if self.minimum_translation_ratio < 0:
            raise ValueError("minimum_translation_ratio cannot be negative")
        if not 0 < self.maximum_translation_rotation_degrees <= 180:
            raise ValueError(
                "maximum_translation_rotation_degrees must be between 0 and 180"
            )
        if not 0 <= self.normal_tracking_alpha <= 1:
            raise ValueError("normal_tracking_alpha must be between 0 and 1")
        if np.linalg.norm(self.initial_ground_normal_camera) <= 1e-9:
            raise ValueError("initial_ground_normal_camera cannot be zero")


@dataclass(frozen=True, slots=True)
class VisualMotionEstimate:
    start_seconds: float
    end_seconds: float
    state: str
    matched_features: int
    inlier_count: int
    homography_inlier_ratio: float
    median_reprojection_error_pixels: float
    ground_normal_alignment: float
    translation_ratio: float
    rotation_degrees: float
    ground_yaw_degrees: float
    translation_direction_camera: tuple[float, float, float] | None
    translation_direction_world: tuple[float, float, float] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "state": self.state,
            "matched_features": self.matched_features,
            "inlier_count": self.inlier_count,
            "homography_inlier_ratio": self.homography_inlier_ratio,
            "median_reprojection_error_pixels": (
                self.median_reprojection_error_pixels
            ),
            "ground_normal_alignment": self.ground_normal_alignment,
            "translation_ratio": self.translation_ratio,
            "rotation_degrees": self.rotation_degrees,
            "ground_yaw_degrees": self.ground_yaw_degrees,
            "translation_direction_camera": self.translation_direction_camera,
            "translation_direction_world": self.translation_direction_world,
        }


@dataclass(frozen=True, slots=True)
class VisualOdometryResult:
    video_path: Path
    start_seconds: float
    processed_duration_seconds: float
    source_fps: float
    sample_interval_seconds: float
    analysis_size: tuple[int, int]
    positions: tuple[tuple[float, float, float, float], ...]
    motions: tuple[VisualMotionEstimate, ...]

    @property
    def translation_count(self) -> int:
        return sum(motion.state == TRANSLATION for motion in self.motions)

    @property
    def rotation_only_count(self) -> int:
        return sum(motion.state == ROTATION_ONLY for motion in self.motions)

    @property
    def unreliable_count(self) -> int:
        return sum(motion.state == UNRELIABLE for motion in self.motions)

    @property
    def rotation_only_ground_yaw_degrees(self) -> float:
        return sum(
            motion.ground_yaw_degrees
            for motion in self.motions
            if motion.state == ROTATION_ONLY
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "video_path": str(self.video_path),
            "start_seconds": self.start_seconds,
            "processed_duration_seconds": self.processed_duration_seconds,
            "source_fps": self.source_fps,
            "sample_interval_seconds": self.sample_interval_seconds,
            "analysis_size": list(self.analysis_size),
            "coordinate_note": (
                "unscaled initial-camera coordinates; each reliable translation "
                "has unit length"
            ),
            "positions": [list(position) for position in self.positions],
            "motions": [motion.as_dict() for motion in self.motions],
        }


@dataclass(frozen=True, slots=True)
class VisualOdometryProgress:
    processed_pairs: int
    total_pairs: int

    @property
    def percentage(self) -> int:
        if self.total_pairs <= 0:
            return 100
        return min(100, round(self.processed_pairs * 100 / self.total_pairs))


@dataclass(frozen=True, slots=True)
class _PairGeometry:
    state: str
    coordinate_rotation: np.ndarray
    translation_camera: np.ndarray | None
    ground_normal_camera: np.ndarray
    ground_normal_alignment: float
    translation_ratio: float
    rotation_degrees: float
    ground_yaw_degrees: float


def estimate_visual_odometry(
    video_path: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    config: VisualOdometryConfig | None = None,
    progress_callback: Callable[[VisualOdometryProgress], None] | None = None,
) -> VisualOdometryResult:
    """Estimate independent, unscaled camera motion over a short interval."""
    settings = config or VisualOdometryConfig()
    if start_seconds < 0:
        raise ValueError("start_seconds cannot be negative")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise VisualOdometryError(f"video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VisualOdometryError(f"OpenCV could not open the video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise VisualOdometryError("video does not report a valid frame rate")
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        start_frame = round(start_seconds * fps)
        if frame_count and start_frame >= frame_count:
            raise VisualOdometryError("start time is outside the video")
        if duration_seconds is None:
            if frame_count <= 0:
                raise VisualOdometryError(
                    "video does not report a frame count; pass --duration explicitly"
                )
            end_frame = frame_count - 1
        else:
            requested_end_frame = round((start_seconds + duration_seconds) * fps)
            end_frame = (
                min(requested_end_frame, frame_count - 1)
                if frame_count
                else requested_end_frame
            )
        stride = max(1, round(settings.sample_interval_seconds * fps))
        keyframe_count = max(1, (end_frame - start_frame) // stride + 1)
        total_pairs = max(0, keyframe_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        sift = cv2.SIFT_create(
            nfeatures=settings.maximum_features,
            contrastThreshold=settings.sift_contrast_threshold,
        )
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        previous_keypoints = None
        previous_descriptors = None
        previous_time = start_frame / fps
        orientation_camera_to_world = np.eye(3, dtype=np.float64)
        position_world = np.zeros(3, dtype=np.float64)
        expected_ground_normal = _unit_vector(
            np.asarray(settings.initial_ground_normal_camera, dtype=np.float64)
        )
        motions: list[VisualMotionEstimate] = []
        positions: list[tuple[float, float, float, float]] = [
            (previous_time, 0.0, 0.0, 0.0)
        ]
        analysis_size: tuple[int, int] | None = None
        frame_number = start_frame
        next_sample_frame = start_frame

        while frame_number <= end_frame:
            decoded, frame = capture.read()
            if not decoded or frame is None:
                break
            if frame_number < next_sample_frame:
                frame_number += 1
                continue

            gray = _analysis_gray(frame, settings.analysis_width)
            height, width = gray.shape
            analysis_size = (width, height)
            camera_matrix = _camera_matrix(
                width,
                height,
                settings.horizontal_fov_degrees,
            )
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            current_time = frame_number / fps

            if previous_keypoints is not None:
                estimate, coordinate_rotation, next_normal = _estimate_pair(
                    previous_keypoints,
                    previous_descriptors,
                    keypoints,
                    descriptors,
                    matcher,
                    camera_matrix,
                    expected_ground_normal,
                    orientation_camera_to_world,
                    previous_time,
                    current_time,
                    settings,
                )
                if estimate.translation_direction_world is not None:
                    position_world += np.asarray(
                        estimate.translation_direction_world,
                        dtype=np.float64,
                    )
                if coordinate_rotation is not None:
                    orientation_camera_to_world = (
                        orientation_camera_to_world @ coordinate_rotation.T
                    )
                    orientation_camera_to_world = _nearest_rotation(
                        orientation_camera_to_world
                    )
                    expected_ground_normal = next_normal
                motions.append(estimate)
                positions.append(
                    (
                        current_time,
                        float(position_world[0]),
                        float(position_world[1]),
                        float(position_world[2]),
                    )
                )
                if progress_callback is not None:
                    progress_callback(
                        VisualOdometryProgress(
                            processed_pairs=len(motions),
                            total_pairs=total_pairs,
                        )
                    )

            previous_keypoints = keypoints
            previous_descriptors = descriptors
            previous_time = current_time
            next_sample_frame += stride
            frame_number += 1
    finally:
        capture.release()

    if analysis_size is None or previous_keypoints is None:
        raise VisualOdometryError("OpenCV could not decode the requested interval")
    if not motions:
        raise VisualOdometryError("requested interval is too short for visual odometry")
    processed_duration = positions[-1][0] - positions[0][0]
    return VisualOdometryResult(
        video_path=path,
        start_seconds=positions[0][0],
        processed_duration_seconds=processed_duration,
        source_fps=fps,
        sample_interval_seconds=stride / fps,
        analysis_size=analysis_size,
        positions=tuple(positions),
        motions=tuple(motions),
    )


def _estimate_pair(
    previous_keypoints,
    previous_descriptors: np.ndarray | None,
    current_keypoints,
    current_descriptors: np.ndarray | None,
    matcher,
    camera_matrix: np.ndarray,
    expected_ground_normal: np.ndarray,
    orientation_camera_to_world: np.ndarray,
    start_seconds: float,
    end_seconds: float,
    config: VisualOdometryConfig,
) -> tuple[VisualMotionEstimate, np.ndarray | None, np.ndarray]:
    if previous_descriptors is None or current_descriptors is None:
        return (
            _unreliable_motion(start_seconds, end_seconds),
            None,
            expected_ground_normal,
        )
    raw_matches = matcher.knnMatch(previous_descriptors, current_descriptors, k=2)
    matches = [
        best
        for pair in raw_matches
        if len(pair) == 2
        for best, second in [pair]
        if best.distance < config.descriptor_ratio_threshold * second.distance
    ]
    if len(matches) < config.minimum_matches:
        return (
            _unreliable_motion(
                start_seconds,
                end_seconds,
                matched_features=len(matches),
            ),
            None,
            expected_ground_normal,
        )

    previous_points = np.float64(
        [previous_keypoints[match.queryIdx].pt for match in matches]
    )
    current_points = np.float64(
        [current_keypoints[match.trainIdx].pt for match in matches]
    )
    homography, mask = cv2.findHomography(
        previous_points,
        current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.homography_reprojection_threshold_pixels,
        maxIters=5_000,
        confidence=0.999,
    )
    if homography is None or mask is None or not np.isfinite(homography).all():
        return (
            _unreliable_motion(
                start_seconds,
                end_seconds,
                matched_features=len(matches),
            ),
            None,
            expected_ground_normal,
        )

    inliers = mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / len(matches)
    projected = cv2.perspectiveTransform(
        previous_points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected[inliers] - current_points[inliers], axis=1)
    median_error = float(np.median(errors)) if errors.size else math.inf
    if (
        inlier_count < config.minimum_matches
        or inlier_ratio < config.minimum_homography_inlier_ratio
    ):
        return (
            _unreliable_motion(
                start_seconds,
                end_seconds,
                matched_features=len(matches),
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                median_error=median_error,
            ),
            None,
            expected_ground_normal,
        )

    geometry = _decompose_planar_motion(
        homography,
        camera_matrix,
        expected_ground_normal,
        config,
    )
    translation_camera = None
    translation_world = None
    if geometry.state == TRANSLATION and geometry.translation_camera is not None:
        translation_camera_array = _unit_vector(geometry.translation_camera)
        translation_world_array = _unit_vector(
            orientation_camera_to_world @ translation_camera_array
        )
        translation_camera = tuple(float(value) for value in translation_camera_array)
        translation_world = tuple(float(value) for value in translation_world_array)

    estimate = VisualMotionEstimate(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        state=geometry.state,
        matched_features=len(matches),
        inlier_count=inlier_count,
        homography_inlier_ratio=inlier_ratio,
        median_reprojection_error_pixels=median_error,
        ground_normal_alignment=geometry.ground_normal_alignment,
        translation_ratio=geometry.translation_ratio,
        rotation_degrees=geometry.rotation_degrees,
        ground_yaw_degrees=geometry.ground_yaw_degrees,
        translation_direction_camera=translation_camera,
        translation_direction_world=translation_world,
    )
    return estimate, geometry.coordinate_rotation, geometry.ground_normal_camera


def _decompose_planar_motion(
    homography: np.ndarray,
    camera_matrix: np.ndarray,
    expected_ground_normal: np.ndarray,
    config: VisualOdometryConfig,
) -> _PairGeometry:
    pure_rotation = _rotation_from_homography(homography, camera_matrix)
    if pure_rotation is None:
        return _PairGeometry(
            state=UNRELIABLE,
            coordinate_rotation=np.eye(3),
            translation_camera=None,
            ground_normal_camera=expected_ground_normal,
            ground_normal_alignment=0.0,
            translation_ratio=0.0,
            rotation_degrees=0.0,
            ground_yaw_degrees=0.0,
        )

    try:
        solution_count, rotations, translations, normals = cv2.decomposeHomographyMat(
            homography,
            camera_matrix,
        )
    except cv2.error:
        solution_count = 0
        rotations, translations, normals = (), (), ()

    best = None
    best_alignment = -1.0
    for index in range(solution_count):
        raw_normal = normals[index].reshape(3)
        if np.linalg.norm(raw_normal) <= 1e-12:
            continue
        normal = _unit_vector(raw_normal)
        alignment = float(np.dot(normal, expected_ground_normal))
        if alignment > best_alignment:
            best_alignment = alignment
            best = (
                _nearest_rotation(rotations[index]),
                translations[index].reshape(3),
                normal,
            )

    if best is not None:
        coordinate_rotation, translation, measured_normal = best
        translation_ratio = float(np.linalg.norm(translation))
    else:
        coordinate_rotation = pure_rotation
        translation = None
        measured_normal = expected_ground_normal
        translation_ratio = 0.0

    candidate_rotation_vector, _ = cv2.Rodrigues(coordinate_rotation)
    candidate_rotation_degrees = math.degrees(
        float(np.linalg.norm(candidate_rotation_vector))
    )
    is_translation = (
        translation is not None
        and best_alignment >= config.minimum_ground_normal_alignment
        and translation_ratio >= config.minimum_translation_ratio
        and candidate_rotation_degrees
        <= config.maximum_translation_rotation_degrees
    )
    if is_translation:
        blended_normal = _unit_vector(
            (1 - config.normal_tracking_alpha) * expected_ground_normal
            + config.normal_tracking_alpha * measured_normal
        )
        next_normal = _unit_vector(coordinate_rotation @ blended_normal)
        camera_displacement = -coordinate_rotation.T @ translation
        # Homography translation is only observable inside the dominant plane.
        # Remove its normal component before GPS later supplies metric scale.
        camera_displacement -= (
            np.dot(camera_displacement, blended_normal) * blended_normal
        )
        if np.linalg.norm(camera_displacement) <= 1e-12:
            camera_displacement = None
            state = ROTATION_ONLY
            chosen_rotation = pure_rotation
            next_normal = _unit_vector(pure_rotation @ expected_ground_normal)
        else:
            state = TRANSLATION
            chosen_rotation = coordinate_rotation
    else:
        next_normal = _unit_vector(pure_rotation @ expected_ground_normal)
        camera_displacement = None
        state = ROTATION_ONLY
        chosen_rotation = pure_rotation

    rotation_vector, _ = cv2.Rodrigues(chosen_rotation)
    rotation_degrees = math.degrees(float(np.linalg.norm(rotation_vector)))
    physical_rotation_vector, _ = cv2.Rodrigues(chosen_rotation.T)
    ground_yaw_degrees = math.degrees(
        float(physical_rotation_vector.reshape(3) @ expected_ground_normal)
    )
    return _PairGeometry(
        state=state,
        coordinate_rotation=chosen_rotation,
        translation_camera=camera_displacement,
        ground_normal_camera=next_normal,
        ground_normal_alignment=max(0.0, best_alignment),
        translation_ratio=translation_ratio,
        rotation_degrees=rotation_degrees,
        ground_yaw_degrees=ground_yaw_degrees,
    )


def _unreliable_motion(
    start_seconds: float,
    end_seconds: float,
    *,
    matched_features: int = 0,
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
    median_error: float = 0.0,
) -> VisualMotionEstimate:
    return VisualMotionEstimate(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        state=UNRELIABLE,
        matched_features=matched_features,
        inlier_count=inlier_count,
        homography_inlier_ratio=inlier_ratio,
        median_reprojection_error_pixels=median_error,
        ground_normal_alignment=0.0,
        translation_ratio=0.0,
        rotation_degrees=0.0,
        ground_yaw_degrees=0.0,
        translation_direction_camera=None,
        translation_direction_world=None,
    )


def _analysis_gray(frame: np.ndarray, analysis_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    output_width = min(width, analysis_width)
    output_height = max(1, round(height * output_width / width))
    resized = cv2.resize(
        frame,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _camera_matrix(
    width: int,
    height: int,
    horizontal_fov_degrees: float,
) -> np.ndarray:
    focal_length = width / (
        2 * math.tan(math.radians(horizontal_fov_degrees / 2))
    )
    return np.array(
        (
            (focal_length, 0.0, (width - 1) / 2),
            (0.0, focal_length, (height - 1) / 2),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _rotation_from_homography(
    homography: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray | None:
    normalized = np.linalg.inv(camera_matrix) @ homography @ camera_matrix
    determinant = float(np.linalg.det(normalized))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        return None
    if determinant < 0:
        normalized = -normalized
        determinant = -determinant
    normalized /= math.cbrt(determinant)
    return _nearest_rotation(normalized)


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right_transposed = np.linalg.svd(matrix)
    rotation = left @ right_transposed
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transposed
    return rotation


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm
