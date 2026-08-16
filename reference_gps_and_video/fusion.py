"""Metric GPS/video trajectory fusion for an independent reference path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from reference_gps.extractor import GpsSample, to_local_metres
from reference_gps_and_video.visual_odometry import (
    TRANSLATION,
    VisualOdometryResult,
)


class ReferenceFusionError(ValueError):
    """Raised when GPS and visual motion cannot form a reference trajectory."""


@dataclass(frozen=True, slots=True)
class ReferenceFusionConfig:
    gps_east_sigma_m: float = 4.0
    gps_north_sigma_m: float = 6.0
    origin_sigma_m: float = 0.50
    acceleration_noise_m2_s3: float = 2.25
    missing_direction_process_multiplier: float = 4.0
    visual_parallel_sigma_m_s: float = 1.0
    visual_direction_sigma_degrees: float = 15.0
    stationary_velocity_sigma_m_s: float = 0.50
    stationary_speed_threshold_m_s: float = 0.50
    minimum_alignment_displacement_m: float = 20.0
    minimum_alignment_visual_samples: int = 8
    alignment_window_seconds: float = 5.0
    minimum_alignment_coverage: float = 0.60
    minimum_alignment_coherence: float = 0.50
    minimum_alignment_windows: int = 3

    def __post_init__(self) -> None:
        positive = {
            "gps_east_sigma_m": self.gps_east_sigma_m,
            "gps_north_sigma_m": self.gps_north_sigma_m,
            "origin_sigma_m": self.origin_sigma_m,
            "acceleration_noise_m2_s3": self.acceleration_noise_m2_s3,
            "missing_direction_process_multiplier": (
                self.missing_direction_process_multiplier
            ),
            "visual_parallel_sigma_m_s": self.visual_parallel_sigma_m_s,
            "visual_direction_sigma_degrees": (
                self.visual_direction_sigma_degrees
            ),
            "stationary_velocity_sigma_m_s": self.stationary_velocity_sigma_m_s,
            "minimum_alignment_displacement_m": (
                self.minimum_alignment_displacement_m
            ),
            "alignment_window_seconds": self.alignment_window_seconds,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.stationary_speed_threshold_m_s < 0:
            raise ValueError("stationary_speed_threshold_m_s cannot be negative")
        if self.minimum_alignment_visual_samples < 1:
            raise ValueError("minimum_alignment_visual_samples must be positive")
        if not 0 < self.minimum_alignment_coverage <= 1:
            raise ValueError("minimum_alignment_coverage must be between 0 and 1")
        if not 0 < self.minimum_alignment_coherence <= 1:
            raise ValueError("minimum_alignment_coherence must be between 0 and 1")
        if self.minimum_alignment_windows < 1:
            raise ValueError("minimum_alignment_windows must be positive")


@dataclass(frozen=True, slots=True)
class FusedReferenceResult:
    video_path: Path
    start_seconds: float
    processed_duration_seconds: float
    timestamps: tuple[float, ...]
    positions_m: tuple[tuple[float, float], ...]
    velocities_m_s: tuple[tuple[float, float], ...]
    motion_states: tuple[str, ...]
    raw_gps_points_m: tuple[tuple[float, float, float], ...]
    gps_anchor_count: int
    visual_velocity_count: int
    stationary_velocity_count: int
    alignment_interval: tuple[float, float]
    visual_to_gps_rotation_degrees: float
    alignment_window_count: int
    visual_alignment_drift_degrees: float
    gps_rms_residual_m: float
    endpoint_gps_residual_m: float

    @property
    def start_to_end_distance_m(self) -> float:
        start = np.asarray(self.positions_m[0], dtype=np.float64)
        end = np.asarray(self.positions_m[-1], dtype=np.float64)
        return float(np.linalg.norm(end - start))

    def as_dict(self) -> dict[str, object]:
        return {
            "video_path": str(self.video_path),
            "start_seconds": self.start_seconds,
            "processed_duration_seconds": self.processed_duration_seconds,
            "coordinate_system": "local metres: East positive X, North positive Y",
            "timestamps": list(self.timestamps),
            "positions_m": [list(point) for point in self.positions_m],
            "velocities_m_s": [list(velocity) for velocity in self.velocities_m_s],
            "motion_states": list(self.motion_states),
            "raw_gps_points_m": [list(point) for point in self.raw_gps_points_m],
            "gps_anchor_count": self.gps_anchor_count,
            "visual_velocity_count": self.visual_velocity_count,
            "stationary_velocity_count": self.stationary_velocity_count,
            "alignment_interval": list(self.alignment_interval),
            "visual_to_gps_rotation_degrees": (
                self.visual_to_gps_rotation_degrees
            ),
            "alignment_window_count": self.alignment_window_count,
            "visual_alignment_drift_degrees": (
                self.visual_alignment_drift_degrees
            ),
            "gps_rms_residual_m": self.gps_rms_residual_m,
            "endpoint_gps_residual_m": self.endpoint_gps_residual_m,
            "start_to_end_distance_m": self.start_to_end_distance_m,
        }


def fuse_reference_trajectory(
    gps_samples: list[GpsSample],
    visual_odometry: VisualOdometryResult,
    *,
    alignment_interval: tuple[float, float] | None = None,
    config: ReferenceFusionConfig | None = None,
) -> FusedReferenceResult:
    """Fuse coarse GPS anchors with independent visual movement directions."""
    settings = config or ReferenceFusionConfig()
    if len(gps_samples) < 2:
        raise ReferenceFusionError("at least two GPS samples are required")
    if not visual_odometry.motions:
        raise ReferenceFusionError("visual odometry contains no motion estimates")

    gps_times = np.asarray(
        [sample.time_seconds for sample in gps_samples],
        dtype=np.float64,
    )
    if np.any(np.diff(gps_times) <= 0):
        raise ReferenceFusionError("GPS sample timestamps must increase")
    gps_points = np.asarray(to_local_metres(gps_samples), dtype=np.float64)
    timestamps = np.asarray(
        [position[0] for position in visual_odometry.positions],
        dtype=np.float64,
    )
    if len(timestamps) != len(visual_odometry.motions) + 1:
        raise ReferenceFusionError("visual positions and motions are inconsistent")
    if np.any(np.diff(timestamps) <= 0):
        raise ReferenceFusionError("visual timestamps must increase")
    if timestamps[0] > gps_times[-1] or timestamps[-1] < gps_times[0]:
        raise ReferenceFusionError("GPS and visual-odometry intervals do not overlap")

    origin = _interpolate_points(gps_times, gps_points, timestamps[0])
    gps_points = gps_points - origin
    interval = (
        _validate_alignment_interval(alignment_interval, visual_odometry)
        if alignment_interval is not None
        else _choose_alignment_interval(
            gps_times,
            gps_points,
            visual_odometry,
            settings,
        )
    )
    gps_speeds = np.asarray(
        [sample.horizontal_speed_m_s for sample in gps_samples],
        dtype=np.float64,
    )
    alignment_times, alignment_angles = _estimate_alignment_series(
        gps_times,
        gps_points,
        gps_speeds,
        visual_odometry,
        settings,
    )

    velocity_measurements: list[np.ndarray | None] = [None] * len(timestamps)
    velocity_noises: list[np.ndarray | None] = [None] * len(timestamps)
    process_multipliers = np.ones(len(timestamps), dtype=np.float64)
    visual_velocity_count = 0
    stationary_velocity_count = 0
    for index, motion in enumerate(visual_odometry.motions, start=1):
        speed = float(np.interp(timestamps[index], gps_times, gps_speeds))
        sample_delta = timestamps[index] - timestamps[index - 1]
        correlation_inflation = max(1.0, 1.0 / sample_delta)
        if speed <= settings.stationary_speed_threshold_m_s:
            velocity_measurements[index] = np.zeros(2, dtype=np.float64)
            velocity_noises[index] = (
                np.eye(2)
                * settings.stationary_velocity_sigma_m_s**2
                * correlation_inflation
            )
            stationary_velocity_count += 1
            continue
        direction = _motion_direction_2d(motion.translation_direction_world)
        if motion.state != TRANSLATION or direction is None:
            process_multipliers[index] = (
                settings.missing_direction_process_multiplier
            )
            continue
        local_angle = float(
            np.interp(timestamps[index], alignment_times, alignment_angles)
        )
        earth_direction = _rotate(direction, local_angle)
        velocity_measurements[index] = earth_direction * speed
        normal = np.array((-earth_direction[1], earth_direction[0]))
        perpendicular_sigma = max(
            0.75,
            speed * math.tan(math.radians(settings.visual_direction_sigma_degrees)),
        )
        quality = max(0.20, motion.homography_inlier_ratio) / (
            1.0 + motion.median_reprojection_error_pixels / 2.0
        )
        velocity_noises[index] = (
            (
                settings.visual_parallel_sigma_m_s**2
                * np.outer(earth_direction, earth_direction)
                + perpendicular_sigma**2 * np.outer(normal, normal)
            )
            * correlation_inflation
            / quality
        )
        visual_velocity_count += 1

    gps_measurements = _gps_measurements_by_state(
        gps_times,
        gps_points,
        timestamps,
    )
    states, covariances = _forward_filter(
        timestamps,
        gps_times,
        gps_points,
        gps_measurements,
        velocity_measurements,
        velocity_noises,
        process_multipliers,
        settings,
    )
    smoothed_states = _backward_smooth(
        timestamps,
        states,
        covariances,
        process_multipliers,
        settings,
    )

    positions = smoothed_states[:, :2].copy()
    # The local coordinate origin is arbitrary; translate without deforming the
    # smoothed trajectory or manufacturing endpoint agreement.
    positions -= positions[0]
    end_target = _interpolate_points(gps_times, gps_points, timestamps[-1])
    endpoint_residual = float(np.linalg.norm(positions[-1] - end_target))

    valid_gps = (gps_times >= timestamps[0]) & (gps_times <= timestamps[-1])
    raw_points = [
        (float(time), float(point[0]), float(point[1]))
        for time, point in zip(gps_times[valid_gps], gps_points[valid_gps], strict=True)
    ]
    if not raw_points or raw_points[0][0] > timestamps[0] + 1e-6:
        raw_points.insert(0, (float(timestamps[0]), 0.0, 0.0))
    if raw_points[-1][0] < timestamps[-1] - 1e-6:
        raw_points.append((float(timestamps[-1]), *map(float, end_target)))

    gps_residuals = []
    for time, east, north in raw_points:
        fused = np.array(
            (
                np.interp(time, timestamps, positions[:, 0]),
                np.interp(time, timestamps, positions[:, 1]),
            )
        )
        gps_residuals.append(np.linalg.norm(fused - np.array((east, north))))
    gps_rms_residual = float(np.sqrt(np.mean(np.square(gps_residuals))))

    return FusedReferenceResult(
        video_path=visual_odometry.video_path,
        start_seconds=float(timestamps[0]),
        processed_duration_seconds=float(timestamps[-1] - timestamps[0]),
        timestamps=tuple(float(value) for value in timestamps),
        positions_m=tuple(tuple(float(value) for value in point) for point in positions),
        velocities_m_s=tuple(
            tuple(float(value) for value in velocity)
            for velocity in smoothed_states[:, 2:]
        ),
        motion_states=tuple(motion.state for motion in visual_odometry.motions),
        raw_gps_points_m=tuple(raw_points),
        gps_anchor_count=int(np.count_nonzero(valid_gps)),
        visual_velocity_count=visual_velocity_count,
        stationary_velocity_count=stationary_velocity_count,
        alignment_interval=interval,
        visual_to_gps_rotation_degrees=math.degrees(float(alignment_angles[0])),
        alignment_window_count=len(alignment_times),
        visual_alignment_drift_degrees=math.degrees(
            float(alignment_angles[-1] - alignment_angles[0])
        ),
        gps_rms_residual_m=gps_rms_residual,
        endpoint_gps_residual_m=endpoint_residual,
    )


def _choose_alignment_interval(
    gps_times: np.ndarray,
    gps_points: np.ndarray,
    visual_odometry: VisualOdometryResult,
    config: ReferenceFusionConfig,
) -> tuple[float, float]:
    lower = max(float(gps_times[0]), visual_odometry.positions[0][0])
    upper = min(float(gps_times[-1]), visual_odometry.positions[-1][0])
    window_seconds = config.alignment_window_seconds
    start = math.ceil(lower)
    while start + window_seconds <= upper:
        end = start + window_seconds
        gps_displacement = np.linalg.norm(
            _interpolate_points(gps_times, gps_points, end)
            - _interpolate_points(gps_times, gps_points, start)
        )
        directions = _visual_directions_in_interval(visual_odometry, start, end)
        if (
            gps_displacement >= config.minimum_alignment_displacement_m
            and len(directions) >= config.minimum_alignment_visual_samples
            and np.linalg.norm(np.sum(directions, axis=0))
            >= len(directions) * config.minimum_alignment_coherence
        ):
            return float(start), float(end)
        start += 1
    raise ReferenceFusionError(
        "could not find a stable GPS/video alignment interval; pass --align START END"
    )


def _validate_alignment_interval(
    interval: tuple[float, float],
    visual_odometry: VisualOdometryResult,
) -> tuple[float, float]:
    start, end = interval
    if end <= start:
        raise ReferenceFusionError("alignment end must be greater than its start")
    first_time = visual_odometry.positions[0][0]
    last_time = visual_odometry.positions[-1][0]
    frame_tolerance = max(
        visual_odometry.sample_interval_seconds / 2,
        1 / visual_odometry.source_fps,
    )
    if start < first_time - frame_tolerance or end > last_time + frame_tolerance:
        raise ReferenceFusionError(
            f"alignment interval must be within {first_time:.1f}-{last_time:.1f} seconds"
        )
    return max(start, first_time), min(end, last_time)


def _estimate_alignment_series(
    gps_times: np.ndarray,
    gps_points: np.ndarray,
    gps_speeds: np.ndarray,
    visual_odometry: VisualOdometryResult,
    config: ReferenceFusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a slowly varying VO-to-East/North rotation from 5 s windows."""
    lower = max(float(gps_times[0]), visual_odometry.positions[0][0])
    upper = min(float(gps_times[-1]), visual_odometry.positions[-1][0])
    window = config.alignment_window_seconds
    measurement_times: list[float] = []
    measurement_angles: list[float] = []
    measurement_variances: list[float] = []
    window_start = math.ceil(lower)
    average_gps_sigma = math.hypot(
        config.gps_east_sigma_m,
        config.gps_north_sigma_m,
    ) / math.sqrt(2)

    while window_start + window <= upper:
        window_end = window_start + window
        visual_vector = np.zeros(2, dtype=np.float64)
        integrated_distance = 0.0
        covered_seconds = 0.0
        visual_samples = 0
        for motion in visual_odometry.motions:
            overlap = max(
                0.0,
                min(window_end, motion.end_seconds)
                - max(window_start, motion.start_seconds),
            )
            if overlap <= 0 or motion.state != TRANSLATION:
                continue
            direction = _motion_direction_2d(motion.translation_direction_world)
            if direction is None:
                continue
            midpoint = (motion.start_seconds + motion.end_seconds) / 2
            speed = float(np.interp(midpoint, gps_times, gps_speeds))
            if speed <= config.stationary_speed_threshold_m_s:
                continue
            distance = speed * overlap
            visual_vector += direction * distance
            integrated_distance += distance
            covered_seconds += overlap
            visual_samples += 1

        gps_vector = (
            _interpolate_points(gps_times, gps_points, window_end)
            - _interpolate_points(gps_times, gps_points, window_start)
        )
        gps_distance = float(np.linalg.norm(gps_vector))
        coherence = (
            float(np.linalg.norm(visual_vector)) / integrated_distance
            if integrated_distance > 1e-9
            else 0.0
        )
        coverage = covered_seconds / window
        if (
            visual_samples >= config.minimum_alignment_visual_samples
            and coverage >= config.minimum_alignment_coverage
            and coherence >= config.minimum_alignment_coherence
            and integrated_distance >= 15.0
            and gps_distance >= 10.0
        ):
            angle = math.atan2(gps_vector[1], gps_vector[0]) - math.atan2(
                visual_vector[1], visual_vector[0]
            )
            angular_gps_sigma = math.atan2(
                math.sqrt(2) * average_gps_sigma,
                gps_distance,
            )
            base_sigma = math.radians(10.0)
            quality = max(0.20, coverage * coherence)
            variance = (base_sigma**2 + angular_gps_sigma**2) / quality
            measurement_times.append(window_start + window / 2)
            measurement_angles.append(_wrap_radians(angle))
            measurement_variances.append(variance)
        window_start += 1

    if len(measurement_times) < config.minimum_alignment_windows:
        raise ReferenceFusionError(
            "too few stable GPS/video windows to calibrate direction"
        )
    return _smooth_alignment_angles(
        np.asarray(measurement_times, dtype=np.float64),
        np.asarray(measurement_angles, dtype=np.float64),
        np.asarray(measurement_variances, dtype=np.float64),
    )


def _smooth_alignment_angles(
    times: np.ndarray,
    measurements: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(times)
    filtered = np.zeros(count, dtype=np.float64)
    filtered_variance = np.zeros(count, dtype=np.float64)
    predicted = np.zeros(count, dtype=np.float64)
    predicted_variance = np.zeros(count, dtype=np.float64)
    state = float(measurements[0])
    covariance = math.radians(30.0) ** 2
    random_walk_variance_per_second = math.radians(2.0) ** 2

    for index in range(count):
        if index > 0:
            covariance += random_walk_variance_per_second * (
                times[index] - times[index - 1]
            )
        predicted[index] = state
        predicted_variance[index] = covariance
        measurement = state + _wrap_radians(measurements[index] - state)
        noise = float(variances[index])
        innovation = measurement - state
        innovation_variance = covariance + noise
        normalized_squared = innovation**2 / innovation_variance
        if normalized_squared > 9.0:
            noise *= normalized_squared / 9.0
            innovation_variance = covariance + noise
        gain = covariance / innovation_variance
        state += gain * innovation
        covariance = (1 - gain) * covariance
        filtered[index] = state
        filtered_variance[index] = covariance

    smoothed = filtered.copy()
    smoothed_variance = filtered_variance.copy()
    for index in range(count - 2, -1, -1):
        next_prediction_variance = predicted_variance[index + 1]
        gain = filtered_variance[index] / next_prediction_variance
        smoothed[index] = filtered[index] + gain * (
            smoothed[index + 1] - predicted[index + 1]
        )
        smoothed_variance[index] = filtered_variance[index] + gain**2 * (
            smoothed_variance[index + 1] - next_prediction_variance
        )
    return times, smoothed


def _visual_to_gps_alignment(
    gps_times: np.ndarray,
    gps_points: np.ndarray,
    visual_odometry: VisualOdometryResult,
    interval: tuple[float, float],
    config: ReferenceFusionConfig,
) -> float:
    start, end = interval
    if end <= start:
        raise ReferenceFusionError("alignment end must be greater than its start")
    if (
        start < max(gps_times[0], visual_odometry.positions[0][0]) - 0.1
        or end > min(gps_times[-1], visual_odometry.positions[-1][0]) + 0.1
    ):
        raise ReferenceFusionError("alignment interval is outside the fused range")
    gps_vector = (
        _interpolate_points(gps_times, gps_points, end)
        - _interpolate_points(gps_times, gps_points, start)
    )
    if np.linalg.norm(gps_vector) < config.minimum_alignment_displacement_m:
        raise ReferenceFusionError(
            "alignment interval has too little GPS displacement"
        )
    directions = _visual_directions_in_interval(visual_odometry, start, end)
    if len(directions) < config.minimum_alignment_visual_samples:
        raise ReferenceFusionError(
            "alignment interval has too few visual translation measurements"
        )
    visual_vector = np.sum(directions, axis=0)
    if np.linalg.norm(visual_vector) <= 1e-9:
        raise ReferenceFusionError("alignment visual directions cancel each other")
    return math.atan2(gps_vector[1], gps_vector[0]) - math.atan2(
        visual_vector[1],
        visual_vector[0],
    )


def _visual_directions_in_interval(
    visual_odometry: VisualOdometryResult,
    start: float,
    end: float,
) -> np.ndarray:
    directions = []
    for motion in visual_odometry.motions:
        if motion.end_seconds < start or motion.end_seconds > end:
            continue
        if motion.state != TRANSLATION:
            continue
        direction = _motion_direction_2d(motion.translation_direction_world)
        if direction is not None:
            directions.append(direction)
    return np.asarray(directions, dtype=np.float64).reshape(-1, 2)


def _motion_direction_2d(
    direction_world: tuple[float, float, float] | None,
) -> np.ndarray | None:
    if direction_world is None:
        return None
    direction = np.asarray((direction_world[0], direction_world[2]), dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-9 else None


def _gps_measurements_by_state(
    gps_times: np.ndarray,
    gps_points: np.ndarray,
    timestamps: np.ndarray,
) -> dict[int, list[np.ndarray]]:
    output: dict[int, list[np.ndarray]] = {}
    half_step = float(np.median(np.diff(timestamps))) / 2
    for time, point in zip(gps_times, gps_points, strict=True):
        if time < timestamps[0] - half_step or time > timestamps[-1] + half_step:
            continue
        insertion = int(np.searchsorted(timestamps, time))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(timestamps)
        ]
        index = min(candidates, key=lambda value: abs(timestamps[value] - time))
        output.setdefault(index, []).append(point)
    return output


def _forward_filter(
    timestamps: np.ndarray,
    gps_times: np.ndarray,
    gps_points: np.ndarray,
    gps_measurements: dict[int, list[np.ndarray]],
    velocity_measurements: list[np.ndarray | None],
    velocity_noises: list[np.ndarray | None],
    process_multipliers: np.ndarray,
    config: ReferenceFusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(timestamps)
    filtered_states = np.zeros((count, 4), dtype=np.float64)
    filtered_covariances = np.zeros((count, 4, 4), dtype=np.float64)
    start_position = _interpolate_points(gps_times, gps_points, timestamps[0])
    if velocity_measurements[0] is not None:
        start_velocity = velocity_measurements[0]
    else:
        later = next(
            (value for value in velocity_measurements[1:] if value is not None),
            np.zeros(2, dtype=np.float64),
        )
        start_velocity = later
    state = np.array((*start_position, *start_velocity), dtype=np.float64)
    covariance = np.diag(
        (
            config.origin_sigma_m**2,
            config.origin_sigma_m**2,
            5.0**2,
            5.0**2,
        )
    )
    position_matrix = np.array(((1, 0, 0, 0), (0, 1, 0, 0)), dtype=np.float64)
    velocity_matrix = np.array(((0, 0, 1, 0), (0, 0, 0, 1)), dtype=np.float64)

    for index in range(count):
        if index > 0:
            transition, process_noise = _transition_and_noise(
                timestamps[index] - timestamps[index - 1],
                config.acceleration_noise_m2_s3,
                process_multipliers[index],
            )
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
        velocity = velocity_measurements[index]
        velocity_noise = velocity_noises[index]
        if velocity is not None and velocity_noise is not None:
            state, covariance = _measurement_update(
                state,
                covariance,
                velocity,
                velocity_matrix,
                velocity_noise,
            )
        for gps_position in gps_measurements.get(index, ()):
            state, covariance = _measurement_update(
                state,
                covariance,
                gps_position,
                position_matrix,
                np.diag(
                    (
                        config.gps_east_sigma_m**2,
                        config.gps_north_sigma_m**2,
                    )
                ),
            )
        filtered_states[index] = state
        filtered_covariances[index] = covariance
    return filtered_states, filtered_covariances


def _backward_smooth(
    timestamps: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    process_multipliers: np.ndarray,
    config: ReferenceFusionConfig,
) -> np.ndarray:
    smoothed = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    for index in range(len(timestamps) - 2, -1, -1):
        transition, process_noise = _transition_and_noise(
            timestamps[index + 1] - timestamps[index],
            config.acceleration_noise_m2_s3,
            process_multipliers[index + 1],
        )
        predicted_state = transition @ filtered_states[index]
        predicted_covariance = (
            transition @ filtered_covariances[index] @ transition.T
            + process_noise
        )
        gain = np.linalg.solve(
            predicted_covariance.T,
            (filtered_covariances[index] @ transition.T).T,
        ).T
        smoothed[index] = filtered_states[index] + gain @ (
            smoothed[index + 1] - predicted_state
        )
        smoothed_covariances[index] = filtered_covariances[index] + gain @ (
            smoothed_covariances[index + 1] - predicted_covariance
        ) @ gain.T
    return smoothed


def _transition_and_noise(
    delta_seconds: float,
    acceleration_noise_m2_s3: float,
    multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    transition = np.array(
        (
            (1, 0, delta_seconds, 0),
            (0, 1, 0, delta_seconds),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ),
        dtype=np.float64,
    )
    delta2 = delta_seconds**2
    delta3 = delta_seconds**3
    process_noise = acceleration_noise_m2_s3 * multiplier * np.array(
        (
            (delta3 / 3, 0, delta2 / 2, 0),
            (0, delta3 / 3, 0, delta2 / 2),
            (delta2 / 2, 0, delta_seconds, 0),
            (0, delta2 / 2, 0, delta_seconds),
        ),
        dtype=np.float64,
    )
    return transition, process_noise


def _measurement_update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    innovation = measurement - measurement_matrix @ state
    innovation_covariance = (
        measurement_matrix @ covariance @ measurement_matrix.T
        + measurement_noise
    )
    normalized_squared = float(
        innovation
        @ np.linalg.solve(innovation_covariance, innovation)
    )
    if normalized_squared > 9.21:
        measurement_noise = measurement_noise * (normalized_squared / 9.21)
        innovation_covariance = (
            measurement_matrix @ covariance @ measurement_matrix.T
            + measurement_noise
        )
    gain = np.linalg.solve(
        innovation_covariance.T,
        (covariance @ measurement_matrix.T).T,
    ).T
    updated_state = state + gain @ innovation
    identity = np.eye(len(state))
    residual = identity - gain @ measurement_matrix
    updated_covariance = (
        residual @ covariance @ residual.T + gain @ measurement_noise @ gain.T
    )
    return updated_state, updated_covariance


def _interpolate_points(
    times: np.ndarray,
    points: np.ndarray,
    timestamp: float,
) -> np.ndarray:
    timestamp = float(np.clip(timestamp, times[0], times[-1]))
    return np.array(
        (
            np.interp(timestamp, times, points[:, 0]),
            np.interp(timestamp, times, points[:, 1]),
        ),
        dtype=np.float64,
    )


def _rotate(vector: np.ndarray, radians: float) -> np.ndarray:
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.array(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ),
        dtype=np.float64,
    )


def _wrap_radians(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi
