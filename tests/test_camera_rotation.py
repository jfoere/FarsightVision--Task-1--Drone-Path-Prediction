"""Tests for pure-camera-rotation estimation and accumulation."""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from drone_path.algorithm import (
    CameraRotationConfig,
    CameraRotationHandler,
    OpticalFlowResult,
)


FRAME_SIZE = (960, 540)
FOV_DEGREES = 90.0


def _flow(previous: np.ndarray, current: np.ndarray) -> OpticalFlowResult:
    count = previous.shape[0]
    zeros = np.zeros((count,), dtype=np.float32)
    return OpticalFlowResult(
        previous_points=previous.astype(np.float32),
        current_points=current.astype(np.float32),
        tracking_errors=zeros,
        forward_backward_errors=zeros.copy(),
        detected_count=count,
    )


def _orientation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)
    yaw_matrix = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ]
    )
    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), math.sin(pitch)],
            [0.0, -math.sin(pitch), math.cos(pitch)],
        ]
    )
    roll_matrix = np.array(
        [
            [math.cos(roll), -math.sin(roll), 0.0],
            [math.sin(roll), math.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return yaw_matrix @ pitch_matrix @ roll_matrix


def _rotation_flow(yaw: float, pitch: float, roll: float) -> OpticalFlowResult:
    width, height = FRAME_SIZE
    focal = width / (2 * math.tan(math.radians(FOV_DEGREES / 2)))
    camera_matrix = np.array(
        [[focal, 0.0, 479.5], [0.0, focal, 269.5], [0.0, 0.0, 1.0]]
    )
    random = np.random.default_rng(12)
    previous = random.uniform((40.0, 40.0), (920.0, 500.0), size=(200, 2))
    physical_rotation = _orientation(yaw, pitch, roll)
    homography = camera_matrix @ physical_rotation.T @ np.linalg.inv(camera_matrix)
    current = cv2.perspectiveTransform(
        previous.astype(np.float32).reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    return _flow(previous, current)


class CameraRotationHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = CameraRotationHandler(
            CameraRotationConfig(maximum_delta_rotation_degrees=30.0)
        )

    def test_recovers_yaw_pitch_and_roll(self) -> None:
        measurement = self.handler.update(
            _rotation_flow(8.0, 5.0, -4.0),
            FRAME_SIZE,
            horizontal_fov_degrees=FOV_DEGREES,
        )

        self.assertTrue(measurement.valid)
        self.assertAlmostEqual(measurement.yaw_degrees, 8.0, delta=0.1)
        self.assertAlmostEqual(measurement.pitch_degrees, 5.0, delta=0.1)
        self.assertAlmostEqual(measurement.roll_degrees, -4.0, delta=0.1)
        self.assertGreater(measurement.inlier_ratio, 0.99)
        self.assertGreater(
            math.sqrt(sum(value * value for value in measurement.rotation_vector_degrees)),
            9.0,
        )

    def test_accumulates_consecutive_rotation_updates(self) -> None:
        flow = _rotation_flow(7.0, 0.0, 0.0)

        self.handler.update(flow, FRAME_SIZE, horizontal_fov_degrees=FOV_DEGREES)
        measurement = self.handler.update(
            flow,
            FRAME_SIZE,
            horizontal_fov_degrees=FOV_DEGREES,
        )

        self.assertAlmostEqual(measurement.yaw_degrees, 14.0, delta=0.1)
        self.assertEqual(measurement.sample_count, 2)

    def test_ignores_too_few_tracks(self) -> None:
        points = np.zeros((5, 2), dtype=np.float32)

        measurement = self.handler.update(
            _flow(points, points),
            FRAME_SIZE,
            horizontal_fov_degrees=FOV_DEGREES,
        )

        self.assertFalse(measurement.valid)
        self.assertEqual(measurement.sample_count, 0)

    def test_reset_discards_accumulated_rotation(self) -> None:
        self.handler.update(
            _rotation_flow(7.0, 0.0, 0.0),
            FRAME_SIZE,
            horizontal_fov_degrees=FOV_DEGREES,
        )

        self.handler.reset()

        self.assertFalse(self.handler.measurement.valid)


if __name__ == "__main__":
    unittest.main()
