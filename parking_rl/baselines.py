from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from parking_env import angle_wrap_deg


class ParkingPolicy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, observation: np.ndarray, env) -> np.ndarray | int: ...


@dataclass
class ControllerTelemetry:
    target_bearing_deg: float = 0.0
    desired_heading_deg: float = 0.0
    heading_error_deg: float = 0.0
    distance: float = 0.0
    steering: float = 0.0
    throttle: float = 0.0
    forward_clearance: float = 1.0
    phase: str = "approach"


class ZeroPolicy:
    """No-op reference policy for environment regression checks."""

    name = "zero"

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: np.ndarray, env) -> np.ndarray | int:
        del observation
        if hasattr(env.action_space, "n"):
            table = env.action_table
            assert table is not None
            scores = [abs(float(steer)) + abs(float(throttle)) for steer, throttle in table]
            return int(np.argmin(scores))
        return np.zeros(2, dtype=np.float32)


class RandomPolicy:
    """Seeded random policy independent of Gym's action-space RNG."""

    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, env) -> np.ndarray | int:
        del observation
        if hasattr(env.action_space, "n"):
            return int(self._rng.integers(0, int(env.action_space.n)))
        return self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


class GreedyParkingController:
    """Privileged geometric controller used as an engineering reference baseline.

    This is not presented as a learned agent. It reads the environment's true
    pose/goal state and therefore acts as an oracle-style reference controller.
    Comparing PPO/DQN/SAC against it answers a more useful question than comparing
    against random alone: does a learned policy acquire at least simple geometric
    parking behavior while coping with observations, obstacles and action limits?
    """

    name = "privileged-greedy"

    def __init__(
        self,
        *,
        align_distance: float = 5.0,
        final_distance: float = 1.8,
        heading_gain: float = 1.0 / 45.0,
        cruise_throttle: float = 0.85,
        reverse_threshold_deg: float = 105.0,
        brake_gain: float = 0.9,
        obstacle_brake_lidar: float = 0.12,
    ) -> None:
        if align_distance <= final_distance:
            raise ValueError("align_distance must exceed final_distance")
        self.align_distance = align_distance
        self.final_distance = final_distance
        self.heading_gain = heading_gain
        self.cruise_throttle = cruise_throttle
        self.reverse_threshold_deg = reverse_threshold_deg
        self.brake_gain = brake_gain
        self.obstacle_brake_lidar = obstacle_brake_lidar
        self.telemetry = ControllerTelemetry()
        self._last_steering = 0.0

    def reset(self, seed: int) -> None:
        del seed
        self.telemetry = ControllerTelemetry()
        self._last_steering = 0.0

    @staticmethod
    def _bearing(dx: float, dy: float) -> float:
        return math.degrees(math.atan2(dy, dx))

    def _desired_heading(
        self,
        *,
        x: float,
        y: float,
        heading: float,
        target_x: float,
        target_y: float,
        target_heading: float,
        distance: float,
    ) -> tuple[float, str]:
        del heading
        bearing = self._bearing(target_x - x, target_y - y)
        if distance > self.align_distance:
            return bearing, "approach"
        if distance > self.final_distance:
            blend = (self.align_distance - distance) / (
                self.align_distance - self.final_distance
            )
            delta = angle_wrap_deg(target_heading - bearing)
            return angle_wrap_deg(bearing + blend * delta), "align"
        return target_heading, "settle"

    @staticmethod
    def _forward_clearance(observation: np.ndarray, lidar_rays: int) -> float:
        if lidar_rays <= 0 or observation.size < lidar_rays:
            return 1.0
        lidar = np.asarray(observation[-lidar_rays:], dtype=float)
        midpoint = lidar.size // 2
        indices = [
            int(np.clip(midpoint + offset, 0, lidar.size - 1))
            for offset in (-1, 0, 1)
        ]
        return float(np.min(lidar[indices]))

    def _continuous_action(
        self,
        observation: np.ndarray,
        env,
    ) -> tuple[float, float]:
        x, y, heading, speed = map(float, env.state)
        target_x, target_y, target_heading = map(float, env.target)
        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)
        target_bearing = self._bearing(dx, dy)
        desired_heading, phase = self._desired_heading(
            x=x,
            y=y,
            heading=heading,
            target_x=target_x,
            target_y=target_y,
            target_heading=target_heading,
            distance=distance,
        )
        heading_error = angle_wrap_deg(desired_heading - heading)

        direction = 1.0
        if abs(heading_error) > self.reverse_threshold_deg and distance > self.final_distance:
            direction = -1.0
            desired_reverse_heading = angle_wrap_deg(desired_heading + 180.0)
            heading_error = angle_wrap_deg(desired_reverse_heading - heading)

        raw_steering = float(np.clip(heading_error * self.heading_gain, -1.0, 1.0))
        steering = 0.72 * raw_steering + 0.28 * self._last_steering
        self._last_steering = steering

        if phase == "approach":
            throttle = self.cruise_throttle * direction
        elif phase == "align":
            distance_scale = float(np.clip(distance / self.align_distance, 0.25, 0.75))
            heading_scale = float(np.clip(1.0 - abs(heading_error) / 120.0, 0.20, 1.0))
            throttle = direction * distance_scale * heading_scale
        else:
            position_command = float(np.clip(distance / self.final_distance, 0.0, 0.45))
            if distance < env.cfg.success_distance * 0.8:
                position_command = 0.0
            throttle = (
                direction * position_command
                - self.brake_gain * speed / env.cfg.max_speed
            )

        forward_clearance = self._forward_clearance(observation, env.cfg.lidar_rays)
        if forward_clearance < self.obstacle_brake_lidar and throttle > 0:
            throttle = min(throttle, -0.35)

        throttle = float(np.clip(throttle, -1.0, 1.0))
        self.telemetry = ControllerTelemetry(
            target_bearing_deg=target_bearing,
            desired_heading_deg=desired_heading,
            heading_error_deg=heading_error,
            distance=distance,
            steering=steering,
            throttle=throttle,
            forward_clearance=forward_clearance,
            phase=phase,
        )
        return steering, throttle

    @staticmethod
    def _nearest_discrete_action(env, steering: float, throttle: float) -> int:
        table = env.action_table
        if table is None:
            raise ValueError("discrete action requested without an action table")
        requested = np.asarray([steering, throttle], dtype=float)
        distances = [
            float(np.linalg.norm(np.asarray(candidate, dtype=float) - requested))
            for candidate in table
        ]
        return int(np.argmin(distances))

    def act(self, observation: np.ndarray, env) -> np.ndarray | int:
        steering, throttle = self._continuous_action(observation, env)
        if env.action_table is not None:
            return self._nearest_discrete_action(env, steering, throttle)
        return np.asarray([steering, throttle], dtype=np.float32)
