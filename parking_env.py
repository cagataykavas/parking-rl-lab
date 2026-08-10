from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class ParkingConfig:
    action_mode: str = "discrete9"  # discrete9 | discrete43 | continuous
    world_size: float = 50.0
    max_steps: int = 300
    car_length: float = 4.5
    car_width: float = 2.0
    lot_length: float = 6.5
    lot_width: float = 3.0
    max_speed: float = 1.5
    max_turn_deg: float = 8.0


def angle_wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def build_action_table(levels: int) -> np.ndarray:
    if levels == 9:
        steering = [-1.0, 0.0, 1.0]
        throttle = [-1.0, 0.0, 1.0]
        return np.asarray([(s, t) for s in steering for t in throttle], dtype=np.float32)
    if levels == 43:
        steering = np.linspace(-1.0, 1.0, 7)
        throttle = np.linspace(-1.0, 1.0, 6)
        pairs = [(float(s), float(t)) for s in steering for t in throttle]
        pairs.append((0.0, 0.0))
        return np.asarray(pairs, dtype=np.float32)
    raise ValueError("levels must be 9 or 43")


class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: ParkingConfig | None = None):
        super().__init__()
        self.cfg = config or ParkingConfig()
        self.action_mode = self.cfg.action_mode.lower()
        if self.action_mode == "discrete9":
            self.action_table = build_action_table(9)
            self.action_space = spaces.Discrete(9)
        elif self.action_mode == "discrete43":
            self.action_table = build_action_table(43)
            self.action_space = spaces.Discrete(43)
        elif self.action_mode == "continuous":
            self.action_table = None
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            raise ValueError("action_mode must be discrete9, discrete43 or continuous")

        # x, y, cos/sin heading, speed, dx, dy, distance, cos/sin heading error, phase one-hot
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(13,), dtype=np.float32)
        self.np_random = None
        self.steps = 0
        self.state = np.zeros(5, dtype=np.float32)
        self.target = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        rng = self.np_random
        self.target = np.asarray([
            rng.uniform(-20, 20),
            rng.uniform(-20, 20),
            rng.uniform(-180, 180),
        ], dtype=np.float32)
        for _ in range(100):
            candidate = np.asarray([
                rng.uniform(-22, 22),
                rng.uniform(-22, 22),
                rng.uniform(-180, 180),
                0.0,
                0.0,
            ], dtype=np.float32)
            if np.linalg.norm(candidate[:2] - self.target[:2]) > 12:
                self.state = candidate
                break
        return self._obs(), {}

    def _decode_action(self, action) -> tuple[float, float]:
        if self.action_table is not None:
            steering, throttle = self.action_table[int(action)]
        else:
            steering, throttle = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        return float(steering), float(throttle)

    def _phase(self, distance: float, heading_error: float) -> np.ndarray:
        if distance > 8.0:
            return np.asarray([1, 0, 0], dtype=np.float32)
        if abs(heading_error) > 15.0:
            return np.asarray([0, 1, 0], dtype=np.float32)
        return np.asarray([0, 0, 1], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        x, y, heading, speed, _ = self.state
        dx, dy = self.target[:2] - self.state[:2]
        distance = float(np.linalg.norm([dx, dy]))
        heading_error = angle_wrap_deg(float(self.target[2] - heading))
        phase = self._phase(distance, heading_error)
        return np.asarray([
            x / self.cfg.world_size,
            y / self.cfg.world_size,
            math.cos(math.radians(float(heading))),
            math.sin(math.radians(float(heading))),
            speed / self.cfg.max_speed,
            dx / self.cfg.world_size,
            dy / self.cfg.world_size,
            distance / self.cfg.world_size,
            math.cos(math.radians(heading_error)),
            math.sin(math.radians(heading_error)),
            *phase,
        ], dtype=np.float32)

    def step(self, action):
        steering, throttle = self._decode_action(action)
        x, y, heading, speed, previous_distance = map(float, self.state)
        distance_before = float(np.linalg.norm(self.target[:2] - self.state[:2]))
        speed = float(np.clip(speed + 0.18 * throttle, -self.cfg.max_speed, self.cfg.max_speed))
        heading = angle_wrap_deg(heading + self.cfg.max_turn_deg * steering * (0.25 + abs(speed) / self.cfg.max_speed))
        x += speed * math.cos(math.radians(heading))
        y += speed * math.sin(math.radians(heading))
        self.state = np.asarray([x, y, heading, speed, distance_before], dtype=np.float32)
        self.steps += 1

        distance_after = float(np.linalg.norm(self.target[:2] - self.state[:2]))
        heading_error = abs(angle_wrap_deg(float(self.target[2] - heading)))
        progress = distance_before - distance_after
        reward = 6.0 * progress - 0.015 - 0.004 * abs(steering)

        success = distance_after < 1.2 and heading_error < 12.0 and abs(speed) < 0.35
        out_of_bounds = abs(x) > self.cfg.world_size or abs(y) > self.cfg.world_size
        timeout = self.steps >= self.cfg.max_steps
        if success:
            reward += 120.0
        if out_of_bounds:
            reward -= 80.0
        terminated = bool(success or out_of_bounds)
        truncated = bool(timeout and not terminated)
        info = {
            "success": success,
            "distance": distance_after,
            "heading_error_deg": heading_error,
            "out_of_bounds": out_of_bounds,
        }
        return self._obs(), float(reward), terminated, truncated, info
