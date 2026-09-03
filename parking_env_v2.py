from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from parking_env import angle_wrap_deg, build_action_table


@dataclass(frozen=True)
class Rect:
    cx: float
    cy: float
    width: float
    height: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.cx - self.width / 2,
            self.cx + self.width / 2,
            self.cy - self.height / 2,
            self.cy + self.height / 2,
        )


@dataclass
class ParkingV2Config:
    action_mode: str = "discrete9"
    curriculum_level: int = 0
    world_size: float = 32.0
    max_steps: int = 350
    car_length: float = 4.5
    car_width: float = 2.0
    slot_length: float = 6.5
    slot_width: float = 3.0
    max_speed: float = 1.5
    max_turn_deg: float = 8.0
    lidar_rays: int = 16
    lidar_range: float = 16.0
    obstacle_count: int = 6
    success_distance: float = 1.0
    success_heading_deg: float = 10.0
    success_speed: float = 0.30


@dataclass(frozen=True)
class RewardTerms:
    progress: float
    alignment: float
    speed: float
    smoothness: float
    time: float
    terminal: float

    @property
    def total(self) -> float:
        return self.progress + self.alignment + self.speed + self.smoothness + self.time + self.terminal


class ParkingEnvV2(gym.Env):
    """Parking environment with obstacle geometry, LIDAR and curriculum levels.

    The dynamics remain intentionally lightweight so algorithm comparisons focus
    on RL design choices rather than a heavyweight vehicle simulator.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array"]}

    def __init__(self, config: ParkingV2Config | None = None):
        super().__init__()
        self.cfg = config or ParkingV2Config()
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

        # 10 kinematic/goal features + 3 phase one-hot + N lidar distances.
        self.observation_space = spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(13 + self.cfg.lidar_rays,),
            dtype=np.float32,
        )
        self.steps = 0
        self.state = np.zeros(4, dtype=np.float32)  # x, y, heading deg, speed
        self.target = np.zeros(3, dtype=np.float32)  # x, y, heading deg
        self.obstacles: list[Rect] = []
        self.last_action = np.zeros(2, dtype=np.float32)

    def set_curriculum_level(self, level: int) -> None:
        self.cfg.curriculum_level = int(np.clip(level, 0, 3))

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action[:] = 0
        level = self.cfg.curriculum_level
        rng = self.np_random

        self.target = np.asarray(
            [rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(-180, 180)],
            dtype=np.float32,
        )
        min_distance, max_distance = [(7, 13), (10, 20), (14, 27), (18, 30)][level]
        for _ in range(250):
            angle = rng.uniform(-math.pi, math.pi)
            distance = rng.uniform(min_distance, max_distance)
            candidate = np.asarray(
                [
                    self.target[0] + distance * math.cos(angle),
                    self.target[1] + distance * math.sin(angle),
                    rng.uniform(-180, 180),
                    0.0,
                ],
                dtype=np.float32,
            )
            if max(abs(candidate[0]), abs(candidate[1])) < self.cfg.world_size - 2:
                self.state = candidate
                break

        self.obstacles = self._generate_obstacles(level)
        return self._obs(), {"curriculum_level": level}

    def _generate_obstacles(self, level: int) -> list[Rect]:
        if level == 0:
            return []
        rng = self.np_random
        count = max(2, self.cfg.obstacle_count - 2 + level)
        obstacles: list[Rect] = []
        for _ in range(500):
            if len(obstacles) >= count:
                break
            candidate = Rect(
                cx=float(rng.uniform(-self.cfg.world_size + 4, self.cfg.world_size - 4)),
                cy=float(rng.uniform(-self.cfg.world_size + 4, self.cfg.world_size - 4)),
                width=float(rng.uniform(2.5, 6.0)),
                height=float(rng.uniform(2.5, 6.0)),
            )
            if self._point_rect_distance(self.target[:2], candidate) < 4.5:
                continue
            if self._point_rect_distance(self.state[:2], candidate) < 4.5:
                continue
            if any(self._rect_overlap(candidate, existing, padding=1.0) for existing in obstacles):
                continue
            obstacles.append(candidate)
        return obstacles

    @staticmethod
    def _rect_overlap(a: Rect, b: Rect, padding: float = 0.0) -> bool:
        amin_x, amax_x, amin_y, amax_y = a.bounds
        bmin_x, bmax_x, bmin_y, bmax_y = b.bounds
        return not (
            amax_x + padding < bmin_x
            or bmax_x + padding < amin_x
            or amax_y + padding < bmin_y
            or bmax_y + padding < amin_y
        )

    @staticmethod
    def _point_rect_distance(point: np.ndarray, rect: Rect) -> float:
        min_x, max_x, min_y, max_y = rect.bounds
        dx = max(min_x - float(point[0]), 0.0, float(point[0]) - max_x)
        dy = max(min_y - float(point[1]), 0.0, float(point[1]) - max_y)
        return math.hypot(dx, dy)

    def _decode_action(self, action) -> tuple[float, float]:
        if self.action_table is not None:
            steering, throttle = self.action_table[int(action)]
        else:
            steering, throttle = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        return float(steering), float(throttle)

    def _phase(self, distance: float, heading_error: float) -> np.ndarray:
        if distance > 8:
            return np.asarray([1, 0, 0], dtype=np.float32)
        if abs(heading_error) > 15:
            return np.asarray([0, 1, 0], dtype=np.float32)
        return np.asarray([0, 0, 1], dtype=np.float32)

    def _ray_distance(self, angle_deg: float) -> float:
        origin = self.state[:2].astype(float)
        direction = np.asarray(
            [math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))],
            dtype=float,
        )
        # Small step ray marching is sufficient for the portfolio environment and
        # makes the geometry easy to inspect and modify.
        for distance in np.linspace(0.25, self.cfg.lidar_range, 80):
            point = origin + direction * distance
            if max(abs(point[0]), abs(point[1])) >= self.cfg.world_size:
                return float(distance)
            if any(self._point_rect_distance(point, obstacle) <= 0.01 for obstacle in self.obstacles):
                return float(distance)
        return self.cfg.lidar_range

    def _lidar(self) -> np.ndarray:
        heading = float(self.state[2])
        offsets = np.linspace(-180, 180, self.cfg.lidar_rays, endpoint=False)
        distances = [self._ray_distance(heading + float(offset)) / self.cfg.lidar_range for offset in offsets]
        return np.asarray(distances, dtype=np.float32)

    def _obs(self) -> np.ndarray:
        x, y, heading, speed = map(float, self.state)
        dx, dy = self.target[:2] - self.state[:2]
        distance = float(np.linalg.norm([dx, dy]))
        heading_error = angle_wrap_deg(float(self.target[2] - heading))
        phase = self._phase(distance, heading_error)
        core = np.asarray(
            [
                x / self.cfg.world_size,
                y / self.cfg.world_size,
                math.cos(math.radians(heading)),
                math.sin(math.radians(heading)),
                speed / self.cfg.max_speed,
                float(dx) / self.cfg.world_size,
                float(dy) / self.cfg.world_size,
                distance / (2 * self.cfg.world_size),
                math.cos(math.radians(heading_error)),
                math.sin(math.radians(heading_error)),
                *phase,
            ],
            dtype=np.float32,
        )
        return np.concatenate([core, self._lidar()]).astype(np.float32)

    def _car_collision(self) -> bool:
        x, y = map(float, self.state[:2])
        radius = math.hypot(self.cfg.car_length / 2, self.cfg.car_width / 2) * 0.72
        if abs(x) + radius >= self.cfg.world_size or abs(y) + radius >= self.cfg.world_size:
            return True
        return any(self._point_rect_distance(self.state[:2], obstacle) <= radius for obstacle in self.obstacles)

    def step(self, action):
        steering, throttle = self._decode_action(action)
        x, y, heading, speed = map(float, self.state)
        distance_before = float(np.linalg.norm(self.target[:2] - self.state[:2]))
        heading_error_before = abs(angle_wrap_deg(float(self.target[2] - heading)))

        speed = float(np.clip(speed + 0.18 * throttle, -self.cfg.max_speed, self.cfg.max_speed))
        heading = angle_wrap_deg(
            heading + self.cfg.max_turn_deg * steering * (0.25 + abs(speed) / self.cfg.max_speed)
        )
        x += speed * math.cos(math.radians(heading))
        y += speed * math.sin(math.radians(heading))
        self.state = np.asarray([x, y, heading, speed], dtype=np.float32)
        self.steps += 1

        distance_after = float(np.linalg.norm(self.target[:2] - self.state[:2]))
        heading_error = abs(angle_wrap_deg(float(self.target[2] - heading)))
        collision = self._car_collision()
        success = (
            distance_after < self.cfg.success_distance
            and heading_error < self.cfg.success_heading_deg
            and abs(speed) < self.cfg.success_speed
            and not collision
        )
        timeout = self.steps >= self.cfg.max_steps

        progress_reward = 8.0 * (distance_before - distance_after)
        alignment_improvement = max(-30.0, min(30.0, heading_error_before - heading_error)) / 30.0
        alignment_reward = (1.5 * alignment_improvement) if distance_after < 8 else 0.0
        speed_reward = -0.3 * abs(speed) if distance_after < 2.5 else 0.0
        smoothness = -0.04 * abs(steering - float(self.last_action[0]))
        time_penalty = -0.015
        terminal = 160.0 if success else (-120.0 if collision else 0.0)
        terms = RewardTerms(progress_reward, alignment_reward, speed_reward, smoothness, time_penalty, terminal)
        self.last_action[:] = (steering, throttle)

        terminated = bool(success or collision)
        truncated = bool(timeout and not terminated)
        info = {
            "success": success,
            "collision": collision,
            "distance": distance_after,
            "heading_error_deg": heading_error,
            "curriculum_level": self.cfg.curriculum_level,
            "reward_terms": terms.__dict__,
        }
        return self._obs(), float(terms.total), terminated, truncated, info
