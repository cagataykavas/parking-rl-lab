from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from parking_env_v2 import ParkingEnvV2, ParkingV2Config

from .baselines import ParkingPolicy
from .geometry import (
    AxisAlignedRect,
    ParkingSlot,
    Pose2D,
    VehicleFootprint,
    parking_quality,
)


@dataclass(frozen=True)
class EvaluationConfig:
    action_mode: str = "continuous"
    levels: tuple[int, ...] = (0, 1, 2, 3)
    seeds: tuple[int, ...] = (11, 23, 37)
    episodes_per_seed: int = 5
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 2026

    def __post_init__(self) -> None:
        if self.action_mode not in {"continuous", "discrete9", "discrete43"}:
            raise ValueError("unsupported action mode")
        if not self.levels or any(level not in {0, 1, 2, 3} for level in self.levels):
            raise ValueError("levels must contain curriculum levels 0..3")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.episodes_per_seed < 1:
            raise ValueError("episodes_per_seed must be >= 1")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be >= 100")


@dataclass(frozen=True)
class EpisodeMetrics:
    policy: str
    curriculum_level: int
    seed: int
    episode_index: int
    success: bool
    collision: bool
    timeout: bool
    reward: float
    steps: int
    path_length: float
    final_distance: float
    final_heading_error_deg: float
    final_speed: float
    mean_abs_steering: float
    mean_abs_throttle: float
    mean_action_delta: float
    min_normalized_lidar: float
    final_pose_score: float
    fully_inside_slot: bool
    min_slot_clearance: float
    obstacle_clearance: float
    world_clearance: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _bootstrap_mean_interval(
    values: Iterable[float],
    *,
    samples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return ConfidenceInterval(math.nan, math.nan, math.nan)
    if array.size == 1:
        value = float(array[0])
        return ConfidenceInterval(value, value, value)
    draws = rng.choice(array, size=(samples, array.size), replace=True)
    boot_means = np.mean(draws, axis=1)
    return ConfidenceInterval(
        mean=float(np.mean(array)),
        lower=float(np.quantile(boot_means, alpha / 2.0)),
        upper=float(np.quantile(boot_means, 1.0 - alpha / 2.0)),
    )


def _action_components(env: ParkingEnvV2, action: np.ndarray | int) -> tuple[float, float]:
    if env.action_table is not None:
        steering, throttle = env.action_table[int(action)]
        return float(steering), float(throttle)
    materialized = np.asarray(action, dtype=float).reshape(-1)
    if materialized.size != 2:
        raise ValueError("continuous parking actions must contain steering and throttle")
    return float(materialized[0]), float(materialized[1])


def _min_lidar(observation: np.ndarray, lidar_rays: int) -> float:
    if lidar_rays <= 0 or observation.size < lidar_rays:
        return 1.0
    lidar = np.asarray(observation[-lidar_rays:], dtype=float)
    return float(np.min(lidar)) if lidar.size else 1.0


def _quality(env: ParkingEnvV2):
    pose = Pose2D(
        x=float(env.state[0]),
        y=float(env.state[1]),
        heading_deg=float(env.state[2]),
    )
    footprint = VehicleFootprint(length=env.cfg.car_length, width=env.cfg.car_width)
    slot = ParkingSlot(
        center_x=float(env.target[0]),
        center_y=float(env.target[1]),
        heading_deg=float(env.target[2]),
        length=env.cfg.slot_length,
        width=env.cfg.slot_width,
    )
    obstacles = [
        AxisAlignedRect(
            center_x=obstacle.cx,
            center_y=obstacle.cy,
            width=obstacle.width,
            height=obstacle.height,
        )
        for obstacle in env.obstacles
    ]
    return parking_quality(
        pose=pose,
        footprint=footprint,
        slot=slot,
        obstacles=obstacles,
        world_size=env.cfg.world_size,
    )


def run_episode(
    policy: ParkingPolicy,
    *,
    level: int,
    seed: int,
    episode_index: int,
    action_mode: str,
) -> EpisodeMetrics:
    env = ParkingEnvV2(
        ParkingV2Config(action_mode=action_mode, curriculum_level=level)
    )
    observation, _ = env.reset(seed=seed)
    policy.reset(seed)

    previous_position = env.state[:2].astype(float).copy()
    previous_action = np.zeros(2, dtype=float)
    steering_values: list[float] = []
    throttle_values: list[float] = []
    action_deltas: list[float] = []
    lidar_minima: list[float] = []
    reward_sum = 0.0
    path_length = 0.0
    info: dict[str, object] = {
        "success": False,
        "collision": False,
        "distance": float(np.linalg.norm(env.target[:2] - env.state[:2])),
        "heading_error_deg": 180.0,
    }
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy.act(observation, env)
        steering, throttle = _action_components(env, action)
        observation, reward, terminated, truncated, info = env.step(action)

        position = env.state[:2].astype(float)
        path_length += float(np.linalg.norm(position - previous_position))
        previous_position = position.copy()
        action_vector = np.asarray([steering, throttle], dtype=float)
        action_deltas.append(float(np.linalg.norm(action_vector - previous_action)))
        previous_action = action_vector
        steering_values.append(abs(steering))
        throttle_values.append(abs(throttle))
        lidar_minima.append(_min_lidar(observation, env.cfg.lidar_rays))
        reward_sum += float(reward)

    quality = _quality(env)
    return EpisodeMetrics(
        policy=policy.name,
        curriculum_level=level,
        seed=seed,
        episode_index=episode_index,
        success=bool(info["success"]),
        collision=bool(info["collision"]),
        timeout=bool(truncated and not terminated),
        reward=reward_sum,
        steps=env.steps,
        path_length=path_length,
        final_distance=float(info["distance"]),
        final_heading_error_deg=float(info["heading_error_deg"]),
        final_speed=abs(float(env.state[3])),
        mean_abs_steering=mean(steering_values) if steering_values else 0.0,
        mean_abs_throttle=mean(throttle_values) if throttle_values else 0.0,
        mean_action_delta=mean(action_deltas) if action_deltas else 0.0,
        min_normalized_lidar=min(lidar_minima, default=1.0),
        final_pose_score=quality.pose_score,
        fully_inside_slot=quality.fully_inside_slot,
        min_slot_clearance=quality.min_slot_clearance,
        obstacle_clearance=quality.obstacle_clearance,
        world_clearance=quality.world_clearance,
    )


def _metric_values(episodes: list[EpisodeMetrics]) -> dict[str, list[float]]:
    return {
        "success_rate": [float(item.success) for item in episodes],
        "collision_rate": [float(item.collision) for item in episodes],
        "timeout_rate": [float(item.timeout) for item in episodes],
        "reward": [item.reward for item in episodes],
        "steps": [float(item.steps) for item in episodes],
        "path_length": [item.path_length for item in episodes],
        "final_distance": [item.final_distance for item in episodes],
        "final_heading_error_deg": [item.final_heading_error_deg for item in episodes],
        "final_speed": [item.final_speed for item in episodes],
        "mean_action_delta": [item.mean_action_delta for item in episodes],
        "min_normalized_lidar": [item.min_normalized_lidar for item in episodes],
        "final_pose_score": [item.final_pose_score for item in episodes],
        "fully_inside_slot_rate": [float(item.fully_inside_slot) for item in episodes],
        "min_slot_clearance": [item.min_slot_clearance for item in episodes],
    }


def aggregate_episodes(
    episodes: list[EpisodeMetrics],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if not episodes:
        raise ValueError("cannot aggregate an empty episode list")
    rng = np.random.default_rng(bootstrap_seed)
    metrics = {
        name: _bootstrap_mean_interval(values, samples=bootstrap_samples, rng=rng).as_dict()
        for name, values in _metric_values(episodes).items()
    }
    successful = [item for item in episodes if item.success]
    metrics["successful_steps"] = _bootstrap_mean_interval(
        [float(item.steps) for item in successful],
        samples=bootstrap_samples,
        rng=rng,
    ).as_dict()
    metrics["successful_path_length"] = _bootstrap_mean_interval(
        [item.path_length for item in successful],
        samples=bootstrap_samples,
        rng=rng,
    ).as_dict()
    return {
        "episodes": len(episodes),
        "metrics": metrics,
    }


def evaluate_policy(policy: ParkingPolicy, config: EvaluationConfig) -> dict[str, object]:
    episodes: list[EpisodeMetrics] = []
    episode_index = 0
    for level in config.levels:
        for base_seed in config.seeds:
            for repeat in range(config.episodes_per_seed):
                seed = int(base_seed + level * 100_000 + repeat * 1_003)
                episodes.append(
                    run_episode(
                        policy,
                        level=level,
                        seed=seed,
                        episode_index=episode_index,
                        action_mode=config.action_mode,
                    )
                )
                episode_index += 1

    level_reports: dict[str, object] = {}
    for level in config.levels:
        subset = [item for item in episodes if item.curriculum_level == level]
        level_reports[f"level_{level}"] = aggregate_episodes(
            subset,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed + level,
        )

    return {
        "policy": policy.name,
        "evaluation": asdict(config),
        "overall": aggregate_episodes(
            episodes,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed + 99,
        ),
        "by_curriculum_level": level_reports,
        "episodes": [item.as_dict() for item in episodes],
    }


def evaluate_policies(
    policies: Iterable[ParkingPolicy],
    config: EvaluationConfig,
) -> dict[str, object]:
    reports = [evaluate_policy(policy, config) for policy in policies]
    return {
        "schema_version": 1,
        "note": (
            "Reference-policy evaluation generated from deterministic environment rollouts. "
            "It is not a claim about trained PPO/DQN/SAC checkpoint performance."
        ),
        "reports": reports,
    }


def write_report(report: dict[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    return destination
