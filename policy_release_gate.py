"""Risk-aware release decisions for parking-policy evaluation evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    level: int
    success: bool
    collision: bool
    episode_return: float


@dataclass(frozen=True)
class ReleasePolicy:
    expected_levels: tuple[int, ...] = (0, 1, 2, 3)
    min_episodes_per_level: int = 20
    min_success_rate: float = 0.70
    min_level_success_rate: float = 0.55
    max_collision_rate: float = 0.05
    min_worst_seed_success_rate: float = 0.50
    tail_fraction: float = 0.10
    min_tail_mean_return: float = -50.0


def _validate_policy(policy: ReleasePolicy) -> None:
    if not policy.expected_levels or len(set(policy.expected_levels)) != len(
        policy.expected_levels
    ):
        raise ValueError("expected_levels must be non-empty and unique")
    if policy.min_episodes_per_level < 1:
        raise ValueError("min_episodes_per_level must be positive")
    for name in (
        "min_success_rate",
        "min_level_success_rate",
        "max_collision_rate",
        "min_worst_seed_success_rate",
        "tail_fraction",
    ):
        value = getattr(policy, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if policy.tail_fraction == 0.0:
        raise ValueError("tail_fraction must be greater than zero")
    if not math.isfinite(policy.min_tail_mean_return):
        raise ValueError("min_tail_mean_return must be finite")


def evaluate_policy_release(
    episodes: Iterable[EpisodeResult], policy: ReleasePolicy | None = None
) -> dict[str, object]:
    """Evaluate rollout evidence, returning a deterministic JSON-ready report."""
    policy = policy or ReleasePolicy()
    _validate_policy(policy)
    rows = list(episodes)
    if not rows:
        raise ValueError("episodes must not be empty")

    identities: set[tuple[int, int]] = set()
    expected = set(policy.expected_levels)
    for row in rows:
        identity = (row.seed, row.level)
        if identity in identities:
            raise ValueError(f"duplicate seed/level result: {identity}")
        identities.add(identity)
        if row.level not in expected:
            raise ValueError(f"unexpected curriculum level: {row.level}")
        if not isinstance(row.success, bool) or not isinstance(row.collision, bool):
            raise TypeError("success and collision must be booleans")
        if row.success and row.collision:
            raise ValueError("an episode cannot be both successful and collided")
        if not math.isfinite(row.episode_return):
            raise ValueError("episode_return must be finite")

    def rates(group: list[EpisodeResult]) -> dict[str, float | int]:
        return {
            "episodes": len(group),
            "success_rate": sum(row.success for row in group) / len(group),
            "collision_rate": sum(row.collision for row in group) / len(group),
        }

    by_level = {
        str(level): rates([row for row in rows if row.level == level])
        for level in sorted(expected)
        if any(row.level == level for row in rows)
    }
    by_seed = {
        str(seed): rates([row for row in rows if row.seed == seed])
        for seed in sorted({row.seed for row in rows})
    }
    overall = rates(rows)
    tail_count = max(1, math.ceil(len(rows) * policy.tail_fraction))
    tail = sorted(row.episode_return for row in rows)[:tail_count]
    tail_mean_return = sum(tail) / tail_count

    reasons: list[str] = []
    for level in sorted(expected):
        metric = by_level.get(str(level))
        if metric is None or metric["episodes"] < policy.min_episodes_per_level:
            reasons.append(f"insufficient_level_evidence:{level}")
        elif metric["success_rate"] < policy.min_level_success_rate:
            reasons.append(f"level_success_below_floor:{level}")
    if overall["success_rate"] < policy.min_success_rate:
        reasons.append("overall_success_below_floor")
    if overall["collision_rate"] > policy.max_collision_rate:
        reasons.append("collision_rate_above_ceiling")
    worst_seed_success = min(metric["success_rate"] for metric in by_seed.values())
    if worst_seed_success < policy.min_worst_seed_success_rate:
        reasons.append("worst_seed_success_below_floor")
    if tail_mean_return < policy.min_tail_mean_return:
        reasons.append("tail_return_below_floor")

    return {
        "schema_version": 1,
        "decision": "promote" if not reasons else "reject",
        "reasons": reasons,
        "policy": asdict(policy),
        "overall": overall,
        "by_level": by_level,
        "by_seed": by_seed,
        "worst_seed_success_rate": worst_seed_success,
        "tail_mean_return": tail_mean_return,
        "tail_episode_count": tail_count,
    }
