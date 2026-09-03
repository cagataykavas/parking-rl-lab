from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from parking_env_v2 import ParkingEnvV2, Rect

from .geometry import (
    AxisAlignedRect,
    ParkingSlot,
    Pose2D,
    VehicleFootprint,
    parking_quality,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    start: Pose2D
    target: Pose2D
    obstacles: tuple[AxisAlignedRect, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioValidation:
    valid: bool
    reasons: tuple[str, ...]


def default_scenarios() -> tuple[Scenario, ...]:
    """Deterministic scenarios for regression/evaluation beyond random resets."""
    return (
        Scenario(
            name="straight_approach",
            description="Open-space forward approach with target already aligned.",
            start=Pose2D(-10.0, 0.0, 0.0),
            target=Pose2D(0.0, 0.0, 0.0),
            tags=("level0", "approach"),
        ),
        Scenario(
            name="reverse_alignment",
            description="Target lies behind the vehicle and rewards reverse reasoning.",
            start=Pose2D(8.0, 0.0, 0.0),
            target=Pose2D(0.0, 0.0, 0.0),
            tags=("reverse", "heading"),
        ),
        Scenario(
            name="perpendicular_entry",
            description="Large heading mismatch close enough to require an alignment phase.",
            start=Pose2D(-7.0, -5.0, 90.0),
            target=Pose2D(0.0, 0.0, 0.0),
            tags=("alignment", "turning"),
        ),
        Scenario(
            name="single_obstacle_detour",
            description="One parked-car rectangle blocks the geometric straight line.",
            start=Pose2D(-12.0, 0.0, 0.0),
            target=Pose2D(2.0, 0.0, 0.0),
            obstacles=(AxisAlignedRect(-4.0, 0.0, 3.5, 5.0),),
            tags=("obstacle", "lidar"),
        ),
        Scenario(
            name="narrow_final_slot",
            description="Two parked cars flank the final target and expose clearance quality.",
            start=Pose2D(-10.0, -7.0, 25.0),
            target=Pose2D(0.0, 0.0, 90.0),
            obstacles=(
                AxisAlignedRect(-3.1, 0.0, 2.4, 5.4),
                AxisAlignedRect(3.1, 0.0, 2.4, 5.4),
            ),
            tags=("clearance", "obstacle", "final-pose"),
        ),
        Scenario(
            name="boundary_recovery",
            description="Start near the world edge facing outward; policy must recover inward.",
            start=Pose2D(27.0, -10.0, -20.0),
            target=Pose2D(10.0, -8.0, 160.0),
            tags=("boundary", "recovery"),
        ),
    )


def validate_scenario(scenario: Scenario, env: ParkingEnvV2) -> ScenarioValidation:
    reasons: list[str] = []
    margin = max(env.cfg.car_length, env.cfg.car_width) / 2.0
    for label, pose in (("start", scenario.start), ("target", scenario.target)):
        if abs(pose.x) >= env.cfg.world_size - margin:
            reasons.append(f"{label} x is too close to/outside the world boundary")
        if abs(pose.y) >= env.cfg.world_size - margin:
            reasons.append(f"{label} y is too close to/outside the world boundary")

    footprint = VehicleFootprint(env.cfg.car_length, env.cfg.car_width)
    slot = ParkingSlot(
        scenario.target.x,
        scenario.target.y,
        scenario.target.heading_deg,
        env.cfg.slot_length,
        env.cfg.slot_width,
    )
    start_quality = parking_quality(
        pose=scenario.start,
        footprint=footprint,
        slot=slot,
        obstacles=scenario.obstacles,
        world_size=env.cfg.world_size,
    )
    if start_quality.collision:
        reasons.append("start pose collides with an obstacle or world boundary")

    target_quality = parking_quality(
        pose=scenario.target,
        footprint=footprint,
        slot=slot,
        obstacles=scenario.obstacles,
        world_size=env.cfg.world_size,
    )
    if target_quality.collision:
        reasons.append("target pose collides with an obstacle or world boundary")

    return ScenarioValidation(valid=not reasons, reasons=tuple(reasons))


def apply_scenario(
    env: ParkingEnvV2,
    scenario: Scenario,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reset Gym bookkeeping, then replace random layout with a validated fixture."""
    env.reset(seed=seed)
    validation = validate_scenario(scenario, env)
    if not validation.valid:
        raise ValueError(
            f"invalid scenario {scenario.name!r}: {'; '.join(validation.reasons)}"
        )

    env.steps = 0
    env.last_action[:] = 0.0
    env.state = np.asarray(
        [scenario.start.x, scenario.start.y, scenario.start.heading_deg, 0.0],
        dtype=np.float32,
    )
    env.target = np.asarray(
        [scenario.target.x, scenario.target.y, scenario.target.heading_deg],
        dtype=np.float32,
    )
    env.obstacles = [
        Rect(
            cx=obstacle.center_x,
            cy=obstacle.center_y,
            width=obstacle.width,
            height=obstacle.height,
        )
        for obstacle in scenario.obstacles
    ]
    return env._obs(), {  # noqa: SLF001 - fixture adapter intentionally materializes observation
        "scenario": scenario.name,
        "description": scenario.description,
        "tags": list(scenario.tags),
    }


def scenario_by_name(name: str, scenarios: Iterable[Scenario] | None = None) -> Scenario:
    materialized = tuple(scenarios) if scenarios is not None else default_scenarios()
    for scenario in materialized:
        if scenario.name == name:
            return scenario
    available = ", ".join(item.name for item in materialized)
    raise KeyError(f"unknown scenario {name!r}; available: {available}")
