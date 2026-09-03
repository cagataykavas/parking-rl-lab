from __future__ import annotations

import numpy as np
import pytest

from parking_env_v2 import ParkingEnvV2, ParkingV2Config
from parking_rl.geometry import AxisAlignedRect, Pose2D
from parking_rl.scenarios import (
    Scenario,
    apply_scenario,
    default_scenarios,
    scenario_by_name,
    validate_scenario,
)


def test_default_scenarios_are_unique_and_valid() -> None:
    env = ParkingEnvV2(ParkingV2Config(action_mode="continuous", curriculum_level=0))
    scenarios = default_scenarios()
    names = [scenario.name for scenario in scenarios]
    assert len(names) == len(set(names))
    assert len(scenarios) >= 6
    for scenario in scenarios:
        validation = validate_scenario(scenario, env)
        assert validation.valid, (scenario.name, validation.reasons)


def test_apply_scenario_replaces_random_state_and_obstacles() -> None:
    env = ParkingEnvV2(ParkingV2Config(action_mode="continuous", curriculum_level=3))
    scenario = scenario_by_name("single_obstacle_detour")
    observation, info = apply_scenario(env, scenario, seed=7)

    assert np.allclose(env.state[:3], [-12.0, 0.0, 0.0])
    assert np.allclose(env.target, [2.0, 0.0, 0.0])
    assert len(env.obstacles) == 1
    assert observation.shape == env.observation_space.shape
    assert info["scenario"] == scenario.name
    assert "obstacle" in info["tags"]


def test_scenario_application_is_deterministic() -> None:
    scenario = scenario_by_name("perpendicular_entry")
    first = ParkingEnvV2(ParkingV2Config(action_mode="continuous"))
    second = ParkingEnvV2(ParkingV2Config(action_mode="continuous"))
    obs_a, _ = apply_scenario(first, scenario, seed=99)
    obs_b, _ = apply_scenario(second, scenario, seed=99)
    assert np.allclose(first.state, second.state)
    assert np.allclose(first.target, second.target)
    assert np.allclose(obs_a, obs_b)


def test_invalid_boundary_start_is_rejected() -> None:
    env = ParkingEnvV2(ParkingV2Config(action_mode="continuous"))
    invalid = Scenario(
        name="outside",
        description="invalid fixture",
        start=Pose2D(32.0, 0.0, 0.0),
        target=Pose2D(0.0, 0.0, 0.0),
    )
    validation = validate_scenario(invalid, env)
    assert not validation.valid
    assert any("boundary" in reason for reason in validation.reasons)
    with pytest.raises(ValueError, match="invalid scenario"):
        apply_scenario(env, invalid)


def test_invalid_colliding_target_is_rejected() -> None:
    env = ParkingEnvV2(ParkingV2Config(action_mode="continuous"))
    invalid = Scenario(
        name="blocked-target",
        description="obstacle occupies target",
        start=Pose2D(-10.0, 0.0, 0.0),
        target=Pose2D(0.0, 0.0, 0.0),
        obstacles=(AxisAlignedRect(0.0, 0.0, 3.0, 3.0),),
    )
    validation = validate_scenario(invalid, env)
    assert not validation.valid
    assert any("target pose collides" in reason for reason in validation.reasons)


def test_scenario_lookup_reports_available_names() -> None:
    with pytest.raises(KeyError, match="available"):
        scenario_by_name("definitely-missing")
