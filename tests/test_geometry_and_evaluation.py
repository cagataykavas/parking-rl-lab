from __future__ import annotations

import math

import numpy as np

from parking_env_v2 import ParkingEnvV2, ParkingV2Config
from parking_rl.baselines import GreedyParkingController, RandomPolicy, ZeroPolicy
from parking_rl.evaluation import EvaluationConfig, evaluate_policy, run_episode
from parking_rl.geometry import (
    AxisAlignedRect,
    ParkingSlot,
    Pose2D,
    VehicleFootprint,
    inside_slot_fraction,
    parking_quality,
    polygon_distance,
    polygons_overlap,
    rect_polygon,
    signed_slot_clearance,
    slot_corners,
    vehicle_corners,
    wrap_angle_deg,
)


def test_angle_wrapping() -> None:
    assert wrap_angle_deg(0) == 0
    assert wrap_angle_deg(360) == 0
    assert wrap_angle_deg(190) == -170
    assert wrap_angle_deg(-190) == 170


def test_vehicle_corners_preserve_center_and_dimensions() -> None:
    footprint = VehicleFootprint(length=4.0, width=2.0)
    corners = vehicle_corners(Pose2D(10.0, -2.0, 90.0), footprint)
    assert np.allclose(corners.mean(axis=0), [10.0, -2.0])
    edge_lengths = sorted(
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    )
    assert np.allclose(edge_lengths, [2.0, 2.0, 4.0, 4.0])


def test_polygon_overlap_detects_separation_rotation_and_contact() -> None:
    footprint = VehicleFootprint(length=4.5, width=2.0)
    origin = vehicle_corners(Pose2D(0.0, 0.0, 35.0), footprint)
    near = vehicle_corners(Pose2D(1.0, 0.2, -20.0), footprint)
    far = vehicle_corners(Pose2D(12.0, 12.0, 0.0), footprint)
    assert polygons_overlap(origin, near)
    assert not polygons_overlap(origin, far)
    assert polygon_distance(origin, far) > 0


def test_slot_containment_and_clearance() -> None:
    footprint = VehicleFootprint(length=4.5, width=2.0)
    slot = ParkingSlot(0.0, 0.0, 0.0, length=6.5, width=3.0)
    centered = vehicle_corners(Pose2D(0.0, 0.0, 0.0), footprint)
    outside = vehicle_corners(Pose2D(5.0, 0.0, 0.0), footprint)

    assert inside_slot_fraction(centered, slot) == 1.0
    assert signed_slot_clearance(centered, slot) > 0
    assert inside_slot_fraction(outside, slot) < 1.0
    assert signed_slot_clearance(outside, slot) < 0
    assert slot_corners(slot).shape == (4, 2)


def test_parking_quality_reports_obstacle_and_boundary_collisions() -> None:
    footprint = VehicleFootprint(length=4.5, width=2.0)
    slot = ParkingSlot(0.0, 0.0, 0.0, length=6.5, width=3.0)
    safe = parking_quality(
        pose=Pose2D(0.0, 0.0, 0.0),
        footprint=footprint,
        slot=slot,
        obstacles=[AxisAlignedRect(8.0, 8.0, 2.0, 2.0)],
        world_size=20.0,
    )
    blocked = parking_quality(
        pose=Pose2D(0.0, 0.0, 0.0),
        footprint=footprint,
        slot=slot,
        obstacles=[AxisAlignedRect(1.0, 0.0, 2.0, 2.0)],
        world_size=20.0,
    )
    boundary = parking_quality(
        pose=Pose2D(19.5, 0.0, 0.0),
        footprint=footprint,
        slot=slot,
        world_size=20.0,
    )

    assert safe.fully_inside_slot
    assert not safe.collision
    assert safe.pose_score > 0.95
    assert blocked.collision
    assert blocked.obstacle_clearance == 0.0
    assert boundary.collision
    assert boundary.world_clearance < 0


def test_rect_polygon_distance_is_symmetric() -> None:
    left = rect_polygon(AxisAlignedRect(0.0, 0.0, 2.0, 2.0))
    right = rect_polygon(AxisAlignedRect(5.0, 0.0, 2.0, 2.0))
    assert math.isclose(polygon_distance(left, right), 3.0)
    assert math.isclose(polygon_distance(right, left), 3.0)


def test_seeded_random_policy_is_reproducible() -> None:
    env = ParkingEnvV2(ParkingV2Config(action_mode="continuous", curriculum_level=0))
    obs, _ = env.reset(seed=99)
    first = RandomPolicy()
    second = RandomPolicy()
    first.reset(123)
    second.reset(123)
    actions_a = [first.act(obs, env) for _ in range(5)]
    actions_b = [second.act(obs, env) for _ in range(5)]
    for left, right in zip(actions_a, actions_b, strict=True):
        assert np.allclose(left, right)


def test_zero_policy_maps_to_nearest_discrete_noop() -> None:
    env = ParkingEnvV2(ParkingV2Config(action_mode="discrete9", curriculum_level=0))
    obs, _ = env.reset(seed=4)
    action = ZeroPolicy().act(obs, env)
    steering, throttle = env.action_table[int(action)]
    assert abs(float(steering)) + abs(float(throttle)) == 0.0


def test_greedy_controller_outputs_valid_continuous_action() -> None:
    env = ParkingEnvV2(ParkingV2Config(action_mode="continuous", curriculum_level=0))
    obs, _ = env.reset(seed=8)
    controller = GreedyParkingController()
    controller.reset(8)
    action = np.asarray(controller.act(obs, env), dtype=float)
    assert action.shape == (2,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    assert controller.telemetry.phase in {"approach", "align", "settle"}


def test_episode_evaluation_produces_operational_metrics() -> None:
    episode = run_episode(
        ZeroPolicy(),
        level=0,
        seed=101,
        episode_index=0,
        action_mode="continuous",
    )
    assert episode.steps > 0
    assert not episode.success
    assert episode.timeout
    assert episode.path_length == 0.0
    assert 0.0 <= episode.final_pose_score <= 1.0
    assert 0.0 <= episode.min_normalized_lidar <= 1.0


def test_multi_seed_report_has_bootstrap_intervals() -> None:
    report = evaluate_policy(
        ZeroPolicy(),
        EvaluationConfig(
            action_mode="continuous",
            levels=(0,),
            seeds=(1,),
            episodes_per_seed=1,
            bootstrap_samples=100,
        ),
    )
    assert report["policy"] == "zero"
    assert report["overall"]["episodes"] == 1
    metrics = report["overall"]["metrics"]
    assert metrics["success_rate"]["mean"] == 0.0
    assert metrics["timeout_rate"]["mean"] == 1.0
    assert metrics["path_length"]["mean"] == 0.0
