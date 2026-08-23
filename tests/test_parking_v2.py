import numpy as np

from parking_env_v2 import ParkingEnvV2, ParkingV2Config, Rect


def test_discrete_environment_reset_and_step() -> None:
    env = ParkingEnvV2(ParkingV2Config(action_mode="discrete9", lidar_rays=12))
    observation, info = env.reset(seed=42)
    assert observation.shape == (25,)
    assert info["curriculum_level"] == 0
    next_observation, reward, terminated, truncated, step_info = env.step(4)
    assert next_observation.shape == observation.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_terms" in step_info


def test_lidar_is_normalized() -> None:
    env = ParkingEnvV2(ParkingV2Config(lidar_rays=16, curriculum_level=2))
    observation, _ = env.reset(seed=7)
    lidar = observation[-16:]
    assert np.all(lidar >= 0.0)
    assert np.all(lidar <= 1.0)


def test_collision_with_obstacle() -> None:
    env = ParkingEnvV2(ParkingV2Config())
    env.reset(seed=3)
    env.obstacles = [Rect(float(env.state[0]), float(env.state[1]), 3.0, 3.0)]
    assert env._car_collision()


def test_curriculum_level_is_bounded() -> None:
    env = ParkingEnvV2(ParkingV2Config())
    env.set_curriculum_level(99)
    assert env.cfg.curriculum_level == 3
    env.set_curriculum_level(-5)
    assert env.cfg.curriculum_level == 0


def test_continuous_action_mode() -> None:
    env = ParkingEnvV2(ParkingV2Config(action_mode="continuous"))
    env.reset(seed=12)
    _, _, _, _, info = env.step(np.asarray([0.2, 0.5], dtype=np.float32))
    assert "distance" in info
