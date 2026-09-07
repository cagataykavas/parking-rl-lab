from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from parking_rl.agents.dqn import (
    DQNConfig,
    DoubleDQNAgent,
    DuelingQNetwork,
    ReplayBatch,
    ReplayBuffer,
    evaluate_dqn,
)
from parking_env_v2 import ParkingEnvV2, ParkingV2Config


def tiny_config(**overrides) -> DQNConfig:
    values = {
        "action_mode": "discrete9",
        "episodes": 2,
        "seed": 7,
        "batch_size": 4,
        "replay_capacity": 32,
        "replay_warmup": 4,
        "train_every": 1,
        "target_sync_every": 2,
        "epsilon_start": 0.4,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 10,
        "hidden_size": 32,
        "checkpoint_every": 1,
        "device": "cpu",
    }
    values.update(overrides)
    return DQNConfig(**values)


def transition(index: int, dimension: int = 5):
    state = np.linspace(index, index + 1, dimension, dtype=np.float32)
    next_state = state + 0.25
    return state, index % 3, float(index), next_state, index % 2 == 0


def test_dqn_rejects_continuous_action_space() -> None:
    with pytest.raises(ValueError, match="discrete9 and discrete43"):
        DQNConfig(action_mode="continuous")


def test_dqn_config_validates_training_invariants() -> None:
    with pytest.raises(ValueError, match="replay_capacity"):
        tiny_config(replay_capacity=2)
    with pytest.raises(ValueError, match="replay_warmup"):
        tiny_config(replay_warmup=2)
    with pytest.raises(ValueError, match="epsilon"):
        tiny_config(epsilon_start=0.1, epsilon_end=0.4)
    with pytest.raises(ValueError, match="curriculum_level"):
        tiny_config(curriculum_level=9)


def test_replay_buffer_is_bounded_and_copies_input() -> None:
    buffer = ReplayBuffer(3, seed=11)
    state = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    next_state = state + 1
    buffer.add(state, 0, 1.0, next_state, False)
    state[:] = 99
    next_state[:] = 99

    for index in range(1, 5):
        buffer.add(*transition(index, dimension=3))

    assert len(buffer) == 3
    snapshot = buffer.snapshot()
    assert [item["reward"] for item in snapshot] == [2.0, 3.0, 4.0]
    assert all(item["state_norm"] < 20 for item in snapshot)


def test_replay_sampling_is_seeded_and_shape_safe() -> None:
    left = ReplayBuffer(10, seed=123)
    right = ReplayBuffer(10, seed=123)
    for index in range(8):
        item = transition(index)
        left.add(*item)
        right.add(*item)

    left_batch = left.sample(4, device=torch.device("cpu"))
    right_batch = right.sample(4, device=torch.device("cpu"))

    assert left_batch.states.shape == (4, 5)
    assert left_batch.actions.shape == (4,)
    assert torch.equal(left_batch.states, right_batch.states)
    assert torch.equal(left_batch.actions, right_batch.actions)


def test_replay_rejects_invalid_sample_size_and_state_shape() -> None:
    buffer = ReplayBuffer(4)
    state = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="dimensions"):
        buffer.add(state, 0, 0.0, np.zeros(4, dtype=np.float32), False)
    buffer.add(state, 0, 0.0, state, False)
    with pytest.raises(ValueError, match="more transitions"):
        buffer.sample(2, device=torch.device("cpu"))


def test_dueling_network_has_expected_action_shape() -> None:
    network = DuelingQNetwork(state_dim=29, action_dim=43, hidden_size=64)
    values = network(torch.zeros((6, 29), dtype=torch.float32))

    assert values.shape == (6, 43)
    assert torch.isfinite(values).all()


class FixedQNetwork(nn.Module):
    def __init__(self, rows: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("rows", rows)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.rows[: states.shape[0]]


def test_double_dqn_selects_online_action_but_uses_target_value() -> None:
    config = tiny_config(gamma=0.5)
    agent = DoubleDQNAgent(state_dim=2, action_dim=3, config=config)
    agent.online = FixedQNetwork(
        torch.tensor([[1.0, 9.0, 3.0], [8.0, 2.0, 1.0]])
    )
    agent.target = FixedQNetwork(
        torch.tensor([[100.0, 7.0, 2.0], [5.0, 200.0, 4.0]])
    )
    batch = ReplayBatch(
        states=torch.zeros((2, 2)),
        actions=torch.tensor([0, 1]),
        rewards=torch.tensor([1.0, 2.0]),
        next_states=torch.zeros((2, 2)),
        terminated=torch.tensor([0.0, 1.0]),
    )

    targets = agent._double_dqn_targets(batch)

    # Online chooses action 1 for row 0; target evaluates action 1 as 7.
    assert targets[0].item() == pytest.approx(1.0 + 0.5 * 7.0)
    # Terminal transitions never bootstrap.
    assert targets[1].item() == pytest.approx(2.0)


def test_epsilon_schedule_decays_with_environment_steps() -> None:
    config = tiny_config(
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay_steps=100,
    )
    agent = DoubleDQNAgent(state_dim=4, action_dim=9, config=config)

    assert agent.epsilon() == pytest.approx(1.0)
    agent.environment_steps = 50
    assert agent.epsilon() == pytest.approx(0.6)
    agent.environment_steps = 1000
    assert agent.epsilon() == pytest.approx(0.2)


def test_observe_checks_action_range() -> None:
    agent = DoubleDQNAgent(state_dim=3, action_dim=9, config=tiny_config())
    state = np.zeros(3, dtype=np.float32)

    with pytest.raises(ValueError, match="action index"):
        agent.observe(state, 9, 0.0, state, False)


def test_update_changes_online_weights_and_syncs_target() -> None:
    torch.manual_seed(5)
    config = tiny_config(target_sync_every=1)
    agent = DoubleDQNAgent(state_dim=5, action_dim=9, config=config)
    before = {
        name: tensor.detach().clone()
        for name, tensor in agent.online.state_dict().items()
    }

    for index in range(8):
        state = np.full(5, index / 10.0, dtype=np.float32)
        next_state = state + 0.05
        agent.observe(state, index % 9, 1.0 - index * 0.1, next_state, index == 7)

    update = agent.update()

    assert update is not None
    assert update.loss >= 0.0
    assert update.gradient_step == 1
    assert update.mean_abs_td_error >= 0.0
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in agent.online.state_dict().items()
        if tensor.dtype.is_floating_point
    )
    for name, tensor in agent.online.state_dict().items():
        assert torch.equal(tensor, agent.target.state_dict()[name])


def test_update_waits_for_replay_warmup() -> None:
    agent = DoubleDQNAgent(state_dim=5, action_dim=9, config=tiny_config())
    for index in range(3):
        agent.observe(*transition(index))
    assert agent.update() is None


def test_checkpoint_round_trip_restores_predictions_and_counters(tmp_path: Path) -> None:
    torch.manual_seed(9)
    config = tiny_config()
    first = DoubleDQNAgent(state_dim=5, action_dim=9, config=config)
    observation = np.arange(5, dtype=np.float32)
    first.environment_steps = 123
    first.gradient_steps = 17
    expected = first.greedy_actions(observation)
    checkpoint = first.save(
        tmp_path / "agent.pt",
        metadata={"purpose": "round-trip-test"},
    )

    second = DoubleDQNAgent(state_dim=5, action_dim=9, config=config)
    metadata = second.load(checkpoint)

    assert metadata == {"purpose": "round-trip-test"}
    assert second.environment_steps == 123
    assert second.gradient_steps == 17
    assert np.array_equal(second.greedy_actions(observation), expected)


def test_checkpoint_rejects_wrong_action_dimension(tmp_path: Path) -> None:
    source = DoubleDQNAgent(state_dim=5, action_dim=9, config=tiny_config())
    path = source.save(tmp_path / "nine.pt")
    target = DoubleDQNAgent(
        state_dim=5,
        action_dim=43,
        config=tiny_config(action_mode="discrete43"),
    )

    with pytest.raises(ValueError, match="action dimension"):
        target.load(path)


def test_agent_runs_against_real_parking_environment() -> None:
    config = tiny_config()
    env = ParkingEnvV2(
        ParkingV2Config(action_mode="discrete9", curriculum_level=0, max_steps=5)
    )
    observation, _ = env.reset(seed=13)
    agent = DoubleDQNAgent(
        state_dim=observation.shape[0], action_dim=9, config=config
    )

    total_reward = 0.0
    for _ in range(5):
        action = agent.act(observation, deterministic=True)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        agent.observe(observation, action, reward, next_observation, terminated)
        observation = next_observation
        total_reward += reward
        if terminated or truncated:
            break

    assert agent.environment_steps >= 1
    assert np.isfinite(total_reward)


def test_evaluation_uses_disjoint_deterministic_rollouts() -> None:
    config = tiny_config()
    env = ParkingEnvV2(
        ParkingV2Config(action_mode="discrete9", curriculum_level=0)
    )
    observation, _ = env.reset(seed=config.seed)
    agent = DoubleDQNAgent(observation.shape[0], 9, config)

    first = evaluate_dqn(agent, config, episodes=2, seed_offset=200_000)
    second = evaluate_dqn(agent, config, episodes=2, seed_offset=200_000)

    assert first == second
    assert first.episodes == 2
    assert 0.0 <= first.success_rate <= 1.0
    assert 0.0 <= first.collision_rate <= 1.0
    assert np.isfinite(first.mean_reward)
