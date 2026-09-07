from __future__ import annotations

import numpy as np
import pytest
import torch

from parking_rl.control import (
    DEFAULT_MOTOR_PHASES,
    ControlStage,
    ExecutionKind,
    LatentMotorAdapter,
    LatentMotorCurriculum,
    MotorCurriculumPhase,
)


def test_default_motor_curriculum_preserves_recovered_phase_lengths() -> None:
    curriculum = LatentMotorCurriculum()

    assert curriculum.boundaries() == (400, 1000, 1600)
    assert curriculum.stage_for_episode(1).kind is ExecutionKind.DISCRETE_9
    assert curriculum.stage_for_episode(400).kind is ExecutionKind.DISCRETE_9
    assert curriculum.stage_for_episode(401).kind is ExecutionKind.DISCRETE_43
    assert curriculum.stage_for_episode(1000).kind is ExecutionKind.DISCRETE_43
    assert (
        curriculum.stage_for_episode(1001).kind
        is ExecutionKind.ANNEAL_43_TO_CONTINUOUS
    )
    assert curriculum.stage_for_episode(1601).kind is ExecutionKind.CONTINUOUS


def test_annealing_phase_moves_from_quantized_to_continuous() -> None:
    curriculum = LatentMotorCurriculum()

    start = curriculum.stage_for_episode(1001)
    middle = curriculum.stage_for_episode(1300)
    end = curriculum.stage_for_episode(1600)

    assert start.continuous_mix == pytest.approx(0.0)
    assert middle.continuous_mix == pytest.approx(299 / 599)
    assert end.continuous_mix == pytest.approx(1.0)
    assert start.action_count == 43
    assert end.action_count == 43


def test_curriculum_uses_one_stable_policy_contract() -> None:
    contract = LatentMotorCurriculum().as_dict()["policy_contract"]

    assert contract == {
        "shape": [2],
        "fields": ["steering", "throttle"],
        "range": [-1.0, 1.0],
    }


def test_curriculum_rejects_bad_episode_and_duplicate_phase_names() -> None:
    curriculum = LatentMotorCurriculum()
    with pytest.raises(ValueError, match="starts at one"):
        curriculum.stage_for_episode(0)

    duplicate = (
        MotorCurriculumPhase("same", ExecutionKind.DISCRETE_9, 10),
        MotorCurriculumPhase("same", ExecutionKind.CONTINUOUS, 10),
    )
    with pytest.raises(ValueError, match="unique"):
        LatentMotorCurriculum(duplicate)


def test_default_action_tables_match_environment_contracts() -> None:
    adapter = LatentMotorAdapter()

    assert adapter.table9.shape == (9, 2)
    assert adapter.table43.shape == (43, 2)
    assert np.all(adapter.table9 >= -1.0)
    assert np.all(adapter.table9 <= 1.0)
    assert np.all(adapter.table43 >= -1.0)
    assert np.all(adapter.table43 <= 1.0)


def test_discrete9_decoding_returns_exact_table_member() -> None:
    adapter = LatentMotorAdapter()
    stage = LatentMotorCurriculum().stage_for_episode(1)
    result = adapter.decode(np.asarray([0.81, -0.72], dtype=np.float32), stage)

    assert result.discrete_index is not None
    assert np.array_equal(result.executed, adapter.table9[result.discrete_index])
    assert result.kind is ExecutionKind.DISCRETE_9
    assert result.quantization_error() >= 0.0


def test_discrete43_has_finer_or_equal_resolution_for_generic_command() -> None:
    adapter = LatentMotorAdapter()
    command = np.asarray([0.28, 0.37], dtype=np.float32)
    coarse = adapter.decode(
        command, LatentMotorCurriculum().stage_for_episode(1)
    )
    fine = adapter.decode(
        command, LatentMotorCurriculum().stage_for_episode(401)
    )

    assert fine.quantization_error() <= coarse.quantization_error() + 1e-7


def test_annealed_execution_interpolates_between_grid_and_request() -> None:
    adapter = LatentMotorAdapter()
    curriculum = LatentMotorCurriculum()
    command = np.asarray([0.28, 0.37], dtype=np.float32)
    stage = curriculum.stage_for_episode(1300)
    result = adapter.decode(command, stage)

    index = adapter.nearest_index(command, adapter.table43)
    discrete = adapter.table43[index]
    expected = (
        (1.0 - stage.continuous_mix) * discrete
        + stage.continuous_mix * command
    )

    assert np.allclose(result.executed, expected)
    assert result.discrete_index == index


def test_continuous_phase_is_identity_after_clipping() -> None:
    adapter = LatentMotorAdapter()
    stage = LatentMotorCurriculum().stage_for_episode(2000)
    result = adapter.decode([1.4, -1.7], stage)

    assert result.discrete_index is None
    assert np.allclose(result.requested, [1.0, -1.0])
    assert np.array_equal(result.executed, result.requested)


def test_residual_policy_is_added_to_base_before_execution() -> None:
    adapter = LatentMotorAdapter(residual_scale=0.25)
    stage = LatentMotorCurriculum().stage_for_episode(2000)
    result = adapter.decode(
        [0.8, -0.4],
        stage,
        base_command=np.asarray([0.1, 0.2], dtype=np.float32),
    )

    assert np.allclose(result.executed, [0.3, 0.1], atol=1e-6)


def test_residual_scale_is_validated() -> None:
    with pytest.raises(ValueError, match="residual_scale"):
        LatentMotorAdapter(residual_scale=1.1)


def test_discrete_environment_index_matches_nearest_table_member() -> None:
    adapter = LatentMotorAdapter()
    command = [0.32, -0.68]

    index9 = adapter.discrete_env_index(command, action_mode="discrete9")
    index43 = adapter.discrete_env_index(command, action_mode="discrete43")

    assert 0 <= index9 < 9
    assert 0 <= index43 < 43
    assert index9 == adapter.nearest_index(
        adapter.normalize(command), adapter.table9
    )
    assert index43 == adapter.nearest_index(
        adapter.normalize(command), adapter.table43
    )


def test_discrete_environment_index_rejects_continuous_mode() -> None:
    adapter = LatentMotorAdapter()
    with pytest.raises(ValueError, match="discrete9 or discrete43"):
        adapter.discrete_env_index([0.0, 0.0], action_mode="continuous")


def test_motor_command_shape_validation() -> None:
    adapter = LatentMotorAdapter()
    stage = LatentMotorCurriculum().stage_for_episode(1)
    with pytest.raises(ValueError, match="steering and throttle"):
        adapter.decode([0.0, 0.2, 0.4], stage)


def test_torch_quantization_has_straight_through_gradient() -> None:
    adapter = LatentMotorAdapter()
    stage = LatentMotorCurriculum().stage_for_episode(1)
    command = torch.tensor([[0.31, -0.66]], dtype=torch.float32, requires_grad=True)

    executed = adapter.decode_torch(command, stage)
    loss = executed.sum()
    loss.backward()

    assert command.grad is not None
    assert torch.allclose(command.grad, torch.ones_like(command))
    assert any(
        torch.allclose(executed.detach()[0], candidate)
        for candidate in torch.as_tensor(adapter.table9)
    )


def test_torch_annealing_preserves_gradient_path() -> None:
    adapter = LatentMotorAdapter()
    stage = LatentMotorCurriculum().stage_for_episode(1300)
    command = torch.tensor([[0.31, -0.66]], dtype=torch.float32, requires_grad=True)

    executed = adapter.decode_torch(command, stage)
    executed.sum().backward()

    assert command.grad is not None
    assert torch.isfinite(command.grad).all()
    assert torch.all(command.grad > 0)


def test_torch_residual_requires_matching_shape() -> None:
    adapter = LatentMotorAdapter()
    stage = LatentMotorCurriculum().stage_for_episode(2000)
    command = torch.zeros((2, 2))
    base = torch.zeros((1, 2))

    with pytest.raises(ValueError, match="shape must match"):
        adapter.decode_torch(command, stage, base_command=base)


def test_custom_curriculum_can_skip_discrete43_phase() -> None:
    curriculum = LatentMotorCurriculum(
        (
            MotorCurriculumPhase("coarse", ExecutionKind.DISCRETE_9, 5),
            MotorCurriculumPhase("continuous", ExecutionKind.CONTINUOUS, 100),
        )
    )

    assert curriculum.stage_for_episode(5).kind is ExecutionKind.DISCRETE_9
    assert curriculum.stage_for_episode(6).kind is ExecutionKind.CONTINUOUS
    assert curriculum.boundaries() == (5,)


def test_control_stage_serialization_uses_string_kind() -> None:
    stage = ControlStage(
        episode=1,
        phase_index=0,
        name="test",
        kind=ExecutionKind.DISCRETE_9,
        phase_episode=1,
        phase_duration=10,
        continuous_mix=0.0,
        action_count=9,
    )

    payload = stage.as_dict()

    assert payload["kind"] == "discrete9"
    assert payload["action_count"] == 9


def test_default_phase_tuple_is_immutable_and_named() -> None:
    assert isinstance(DEFAULT_MOTOR_PHASES, tuple)
    assert [phase.name for phase in DEFAULT_MOTOR_PHASES] == [
        "coarse_discrete_9",
        "fine_discrete_43",
        "anneal_43_to_continuous",
        "continuous",
    ]
