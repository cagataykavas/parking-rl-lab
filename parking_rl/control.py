from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
import torch

from parking_env import build_action_table


class ExecutionKind(str, Enum):
    DISCRETE_9 = "discrete9"
    DISCRETE_43 = "discrete43"
    ANNEAL_43_TO_CONTINUOUS = "anneal43"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class MotorCurriculumPhase:
    name: str
    kind: ExecutionKind
    episodes: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("phase name must not be empty")
        if self.episodes < 1:
            raise ValueError("phase duration must be positive")


DEFAULT_MOTOR_PHASES: tuple[MotorCurriculumPhase, ...] = (
    MotorCurriculumPhase("coarse_discrete_9", ExecutionKind.DISCRETE_9, 400),
    MotorCurriculumPhase("fine_discrete_43", ExecutionKind.DISCRETE_43, 600),
    MotorCurriculumPhase(
        "anneal_43_to_continuous",
        ExecutionKind.ANNEAL_43_TO_CONTINUOUS,
        600,
    ),
    MotorCurriculumPhase("continuous", ExecutionKind.CONTINUOUS, 10_000_000),
)


@dataclass(frozen=True)
class ControlStage:
    episode: int
    phase_index: int
    name: str
    kind: ExecutionKind
    phase_episode: int
    phase_duration: int
    continuous_mix: float
    action_count: int | None

    @property
    def progress(self) -> float:
        if self.phase_duration <= 1:
            return 1.0
        return min(1.0, max(0.0, (self.phase_episode - 1) / (self.phase_duration - 1)))

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


class LatentMotorCurriculum:
    """Keep a stable 2-D motor policy while changing execution resolution.

    The recovered historical implementation used this idea to avoid an important
    transfer problem: a categorical 9-action head, categorical 43-action head, and
    2-D continuous head do not have compatible checkpoint shapes. Here the learned
    policy always emits ``[steering, throttle]`` in ``[-1, 1]``. The execution layer
    first snaps that command to 9 actions, then 43 actions, gradually removes the
    43-action quantization, and finally executes it continuously.
    """

    def __init__(
        self,
        phases: tuple[MotorCurriculumPhase, ...] = DEFAULT_MOTOR_PHASES,
    ) -> None:
        if not phases:
            raise ValueError("motor curriculum must contain at least one phase")
        names = [phase.name for phase in phases]
        if len(names) != len(set(names)):
            raise ValueError("motor curriculum phase names must be unique")
        self._phases = tuple(phases)
        self._ends = self._build_endpoints()

    @property
    def phases(self) -> tuple[MotorCurriculumPhase, ...]:
        return self._phases

    def _build_endpoints(self) -> tuple[int, ...]:
        total = 0
        ends: list[int] = []
        for phase in self._phases:
            total += phase.episodes
            ends.append(total)
        return tuple(ends)

    def stage_for_episode(self, episode: int) -> ControlStage:
        if episode < 1:
            raise ValueError("episode numbering starts at one")
        phase_start = 1
        for index, (phase, phase_end) in enumerate(
            zip(self._phases, self._ends, strict=True)
        ):
            if episode <= phase_end:
                phase_episode = episode - phase_start + 1
                if phase.kind is ExecutionKind.ANNEAL_43_TO_CONTINUOUS:
                    denominator = max(1, phase.episodes - 1)
                    mix = (phase_episode - 1) / denominator
                    action_count: int | None = 43
                elif phase.kind is ExecutionKind.DISCRETE_9:
                    mix = 0.0
                    action_count = 9
                elif phase.kind is ExecutionKind.DISCRETE_43:
                    mix = 0.0
                    action_count = 43
                else:
                    mix = 1.0
                    action_count = None
                return ControlStage(
                    episode=episode,
                    phase_index=index,
                    name=phase.name,
                    kind=phase.kind,
                    phase_episode=phase_episode,
                    phase_duration=phase.episodes,
                    continuous_mix=float(mix),
                    action_count=action_count,
                )
            phase_start = phase_end + 1

        final = self._phases[-1]
        return ControlStage(
            episode=episode,
            phase_index=len(self._phases) - 1,
            name=final.name,
            kind=ExecutionKind.CONTINUOUS,
            phase_episode=episode - self._ends[-2] if len(self._ends) > 1 else episode,
            phase_duration=final.episodes,
            continuous_mix=1.0,
            action_count=None,
        )

    def boundaries(self) -> tuple[int, ...]:
        return self._ends[:-1]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_contract": {
                "shape": [2],
                "fields": ["steering", "throttle"],
                "range": [-1.0, 1.0],
            },
            "phases": [
                {
                    "name": phase.name,
                    "kind": phase.kind.value,
                    "episodes": phase.episodes,
                }
                for phase in self._phases
            ],
            "boundaries": list(self.boundaries()),
        }


@dataclass(frozen=True)
class QuantizedAction:
    requested: np.ndarray
    executed: np.ndarray
    discrete_index: int | None
    continuous_mix: float
    kind: ExecutionKind

    def quantization_error(self) -> float:
        return float(np.linalg.norm(self.executed - self.requested))


class LatentMotorAdapter:
    """Map stable latent motor commands to curriculum-dependent execution commands."""

    def __init__(
        self,
        curriculum: LatentMotorCurriculum | None = None,
        *,
        residual_scale: float = 0.35,
    ) -> None:
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError("residual_scale must be in [0, 1]")
        self.curriculum = curriculum or LatentMotorCurriculum()
        self.residual_scale = float(residual_scale)
        self.table9 = np.asarray(build_action_table(9), dtype=np.float32)
        self.table43 = np.asarray(build_action_table(43), dtype=np.float32)

    @staticmethod
    def normalize(command: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
        values = np.asarray(command, dtype=np.float32).reshape(-1)
        if values.shape != (2,):
            raise ValueError("motor command must contain steering and throttle")
        return np.clip(values, -1.0, 1.0)

    @staticmethod
    def nearest_index(command: np.ndarray, table: np.ndarray) -> int:
        values = np.asarray(command, dtype=np.float32).reshape(2)
        candidates = np.asarray(table, dtype=np.float32)
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError("action table must have shape (n, 2)")
        squared_distance = np.sum((candidates - values[None, :]) ** 2, axis=1)
        return int(np.argmin(squared_distance))

    def _table_for_stage(self, stage: ControlStage) -> np.ndarray | None:
        if stage.kind is ExecutionKind.DISCRETE_9:
            return self.table9
        if stage.kind in {
            ExecutionKind.DISCRETE_43,
            ExecutionKind.ANNEAL_43_TO_CONTINUOUS,
        }:
            return self.table43
        return None

    def decode(
        self,
        command: np.ndarray | list[float] | tuple[float, float],
        stage: ControlStage,
        *,
        base_command: np.ndarray | None = None,
    ) -> QuantizedAction:
        requested = self.normalize(command)
        if base_command is not None:
            base = self.normalize(base_command)
            requested = np.clip(
                base + self.residual_scale * requested,
                -1.0,
                1.0,
            ).astype(np.float32)

        table = self._table_for_stage(stage)
        if table is None:
            return QuantizedAction(
                requested=requested.copy(),
                executed=requested.copy(),
                discrete_index=None,
                continuous_mix=1.0,
                kind=stage.kind,
            )

        index = self.nearest_index(requested, table)
        discrete = table[index].copy()
        if stage.kind is ExecutionKind.ANNEAL_43_TO_CONTINUOUS:
            executed = (
                (1.0 - stage.continuous_mix) * discrete
                + stage.continuous_mix * requested
            )
        else:
            executed = discrete
        executed = np.clip(executed, -1.0, 1.0).astype(np.float32)
        return QuantizedAction(
            requested=requested.copy(),
            executed=executed,
            discrete_index=index,
            continuous_mix=stage.continuous_mix,
            kind=stage.kind,
        )

    def discrete_env_index(
        self,
        command: np.ndarray | list[float] | tuple[float, float],
        *,
        action_mode: str,
    ) -> int:
        """Convert a latent command to an index for fixed discrete Gym environments."""
        normalized = self.normalize(command)
        mode = action_mode.lower()
        if mode == "discrete9":
            return self.nearest_index(normalized, self.table9)
        if mode == "discrete43":
            return self.nearest_index(normalized, self.table43)
        raise ValueError("discrete_env_index requires discrete9 or discrete43")

    @staticmethod
    def _nearest_torch(command: torch.Tensor, table: np.ndarray) -> torch.Tensor:
        if command.ndim != 2 or command.shape[1] != 2:
            raise ValueError("torch motor commands must have shape (batch, 2)")
        table_tensor = torch.as_tensor(
            table, dtype=command.dtype, device=command.device
        )
        distances = torch.sum(
            (command.unsqueeze(1) - table_tensor.unsqueeze(0)) ** 2,
            dim=-1,
        )
        nearest = table_tensor[torch.argmin(distances, dim=1)]
        # Straight-through estimator: forward is quantized; gradient is identity.
        return command + (nearest - command).detach()

    def decode_torch(
        self,
        command: torch.Tensor,
        stage: ControlStage,
        *,
        base_command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = torch.clamp(command, -1.0, 1.0)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("torch motor commands must have shape (batch, 2)")
        if base_command is not None:
            if base_command.shape != values.shape:
                raise ValueError("base command shape must match policy command")
            values = torch.clamp(
                base_command + self.residual_scale * values,
                -1.0,
                1.0,
            )

        if stage.kind is ExecutionKind.DISCRETE_9:
            return self._nearest_torch(values, self.table9)
        if stage.kind is ExecutionKind.DISCRETE_43:
            return self._nearest_torch(values, self.table43)
        if stage.kind is ExecutionKind.ANNEAL_43_TO_CONTINUOUS:
            quantized = self._nearest_torch(values, self.table43)
            return torch.clamp(
                (1.0 - stage.continuous_mix) * quantized
                + stage.continuous_mix * values,
                -1.0,
                1.0,
            )
        return values
