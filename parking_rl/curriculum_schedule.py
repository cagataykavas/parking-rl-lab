from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ActionMode = Literal["discrete9", "discrete43", "continuous"]
TransitionKind = Literal["hard", "blend"]


@dataclass(frozen=True)
class CurriculumStage:
    """One phase of a coarse-to-fine parking curriculum.

    `until_episode` is inclusive. The schedule intentionally describes *environment
    control resolution*, not a promise that a single Stable-Baselines checkpoint can
    be reused across incompatible action spaces. A caller must explicitly decide how
    model transfer is performed at stage boundaries.
    """

    name: str
    action_mode: ActionMode
    until_episode: int
    transition: TransitionKind = "hard"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stage name must not be empty")
        if self.until_episode < 0:
            raise ValueError("until_episode must be non-negative")
        if self.action_mode not in {"discrete9", "discrete43", "continuous"}:
            raise ValueError(f"unsupported action mode: {self.action_mode}")
        if self.transition not in {"hard", "blend"}:
            raise ValueError(f"unsupported transition: {self.transition}")


DEFAULT_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        "coarse",
        "discrete9",
        400,
        notes="Learn broad approach/reverse decisions with the compact action set.",
    ),
    CurriculumStage(
        "fine",
        "discrete43",
        1000,
        transition="hard",
        notes="Increase steering/throttle resolution for alignment and settling.",
    ),
    CurriculumStage(
        "continuous",
        "continuous",
        10_000_000,
        transition="blend",
        notes="Experimental continuous-control phase; requires explicit model transfer.",
    ),
)


@dataclass(frozen=True)
class CurriculumPosition:
    episode: int
    stage_index: int
    stage: CurriculumStage
    stage_start: int
    stage_end: int
    progress: float

    @property
    def remaining_episodes(self) -> int:
        return max(0, self.stage_end - self.episode)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stage"] = asdict(self.stage)
        return payload


@dataclass(frozen=True)
class TransitionWindow:
    from_stage: CurriculumStage
    to_stage: CurriculumStage
    start_episode: int
    end_episode: int

    def __post_init__(self) -> None:
        if self.start_episode < 0:
            raise ValueError("transition start must be non-negative")
        if self.end_episode <= self.start_episode:
            raise ValueError("transition end must be greater than transition start")

    def ratio(self, episode: int) -> float:
        """Return the normalized interpolation ratio for a transition window."""
        if episode <= self.start_episode:
            return 0.0
        if episode >= self.end_episode:
            return 1.0
        return (episode - self.start_episode) / (self.end_episode - self.start_episode)


class CoarseToFineSchedule:
    """Validated schedule recovered from the historical `experimental` branch.

    The old branch contained only three stage thresholds and a free function for a
    blend ratio. This class keeps that idea while making stage ordering, transition
    windows, serialization and resume semantics explicit and testable.
    """

    def __init__(self, stages: tuple[CurriculumStage, ...] = DEFAULT_STAGES) -> None:
        if not stages:
            raise ValueError("curriculum must contain at least one stage")
        self._stages = tuple(stages)
        self._validate()

    @property
    def stages(self) -> tuple[CurriculumStage, ...]:
        return self._stages

    def _validate(self) -> None:
        names: set[str] = set()
        previous_until = -1
        for stage in self._stages:
            if stage.name in names:
                raise ValueError(f"duplicate curriculum stage name: {stage.name}")
            names.add(stage.name)
            if stage.until_episode <= previous_until:
                raise ValueError("stage thresholds must be strictly increasing")
            previous_until = stage.until_episode

    def stage_index_for_episode(self, episode: int) -> int:
        if episode < 0:
            raise ValueError("episode must be non-negative")
        for index, stage in enumerate(self._stages):
            if episode <= stage.until_episode:
                return index
        return len(self._stages) - 1

    def stage_for_episode(self, episode: int) -> CurriculumStage:
        return self._stages[self.stage_index_for_episode(episode)]

    def position(self, episode: int) -> CurriculumPosition:
        index = self.stage_index_for_episode(episode)
        stage = self._stages[index]
        start = 0 if index == 0 else self._stages[index - 1].until_episode + 1
        end = stage.until_episode
        width = max(1, end - start)
        progress = min(1.0, max(0.0, (episode - start) / width))
        return CurriculumPosition(
            episode=episode,
            stage_index=index,
            stage=stage,
            stage_start=start,
            stage_end=end,
            progress=progress,
        )

    def next_stage(self, episode: int) -> CurriculumStage | None:
        index = self.stage_index_for_episode(episode)
        if index + 1 >= len(self._stages):
            return None
        return self._stages[index + 1]

    def boundary_episodes(self) -> tuple[int, ...]:
        return tuple(stage.until_episode for stage in self._stages[:-1])

    def transition_window(
        self,
        *,
        from_stage_name: str,
        width: int = 400,
    ) -> TransitionWindow:
        if width < 2:
            raise ValueError("transition width must be at least two episodes")
        index = next(
            (i for i, stage in enumerate(self._stages) if stage.name == from_stage_name),
            None,
        )
        if index is None:
            raise KeyError(f"unknown stage: {from_stage_name}")
        if index + 1 >= len(self._stages):
            raise ValueError("final stage has no outgoing transition")

        boundary = self._stages[index].until_episode
        half = width // 2
        return TransitionWindow(
            from_stage=self._stages[index],
            to_stage=self._stages[index + 1],
            start_episode=max(0, boundary - half),
            end_episode=boundary + (width - half),
        )

    def action_mode_changed_at(self, episode: int) -> bool:
        """Return True on the first episode of a stage with a new action mode."""
        if episode <= 0:
            return False
        current = self.stage_for_episode(episode)
        previous = self.stage_for_episode(episode - 1)
        return current.action_mode != previous.action_mode

    def transfer_required_at(self, episode: int) -> bool:
        """Alias with model-training semantics made explicit.

        Stable-Baselines policies generally cannot be blindly reused when the action
        space changes from Discrete(9) to Discrete(43) or Box(2). The training layer
        should create a new model or an explicit transfer adapter at these episodes.
        """
        return self.action_mode_changed_at(episode)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "stages": [asdict(stage) for stage in self._stages],
            "boundaries": list(self.boundary_episodes()),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CoarseToFineSchedule:
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list):
            raise TypeError("payload must contain a stages list")
        stages: list[CurriculumStage] = []
        for item in raw_stages:
            if not isinstance(item, dict):
                raise TypeError("each stage must be an object")
            stages.append(
                CurriculumStage(
                    name=str(item["name"]),
                    action_mode=str(item["action_mode"]),  # type: ignore[arg-type]
                    until_episode=int(item["until_episode"]),
                    transition=str(item.get("transition", "hard")),  # type: ignore[arg-type]
                    notes=str(item.get("notes", "")),
                )
            )
        return cls(tuple(stages))


def stage_for_episode(
    episode: int,
    stages: tuple[CurriculumStage, ...] = DEFAULT_STAGES,
) -> CurriculumStage:
    """Compatibility helper for the original experimental-branch API."""
    return CoarseToFineSchedule(stages).stage_for_episode(episode)


def blend_ratio(episode: int, start: int = 800, end: int = 1200) -> float:
    """Compatibility helper preserving the historical annealing calculation."""
    if end <= start:
        raise ValueError("end must be greater than start")
    if episode <= start:
        return 0.0
    if episode >= end:
        return 1.0
    return (episode - start) / (end - start)
