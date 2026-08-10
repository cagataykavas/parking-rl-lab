from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    action_mode: str
    until_episode: int


DEFAULT_CURRICULUM = [
    CurriculumStage("coarse", "discrete9", 400),
    CurriculumStage("fine", "discrete43", 1000),
    CurriculumStage("continuous", "continuous", 10_000_000),
]


def stage_for_episode(episode: int, stages=DEFAULT_CURRICULUM) -> CurriculumStage:
    for stage in stages:
        if episode <= stage.until_episode:
            return stage
    return stages[-1]


def blend_ratio(episode: int, start: int = 800, end: int = 1200) -> float:
    """Anneal from discrete execution toward continuous execution."""
    if episode <= start:
        return 0.0
    if episode >= end:
        return 1.0
    return (episode - start) / (end - start)
