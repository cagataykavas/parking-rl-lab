from __future__ import annotations

import pytest

from parking_rl.curriculum_schedule import (
    CoarseToFineSchedule,
    CurriculumStage,
    blend_ratio,
    stage_for_episode,
)


def test_default_schedule_preserves_historical_thresholds() -> None:
    schedule = CoarseToFineSchedule()

    assert schedule.stage_for_episode(0).name == "coarse"
    assert schedule.stage_for_episode(400).action_mode == "discrete9"
    assert schedule.stage_for_episode(401).action_mode == "discrete43"
    assert schedule.stage_for_episode(1000).name == "fine"
    assert schedule.stage_for_episode(1001).action_mode == "continuous"
    assert schedule.stage_for_episode(50_000_000).name == "continuous"
    assert schedule.boundary_episodes() == (400, 1000)


def test_compatibility_helpers_match_original_experimental_branch_behavior() -> None:
    assert stage_for_episode(1).name == "coarse"
    assert stage_for_episode(700).name == "fine"
    assert stage_for_episode(2000).name == "continuous"

    assert blend_ratio(799) == 0.0
    assert blend_ratio(800) == 0.0
    assert blend_ratio(1000) == pytest.approx(0.5)
    assert blend_ratio(1200) == 1.0
    assert blend_ratio(2000) == 1.0


def test_position_reports_stage_progress_and_remaining_episodes() -> None:
    schedule = CoarseToFineSchedule()

    start = schedule.position(0)
    assert start.stage.name == "coarse"
    assert start.progress == 0.0
    assert start.remaining_episodes == 400

    middle = schedule.position(200)
    assert middle.progress == pytest.approx(0.5)
    assert middle.remaining_episodes == 200

    fine_start = schedule.position(401)
    assert fine_start.stage.name == "fine"
    assert fine_start.stage_start == 401
    assert fine_start.progress == 0.0


def test_action_space_transitions_are_explicit_transfer_boundaries() -> None:
    schedule = CoarseToFineSchedule()

    assert not schedule.transfer_required_at(400)
    assert schedule.transfer_required_at(401)
    assert not schedule.transfer_required_at(999)
    assert schedule.transfer_required_at(1001)


def test_transition_window_interpolates_around_boundary() -> None:
    schedule = CoarseToFineSchedule()
    window = schedule.transition_window(from_stage_name="fine", width=400)

    assert window.start_episode == 800
    assert window.end_episode == 1200
    assert window.from_stage.name == "fine"
    assert window.to_stage.name == "continuous"
    assert window.ratio(799) == 0.0
    assert window.ratio(800) == 0.0
    assert window.ratio(1000) == pytest.approx(0.5)
    assert window.ratio(1200) == 1.0


def test_schedule_serialization_round_trip() -> None:
    schedule = CoarseToFineSchedule(
        (
            CurriculumStage("coarse", "discrete9", 10),
            CurriculumStage("fine", "discrete43", 20),
            CurriculumStage("continuous", "continuous", 100, transition="blend"),
        )
    )

    restored = CoarseToFineSchedule.from_dict(schedule.as_dict())

    assert restored.stages == schedule.stages
    assert restored.boundary_episodes() == (10, 20)


def test_schedule_rejects_invalid_stage_order_and_duplicates() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CoarseToFineSchedule(
            (
                CurriculumStage("first", "discrete9", 100),
                CurriculumStage("second", "discrete43", 100),
            )
        )

    with pytest.raises(ValueError, match="duplicate"):
        CoarseToFineSchedule(
            (
                CurriculumStage("same", "discrete9", 100),
                CurriculumStage("same", "continuous", 200),
            )
        )


def test_schedule_rejects_invalid_episode_and_transition_width() -> None:
    schedule = CoarseToFineSchedule()

    with pytest.raises(ValueError, match="non-negative"):
        schedule.stage_for_episode(-1)

    with pytest.raises(ValueError, match="at least two"):
        schedule.transition_window(from_stage_name="coarse", width=1)

    with pytest.raises(ValueError, match="no outgoing transition"):
        schedule.transition_window(from_stage_name="continuous")

    with pytest.raises(KeyError, match="unknown stage"):
        schedule.transition_window(from_stage_name="missing")

    with pytest.raises(ValueError, match="greater than start"):
        blend_ratio(10, start=20, end=20)
