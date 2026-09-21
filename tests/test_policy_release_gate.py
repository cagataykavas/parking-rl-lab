import json

import pytest
from policy_release_gate import EpisodeResult, ReleasePolicy, evaluate_policy_release


def evidence(*, collision_seed: int | None = None, weak_level: int | None = None):
    rows = []
    for level in range(4):
        for seed in range(10, 15):
            weak = level == weak_level and seed in {10, 11, 12}
            collision = seed == collision_seed
            rows.append(EpisodeResult(seed, level, not (weak or collision), collision, 10.0))
    return rows


def policy(**overrides):
    values = {
        "min_episodes_per_level": 5,
        "min_success_rate": 0.7,
        "min_level_success_rate": 0.55,
        "max_collision_rate": 0.1,
        "min_worst_seed_success_rate": 0.5,
        "min_tail_mean_return": 0.0,
    }
    values.update(overrides)
    return ReleasePolicy(**values)


def test_promotes_complete_safe_evidence_and_is_json_ready():
    report = evaluate_policy_release(evidence(), policy())

    assert report["decision"] == "promote"
    assert report["reasons"] == []
    assert report["worst_seed_success_rate"] == 1.0
    assert json.loads(json.dumps(report))["schema_version"] == 1


def test_rejects_level_regression_hidden_by_aggregate_success():
    report = evaluate_policy_release(evidence(weak_level=3), policy())

    assert report["overall"]["success_rate"] == 0.85
    assert report["decision"] == "reject"
    assert "level_success_below_floor:3" in report["reasons"]


def test_rejects_seed_specific_failure():
    report = evaluate_policy_release(evidence(collision_seed=10), policy(max_collision_rate=0.3))

    assert report["overall"]["success_rate"] == 0.8
    assert report["decision"] == "reject"
    assert "worst_seed_success_below_floor" in report["reasons"]


def test_rejects_collision_budget_and_tail_return():
    rows = evidence(collision_seed=10)
    rows[1] = EpisodeResult(rows[1].seed, rows[1].level, False, True, -100.0)
    report = evaluate_policy_release(rows, policy(max_collision_rate=0.05))

    assert "collision_rate_above_ceiling" in report["reasons"]
    assert "tail_return_below_floor" in report["reasons"]


def test_reports_missing_level_evidence():
    report = evaluate_policy_release([row for row in evidence() if row.level != 2], policy())

    assert report["decision"] == "reject"
    assert "insufficient_level_evidence:2" in report["reasons"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "must not be empty"),
        ([EpisodeResult(1, 0, True, True, 1.0)], "both successful and collided"),
        ([EpisodeResult(1, 0, False, False, float("nan"))], "must be finite"),
        (
            [EpisodeResult(1, 0, False, False, 1.0)] * 2,
            "duplicate seed/level",
        ),
        ([EpisodeResult(1, 9, False, False, 1.0)], "unexpected curriculum level"),
    ],
)
def test_rejects_malformed_evidence(rows, message):
    with pytest.raises(ValueError, match=message):
        evaluate_policy_release(rows, policy())


def test_rejects_invalid_policy():
    with pytest.raises(ValueError, match="tail_fraction"):
        evaluate_policy_release(evidence(), policy(tail_fraction=0.0))
