from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_rl.baselines import GreedyParkingController, RandomPolicy, ZeroPolicy
from parking_rl.evaluation import EvaluationConfig, evaluate_policies, write_report


def _fmt_interval(metric: dict[str, float]) -> str:
    mean = metric["mean"]
    lower = metric["lower"]
    upper = metric["upper"]
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"


def markdown_report(report: dict[str, object]) -> str:
    note = (
        "> Generated from deterministic environment rollouts. These are "
        "reference-controller results, not trained PPO/DQN/SAC checkpoint claims."
    )
    lines = [
        "# Parking RL reference-policy benchmark",
        "",
        note,
        "",
        "| Policy | Success | Collision | Timeout | Reward | Final pose score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy_report in report["reports"]:
        metrics = policy_report["overall"]["metrics"]
        lines.append(
            "| {policy} | {success} | {collision} | {timeout} | {reward} | {pose} |".format(
                policy=policy_report["policy"],
                success=_fmt_interval(metrics["success_rate"]),
                collision=_fmt_interval(metrics["collision_rate"]),
                timeout=_fmt_interval(metrics["timeout_rate"]),
                reward=_fmt_interval(metrics["reward"]),
                pose=_fmt_interval(metrics["final_pose_score"]),
            )
        )

    lines.extend(["", "## Curriculum breakdown", ""])
    for policy_report in report["reports"]:
        lines.extend([f"### {policy_report['policy']}", ""])
        lines.append("| Level | Success | Collision | Final distance | Heading error |")
        lines.append("|---|---:|---:|---:|---:|")
        for level_name, level_report in policy_report["by_curriculum_level"].items():
            metrics = level_report["metrics"]
            lines.append(
                "| {level} | {success} | {collision} | {distance} | {heading} |".format(
                    level=level_name,
                    success=_fmt_interval(metrics["success_rate"]),
                    collision=_fmt_interval(metrics["collision_rate"]),
                    distance=_fmt_interval(metrics["final_distance"]),
                    heading=_fmt_interval(metrics["final_heading_error_deg"]),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic reference policies in ParkingEnvV2."
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_benchmark.json"))
    parser.add_argument("--markdown", type=Path, default=Path("artifacts/baseline_benchmark.md"))
    parser.add_argument("--episodes-per-seed", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()

    config = EvaluationConfig(
        action_mode="continuous",
        levels=(0, 1, 2, 3),
        seeds=tuple(args.seeds),
        episodes_per_seed=args.episodes_per_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    policies = [ZeroPolicy(), RandomPolicy(), GreedyParkingController()]
    report = evaluate_policies(policies, config)
    write_report(report, args.output)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_report(report), encoding="utf-8")

    compact = {
        policy_report["policy"]: {
            name: metric["mean"]
            for name, metric in policy_report["overall"]["metrics"].items()
            if name
            in {
                "success_rate",
                "collision_rate",
                "timeout_rate",
                "reward",
                "final_pose_score",
            }
        }
        for policy_report in report["reports"]
    }
    print(json.dumps(compact, indent=2))
    print(f"Wrote {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()
