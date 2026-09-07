from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from parking_env_v2 import ParkingEnvV2, ParkingV2Config
from parking_rl.agents.dqn import (
    DoubleDQNAgent,
    DQNConfig,
    evaluate_dqn,
    train_dqn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate the repository's inspectable native Dueling "
            "Double-DQN baseline. DQN intentionally supports fixed discrete "
            "parking actions only."
        )
    )
    parser.add_argument(
        "--action-mode",
        choices=("discrete9", "discrete43"),
        default="discrete9",
    )
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum-level", type=int, choices=range(4), default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/native_dqn"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-warmup", type=int, default=1000)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--target-sync-every", type=int, default=1000)
    return parser


def config_from_args(args: argparse.Namespace) -> DQNConfig:
    config = DQNConfig(
        action_mode=args.action_mode,
        episodes=args.episodes,
        seed=args.seed,
        curriculum_level=args.curriculum_level,
        batch_size=args.batch_size,
        replay_warmup=args.replay_warmup,
        replay_capacity=args.replay_capacity,
        target_sync_every=args.target_sync_every,
    )
    if args.device:
        config = replace(config, device=args.device)
    return config


def load_agent(config: DQNConfig, checkpoint: Path) -> DoubleDQNAgent:
    env = ParkingEnvV2(
        ParkingV2Config(
            action_mode=config.normalized_action_mode,
            curriculum_level=config.curriculum_level,
        )
    )
    observation, _ = env.reset(seed=config.seed)
    action_count = getattr(env.action_space, "n", None)
    if action_count is None:
        raise RuntimeError("native DQN requires a discrete environment")
    agent = DoubleDQNAgent(
        state_dim=observation.shape[0],
        action_dim=int(action_count),
        config=config,
    )
    agent.load(checkpoint, load_optimizer=False)
    return agent


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is not None:
        agent = load_agent(config, args.checkpoint)
        evaluation = evaluate_dqn(
            agent,
            config,
            episodes=args.eval_episodes,
        )
        payload = {
            "mode": "evaluation",
            "checkpoint": str(args.checkpoint),
            "evaluation": evaluation.as_dict(),
        }
        (args.output / "evaluation.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
        return

    result = train_dqn(config, output_dir=args.output)
    checkpoint = result.best_checkpoint or result.latest_checkpoint
    payload: dict[str, object] = {
        "mode": "training",
        "training": {
            "episodes": len(result.episodes),
            "success_rate": result.success_rate,
            "collision_rate": result.collision_rate,
            "best_reward": result.best_reward,
            "best_successful_reward": result.best_successful_reward,
            "checkpoint": checkpoint,
        },
    }
    if checkpoint is not None:
        agent = load_agent(config, Path(checkpoint))
        evaluation = evaluate_dqn(
            agent,
            config,
            episodes=args.eval_episodes,
        )
        payload["evaluation"] = evaluation.as_dict()
        (args.output / "evaluation.json").write_text(
            json.dumps(evaluation.as_dict(), indent=2), encoding="utf-8"
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
