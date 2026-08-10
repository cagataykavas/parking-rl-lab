from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from parking_env import ParkingConfig, ParkingEnv


def make_env(action_mode: str, seed: int):
    env = Monitor(ParkingEnv(ParkingConfig(action_mode=action_mode)))
    env.reset(seed=seed)
    return env


def build_model(algorithm: str, action_mode: str, env, seed: int):
    algo = algorithm.lower()
    if algo == "dqn":
        if action_mode == "continuous":
            raise ValueError("DQN requires a discrete action space. Use discrete9 or discrete43.")
        return DQN(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=2_000,
            batch_size=256,
            gamma=0.99,
            target_update_interval=1_000,
            exploration_fraction=0.35,
            exploration_final_eps=0.05,
            seed=seed,
            verbose=1,
        )
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            seed=seed,
            verbose=1,
        )
    if algo == "sac":
        if action_mode != "continuous":
            raise ValueError("SAC requires continuous action mode.")
        return SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=2_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            seed=seed,
            verbose=1,
        )
    raise ValueError("algorithm must be dqn, ppo or sac")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a parking RL baseline")
    parser.add_argument("--algorithm", choices=["dqn", "ppo", "sac"], default="ppo")
    parser.add_argument("--action-mode", choices=["discrete9", "discrete43", "continuous"], default="discrete9")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    env = make_env(args.action_mode, args.seed)
    model = build_model(args.algorithm, args.action_mode, env, args.seed)
    model.learn(total_timesteps=args.timesteps)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"{args.algorithm}_{args.action_mode}"
    model.save(checkpoint)

    eval_env = make_env(args.action_mode, args.seed + 10_000)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True)
    print(f"mean_reward={mean_reward:.3f} std_reward={std_reward:.3f}")
    print(f"checkpoint={checkpoint}.zip")


if __name__ == "__main__":
    main()
