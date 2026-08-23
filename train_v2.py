from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor

from curriculum import SuccessCurriculumCallback
from parking_env_v2 import ParkingEnvV2, ParkingV2Config

ALGORITHMS = {"dqn": DQN, "ppo": PPO, "sac": SAC}


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_env(action_mode: str, curriculum_level: int, seed: int):
    env = ParkingEnvV2(
        ParkingV2Config(action_mode=action_mode, curriculum_level=curriculum_level)
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def build_model(config: dict, env):
    algorithm = config["algorithm"].lower()
    action_mode = config["action_mode"].lower()
    if algorithm == "dqn" and action_mode == "continuous":
        raise ValueError("DQN requires a discrete action space")
    if algorithm == "sac" and action_mode != "continuous":
        raise ValueError("SAC requires continuous action mode")
    cls = ALGORITHMS[algorithm]
    return cls(
        "MlpPolicy",
        env,
        seed=int(config.get("seed", 42)),
        verbose=1,
        tensorboard_log=str(Path("artifacts") / "tensorboard"),
        **config.get("model", {}),
    )


def evaluate(model, action_mode: str, level: int, seed: int, episodes: int = 30) -> dict[str, float]:
    env = ParkingEnvV2(ParkingV2Config(action_mode=action_mode, curriculum_level=level))
    rewards: list[float] = []
    successes = collisions = steps_total = 0
    final_distances: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        reward_sum = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            steps_total += 1
            done = terminated or truncated
        rewards.append(reward_sum)
        successes += int(info["success"])
        collisions += int(info["collision"])
        final_distances.append(float(info["distance"]))
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "mean_reward": sum(rewards) / episodes,
        "mean_steps": steps_total / episodes,
        "mean_final_distance": sum(final_distances) / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 parking policies from YAML profiles.")
    parser.add_argument("--config", type=Path, default=Path("configs/stable_ppo.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    curriculum = config.get("curriculum", {})
    initial_level = int(curriculum.get("initial_level", 0))
    env = make_env(config["action_mode"], initial_level, seed)
    model = build_model(config, env)

    callbacks = []
    if curriculum.get("enabled", False):
        callbacks.append(
            SuccessCurriculumCallback(
                initial_level=initial_level,
                max_level=int(curriculum.get("max_level", 3)),
                window=int(curriculum.get("window", 100)),
                minimum_episodes=int(curriculum.get("minimum_episodes", 60)),
                promotion_success_rate=float(curriculum.get("promotion_success_rate", 0.68)),
            )
        )

    model.learn(total_timesteps=int(config["timesteps"]), callback=callbacks or None)
    run_dir = args.output / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = {
        f"level_{level}": evaluate(
            model,
            config["action_mode"],
            level,
            seed + 10000 + level * 1000,
            args.eval_episodes,
        )
        for level in range(4)
    }
    (run_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
