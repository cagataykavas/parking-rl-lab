from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from parking_env_v2 import ParkingEnvV2, ParkingV2Config


@dataclass(frozen=True)
class DQNConfig:
    """Configuration for the native Dueling Double-DQN baseline.

    The implementation intentionally supports only fixed discrete action spaces.
    Native continuous parking belongs to actor-critic methods such as SAC/PPO;
    coercing DQN into a continuous policy would make the comparison misleading.
    """

    action_mode: str = "discrete9"
    episodes: int = 1200
    seed: int = 42
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    replay_warmup: int = 1_000
    train_every: int = 1
    target_sync_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000
    hidden_size: int = 256
    max_grad_norm: float = 1.0
    checkpoint_every: int = 100
    curriculum_level: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        mode = self.action_mode.lower()
        if mode not in {"discrete9", "discrete43"}:
            raise ValueError("native DQN supports only discrete9 and discrete43")
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.replay_warmup < self.batch_size:
            raise ValueError("replay_warmup must be at least batch_size")
        if self.train_every < 1 or self.target_sync_every < 1:
            raise ValueError("update intervals must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_steps < 1:
            raise ValueError("epsilon_decay_steps must be positive")
        if self.hidden_size < 16:
            raise ValueError("hidden_size must be at least 16")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.curriculum_level not in {0, 1, 2, 3}:
            raise ValueError("curriculum_level must be 0..3")

    @property
    def normalized_action_mode(self) -> str:
        return self.action_mode.lower()


@dataclass(frozen=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    """Bounded replay storage with its own deterministic sampler RNG."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.capacity
        )
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        next_state_array = np.asarray(next_state, dtype=np.float32).reshape(-1).copy()
        if state_array.shape != next_state_array.shape:
            raise ValueError("state and next_state dimensions must match")
        self._items.append(
            (state_array, int(action), float(reward), next_state_array, bool(terminated))
        )

    def sample(self, batch_size: int, *, device: torch.device) -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._items):
            raise ValueError("cannot sample more transitions than are stored")
        items = self._rng.sample(list(self._items), batch_size)
        states, actions, rewards, next_states, terminated = zip(*items, strict=True)
        return ReplayBatch(
            states=torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            actions=torch.as_tensor(actions, dtype=torch.long, device=device),
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
            next_states=torch.as_tensor(
                np.stack(next_states), dtype=torch.float32, device=device
            ),
            terminated=torch.as_tensor(
                terminated, dtype=torch.float32, device=device
            ),
        )

    def snapshot(self) -> list[dict[str, object]]:
        """Return JSON-friendly replay metadata without serializing large arrays."""
        return [
            {
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "state_norm": float(np.linalg.norm(state)),
                "next_state_norm": float(np.linalg.norm(next_state)),
            }
            for state, action, reward, next_state, terminated in self._items
        ]


class DuelingQNetwork(nn.Module):
    """MLP dueling architecture: Q(s,a)=V(s)+A(s,a)-mean(A)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        if state_dim < 1 or action_dim < 2:
            raise ValueError("invalid network dimensions")
        if hidden_size < 16:
            raise ValueError("hidden_size must be at least 16")

        half = max(16, hidden_size // 2)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Linear(half, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.encoder(states)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


@dataclass(frozen=True)
class DQNUpdate:
    loss: float
    mean_abs_td_error: float
    mean_q: float
    mean_target: float
    grad_norm: float
    epsilon: float
    gradient_step: int


class DoubleDQNAgent:
    """Inspectable native Dueling Double-DQN implementation."""

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = int(action_dim)
        self.online = DuelingQNetwork(
            state_dim, self.action_dim, config.hidden_size
        ).to(self.device)
        self.target = DuelingQNetwork(
            state_dim, self.action_dim, config.hidden_size
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=config.learning_rate
        )
        self.replay = ReplayBuffer(config.replay_capacity, seed=config.seed + 11)
        self.environment_steps = 0
        self.gradient_steps = 0
        self._action_rng = random.Random(config.seed + 23)

    def epsilon(self) -> float:
        progress = min(
            1.0,
            self.environment_steps / max(1, self.config.epsilon_decay_steps),
        )
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    @torch.no_grad()
    def greedy_actions(self, observations: np.ndarray) -> np.ndarray:
        materialized = np.asarray(observations, dtype=np.float32)
        if materialized.ndim == 1:
            materialized = materialized[None, :]
        if materialized.ndim != 2:
            raise ValueError("observations must be a vector or matrix")
        tensor = torch.as_tensor(
            materialized, dtype=torch.float32, device=self.device
        )
        return (
            torch.argmax(self.online(tensor), dim=-1)
            .cpu()
            .numpy()
            .astype(np.int64)
        )

    def act(self, observation: np.ndarray, *, deterministic: bool = False) -> int:
        if not deterministic and self._action_rng.random() < self.epsilon():
            return self._action_rng.randrange(self.action_dim)
        return int(self.greedy_actions(observation)[0])

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        if not 0 <= int(action) < self.action_dim:
            raise ValueError("action index outside configured action space")
        self.replay.add(state, action, reward, next_state, terminated)
        self.environment_steps += 1

    def _double_dqn_targets(self, batch: ReplayBatch) -> torch.Tensor:
        """Select next actions online, evaluate those actions with target net."""
        with torch.no_grad():
            next_actions = torch.argmax(self.online(batch.next_states), dim=1)
            target_q = self.target(batch.next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            return batch.rewards + self.config.gamma * (
                1.0 - batch.terminated
            ) * target_q

    def ready_to_update(self) -> bool:
        minimum = max(self.config.batch_size, self.config.replay_warmup)
        return (
            len(self.replay) >= minimum
            and self.environment_steps % self.config.train_every == 0
        )

    def update(self) -> DQNUpdate | None:
        if not self.ready_to_update():
            return None

        batch = self.replay.sample(self.config.batch_size, device=self.device)
        q_values = self.online(batch.states).gather(
            1, batch.actions.unsqueeze(1)
        ).squeeze(1)
        targets = self._double_dqn_targets(batch)
        loss = functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.gradient_steps += 1

        if self.gradient_steps % self.config.target_sync_every == 0:
            self.sync_target()

        with torch.no_grad():
            td_error = torch.mean(torch.abs(targets - q_values))
            mean_q = torch.mean(q_values)
            mean_target = torch.mean(targets)

        return DQNUpdate(
            loss=float(loss.detach().cpu()),
            mean_abs_td_error=float(td_error.cpu()),
            mean_q=float(mean_q.cpu()),
            mean_target=float(mean_target.cpu()),
            grad_norm=float(grad_norm),
            epsilon=float(self.epsilon()),
            gradient_step=self.gradient_steps,
        )

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "algorithm": "dueling-double-dqn",
                "config": asdict(self.config),
                "action_dim": self.action_dim,
                "environment_steps": self.environment_steps,
                "gradient_steps": self.gradient_steps,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metadata": metadata or {},
            },
            destination,
        )
        return destination

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, object]:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        if checkpoint.get("algorithm") != "dueling-double-dqn":
            raise ValueError("checkpoint algorithm does not match native DQN")
        if int(checkpoint["action_dim"]) != self.action_dim:
            raise ValueError("checkpoint action dimension does not match agent")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.environment_steps = int(checkpoint.get("environment_steps", 0))
        self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        return dict(checkpoint.get("metadata", {}))


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    seed: int
    reward: float
    steps: int
    success: bool
    collision: bool
    final_distance: float
    epsilon: float
    gradient_steps: int
    mean_update_loss: float | None


@dataclass(frozen=True)
class DQNTrainingResult:
    config: DQNConfig
    episodes: tuple[EpisodeRecord, ...]
    best_reward: float
    best_successful_reward: float | None
    latest_checkpoint: str | None
    best_checkpoint: str | None

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(record.success for record in self.episodes) / len(self.episodes)

    @property
    def collision_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(record.collision for record in self.episodes) / len(self.episodes)

    def as_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "episodes": [asdict(record) for record in self.episodes],
            "best_reward": self.best_reward,
            "best_successful_reward": self.best_successful_reward,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "latest_checkpoint": self.latest_checkpoint,
            "best_checkpoint": self.best_checkpoint,
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2), encoding="utf-8"
        )
        return destination


@dataclass(frozen=True)
class DQNEvaluation:
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_reward: float
    reward_std: float
    mean_steps: float
    mean_final_distance: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(config: DQNConfig) -> ParkingEnvV2:
    return ParkingEnvV2(
        ParkingV2Config(
            action_mode=config.normalized_action_mode,
            curriculum_level=config.curriculum_level,
        )
    )


def _action_dim(env: ParkingEnvV2) -> int:
    action_count = getattr(env.action_space, "n", None)
    if action_count is None:
        raise ValueError("native DQN requires a discrete Gym action space")
    return int(action_count)


def train_dqn(
    config: DQNConfig,
    *,
    output_dir: str | Path | None = None,
) -> DQNTrainingResult:
    """Train the native baseline and return episode-level measured evidence."""
    _seed_everything(config.seed)
    env = _make_env(config)
    first_observation, _ = env.reset(seed=config.seed)
    agent = DoubleDQNAgent(
        state_dim=int(first_observation.shape[0]),
        action_dim=_action_dim(env),
        config=config,
    )

    output = Path(output_dir) if output_dir is not None else None
    latest_checkpoint: Path | None = None
    best_checkpoint: Path | None = None
    best_reward = -math.inf
    best_successful_reward: float | None = None
    records: list[EpisodeRecord] = []

    for episode in range(1, config.episodes + 1):
        episode_seed = config.seed + episode * 1_003
        observation, _ = env.reset(seed=episode_seed)
        reward_sum = 0.0
        update_losses: list[float] = []
        terminated = truncated = False
        info: dict[str, object] = {
            "success": False,
            "collision": False,
            "distance": math.inf,
        }

        while not (terminated or truncated):
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.observe(
                observation,
                action,
                float(reward),
                next_observation,
                terminated,
            )
            update = agent.update()
            if update is not None:
                update_losses.append(update.loss)
            reward_sum += float(reward)
            observation = next_observation

        best_reward = max(best_reward, reward_sum)
        if bool(info["success"]) and (
            best_successful_reward is None or reward_sum > best_successful_reward
        ):
            best_successful_reward = reward_sum
            if output is not None:
                best_checkpoint = agent.save(
                    output / "best_successful.pt",
                    metadata={
                        "episode": episode,
                        "reward": reward_sum,
                        "success": True,
                    },
                )

        if output is not None and (
            episode % config.checkpoint_every == 0 or episode == config.episodes
        ):
            latest_checkpoint = agent.save(
                output / "latest.pt",
                metadata={
                    "episode": episode,
                    "reward": reward_sum,
                    "best_reward": best_reward,
                },
            )

        records.append(
            EpisodeRecord(
                episode=episode,
                seed=episode_seed,
                reward=reward_sum,
                steps=env.steps,
                success=bool(info["success"]),
                collision=bool(info["collision"]),
                final_distance=float(info["distance"]),
                epsilon=agent.epsilon(),
                gradient_steps=agent.gradient_steps,
                mean_update_loss=(
                    float(np.mean(update_losses)) if update_losses else None
                ),
            )
        )

    result = DQNTrainingResult(
        config=config,
        episodes=tuple(records),
        best_reward=best_reward,
        best_successful_reward=best_successful_reward,
        latest_checkpoint=(str(latest_checkpoint) if latest_checkpoint else None),
        best_checkpoint=(str(best_checkpoint) if best_checkpoint else None),
    )
    if output is not None:
        result.write_json(output / "training.json")
    return result


def evaluate_dqn(
    agent: DoubleDQNAgent,
    config: DQNConfig,
    *,
    episodes: int = 30,
    seed_offset: int = 100_000,
) -> DQNEvaluation:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = _make_env(config)
    rewards: list[float] = []
    steps: list[int] = []
    distances: list[float] = []
    successes = collisions = timeouts = 0

    for episode in range(episodes):
        observation, _ = env.reset(
            seed=config.seed + seed_offset + episode * 1_009
        )
        reward_sum = 0.0
        terminated = truncated = False
        info: dict[str, object] = {
            "success": False,
            "collision": False,
            "distance": math.inf,
        }
        while not (terminated or truncated):
            action = agent.act(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)

        rewards.append(reward_sum)
        steps.append(env.steps)
        distances.append(float(info["distance"]))
        successes += int(bool(info["success"]))
        collisions += int(bool(info["collision"]))
        timeouts += int(bool(truncated and not terminated))

    return DQNEvaluation(
        episodes=episodes,
        success_rate=successes / episodes,
        collision_rate=collisions / episodes,
        timeout_rate=timeouts / episodes,
        mean_reward=float(np.mean(rewards)),
        reward_std=float(np.std(rewards)),
        mean_steps=float(np.mean(steps)),
        mean_final_distance=float(np.mean(distances)),
    )
