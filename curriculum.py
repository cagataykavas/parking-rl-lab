from __future__ import annotations

from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class SuccessCurriculumCallback(BaseCallback):
    """Raise environment difficulty when recent terminal success is sustained."""

    def __init__(
        self,
        *,
        initial_level: int = 0,
        max_level: int = 3,
        window: int = 100,
        minimum_episodes: int = 60,
        promotion_success_rate: float = 0.68,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.level = initial_level
        self.max_level = max_level
        self.window = window
        self.minimum_episodes = minimum_episodes
        self.promotion_success_rate = promotion_success_rate
        self.results: deque[bool] = deque(maxlen=window)
        self.episodes_at_level = 0

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_curriculum_level", self.level)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.results.append(bool(info.get("success", False)))
            self.episodes_at_level += 1

        if (
            self.level < self.max_level
            and self.episodes_at_level >= self.minimum_episodes
            and len(self.results) >= min(self.window, self.minimum_episodes)
        ):
            rate = sum(self.results) / len(self.results)
            if rate >= self.promotion_success_rate:
                self.level += 1
                self.training_env.env_method("set_curriculum_level", self.level)
                if self.verbose:
                    print(
                        f"[curriculum] promoted to level={self.level} "
                        f"after success_rate={rate:.3f} over {len(self.results)} episodes"
                    )
                self.results.clear()
                self.episodes_at_level = 0
        return True
