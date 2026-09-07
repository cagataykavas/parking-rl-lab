"""Native learning agents used by the parking benchmark.

Stable-Baselines remains the convenient comparison layer in ``train_v2.py``.
Modules in this package expose the mechanics directly so replay, target updates,
checkpointing, and evaluation can be inspected and regression-tested.
"""

from .dqn import (
    DoubleDQNAgent,
    DQNConfig,
    DQNEvaluation,
    DQNTrainingResult,
    DuelingQNetwork,
    ReplayBuffer,
    evaluate_dqn,
    train_dqn,
)

__all__ = [
    "DQNConfig",
    "DQNEvaluation",
    "DQNTrainingResult",
    "DoubleDQNAgent",
    "DuelingQNetwork",
    "ReplayBuffer",
    "evaluate_dqn",
    "train_dqn",
]
