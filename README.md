# Parking RL Lab — Stable

A reinforcement-learning laboratory for autonomous parking with **Double DQN, PPO and SAC**, discrete and continuous control, collision geometry, configurable scenarios, reproducible evaluation, checkpoints and replay visualization.

> Migration note: the recovered full simulator is being separated from its historical `neuralTrainer` branch into this dedicated repository. `main` is the stable portfolio surface; `experimental` carries curriculum and residual-RL work.

## Algorithm suite

| Algorithm | Action modes | Role |
|---|---|---|
| Double DQN + Dueling Network | `DISCRETE_9`, `DISCRETE_43` | value-based discrete baseline |
| PPO | discrete + continuous | general actor-critic baseline |
| SAC | continuous | off-policy continuous-control baseline |

## Stable release goals

- deterministic seed control
- 9- and 43-command action spaces
- native continuous steering/throttle
- rotated-rectangle collision checks using SAT
- SIMPLE / WALLS / RANDOM / custom JSON scenarios
- multi-car environment support
- checkpoints, CSV logs and evaluation utilities
- smoke tests and CI

## Experimental branch

The `experimental` branch is reserved for:

- curriculum `9 → 43 → continuous`
- annealed discrete-to-continuous execution
- residual RL over a geometric controller
- hierarchical `approach → align → settle` state
- harder trap/random scenarios
- multi-car ablations

## Evaluation

Results should be reported over held-out seeds with parking success rate, collision rate, timeout rate, final position/alignment error and episode length. Reward alone is not treated as task success.

## Provenance

This is a personal autonomous-parking RL project recovered from earlier development work and reorganized as a standalone portfolio repository. It is unrelated to employer code or data.
