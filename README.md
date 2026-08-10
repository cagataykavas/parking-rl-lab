# Parking RL Lab — Stable

A runnable autonomous-parking reinforcement-learning benchmark with **DQN, PPO and SAC** over a shared Gymnasium environment.

`main` is deliberately conservative: fixed action spaces, reproducible seeds and standard Stable-Baselines3 implementations. The `experimental` branch is reserved for the recovered coarse-to-fine curriculum, residual control and harder scenario research.

## Algorithm matrix

| Algorithm | Supported action space | Example |
|---|---|---|
| DQN | `discrete9`, `discrete43` | `python train.py --algorithm dqn --action-mode discrete9` |
| PPO | discrete or continuous | `python train.py --algorithm ppo --action-mode continuous` |
| SAC | `continuous` | `python train.py --algorithm sac --action-mode continuous` |

The repository does not force DQN into native continuous control or SAC into a discrete action space merely to make a bigger feature table.

## Environment

`parking_env.py` implements a Gymnasium environment with:

- randomized start and target poses;
- 9-command and 43-command discrete action tables;
- native two-dimensional continuous control;
- lightweight vehicle kinematics;
- normalized geometric observations;
- `approach → align → settle` phase features;
- progress-based shaping;
- terminal parking success;
- out-of-bounds and timeout termination;
- deterministic reset seeds.

The stable observation contains position, heading, speed, target displacement/distance, target heading error and a three-value phase indicator.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --algorithm dqn --action-mode discrete9 --timesteps 100000 --seed 42
python train.py --algorithm ppo --action-mode discrete43 --timesteps 100000 --seed 42
python train.py --algorithm ppo --action-mode continuous --timesteps 100000 --seed 42
python train.py --algorithm sac --action-mode continuous --timesteps 100000 --seed 42
```

Checkpoints are written beneath `artifacts/`. Evaluation uses a different deterministic seed range from training.

## Stable vs experimental

### `main`

Use this branch when you want an easy-to-run algorithm comparison with minimal custom RL machinery.

### `experimental`

The experimental line is intended for the recovered research features:

- curriculum `9 → 43 → continuous`;
- discrete-to-continuous annealing;
- residual RL over a geometric controller;
- harder trap/random scenarios;
- multi-car extensions and ablations.

## Evaluation protocol

Do not report one lucky reward curve. Use multiple train seeds and disjoint evaluation seeds, then report parking success, final position/alignment error, timeout/out-of-bounds rate, episode length and return distribution.

## Provenance

This is a personal parking-RL project reconstructed and reorganized from earlier personal development work. It contains no employer code, data or scenarios.
