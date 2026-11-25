# DQN vs SAC vs PPO on Fetch pick-and-place: problem statement and repo plan

Status: plan only, nothing implemented yet. 2026-11-21: reverted to the single-task plan (Fetch pick-and-place only) after considering a three-task balanced suite.

## Decisions

- Single problem: FetchPickAndPlace, Gymnasium-Robotics on MuJoCo. FetchReach is used only as a warm-up to validate the pipeline.
- All three algorithms run on it and are compared afterwards.
- Library-first: Stable-Baselines3, then single-file re-implementations.
- Compute: local Apple laptop, CPU only. 3 seeds per cell; long runs overnight.

## Known bias (accepted)

This task is goal-conditioned with continuous actions, which suits SAC + HER. PPO cannot use HER in its standard form and DQN needs a hand-made discrete action wrapper. Results describe how the three behave on this problem, not their general ranking. A three-task suite (Taxi-v3 for DQN, FetchReachDense with parallel envs for PPO) was considered and set aside; it can be added later.

Low-cost safeguards kept:
1. Outcomes are stated as hypotheses, not pre-labelled winners.
2. Both environment steps and wall-clock time are reported.
3. Same network size, seeds and evaluation protocol for all; each algorithm gets the same small tuning budget on the FetchReach warm-up.
4. Dense and shaped rewards are available to all three.

## Problem statement

Train a state-based policy that moves a block to a 3D goal with the Fetch arm in simulation.

- Observation: 25-D robot/block state + 3-D achieved goal + 3-D desired goal.
- Action: 4-D continuous in [-1, 1] (dx, dy, dz, gripper). For DQN: discrete wrapper (start with 8 actions: ±x, ±y, ±z, open, close).
- Success: block within 0.05 m of goal. Episode: 50 steps, truncation only.
- Reward: sparse (-1/0), dense (negative distance), optional staged shaping wrapper.
- v4 is the latest listed env version; check the installed version.
- Primary metric: success rate over 100 deterministic eval episodes, 3 seeds.
- Secondary: env steps to 80% success, wall-clock time, seed variance, hyperparameter sensitivity.

Questions to answer:
1. How much does each algorithm depend on reward design?
2. What does HER buy for the off-policy methods, and what does PPO lose without it?
3. What does action discretisation cost DQN in precision and smoothness?
4. Sample efficiency vs wall-clock: SAC vs PPO.
5. Stability across seeds and sensitivity to hyperparameters.

## Experiment matrix

| Algorithm | Sparse | Sparse + HER | Dense | Shaped |
|---|---|---|---|---|
| SAC | run (short) | run | run | run |
| PPO | run (short) | not applicable | run | run |
| DQN (discrete) | run (short) | run | run | run |

Hypotheses: plain sparse fails for all three; SAC + HER solves the task; PPO needs shaped reward; DQN is limited by discretisation.

Rough CPU time (to be measured): FetchReach minutes per run; FetchPickAndPlace a few hours per run at ~1M steps.

## Repo plan

```
rl-pick-place/
  README.md, pyproject.toml
  configs/        one YAML per algorithm x reward cell
  envs/           wrappers: discrete actions, shaped reward, flatten-dict for PPO
  algos/          Phase 4 single-file implementations (dqn.py, sac.py, ppo.py)
  scripts/        train.py, evaluate.py, plot.py, record_video.py
  results/        CSV logs, plots, checkpoints
  tests/          wrapper and smoke tests
  notes/          hypotheses and per-experiment findings
```

Phases:
0. Setup: Python env on macOS, `pip install gymnasium-robotics stable-baselines3`, logging, seeding, smoke test with video.
1. Warm-up on FetchReach: all three algorithms end to end; validate wrappers and eval; small equal tuning pass; measure steps/second to fix budgets.
2. SAC + HER on FetchPickAndPlace.
3. Fill the matrix for PPO and DQN and the remaining SAC cells.
4. Re-implement each algorithm as a single file and check against the library curves.
5. Ablations: HER ratio, entropy coefficient, PPO clip/n_envs, DQN action granularity; observation noise and action delay.
6. Comparison write-up: pros, cons and limitations of each on this task, hypotheses vs results.
