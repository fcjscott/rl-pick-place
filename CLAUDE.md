# CLAUDE.md

A hands-on study comparing DQN, SAC and PPO on one task: `FetchPickAndPlace-v4`
(Gymnasium-Robotics 1.4.2, MuJoCo). Library-first with Stable-Baselines3, then single-file
re-implementations. CPU only, on an Apple laptop. The point is to **learn RL**, not to
solve the task. A ~40-line scripted controller already gets 100%.

Background: [docs/rl-pick-and-place-plan.md](docs/rl-pick-and-place-plan.md) (problem and
experiment matrix), [docs/rl-milestones.md](docs/rl-milestones.md) (who does what),
[docs/environment.md](docs/environment.md) (the task on one page).

## Division of work (most important rule)

Chaojie implements only the parts that carry the learning: seven exercises, **E1-E7**.
Claude prepares everything else.

| # | Exercise | Milestone |
|---|---|---|
| E1 | Discrete action wrapper for DQN | M2 |
| E2 | Shaped reward for pick-and-place | M3 |
| E3 | DQN: TD target and loss, epsilon-greedy | M7 |
| E4 | PPO: GAE | M7 |
| E5 | PPO: clipped policy loss | M7 |
| E6 | SAC: critic, actor and entropy-coefficient losses | M7 |
| E7 | HER goal relabelling (optional) | M7 |

For each exercise:
- Ship a skeleton file with a clearly marked gap (`raise NotImplementedError` plus a
  docstring giving the expected inputs and outputs) and a unit test that passes once the
  gap is filled correctly.
- **Do not fill in the gap**, and do not leave hints in code or comments that give the
  answer away. Reference solutions go in a separate folder, kept apart from the exercises.
- Everything else (scripts, configs, seeding, logging, PPO dict flattening, tuning, plots,
  scorecard) should be finished and run with a single command.

Chaojie also writes the hypotheses, launches the training runs and interprets the
results. Don't pre-write conclusions or name winners. State outcomes as hypotheses.

## Constraints

- Training runs on Chaojie's laptop. Claude can build and smoke-test here but cannot
  reach the laptop. Give long runs as copy-paste commands. Don't start them here.
- Keep smoke tests fast (seconds). Use tiny step budgets in tests.
- Versions are pinned on purpose: `mujoco==3.10.0` (3.11 and later break the Fetch envs
  in gymnasium-robotics 1.4.2) and `gymnasium-robotics==1.4.2`. Don't bump them.
- Python 3.10-3.13. SB3 and tensorboard live in the `train` extra (from M2 onward).

## Fair-comparison rules (apply to every experiment)

- Same network size, seeds (3 per cell) and evaluation protocol for all three algorithms.
- Primary metric: success rate over 100 deterministic eval episodes. Also report env
  steps *and* wall-clock time, steps to 80% success, and seed variance.
- Every algorithm gets the same small tuning budget, spent on the FetchReach warm-up.
- Sparse, dense and shaped rewards are available to all three. HER applies to SAC and DQN
  only, not PPO.
- Known accepted bias: the task favours SAC + HER. Results describe this task only.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ,train for M2+
pytest                             # smoke tests
python scripts/random_rollout.py   # M0: results/random_policy.mp4
python scripts/explore_env.py      # M1: annotated obs/action/reward
python scripts/eval_scripted.py --video
```

## Code layout and conventions

- Package code lives in `src/rlpp/`: `envs.py` (`make_env`, `ENV_IDS` short names
  `reach`/`reach_dense`/`pick`/`pick_dense`, `OBS_LAYOUT`, constants), `scripted.py`
  (baseline), `video.py`. Entry points go in `scripts/`, tests in `tests/`.
- The plan doc sketches top-level `configs/ envs/ algos/ notes/` folders. Treat that as
  intent. Put new library code under `src/rlpp/` unless the user says otherwise.
- `results/` holds generated output (videos, CSVs, plots, checkpoints) and is untracked.
- Style: small modules, a docstring on every script showing its usage line, `argparse`
  for CLIs, `from __future__ import annotations`. Use shared constants
  (`ACTION_SCALE_M`, `SUCCESS_THRESHOLD_M`) rather than repeating numbers. Comments say
  *why*, not what.
- Write docs and READMEs in plain, short language for Chaojie as a learner. Each milestone
  section ends with "Done when ...".

## Environment facts worth remembering

- Obs is a dict: `observation` (25), `achieved_goal` (3), `desired_goal` (3).
- Action is 4-D in [-1, 1]: `dx, dy, dz, gripper` (>0 opens). One unit is 0.05 m per step.
- 50 steps at 25 Hz. Truncation only, never terminated early. Success means the block is
  within 0.05 m of the goal on the final step.
- Reward depends only on `(achieved_goal, desired_goal)`, which is what makes HER work.
  Dense reward is flat while the arm reaches for the block, which is the motivation for E2.
- Baselines: random 6%, scripted 100%.
