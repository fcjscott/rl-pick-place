# rl-pick-place

A hands-on study of DQN, SAC and PPO on one robotics problem: Fetch pick-and-place in
MuJoCo. Library-first (Stable-Baselines3), CPU only, built to run on an Apple laptop.

Status: milestones M0 (setup) and M1 (know the environment) are ready. Training starts in M2.

## Install

Python 3.10 to 3.13.

```bash
cd ~/projects/rl-pick-place
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Versions are pinned on purpose: MuJoCo 3.11 and later break the Fetch environments in
gymnasium-robotics 1.4.2, so `pyproject.toml` pins `mujoco==3.10.0`.

## M0: check it works

```bash
pytest                              # 9 tests, a few seconds
python scripts/random_rollout.py    # writes results/random_policy.mp4
```

To watch live instead of saving a file: `python scripts/random_rollout.py --window`.
If the window does not open on macOS, try `mjpython scripts/random_rollout.py --window`,
or just use the mp4.

Done when the tests pass and you have watched the arm flail in the video.

## M1: know the environment

```bash
python scripts/explore_env.py            # annotated observation, actions, rewards
python scripts/eval_scripted.py --video  # hand-written baseline + results/scripted_policy.mp4
```

Then read `docs/environment.md` and `src/rlpp/scripted.py` (the baseline, about 40 lines).

Done when you can answer these without looking:

1. Which observation entries would a policy need in order to grasp the block?
2. Why does a random policy get zero learning signal from the sparse reward in most episodes?
3. Why is the dense reward flat for the first part of an episode?
4. The scripted controller solves this without learning. What would have to change about
   the task before a hand-written controller became impractical?

## Layout

```
src/rlpp/envs.py      environment creation, observation layout
src/rlpp/scripted.py  hand-written pick-and-place controller (baseline)
src/rlpp/video.py     record episodes to mp4
scripts/              runnable entry points for each milestone
tests/                smoke tests
docs/environment.md   the task on one page
results/              videos, plots, logs (not tracked by git)
```

## What comes next

| Milestone | You do | Prepared for you |
|---|---|---|
| M2 Warm-up on FetchReach | E1: discrete action wrapper for DQN; write hypotheses | Train/evaluate/plot scripts, configs |
| M3 SAC on pick-and-place | E2: shaped reward | Run configs, plots |
| M4 PPO on pick-and-place | Launch runs | Run configs, plots |
| M5 DQN on pick-and-place | Launch runs | Run configs, plots |
| M6 Comparison | Check hypotheses, write conclusions | Figures, scorecard |
| M7 Algorithm internals | E3-E6: DQN, PPO and SAC update code (E7 HER optional) | Skeleton files, tests |
| M8 Ablations and wrap-up | Launch runs, final write-up | Sweep configs |
