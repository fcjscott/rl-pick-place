# Milestones: who does what

Companion to `rl-pick-and-place-plan.md`. Revised 2026-11-21: Chaojie implements only the pieces that carry the learning; Claude prepares everything else.

## Division of work

Claude prepares: repo skeleton, install instructions, environment exploration notebook, scripted baseline controller, train/evaluate/plot scripts, configs, seeding, logging, PPO observation flattening, tuning script, all Stable-Baselines3 runs configured as one-line commands, plotting and scorecard generation, skeleton files with marked gaps, and unit tests for each gap.

Chaojie implements (7 exercises, each a marked gap in a prepared file with a test that passes when it is right):

| # | Exercise | Size | What it teaches |
|---|---|---|---|
| E1 | Discrete action wrapper for DQN | ~30 lines | Action-space design; what discretisation throws away |
| E2 | Shaped reward function for pick-and-place | ~30 lines | Reward design, the most transferable robotics RL skill |
| E3 | DQN: TD target and loss, epsilon-greedy | ~30 lines | Value learning, bootstrapping, target networks |
| E4 | PPO: advantage estimation (GAE) | ~20 lines | Credit assignment, bias-variance trade-off |
| E5 | PPO: clipped policy loss | ~20 lines | Why policy updates are constrained |
| E6 | SAC: critic, actor and entropy-coefficient losses | ~50 lines | Actor-critic, maximum entropy, twin critics |
| E7 | HER goal relabelling (optional stretch) | ~40 lines | Why hindsight solves sparse reward |

Chaojie also does three non-coding things nobody else can: write hypotheses before the runs, launch the training commands on the laptop (Claude cannot run on it), and interpret the results.

## Milestones

| # | Milestone | Claude prepares | Chaojie does | Done when |
|---|---|---|---|---|
| M0 | Setup | Repo, install guide, smoke test | Install, run smoke test | Video of random policy exists |
| M1 | Know the environment | Exploration notebook, scripted controller | Read notebook, run it | Can explain observation, action, reward; baseline success measured |
| M2 | Harness and warm-up | Scripts, configs, SB3 runs on FetchReach, tuning | E1; launch runs; write hypotheses | All three solve FetchReach; throughput measured; budgets fixed |
| M3 | SAC on pick-and-place | Run configs, plots | E2; launch overnight runs | Curves for sparse, HER, dense, shaped |
| M4 | PPO on pick-and-place | Run configs, plots | Launch overnight runs | Curves for sparse, dense, shaped |
| M5 | DQN on pick-and-place | Run configs, plots, second discretisation | Launch overnight runs | Curves per cell; wrapper effect separated |
| M6 | Comparison | Figures, scorecard draft | Check hypotheses vs results; write conclusions | Pros/cons of each backed by curves |
| M7 | Algorithm internals | Skeleton dqn.py, ppo.py, sac.py with tests and reference curves | E3, E4, E5, E6 (E7 optional) | Each matches the library curve on FetchReach |
| M8 | Ablations and wrap-up | Sweep configs, robustness tests, plots | Launch runs; final write-up | Sensitivity plots and report in project |

## Checkpoints

- After M2: go/no-go on budgets from measured laptop throughput.
- After M6: core goal met. M7 can be interleaved earlier (E3 with M5, E4-E5 with M4, E6 with M3) if preferred.

## Constraints

- Training runs execute on Chaojie's laptop. Claude can build and smoke-test the repo in its own workspace but cannot reach the laptop unless it is connected.
- Reference solutions for E1-E7 are kept in a separate folder so the exercises are not spoiled.
