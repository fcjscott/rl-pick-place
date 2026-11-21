# The environment in one page

Task: `FetchPickAndPlace-v4` (Gymnasium-Robotics 1.4.2, MuJoCo). A 7-DoF Fetch arm with a
two-finger gripper must move a block to a target that is on the table or in mid-air.

## What the agent sees

A dictionary with three entries:

| Key | Size | Meaning |
|---|---|---|
| `observation` | 25 | Robot and block state (layout below) |
| `achieved_goal` | 3 | Where the block is now |
| `desired_goal` | 3 | Where the block should be |

`observation` layout (also in `rlpp.OBS_LAYOUT`):

| Index | Name | Meaning |
|---|---|---|
| 0:3 | grip_pos | Gripper position, metres |
| 3:6 | object_pos | Block position, metres |
| 6:9 | object_rel_pos | Block minus gripper position |
| 9:11 | gripper_state | Finger openings |
| 11:14 | object_rot | Block orientation (Euler angles) |
| 14:17 | object_velp | Block linear velocity relative to gripper |
| 17:20 | object_velr | Block angular velocity |
| 20:23 | grip_velp | Gripper linear velocity |
| 23:25 | gripper_vel | Finger velocities |

## What the agent can do

Four numbers in [-1, 1]: `dx, dy, dz, gripper`.

- One unit of `dx/dy/dz` requests a 0.05 m move of the gripper target per step. The arm lags
  the request (about 0.03 m per step measured here).
- `gripper` > 0 opens, < 0 closes.
- Orientation is fixed; inverse kinematics is handled inside the simulator.
- Control rate is 25 Hz, so the 50-step episode is 2 seconds.

## How it is rewarded

- Sparse (`FetchPickAndPlace-v4`): -1 every step until the block is within 0.05 m of the goal, then 0.
- Dense (`FetchPickAndPlaceDense-v4`): minus the block-to-goal distance.
- The episode never terminates early; it is truncated at 50 steps. Success means the block
  is at the goal on the final step.
- The reward depends only on `(achieved_goal, desired_goal)`. That is what lets Hindsight
  Experience Replay recompute rewards for a different goal after the fact.

Something to notice in `results/reward_comparison.png`: the dense reward is also flat while
the arm is reaching for the block, because it measures block-to-goal distance only. Dense
does not mean informative everywhere. Exercise E2 (shaped reward) is about fixing that.

## Baselines measured in this repo (run them yourself to confirm on your machine)

| Policy | Success over 100 episodes |
|---|---|
| Random actions | 6% |
| Hand-written controller (`rlpp/scripted.py`) | 100% |

The hand-written controller is about 40 lines and needs no training. That is the bar: RL
on this task is for learning RL, not because the task needs it.
