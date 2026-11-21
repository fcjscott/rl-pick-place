"""Environment creation and a description of what the observation contains."""

from __future__ import annotations

import contextlib
import io

import gymnasium as gym

# Importing gymnasium_robotics prints a notice about unrelated Adroit environments.
with contextlib.redirect_stderr(io.StringIO()):
    import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

# Short names used throughout the repo -> Gymnasium IDs.
ENV_IDS = {
    "reach": "FetchReach-v4",
    "reach_dense": "FetchReachDense-v4",
    "pick": "FetchPickAndPlace-v4",
    "pick_dense": "FetchPickAndPlaceDense-v4",
}

# Layout of obs["observation"] (25 numbers) for the pick-and-place task.
# (name, start index, end index, meaning)
OBS_LAYOUT = [
    ("grip_pos", 0, 3, "gripper position in the world frame, metres"),
    ("object_pos", 3, 6, "block position in the world frame, metres"),
    ("object_rel_pos", 6, 9, "block position minus gripper position"),
    ("gripper_state", 9, 11, "opening of the two fingers, metres"),
    ("object_rot", 11, 14, "block orientation, Euler angles in radians"),
    ("object_velp", 14, 17, "block linear velocity relative to the gripper"),
    ("object_velr", 17, 20, "block angular velocity"),
    ("grip_velp", 20, 23, "gripper linear velocity"),
    ("gripper_vel", 23, 25, "finger velocities"),
]

# One unit of action moves the gripper target by this many metres per step.
ACTION_SCALE_M = 0.05
# Block within this distance of the goal counts as success.
SUCCESS_THRESHOLD_M = 0.05


def make_env(name: str = "pick", render_mode: str | None = None, **kwargs) -> gym.Env:
    """Create one of the study's environments by short name or full Gymnasium ID."""
    env_id = ENV_IDS.get(name, name)
    return gym.make(env_id, render_mode=render_mode, **kwargs)
