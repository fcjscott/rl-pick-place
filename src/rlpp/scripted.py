"""A hand-written pick-and-place controller: the non-RL baseline.

It reads the same observation an RL policy gets and outputs the same 4-D action,
so its success rate is directly comparable with the learned policies.
"""

from __future__ import annotations

import numpy as np

from rlpp.envs import ACTION_SCALE_M

OPEN, CLOSE = 1.0, -1.0


class ScriptedPickAndPlace:
    """Four phases: hover above the block, descend, grasp, carry to the goal."""

    def __init__(self, hover_height: float = 0.06, xy_tol: float = 0.01,
                 grasp_tol: float = 0.008, grasp_steps: int = 4):
        self.hover_height = hover_height
        self.xy_tol = xy_tol
        self.grasp_tol = grasp_tol
        self.grasp_steps = grasp_steps
        self.reset()

    def reset(self) -> None:
        self.phase = "hover"
        self._closing = 0

    @staticmethod
    def _move(delta: np.ndarray, grip: float) -> np.ndarray:
        """Proportional move: full speed when far, slowing inside one action step."""
        return np.append(np.clip(delta / ACTION_SCALE_M, -1.0, 1.0), grip).astype(np.float32)

    def act(self, obs: dict) -> np.ndarray:
        o = obs["observation"]
        grip_pos, obj_pos = o[0:3], o[3:6]
        goal = obs["desired_goal"]

        if self.phase == "hover":
            target = obj_pos + np.array([0.0, 0.0, self.hover_height])
            if np.linalg.norm((target - grip_pos)[:2]) < self.xy_tol:
                self.phase = "descend"
            return self._move(target - grip_pos, OPEN)

        if self.phase == "descend":
            if np.linalg.norm(obj_pos - grip_pos) < self.grasp_tol:
                self.phase = "grasp"
            return self._move(obj_pos - grip_pos, OPEN)

        if self.phase == "grasp":
            self._closing += 1
            if self._closing >= self.grasp_steps:
                self.phase = "carry"
            return self._move(np.zeros(3), CLOSE)

        # carry: move the block (not the gripper) to the goal, keep squeezing
        return self._move(goal - obj_pos, CLOSE)


def run_episode(env, controller: ScriptedPickAndPlace, seed: int | None = None):
    """Run one episode. Returns (success, list of per-step dicts)."""
    obs, info = env.reset(seed=seed)
    controller.reset()
    trace, done = [], False
    while not done:
        action = controller.act(obs)
        phase = controller.phase
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append({
            "phase": phase,
            "reward": float(reward),
            "distance": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
            "success": bool(info["is_success"]),
        })
        done = terminated or truncated
    return bool(info["is_success"]), trace
