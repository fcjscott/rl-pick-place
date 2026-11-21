"""M1 walkthrough: what the agent sees, what it can do, and how it is rewarded.

    python scripts/explore_env.py

Prints an annotated observation, demonstrates each action dimension, compares the
sparse and dense rewards, and saves results/reward_comparison.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlpp import OBS_LAYOUT, make_env
from rlpp.envs import ACTION_SCALE_M, SUCCESS_THRESHOLD_M
from rlpp.scripted import ScriptedPickAndPlace

np.set_printoptions(precision=3, suppress=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    env = make_env("pick")
    obs, _ = env.reset(seed=0)

    section("1. Spaces")
    print(f"Action space:      {env.action_space}")
    for key, space in env.observation_space.spaces.items():
        print(f"Observation[{key!r}]: shape {space.shape}")
    print(f"Episode length:    {env.spec.max_episode_steps} steps (truncated, never terminated)")

    section("2. One observation, annotated")
    for name, lo, hi, meaning in OBS_LAYOUT:
        print(f"  [{lo:2d}:{hi:2d}] {name:15s} {obs['observation'][lo:hi]}   {meaning}")
    print(f"\n  achieved_goal (block position): {obs['achieved_goal']}")
    print(f"  desired_goal  (target position): {obs['desired_goal']}")
    dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
    print(f"  distance to goal: {dist:.3f} m   (success below {SUCCESS_THRESHOLD_M} m)")

    section("3. What each action dimension does (5 steps of +1.0 each)")
    labels = ["dx", "dy", "dz", "gripper"]
    for i, label in enumerate(labels):
        obs, _ = env.reset(seed=0)
        before_pos = obs["observation"][0:3].copy()
        before_fingers = obs["observation"][9:11].sum()
        action = np.zeros(4, dtype=np.float32)
        action[i] = 1.0
        for _ in range(5):
            obs, *_ = env.step(action)
        moved = obs["observation"][0:3] - before_pos
        fingers = obs["observation"][9:11].sum() - before_fingers
        print(f"  action[{i}] = +1 ({label:7s}) -> gripper moved {moved} m, finger gap changed {fingers:+.3f} m")
    print(f"  One unit of dx/dy/dz asks for {ACTION_SCALE_M} m per step; the arm lags behind the request.")
    print("  Gripper: positive opens, negative closes.")

    section("4. Sparse vs dense reward along one scripted episode")
    # The reward is a pure function of (achieved_goal, desired_goal), so we can
    # compute both versions for the same trajectory. HER relies on exactly this.
    sparse_env, dense_env = env.unwrapped, make_env("pick_dense").unwrapped
    controller = ScriptedPickAndPlace()
    obs, _ = env.reset(seed=1)
    controller.reset()
    sparse, dense, dists, done = [], [], [], False
    while not done:
        obs, _, terminated, truncated, info = env.step(controller.act(obs))
        ag, dg = obs["achieved_goal"], obs["desired_goal"]
        sparse.append(float(sparse_env.compute_reward(ag, dg, info)))
        dense.append(float(dense_env.compute_reward(ag, dg, info)))
        dists.append(float(np.linalg.norm(ag - dg)))
        done = terminated or truncated
    first_success = next((t for t, r in enumerate(sparse) if r == 0.0), None)
    print(f"  Sparse reward: -1 on {sparse.count(-1.0)} steps, 0 on {sparse.count(0.0)} steps")
    print(f"  First success at step: {first_success}")
    print(f"  Dense reward ranged from {min(dense):.3f} to {max(dense):.3f} (negative distance)")
    flat = next((t for t in range(1, len(dense)) if abs(dense[t] - dense[0]) > 1e-3), len(dense))
    print("  Until the block reaches the goal the sparse signal is flat: it says nothing")
    print("  about whether the agent is getting closer. That is the exploration problem.")
    print(f"  The dense reward is also flat for the first {flat} steps: it measures block-to-goal")
    print("  distance only, so reaching for the block earns nothing until the block moves.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    axes[0].step(range(len(sparse)), sparse, where="post", color="#2a6fbb")
    axes[0].set(title="Sparse reward", xlabel="step", ylabel="reward", ylim=(-1.15, 0.15))
    axes[1].plot(dense, color="#c0562a")
    axes[1].set(title="Dense reward (negative distance to goal)", xlabel="step")
    for ax in axes:
        ax.grid(alpha=0.3)
        if first_success is not None:
            ax.axvline(first_success, color="grey", linestyle="--", linewidth=1)
    fig.tight_layout()
    out = Path("results/reward_comparison.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    section("5. Random policy: how often does it succeed by luck?")
    env.action_space.seed(0)
    wins, n = 0, 100
    for ep in range(n):
        env.reset(seed=1000 + ep)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
        wins += int(info["is_success"])
    print(f"  Random policy success: {wins}/{n} episodes")
    print("  These are lucky episodes, e.g. a goal on the table within 5 cm of where the block starts.")


if __name__ == "__main__":
    main()
