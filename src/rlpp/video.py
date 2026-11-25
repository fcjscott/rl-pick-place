"""Record an episode to an mp4 file using offscreen rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio.v2 as imageio

from rlpp.envs import make_env

# The env steps at 25 Hz, so 25 fps is real time.
CONTROL_HZ = 25


def record_episodes(policy: Callable, path: str | Path, env_name: str = "pick",
                    episodes: int = 3, seed: int = 0, on_reset: Callable | None = None,
                    speed: float = 1.0) -> Path:
    """Roll out `policy(obs) -> action` and save the frames. Returns the file path.

    `speed` < 1 plays back slower than real time (0.25 = quarter speed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = make_env(env_name, render_mode="rgb_array")
    frames = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if on_reset is not None:
            on_reset()
        done = False
        frames.append(env.render())
        while not done:
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            frames.append(env.render())
            done = terminated or truncated
    env.close()
    imageio.mimsave(path, frames, fps=CONTROL_HZ * speed, macro_block_size=None)
    return path
