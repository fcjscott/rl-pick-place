"""M0 smoke run: random actions in Fetch pick-and-place, saved as a video.

    python scripts/random_rollout.py            # writes results/random_policy.mp4
    python scripts/random_rollout.py --window   # live viewer instead of a file
"""

import argparse

from rlpp import make_env
from rlpp.video import record_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", action="store_true", help="show a live viewer")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if args.window:
        env = make_env("pick", render_mode="human")
        for ep in range(args.episodes):
            env.reset(seed=ep)
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated
        env.close()
        return

    sampler = make_env("pick")
    sampler.action_space.seed(0)
    out = record_episodes(lambda obs: sampler.action_space.sample(),
                          "results/random_policy.mp4", episodes=args.episodes)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
