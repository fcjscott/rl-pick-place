"""M0 smoke run: random actions in Fetch pick-and-place, saved as a video.

    python scripts/random_rollout.py            # writes results/random_policy.mp4
    python scripts/random_rollout.py --window   # live viewer instead of a file
    python scripts/random_rollout.py --speed 0.25   # quarter speed (video or window)
"""

import argparse
import time

from rlpp import make_env
from rlpp.video import CONTROL_HZ, record_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", action="store_true", help="show a live viewer")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed, e.g. 0.25 for quarter speed")
    args = parser.parse_args()

    if args.window:
        env = make_env("pick", render_mode="human")
        for ep in range(args.episodes):
            env.reset(seed=ep)
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if args.speed < 1.0:
                    # The viewer already runs in real time; sleep the extra on top.
                    time.sleep((1.0 / args.speed - 1.0) / CONTROL_HZ)
                done = terminated or truncated
        env.close()
        return

    sampler = make_env("pick")
    sampler.action_space.seed(0)
    out = record_episodes(lambda obs: sampler.action_space.sample(),
                          "results/random_policy.mp4", episodes=args.episodes,
                          speed=args.speed)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
