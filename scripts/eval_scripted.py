"""M1 baseline: success rate of the hand-written controller.

    python scripts/eval_scripted.py                 # 100 episodes
    python scripts/eval_scripted.py --video         # also save results/scripted_policy.mp4
"""

import argparse
import time
from collections import Counter

import numpy as np

from rlpp import make_env
from rlpp.scripted import ScriptedPickAndPlace, run_episode
from rlpp.video import record_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    env = make_env("pick")
    controller = ScriptedPickAndPlace()
    successes, first_success_steps, stuck_in = [], [], Counter()
    start = time.time()
    for ep in range(args.episodes):
        ok, trace = run_episode(env, controller, seed=args.seed + ep)
        successes.append(ok)
        if ok:
            first_success_steps.append(next(t for t, s in enumerate(trace) if s["success"]) + 1)
        else:
            stuck_in[trace[-1]["phase"]] += 1
    elapsed = time.time() - start
    steps = args.episodes * env.spec.max_episode_steps

    rate = float(np.mean(successes))
    print(f"Scripted controller: {sum(successes)}/{args.episodes} episodes succeeded ({rate:.0%})")
    if first_success_steps:
        print(f"Median steps to first success: {int(np.median(first_success_steps))} of 50")
    if stuck_in:
        print(f"Failures by final phase: {dict(stuck_in)}")
    print(f"Simulation speed: {steps / elapsed:.0f} steps/second on this machine")

    if args.video:
        out = record_episodes(controller.act, "results/scripted_policy.mp4",
                              episodes=3, seed=args.seed, on_reset=controller.reset)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
