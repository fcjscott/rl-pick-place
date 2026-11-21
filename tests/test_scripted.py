"""M1: the hand-written baseline solves the task most of the time."""

from rlpp import make_env
from rlpp.scripted import ScriptedPickAndPlace, run_episode


def test_scripted_controller_mostly_succeeds():
    env = make_env("pick")
    controller = ScriptedPickAndPlace()
    wins = sum(run_episode(env, controller, seed=s)[0] for s in range(20))
    assert wins >= 16, f"scripted controller only solved {wins}/20"
