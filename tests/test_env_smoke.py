"""Integration smoke test for the MuJoCo Go2 environment (the agent's 'plant').

Marked slow: it loads the model and steps physics. Run with -m slow.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.envs.go2_env import Go2LocomotionEnv


@pytest.mark.slow
def test_env_reset_step_shapes():
    env = Go2LocomotionEnv(cfg={"episode_length": 20})
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.shape == (42,)
    assert env.action_space.shape == (12,)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        assert "vx" in info and "height" in info
        if term or trunc:
            break
    env.close()


@pytest.mark.slow
def test_env_is_deterministic_given_seed():
    """Same seed on fresh instances must give bit-identical trajectories."""
    actions = np.linspace(-0.5, 0.5, 5 * 12).reshape(5, 12)  # fixed, RNG-independent

    def rollout():
        env = Go2LocomotionEnv(cfg={"episode_length": 30})
        obs0 = env.reset(seed=7)[0]
        seq = [env.step(actions[i])[0][0] for i in range(5)]
        env.close()
        return obs0, seq

    a0, seq1 = rollout()
    b0, seq2 = rollout()
    assert np.allclose(a0, b0)
    assert np.allclose(seq1, seq2)


@pytest.mark.slow
def test_high_kp_keeps_robot_upright():
    """With good PD gains the robot should not immediately fall."""
    env = Go2LocomotionEnv(cfg={"episode_length": 100, "kp": 80.0, "kd": 1.6})
    env.reset(seed=0)
    fell = False
    for _ in range(100):
        _, _, term, trunc, info = env.step(np.zeros(12))
        if term:
            fell = True
            break
        if trunc:
            break
    h = info["height"]
    env.close()
    assert not fell and h > 0.2, f"robot fell or sagged (height={h})"
