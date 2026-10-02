"""Configurable reward for the Go2 locomotion task.

Every term has a weight supplied by the run config. The weights are part of the
agent's search space, so the reward composition is data, not code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Default weights. Healthy defaults for a joystick-style velocity-tracking task;
# the agent is free to override any of these.
DEFAULT_WEIGHTS: dict[str, float] = {
    "track_lin_vel": 2.0,
    "track_ang_vel": 0.2,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "action_rate": -0.005,
    "torques": -0.0001,
    "base_height": -2.0,
    "alive": 0.2,
}


@dataclass
class RewardState:
    """Signals the env computes each control step and hands to the reward.

    Convention: every field carries a RAW magnitude (not pre-squared), so the
    reward can square uniformly. `lin_vel_track` is forward-progress as a
    fraction of the commanded speed (clipped to [0, 1]); `ang_vel_error` feeds
    the exp yaw-tracking term; the rest are squared.
    """

    lin_vel_track: float   # clip(vx / cmd_vx, 0, 1)
    ang_vel_error: float
    lin_vel_z: float
    ang_vel_xy: float
    gravity_xy: float
    action_rate: float     # ||a_t - a_{t-1}||
    torque: float          # ||torque||_2
    height_error: float


@dataclass
class Reward:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __call__(self, s: RewardState) -> tuple[float, dict[str, float]]:
        w = self.weights
        # Linear forward-progress tracking: 0 when standing still, 1 at the
        # commanded speed. Unlike exp(-error) this leaves no reward floor for
        # standing, which is what makes forward gait discoverable in a short
        # training budget.
        track = float(np.clip(s.lin_vel_track, 0.0, 1.0))
        terms = {
            "track_lin_vel": w.get("track_lin_vel", 0.0) * track,
            "track_ang_vel": w.get("track_ang_vel", 0.0) * float(np.exp(-s.ang_vel_error)),
            "lin_vel_z": w.get("lin_vel_z", 0.0) * s.lin_vel_z ** 2,
            "ang_vel_xy": w.get("ang_vel_xy", 0.0) * s.ang_vel_xy ** 2,
            "orientation": w.get("orientation", 0.0) * s.gravity_xy ** 2,
            "action_rate": w.get("action_rate", 0.0) * s.action_rate ** 2,
            "torques": w.get("torques", 0.0) * s.torque ** 2,
            "base_height": w.get("base_height", 0.0) * s.height_error ** 2,
            "alive": w.get("alive", 0.0),
        }
        return float(sum(terms.values())), terms

    @classmethod
    def from_config(cls, cfg: dict | None) -> "Reward":
        w = dict(DEFAULT_WEIGHTS)
        if cfg:
            w.update({k: float(v) for k, v in cfg.items()})
        return cls(weights=w)
