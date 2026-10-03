"""Configurable reward for the Go2 locomotion task.

Every term has a weight supplied by the run config. The weights are part of the
agent's search space, so the reward composition is data, not code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Default weights, following the standard quadruped (Go2) velocity-tracking
# recipe: a strong positive term for tracking the commanded forward speed and a
# set of penalties that keep the robot upright, smooth and at height.
#
# Crucially there is NO `alive` bonus. A per-step "stay alive" reward is a trap
# here: at standstill the robot still collects it, so the cheapest policy is to
# freeze in place and farm the bonus rather than learn to walk. For the same
# reason `track_ang_vel` is a PENALTY-shaped term (see `__call__`): the usual
# exp(-yaw_err) form pays +w for standing still (yaw_err = 0), which is a free
# bonus that also lets the policy spin in place for free. With the penalty form,
# standing earns 0 and yaw drift costs, so only forward motion is rewarded.
#
# `base_height` is the strong -50 from the reference recipe: without a stiff
# height term the policy happily shuffles along with its belly low, which is the
# first step toward collapse. `termination` is large enough to dominate the
# track reward accumulated across a short episode, so "sprint then fall" cannot
# outscore "walk the full episode".
DEFAULT_WEIGHTS: dict[str, float] = {
    "track_lin_vel": 3.0,
    "track_ang_vel": -0.5,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "action_rate": -0.005,
    "torques": -0.0001,
    "base_height": -50.0,
    "alive": 0.0,
    # One-off penalty on the step that ends the episode by falling. "Sprint
    # forward, crash after ~100 fast steps" earns ~+(100 * 3) from tracking; a
    # penalty smaller than that dominating sum is not enough to make it
    # unprofitable (measured: -50 still let a falling policy net +190).
    "termination": -150.0,
}


@dataclass
class RewardState:
    """Signals the env computes each control step and hands to the reward.

    Convention: every field carries a RAW magnitude (not pre-squared), so the
    reward can square uniformly. `lin_vel_track` is forward speed as a fraction
    of the commanded speed (NOT clipped — the reward's exp kernel needs the raw
    ratio so it can penalize overspeeding); `ang_vel_error` feeds the squared
    yaw penalty; the rest are squared. `fell` is True only on the step that
    terminates the episode (used for the one-off termination penalty).
    """

    lin_vel_track: float   # vx / cmd_vx (raw ratio, unclipped)
    ang_vel_error: float
    lin_vel_z: float
    ang_vel_xy: float
    gravity_xy: float
    action_rate: float     # ||a_t - a_{t-1}||
    torque: float          # ||torque||_2
    height_error: float
    fell: bool = False


@dataclass
class Reward:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # Width of the exp velocity-tracking kernel (matches the standard quadruped
    # recipe's `tracking_sigma`). Smaller = tighter speed regulation.
    tracking_sigma: float = 0.25

    def __call__(self, s: RewardState) -> tuple[float, dict[str, float]]:
        w = self.weights
        # Dense exp velocity tracking: exp(-|vx/cmd - 1| / sigma). Unlike a
        # linear clip (which saturates at the commanded speed and thus pays the
        # SAME reward for "lunge at 2x speed"), this peak is a gradient toward
        # exactly the commanded speed — overspeeding is penalized. That removes
        # the incentive to sprint-and-crash. At standstill it is exp(-1/0.25)
        # ~= 0.018 (a negligible floor), so freezing still earns almost nothing.
        err = abs(float(s.lin_vel_track) - 1.0)
        track = float(np.exp(-err / max(self.tracking_sigma, 1e-6)))
        terms = {
            "track_lin_vel": w.get("track_lin_vel", 0.0) * track,
            # yaw penalty (NOT exp tracking): 0 at zero yaw error, negative when
            # the robot turns. A positive exp-tracking term here would hand the
            # standing-still policy a free bonus and let it spin at no cost.
            "track_ang_vel": w.get("track_ang_vel", 0.0) * s.ang_vel_error ** 2,
            "lin_vel_z": w.get("lin_vel_z", 0.0) * s.lin_vel_z ** 2,
            "ang_vel_xy": w.get("ang_vel_xy", 0.0) * s.ang_vel_xy ** 2,
            "orientation": w.get("orientation", 0.0) * s.gravity_xy ** 2,
            "action_rate": w.get("action_rate", 0.0) * s.action_rate ** 2,
            "torques": w.get("torques", 0.0) * s.torque ** 2,
            "base_height": w.get("base_height", 0.0) * s.height_error ** 2,
            "alive": w.get("alive", 0.0),
            "termination": w.get("termination", 0.0) if s.fell else 0.0,
        }
        return float(sum(terms.values())), terms

    @classmethod
    def from_config(cls, cfg: dict | None) -> "Reward":
        w = dict(DEFAULT_WEIGHTS)
        if cfg:
            w.update({k: float(v) for k, v in cfg.items()})
        return cls(weights=w)
