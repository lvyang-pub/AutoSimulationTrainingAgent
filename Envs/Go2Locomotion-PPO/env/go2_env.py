"""Gymnasium environment: Unitree Go2 forward-locomotion in MuJoCo.

The MJCF model uses torque (motor) actuators. We wrap them with a software PD
controller so actions are desired joint positions; the PD gains (kp/kd) are part
of the config and therefore part of what the agent can tune.

This env is a *supporting fixture* for the agent, so it is deliberately compact:
a fixed forward-velocity command, a configurable reward, and a small set of
signals that the training run records into metrics.csv.
"""
from __future__ import annotations

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .fetch_go2 import fetch, DEST as _GO2_DEST
from .reward import Reward, RewardState

# Local MJCF model, fetched once into env/go2/. Prefer the on-disk copy so a
# normal run never touches the network; only fall back to fetch() if missing.
_SCENE_XML = os.path.join(_GO2_DEST, "scene.xml")

# Actuator order in go2.xml, grouped per leg.
# hip (abduction) / thigh (front-or-back hip) / calf (knee)
DEFAULT_ANGLES = {
    "hip": 0.0,
    "thigh": 0.9,
    "calf": -1.8,
}
# nominal standing base height (matches the "home" keyframe 0.27 + free-joint z)
NOMINAL_HEIGHT = 0.30
# target forward velocity (m/s) for the command
DEFAULT_CMD = (1.0, 0.0, 0.0)


class Go2LocomotionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        cfg: dict | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        cfg = cfg or {}
        self.cfg = cfg
        self.render_mode = render_mode

        scene = _SCENE_XML if os.path.exists(_SCENE_XML) else fetch()
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)

        # ------- task config -------
        self.dt = float(cfg.get("dt", 0.02))          # control period (s)
        self.frame_skip = int(cfg.get("frame_skip", 1))
        if self.frame_skip <= 0:
            self.frame_skip = max(1, round(self.dt / self.model.opt.timestep))
        self.model.opt.timestep = self.dt / self.frame_skip
        self.episode_length = int(cfg.get("episode_length", 500))  # control steps
        self.action_scale = float(cfg.get("action_scale", 0.4))
        self.kp = float(cfg.get("kp", 80.0))
        self.kd = float(cfg.get("kd", 1.6))
        self.max_torque = float(cfg.get("max_torque", 23.7))

        cmd = cfg.get("command", DEFAULT_CMD)
        self.command = np.asarray(cmd, dtype=np.float64)  # [vx, vy, yaw_rate]

        self.terminate_on_fall = bool(cfg.get("terminate_on_fall", True))
        self.min_height = float(cfg.get("min_height", 0.18))
        self.max_tilt = float(cfg.get("max_tilt", 0.7))  # rad, gravity-xy cutoff

        # Initial forward push: the robot starts already moving. Without this the
        # standing pose is a fixed point and PPO never discovers a gait; the push
        # forces it to step to stay upright.
        self.init_vel_forward = float(cfg.get("init_vel_forward", 1.0))
        self.init_vel_noise = float(cfg.get("init_vel_noise", 0.3))

        self.reward_fn = Reward.from_config(cfg.get("reward"))

        # ------- spaces -------
        n_act = self.model.nu
        self._desired = np.array(
            [DEFAULT_ANGLES["hip"], DEFAULT_ANGLES["thigh"], DEFAULT_ANGLES["calf"]] * 4,
            dtype=np.float64,
        )
        obs_dim = 3 + 3 + n_act + n_act + n_act  # ang_vel, grav, q, dq, last_action
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)

        self._last_action = np.zeros(n_act, dtype=np.float64)
        self._step_count = 0
        self._episode_return = 0.0
        self._fall = False

    # ------------------------------------------------------------------ helpers
    def _apply_pd(self, target_q: np.ndarray) -> np.ndarray:
        q = self.data.qpos[7:]      # 12 joint angles (skip free joint: 7)
        dq = self.data.qvel[6:]     # 12 joint velocities
        torque = self.kp * (target_q - q) - self.kd * dq
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        self.data.ctrl[:] = torque
        return torque

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        # base angular velocity (body frame) via the free joint
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        # projected gravity in body frame
        w, x, y, z = quat
        grav = np.array([
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ])
        return np.concatenate([ang_vel, grav, q, dq, self._last_action])

    def _signals(self, action: np.ndarray, fell: bool = False) -> RewardState:
        d = self.data
        vx, vy = d.qvel[0], d.qvel[1]
        yaw_rate = d.qvel[5]
        cmd_vx = max(abs(self.command[0]), 1e-6)
        lin_track = float(vx) / cmd_vx
        ang_err = abs(float(yaw_rate - self.command[2]))
        height = float(d.qpos[2])
        quat = d.qpos[3:7]
        w, x, y, z = quat
        gravity_xy = float(np.hypot(2 * (x * z - w * y), 2 * (y * z + w * x)))
        return RewardState(
            lin_vel_track=lin_track,
            ang_vel_error=ang_err,
            lin_vel_z=float(d.qvel[2]),
            ang_vel_xy=float(np.hypot(d.qvel[3], d.qvel[4])),
            gravity_xy=gravity_xy,
            # penalty on change between consecutive normalized actions
            action_rate=float(np.linalg.norm(action - self._last_action)),
            torque=float(np.linalg.norm(d.ctrl)),
            height_error=height - NOMINAL_HEIGHT,
            fell=fell,
        )

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Full reset first: mj_resetDataKeyframe alone leaves the solver's
        # warm-start acceleration from the previous episode, which makes
        # identical seeds diverge a few steps in (measured: |qacc_warmstart|~685).
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # "home" pose
        # small randomization so the policy cannot overfit a single state
        rng = self.np_random
        self.data.qpos[:7] += rng.uniform(-0.02, 0.02, 7)
        self.data.qvel[:] = rng.uniform(-0.05, 0.05, self.model.nv)
        # give it an initial forward push (linear + a little pitch)
        self.data.qvel[0] = self.init_vel_forward + rng.uniform(-self.init_vel_noise,
                                                                self.init_vel_noise)
        self.data.qvel[4] += rng.uniform(-0.2, 0.2)  # pitch rate
        mujoco.mj_forward(self.model, self.data)
        self._last_action[:] = 0.0
        self._step_count = 0
        self._episode_return = 0.0
        self._fall = False
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        target_q = self._desired + self.action_scale * action
        self._apply_pd(target_q)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        sig = self._signals(action)

        terminated = False
        if self.terminate_on_fall:
            low = self.data.qpos[2] < self.min_height
            tilted = sig.gravity_xy > self.max_tilt
            if low or tilted:
                terminated = True
                self._fall = True
        sig.fell = terminated
        reward, terms = self.reward_fn(sig)

        self._last_action = action
        self._step_count += 1
        truncated = self._step_count >= self.episode_length
        self._episode_return += reward

        info = {
            "vx": float(vx := self.data.qvel[0]),
            "lin_vel_error": abs(float(vx - self.command[0])),
            "lin_vel_track": sig.lin_vel_track,
            "yaw_error": sig.ang_vel_error,
            "height": float(self.data.qpos[2]),
            "fell": self._fall,
            "reward_terms": terms,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if hasattr(self, "_renderer"):
            del self._renderer


def make_env(cfg: dict | None = None, render_mode: str | None = None):
    return Go2LocomotionEnv(cfg=cfg, render_mode=render_mode)
