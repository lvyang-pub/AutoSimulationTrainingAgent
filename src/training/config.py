"""Training config schema: defaults, bounds, and validation.

The schema defines the agent's search space. `normalize` accepts a possibly
partial / slightly-out-of-range config, clamps numeric values to their allowed
range, and returns the normalized config plus a list of warnings — so the agent
gets actionable feedback instead of a hard crash.
"""
from __future__ import annotations

from copy import deepcopy

# --------------------------------------------------------------------------- #
# Default configuration
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: dict = {
    "timesteps": 40000,
    "seed": 0,
    "net_arch": [64, 64],
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "env": {
        "dt": 0.02,
        "frame_skip": 4,
        "episode_length": 500,
        "action_scale": 0.4,
        "kp": 80.0,
        "kd": 1.6,
        "max_torque": 23.7,
        "min_height": 0.18,
        "max_tilt": 0.7,
        "terminate_on_fall": True,
        "command": [1.0, 0.0, 0.0],
        "reward": {
            "track_lin_vel": 2.0,
            "track_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
            "action_rate": -0.005,
            "torques": -0.0001,
            "base_height": -2.0,
            "alive": 0.2,
        },
    },
}

# (low, high) inclusive bounds for every tunable numeric leaf.
SPEC: dict[str, tuple[float, float]] = {
    "timesteps": (4000, 2_000_000),
    "seed": (0, 100_000),
    "ppo.learning_rate": (1e-5, 1e-2),
    "ppo.n_steps": (128, 4096),
    "ppo.batch_size": (32, 1024),
    "ppo.n_epochs": (1, 30),
    "ppo.gamma": (0.9, 0.9999),
    "ppo.gae_lambda": (0.8, 1.0),
    "ppo.clip_range": (0.05, 0.6),
    "ppo.ent_coef": (0.0, 0.05),
    "ppo.vf_coef": (0.1, 1.0),
    "ppo.max_grad_norm": (0.1, 2.0),
    "env.dt": (0.005, 0.05),
    "env.frame_skip": (1, 20),
    "env.episode_length": (100, 2000),
    "env.action_scale": (0.1, 1.5),
    "env.kp": (5.0, 120.0),
    "env.kd": (0.1, 5.0),
    "env.max_torque": (5.0, 45.0),
    "env.min_height": (0.08, 0.26),
    "env.max_tilt": (0.3, 1.2),
    "env.reward.track_lin_vel": (0.0, 5.0),
    "env.reward.track_ang_vel": (0.0, 2.0),
    "env.reward.lin_vel_z": (-10.0, 0.0),
    "env.reward.ang_vel_xy": (-1.0, 0.0),
    "env.reward.orientation": (-10.0, 0.0),
    "env.reward.action_rate": (-0.1, 0.0),
    "env.reward.torques": (-0.005, 0.0),
    "env.reward.base_height": (-50.0, 0.0),
    "env.reward.alive": (0.0, 2.0),
}

REWARD_KEYS = list(DEFAULT_CONFIG["env"]["reward"].keys())


def _get(d: dict, dotted: str):
    cur = d
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def _set(d: dict, dotted: str, val) -> None:
    parts = dotted.split(".")
    cur = d
    for k in parts[:-1]:
        cur = cur[k]
    cur[parts[-1]] = val


def normalize(user_cfg: dict | None) -> tuple[dict, list[str]]:
    """Merge a user config onto defaults, validate, and clamp.

    Returns (config, warnings). Raises ValueError only for structurally
    unusable input (e.g. non-dict ppo/env).
    """
    warnings: list[str] = []
    cfg = deepcopy(DEFAULT_CONFIG)
    if not user_cfg:
        return cfg, warnings

    for section in ("ppo", "env"):
        if section in user_cfg and not isinstance(user_cfg[section], dict):
            raise ValueError(f"'{section}' must be an object, got {type(user_cfg[section]).__name__}")

    def merge(dst: dict, src: dict, prefix: str) -> None:
        for k, v in src.items():
            if k not in dst:
                warnings.append(f"unknown key '{prefix}{k}' ignored")
                continue
            if isinstance(dst[k], dict):
                if not isinstance(v, dict):
                    warnings.append(f"'{prefix}{k}' must be an object; ignored")
                    continue
                merge(dst[k], v, f"{prefix}{k}.")
            else:
                dst[k] = v

    for k, v in user_cfg.items():
        if k not in cfg:
            warnings.append(f"unknown key '{k}' ignored")
            continue
        if isinstance(cfg[k], dict):
            if not isinstance(v, dict):
                warnings.append(f"'{k}' must be an object; ignored")
                continue
            merge(cfg[k], v, f"{k}.")
        else:
            cfg[k] = v

    # net_arch: list of positive ints
    na = cfg.get("net_arch")
    if not (isinstance(na, list) and na and all(isinstance(x, int) and x > 0 for x in na)):
        warnings.append(f"net_arch {na!r} invalid; using {DEFAULT_CONFIG['net_arch']}")
        cfg["net_arch"] = deepcopy(DEFAULT_CONFIG["net_arch"])

    # clamp numeric leaves
    for key, (lo, hi) in SPEC.items():
        try:
            val = float(_get(cfg, key))
        except (KeyError, TypeError, ValueError):
            warnings.append(f"'{key}' non-numeric; using default")
            _set(cfg, key, _get(DEFAULT_CONFIG, key))
            continue
        if val < lo or val > hi:
            clamped = min(max(val, lo), hi)
            warnings.append(f"'{key}'={val} out of [{lo}, {hi}]; clamped to {clamped}")
            val = clamped
        if isinstance(_get(DEFAULT_CONFIG, key), int):
            val = int(round(val))
        _set(cfg, key, val)

    # structural invariants
    if cfg["ppo"]["batch_size"] > cfg["ppo"]["n_steps"]:
        warnings.append(
            f"batch_size {cfg['ppo']['batch_size']} > n_steps {cfg['ppo']['n_steps']}; "
            f"n_steps raised to batch_size")
        cfg["ppo"]["n_steps"] = cfg["ppo"]["batch_size"]

    if len(cfg["env"]["command"]) != 3:
        warnings.append("env.command must have 3 entries; using default")
        cfg["env"]["command"] = deepcopy(DEFAULT_CONFIG["env"]["command"])

    return cfg, warnings


def search_space_description() -> str:
    """Human/LLM-readable summary of the tunable space, for prompts."""
    lines = []
    for key, (lo, hi) in SPEC.items():
        default = _get(DEFAULT_CONFIG, key)
        lines.append(f"  - {key}: [{lo}, {hi}]  (default {default})")
    lines.append(f"  - net_arch: list[int], e.g. [64, 64] or [128, 128]  (default {DEFAULT_CONFIG['net_arch']})")
    return "\n".join(lines)
