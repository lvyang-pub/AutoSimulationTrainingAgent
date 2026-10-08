"""Action interfaces for training operations.

Callable by the agent via Harness/exec.py. Delegates to training/ internals.
"""
from __future__ import annotations

import os
import sys

def _import_training():
    here = os.path.dirname(os.path.abspath(__file__))
    env_root = os.path.abspath(os.path.join(here, ".."))
    if env_root not in sys.path:
        sys.path.insert(0, env_root)
    import training.train_ppo as _t
    import training.config as _c
    return _t, _c


def run_training(config: dict, out_dir: str, verbose: int = 0,
                 visualize: bool = True) -> dict:
    """Train one PPO run. Returns summary dict."""
    _t, _ = _import_training()
    return _t.train(config=config, out_dir=out_dir, verbose=verbose,
                    visualize=visualize)


def normalize_config(user_cfg: dict) -> tuple[dict, list[str]]:
    """Validate and clamp a user config. Returns (normalized_cfg, warnings)."""
    _, _c = _import_training()
    return _c.normalize(user_cfg)


def evaluate_policy(policy_path: str, env_cfg: dict, n_episodes: int = 10,
                    seed: int = 31337, policy_mode: str = "greedy") -> dict:
    """Load a saved policy and evaluate it. Returns stats dict."""
    _t, _ = _import_training()
    return _t.evaluate_from_zip(policy_path=policy_path, env_cfg=env_cfg,
                                n_episodes=n_episodes, seed=seed,
                                policy_mode=policy_mode)
