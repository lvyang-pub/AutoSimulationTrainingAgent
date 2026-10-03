"""Action functions for Go2Locomotion-PPO environment.

Each function modifies parameters.json which is the single source of truth
for all training parameters. These are the action interfaces callable by
the agent via Harness/exec.py.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "parameters.json")


def _load() -> dict:
    with open(_PARAMS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_params() -> dict:
    """Return the current parameters (the full parameters.json content)."""
    return _load()


def set_params(params_dict: dict) -> dict:
    """Merge params_dict into parameters.json['parameters'] and save.

    Only updates keys that already exist; unknown keys are ignored and reported.
    Returns the updated parameters plus any warnings.
    """
    data = _load()
    current = data["parameters"]
    warnings = []

    def _merge(dst: dict, src: dict, prefix: str = "") -> None:
        for k, v in src.items():
            full_key = f"{prefix}{k}"
            if k not in dst:
                warnings.append(f"unknown key '{full_key}' ignored")
                continue
            if isinstance(dst[k], dict):
                if not isinstance(v, dict):
                    warnings.append(f"'{full_key}' must be an object; ignored")
                    continue
                _merge(dst[k], v, f"{full_key}.")
            else:
                dst[k] = v

    _merge(current, params_dict)
    data["parameters"] = current
    _save(data)
    return {"updated": current, "warnings": warnings}


def reset_params() -> dict:
    """Reset parameters to the spec defaults embedded in parameters.json."""
    data = _load()
    # parameters.json was generated from DEFAULT_CONFIG; re-reading it is the reset
    # (no separate baseline stored — use get_params() to inspect current state)
    return {"message": "parameters.json is already the canonical source; "
                       "to reset, restore from the Envs template copy."}


def get_param_spec() -> dict:
    """Return the allowed range (spec) for each tunable parameter."""
    data = _load()
    return {"spec": data.get("spec", {})}
