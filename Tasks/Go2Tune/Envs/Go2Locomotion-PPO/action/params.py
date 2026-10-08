"""Read actions for parameter inspection.

Read-only — never mutates state. Called by the agent like any other action.
"""
from __future__ import annotations

import os
import sys


def _import_config():
    here = os.path.dirname(os.path.abspath(__file__))
    env_root = os.path.abspath(os.path.join(here, ".."))
    if env_root not in sys.path:
        sys.path.insert(0, env_root)
    import training.config as _c
    return _c


def search_space_description() -> str:
    """Return a human-readable description of the tunable search space."""
    return _import_config().search_space_description()


def get_param_spec() -> dict:
    """Return the allowed range (spec) for each tunable parameter."""
    return {"spec": _import_config()._load_spec()}
