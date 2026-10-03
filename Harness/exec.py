"""Harness execution bridge — isolates agent code from env code.

The agent calls exec.call(env_path, module_rel, func_name, **kwargs) instead
of importing env modules directly. This keeps the agent reusable across
different environments without any env-specific imports.

env_path:    absolute or relative path to the env root (e.g. Envs/Go2Locomotion-PPO)
module_rel:  relative path to the .py file within env_path (e.g. observation/metrics.py)
func_name:   name of the function to call
**kwargs:    keyword arguments forwarded to the function

Returns the function's return value.
"""
from __future__ import annotations

import importlib.util
import os
import sys


def call(env_path: str, module_rel: str, func_name: str, **kwargs):
    """Dynamically load a function from env_path/module_rel and call it."""
    env_path = os.path.abspath(env_path)
    module_path = os.path.join(env_path, module_rel)
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")

    # derive a unique module name to avoid cache collisions across envs
    module_name = "harness_dyn_" + module_path.replace(os.sep, "_").replace(":", "").replace(".", "_")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)

    # temporarily add env root to sys.path so relative imports within the
    # env module work (e.g. env code importing from its own sibling packages)
    _added = []
    if env_path not in sys.path:
        sys.path.insert(0, env_path)
        _added.append(env_path)
    parent = os.path.dirname(env_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)
        _added.append(parent)
    try:
        spec.loader.exec_module(mod)
    finally:
        for p in _added:
            if p in sys.path:
                sys.path.remove(p)

    fn = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(f"Function '{func_name}' not found in {module_path}")
    return fn(**kwargs)


def list_functions(env_path: str, module_rel: str) -> list[str]:
    """Return the list of public functions defined in env_path/module_rel."""
    env_path = os.path.abspath(env_path)
    module_path = os.path.join(env_path, module_rel)
    if not os.path.exists(module_path):
        return []
    module_name = "harness_list_" + module_path.replace(os.sep, "_").replace(":", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [name for name in dir(mod)
            if callable(getattr(mod, name)) and not name.startswith("_")]
