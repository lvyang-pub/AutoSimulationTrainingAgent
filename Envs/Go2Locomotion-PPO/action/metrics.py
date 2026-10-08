"""Read-action functions for Go2Locomotion-PPO environment.

Each function returns structured data about completed training runs. Reading is
just another action on the environment, so these live alongside the write
actions in `action/` and are called the same way.
"""
from __future__ import annotations

import csv
import json
import os


def _run_dir(runs_root: str, run_id: str) -> str:
    return os.path.join(runs_root, run_id)


def get_trial_summary(runs_root: str, run_id: str) -> dict:
    """Return the summary.json of a completed training run."""
    path = os.path.join(_run_dir(runs_root, run_id), "summary.json")
    if not os.path.exists(path):
        return {"error": f"summary.json not found for run_id={run_id}"}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # strip heavy fields not needed for agent reasoning
    data.pop("eval_curve", None)
    data.pop("visualization", None)
    return {"run_id": run_id, **data}


def get_eval_curve(runs_root: str, run_id: str) -> dict:
    """Return the full eval curve (timestep → mean_return) for a run."""
    path = os.path.join(_run_dir(runs_root, run_id), "summary.json")
    if not os.path.exists(path):
        return {"error": f"summary.json not found for run_id={run_id}"}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    curve = data.get("eval_curve", [])
    trend = None
    if len(curve) >= 2:
        trend = round(curve[-1]["mean_return"] - curve[0]["mean_return"], 2)
    return {"run_id": run_id, "eval_curve": curve, "return_trend": trend,
            "final": curve[-1] if curve else None}


def list_trials(runs_root: str) -> dict:
    """List all completed trials under runs_root with key metrics."""
    rows = []
    if not os.path.isdir(runs_root):
        return {"trials": [], "n_trials": 0}
    for name in sorted(os.listdir(runs_root)):
        summary_path = os.path.join(runs_root, name, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        rows.append({
            "run_id": name,
            "mean_return": round(s.get("mean_return", 0), 2),
            "fall_rate": s.get("fall_rate"),
            "mean_lin_vel_error": round(s.get("mean_lin_vel_error", 0), 4),
            "train_seconds": s.get("train_seconds"),
        })
    return {"trials": rows, "n_trials": len(rows)}


def get_curve_path(runs_root: str, run_id: str) -> dict:
    """Return the path to the training curve PNG for a run, if it exists."""
    path = os.path.join(_run_dir(runs_root, run_id), "curve.png")
    if os.path.exists(path):
        return {"run_id": run_id, "curve_path": path}
    return {"run_id": run_id, "curve_path": None,
            "error": "curve.png not found (visualize=True required during training)"}


def get_config(runs_root: str, run_id: str) -> dict:
    """Return the normalized config used for a specific run."""
    path = os.path.join(_run_dir(runs_root, run_id), "config.json")
    if not os.path.exists(path):
        return {"error": f"config.json not found for run_id={run_id}"}
    with open(path, encoding="utf-8") as f:
        return {"run_id": run_id, "config": json.load(f)}
