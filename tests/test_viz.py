"""Tests for the post-training visualization helpers.

Fast tests cover the parsing/plotting path and the "never raise" contract on
bad input. The rollout renderer spins up MuJoCo + a policy, so it is marked
slow (run with -m slow).
"""
from __future__ import annotations

import csv
import os

import pytest

from src.training import viz

_METRICS_HEADER = ["kind", "timesteps", "episode_reward", "episode_length",
                   "mean_return", "std_return", "fall_rate", "mean_lin_vel_error"]


def _write_metrics(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_METRICS_HEADER)
        for r in rows:
            w.writerow(r)


def test_plot_curve_writes_png(tmp_path):
    csv_path = str(tmp_path / "metrics.csv")
    _write_metrics(csv_path, [
        ["episode", 100, "1.5", "50", "", "", "", ""],
        ["episode", 200, "-0.3", "20", "", "", "", ""],
        ["eval", 500, "", "", "10.0", "1.0", "0.0", "0.10"],
        ["eval", 1000, "", "", "42.0", "2.0", "0.0", "0.05"],
    ])
    out = str(tmp_path / "curve.png")
    got = viz.plot_training_curve(csv_path, out)
    assert got == out
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_curve_never_raises_on_missing_file(tmp_path):
    """A visualization problem must not break a run."""
    assert viz.plot_training_curve(str(tmp_path / "nope.csv"),
                                   str(tmp_path / "c.png")) is None


def test_plot_curve_never_raises_on_garbage(tmp_path):
    csv_path = str(tmp_path / "bad.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("this,is\nnot,a,valid,metrics,file\n")
    assert viz.plot_training_curve(csv_path, str(tmp_path / "c.png")) is None


def test_write_video_falls_back_or_writes(tmp_path):
    """_write_video writes either an MP4 or a GIF; both beat returning None."""
    import numpy as np
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(4)]
    for i, f in enumerate(frames):
        f[:] = i * 40
    path = viz._write_video(frames, str(tmp_path), "clip", fps=10)
    assert path is not None
    assert path.endswith((".mp4", ".gif"))
    assert os.path.exists(path) and os.path.getsize(path) > 0


@pytest.mark.slow
def test_render_rollout_writes_video(tmp_path):
    """End-to-end: a random policy rolls out and a video file lands on disk."""
    env_cfg = {"episode_length": 30}
    path = viz.render_rollout(None, env_cfg, str(tmp_path), stem="rollout",
                              n_steps=30, fps=10, frame_skip_render=1)
    assert path is not None
    assert path.endswith((".mp4", ".gif"))
    assert os.path.getsize(path) > 0
