"""Post-training visualization: a rollout video and a learning-curve PNG.

Rendering is done off-screen with `mujoco.Renderer`. The rollout is written as
MP4 via OpenCV when available, falling back to an animated GIF via Pillow
otherwise — so this works headless and needs no dependencies beyond what the
project already installs. Both helpers return the path they wrote (or None on
failure) rather than raising, so a visualization problem never breaks a run.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from ..envs.go2_env import Go2LocomotionEnv


def _write_video(frames: list, out_dir: str, stem: str, fps: int) -> str | None:
    """Write frames to <out_dir>/<stem>.mp4 (cv2) or .gif (Pillow). Best-effort."""
    os.makedirs(out_dir or ".", exist_ok=True)
    try:
        import cv2

        path = os.path.join(out_dir, f"{stem}.mp4")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        return path if os.path.exists(path) and os.path.getsize(path) > 0 else None
    except Exception:  # noqa: BLE001 - fall back to GIF
        pass
    try:
        from PIL import Image

        path = os.path.join(out_dir, f"{stem}.gif")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=max(1, int(1000 / max(fps, 1))), loop=0, optimize=True)
        return path
    except Exception:  # noqa: BLE001
        return None


def render_rollout(policy, env_cfg: dict, out_dir: str, stem: str = "rollout",
                   n_steps: int = 500, seed: int = 0, width: int = 320,
                   height: int = 240, fps: int = 30,
                   frame_skip_render: int = 2) -> str | None:
    """Roll out `policy` (an SB3 model; None = random) and save a rollout video.

    `frame_skip_render` renders every Nth control step to keep the file small.
    Returns the written path, or None if rendering failed.
    """
    import mujoco

    env = Go2LocomotionEnv(cfg=env_cfg)
    renderer = None
    try:
        obs, _ = env.reset(seed=seed)
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        frames = []
        for i in range(int(n_steps)):
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, _r, term, trunc, _info = env.step(action)
            if i % frame_skip_render == 0:
                renderer.update_scene(env.data)
                frames.append(renderer.render().copy())
            if term or trunc:
                break
        if not frames:
            return None
        return _write_video(frames, out_dir, stem, fps)
    except Exception:  # noqa: BLE001 - visualization is best-effort
        return None
    finally:
        if renderer is not None:
            del renderer
        env.close()


def plot_training_curve(metrics_csv: str, out_path: str) -> str | None:
    """Plot the periodic eval return (and episode returns) vs timesteps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evals_t, evals_r = [], []
    ep_t, ep_r = [], []
    try:
        with open(metrics_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["kind"] == "eval":
                    evals_t.append(int(row["timesteps"]))
                    evals_r.append(float(row["mean_return"]))
                elif row["kind"] == "episode":
                    ep_t.append(int(row["timesteps"]))
                    ep_r.append(float(row["episode_reward"]))
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=110)
        if ep_r:
            ax.plot(ep_t, ep_r, color="#bbbbbb", lw=0.8, label="episode return")
        if evals_r:
            ax.plot(evals_t, evals_r, color="#1f77b4", lw=2.0, marker="o",
                    ms=3, label="greedy eval return")
        ax.set_xlabel("timesteps")
        ax.set_ylabel("return")
        ax.set_title("Go2 PPO training curve")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception:  # noqa: BLE001
        return None


def live_view(policy, env_cfg: dict, seed: int = 0, n_steps: int = 500) -> None:
    """Open an interactive MuJoCo window. Only for one-off foreground runs."""
    import mujoco.viewer

    env = Go2LocomotionEnv(cfg=env_cfg)
    try:
        obs, _ = env.reset(seed=seed)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for _ in range(int(n_steps)):
                if not viewer.is_running():
                    break
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                obs, _r, term, trunc, _info = env.step(action)
                viewer.sync()
                if term or trunc:
                    obs, _ = env.reset(seed=seed)
    finally:
        env.close()
