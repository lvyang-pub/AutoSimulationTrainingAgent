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

import importlib.util as _ilu
import os as _os

def _load_go2_env():
    import sys, types
    _env_dir = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "env"))
    _pkg_name = "go2_env_pkg"
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [_env_dir]
        _pkg.__package__ = _pkg_name
        sys.modules[_pkg_name] = _pkg
    for _sibling in ("fetch_go2", "reward"):
        _full = f"{_pkg_name}.{_sibling}"
        if _full not in sys.modules:
            _s = _ilu.spec_from_file_location(_full, _os.path.join(_env_dir, f"{_sibling}.py"))
            _m = _ilu.module_from_spec(_s)
            _m.__package__ = _pkg_name
            sys.modules[_full] = _m
            _s.loader.exec_module(_m)
    _full_env = f"{_pkg_name}.go2_env"
    if _full_env not in sys.modules:
        _spec = _ilu.spec_from_file_location(_full_env, _os.path.join(_env_dir, "go2_env.py"))
        _mod = _ilu.module_from_spec(_spec)
        _mod.__package__ = _pkg_name
        sys.modules[_full_env] = _mod
        _spec.loader.exec_module(_mod)
    return sys.modules[_full_env].Go2LocomotionEnv

Go2LocomotionEnv = _load_go2_env()


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


# Follow-camera rig: a FREE camera whose `lookat` is pinned to the robot base
# each frame, with the *offset* (azimuth / elevation / distance) held fixed. That
# gives a camera that tracks the robot while keeping a constant relative position
# and a globally fixed vertical pitch (elevation).
# MuJoCo's `elevation` sign is inverted: a POSITIVE value places the eye *below*
# the horizon looking up (you see the floor from underneath). So a viewpoint above
# the robot, looking down at it, needs a NEGATIVE elevation.
CAM_AZIMUTH = 90.0      # side-on view — best for judging a gait
CAM_ELEVATION = -20.0   # ABOVE the horizon, looking down at the robot
CAM_DISTANCE = 2.6      # metres from the lookat point
CAM_LOOKAT_Z = 0.28     # aim near the torso, not the floor
CAM_SMOOTH = 0.7        # EMA weight on the previous lookat (de-bounce); 0 = raw


def _base_body_id(model):
    """Id of the robot's trunk body ("base" in the Go2 menagerie model)."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    return bid if bid >= 0 else 1          # 1 = first body after world, fallback


def _make_follow_camera(model):
    """A FREE camera we reposition every frame (see the rig constants above)."""
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    cam.distance = CAM_DISTANCE
    cam.lookat[:] = (0.0, 0.0, CAM_LOOKAT_Z)
    return cam


def _render_once(policy, env_cfg, n_steps, seed, width, height,
                 frame_skip_render, use_camera):
    """One episode. Returns (frames, ended_early) or (None, False) on failure."""
    import mujoco

    env = Go2LocomotionEnv(cfg=env_cfg)
    renderer = None
    try:
        obs, _ = env.reset(seed=seed)
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        cam = None
        body_id = None
        if use_camera:
            cam = _make_follow_camera(env.model)
            body_id = _base_body_id(env.model)
        lookat = None
        frames = []
        ended_early = False
        for i in range(int(n_steps)):
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, _r, term, trunc, _info = env.step(action)
            if cam is not None:
                # follow the base, but de-bounce so the view does not judder
                cur = np.asarray(env.data.xpos[body_id], dtype=float).copy()
                cur[2] = CAM_LOOKAT_Z
                lookat = cur if lookat is None else (
                    CAM_SMOOTH * lookat + (1.0 - CAM_SMOOTH) * cur)
                cam.lookat[:] = lookat
            if i % frame_skip_render == 0:
                renderer.update_scene(env.data, camera=cam)
                frames.append(renderer.render().copy())
            if term or trunc:
                ended_early = term              # fell / episode cut short by a fall
                break
        return (frames or None), ended_early
    except Exception:  # noqa: BLE001 - visualization is best-effort
        return None, False
    finally:
        if renderer is not None:
            del renderer
        env.close()


def render_rollout(policy, env_cfg: dict, out_dir: str, stem: str = "rollout",
                   n_steps: int = 500, seed: int = 0, width: int = 320,
                   height: int = 240, fps: int = 30,
                   frame_skip_render: int = 2, follow_camera: bool = True,
                   min_duration_ratio: float = 0.9, max_attempts: int = 3) -> str | None:
    """Roll out `policy` (an SB3 model; None = random) and save a rollout video.

    The camera follows the robot (fixed relative position, fixed pitch — see the
    `CAM_*` rig constants). `frame_skip_render` renders every Nth control step to
    keep the file small.

    A rollout that terminates early almost always means the robot fell. When the
    rendered frame count falls below `min_duration_ratio` of the expected length,
    the episode is re-rendered with the next seed (up to `max_attempts` times) and
    the longest rollout wins — so a healthy policy is not judged on one unlucky
    seed. Returns the written path, or None if rendering failed.
    """
    expected = max(1, int(n_steps) // max(1, int(frame_skip_render)))
    best_frames: list | None = None
    for attempt in range(max(1, int(max_attempts))):
        frames, ended_early = _render_once(
            policy, env_cfg, n_steps, seed + attempt, width, height,
            frame_skip_render, follow_camera)
        if frames is None:
            continue
        if best_frames is None or len(frames) > len(best_frames):
            best_frames = frames
        # good enough: full-ish episode, or the episode never terminated early
        if not ended_early or len(frames) >= expected * min_duration_ratio:
            break
    if not best_frames:
        return None
    return _write_video(best_frames, out_dir, stem, fps)


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
        body_id = _base_body_id(env.model)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            # reuse the follow-camera rig so the window tracks the robot
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.azimuth = CAM_AZIMUTH
            viewer.cam.elevation = CAM_ELEVATION
            viewer.cam.distance = CAM_DISTANCE
            lookat = None
            for _ in range(int(n_steps)):
                if not viewer.is_running():
                    break
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                obs, _r, term, trunc, _info = env.step(action)
                cur = np.asarray(env.data.xpos[body_id], dtype=float).copy()
                cur[2] = CAM_LOOKAT_Z
                lookat = cur if lookat is None else (
                    CAM_SMOOTH * lookat + (1.0 - CAM_SMOOTH) * cur)
                viewer.cam.lookat[:] = lookat
                viewer.sync()
                if term or trunc:
                    obs, _ = env.reset(seed=seed)
                    lookat = None
    finally:
        env.close()
