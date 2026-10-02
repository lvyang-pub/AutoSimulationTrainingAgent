"""Train one PPO run on Go2LocomotionEnv and record its results.

Products of a run (in out_dir):
  config.json   - the normalized config actually used
  metrics.csv   - training log (episode returns) + periodic eval returns
  policy.zip    - final policy
  summary.json  - the structured result the agent reasons over

Both this module and the agent's `run_training` tool call `train()`.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from ..envs.go2_env import Go2LocomotionEnv
from .config import normalize

os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _make_env(cfg: dict):
    return lambda: Go2LocomotionEnv(cfg=cfg)


class _MetricsCallback(BaseCallback):
    """Records episode returns during training and periodic greedy-eval returns."""

    def __init__(self, eval_env_cfg: dict, eval_freq: int, n_eval_episodes: int, seed: int):
        super().__init__()
        self.eval_env_cfg = eval_env_cfg
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.episodes: list[dict] = []   # per finished training episode
        self.evals: list[dict] = []      # periodic greedy evaluation
        self._next_eval = eval_freq
        self._eval_env = None

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episodes.append({
                    "timesteps": int(self.num_timesteps),
                    "episode_reward": float(ep["r"]),
                    "episode_length": int(ep["l"]),
                })
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            stats = evaluate(self.model, self.eval_env_cfg, self.n_eval_episodes,
                             seed=self.seed + 999, policy_mode="greedy")
            stats["timesteps"] = int(self.num_timesteps)
            self.evals.append(stats)
        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()


def evaluate(policy, env_cfg: dict, n_episodes: int, seed: int = 0,
             policy_mode: str = "greedy") -> dict:
    """Run `policy` (an SB3 model) over n_episodes. Returns aggregate stats."""
    deterministic = policy_mode == "greedy"
    returns, lengths, fell, lin_errors = [], [], 0, []
    env = Go2LocomotionEnv(cfg=env_cfg)
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            R, n = 0.0, 0
            while not done:
                action, _ = policy.predict(obs, deterministic=deterministic)
                obs, r, term, trunc, info = env.step(action)
                R += r
                n += 1
                lin_errors.append(info["lin_vel_error"])
                if term:
                    fell += 1
                done = term or trunc
            returns.append(R)
            lengths.append(n)
    finally:
        env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "fall_rate": float(fell / n_episodes),
        "mean_lin_vel_error": float(np.mean(lin_errors)),
        "n_episodes": n_episodes,
    }


def train(config: dict, out_dir: str, verbose: int = 0,
          progress_cb=None, eval_freq: int | None = None,
          visualize: bool = True) -> dict:
    """Train one run. `config` may be partial; it is normalized internally.

    With `visualize=True` (default), one deterministic rollout is rendered to a
    short video and a learning-curve PNG is written into `out_dir` after training.
    Visualization is best-effort and never affects the returned summary.
    """
    cfg, warnings = normalize(config)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    seed = int(cfg["seed"])
    np.random.seed(seed)

    env = VecMonitor(DummyVecEnv([_make_env(cfg["env"])]))
    env.seed(seed)
    ppo = cfg["ppo"]
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo["learning_rate"]),
        n_steps=int(ppo["n_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["ent_coef"]),
        vf_coef=float(ppo["vf_coef"]),
        max_grad_norm=float(ppo["max_grad_norm"]),
        policy_kwargs=dict(net_arch=list(cfg["net_arch"])),
        seed=seed,
        verbose=verbose,
        device="cpu",
    )

    timesteps = int(cfg["timesteps"])
    if eval_freq is None:
        eval_freq = max(int(ppo["n_steps"]) * 4, timesteps // 10)

    cb = _MetricsCallback(cfg["env"], eval_freq=eval_freq, n_eval_episodes=5, seed=seed)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=bool(verbose))
    train_seconds = time.time() - t0

    model.save(os.path.join(out_dir, "policy.zip"))

    # final deterministic evaluation on fresh seeds
    final = evaluate(model, cfg["env"], n_episodes=10, seed=seed + 12345, policy_mode="greedy")

    # write metrics.csv
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "timesteps", "episode_reward", "episode_length",
                    "mean_return", "std_return", "fall_rate", "mean_lin_vel_error"])
        for e in cb.episodes:
            w.writerow(["episode", e["timesteps"], f"{e['episode_reward']:.3f}",
                        e["episode_length"], "", "", "", ""])
        for e in cb.evals:
            w.writerow(["eval", e["timesteps"], "", "", f"{e['mean_return']:.3f}",
                        f"{e['std_return']:.3f}", f"{e['fall_rate']:.3f}",
                        f"{e['mean_lin_vel_error']:.4f}"])

    # post-training visualization: one deterministic rollout video + curve PNG.
    # Best-effort; a rendering failure leaves these as None.
    rollout_path = curve_path = None
    if visualize:
        from . import viz
        rollout_path = viz.render_rollout(
            model, cfg["env"], out_dir, stem="rollout",
            n_steps=int(cfg["env"].get("episode_length", 500)),
            seed=seed + 54321, fps=30)
        curve_path = viz.plot_training_curve(
            os.path.join(out_dir, "metrics.csv"),
            os.path.join(out_dir, "curve.png"))

    # convergence signal: improvement of the last quarter of evals vs the first
    evals = cb.evals
    convergence = None
    if len(evals) >= 4:
        q = max(1, len(evals) // 4)
        first = float(np.mean([e["mean_return"] for e in evals[:q]]))
        last = float(np.mean([e["mean_return"] for e in evals[-q:]]))
        convergence = last - first

    summary = {
        "mean_return": final["mean_return"],
        "std_return": final["std_return"],
        "fall_rate": final["fall_rate"],
        "mean_lin_vel_error": final["mean_lin_vel_error"],
        "mean_episode_length": final["mean_episode_length"],
        "train_seconds": round(train_seconds, 2),
        "timesteps": timesteps,
        "n_episodes_logged": len(cb.episodes),
        "eval_curve": [{"t": e["timesteps"], "mean_return": round(e["mean_return"], 2)}
                       for e in evals],
        "convergence_delta": round(convergence, 2) if convergence is not None else None,
        "visualization": {"rollout": rollout_path, "curve": curve_path},
        "warnings": warnings,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    env.close()
    if progress_cb:
        progress_cb(summary)
    return summary


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train PPO on Go2 locomotion.")
    ap.add_argument("--config", help="path to a JSON config (optional)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--timesteps", type=int, help="override timesteps")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--no-viz", action="store_true",
                    help="skip the post-training rollout video + curve")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            cfg = json.load(fh)
    if args.timesteps:
        cfg["timesteps"] = args.timesteps

    summary = train(cfg, args.out, verbose=args.verbose, visualize=not args.no_viz)
    print(json.dumps(summary, indent=2))
    vis = summary.get("visualization") or {}
    for kind, path in vis.items():
        if path:
            print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
