"""Pilot at the ACTUAL task budget (20k): how much does the config matter?"""
import os, tempfile
import numpy as np
from stable_baselines3 import PPO
from src.training.train_ppo import train, evaluate
from src.training.config import normalize
from src.envs.go2_env import Go2LocomotionEnv


def measure(model, env_cfg, n=6):
    return evaluate(model, env_cfg, n_episodes=n, seed=100, policy_mode="greedy")


CONFIGS = {
    "weak_start": {"env": {"action_scale": 0.3, "kp": 30.0, "kd": 0.6,
                           "reward": {"action_rate": -0.02}}},
    "strong": {"env": {"action_scale": 0.5, "kp": 80.0, "kd": 1.6,
                       "reward": {"alive": 0.0, "track_lin_vel": 8.0, "orientation": -0.5,
                                  "action_rate": -0.001, "lin_vel_z": -0.5}}},
    "strong_hi_ent": {"env": {"action_scale": 0.6, "kp": 80.0, "kd": 1.6,
                              "reward": {"alive": 0.0, "track_lin_vel": 10.0, "orientation": -0.3}},
                      "ppo": {"ent_coef": 0.03, "learning_rate": 5e-4}},
    "low_kp": {"env": {"action_scale": 0.4, "kp": 50.0, "kd": 1.0,
                       "reward": {"alive": 0.0, "track_lin_vel": 6.0}}},
}

for name, cfg in CONFIGS.items():
    cfg = {"timesteps": 20000, "ppo": {"n_steps": 1024, "batch_size": 512}, **cfg}
    out = os.path.join(tempfile.gettempdir(), "pilot20k", name)
    s = train(cfg, out, verbose=0)
    nc, _ = normalize(cfg)
    m = PPO.load(os.path.join(out, "policy.zip"))
    st = measure(m, nc["env"])
    print(f"RESULT {name:14s} ret={s['mean_return']:7.1f} fall={s['fall_rate']:.2f} "
          f"| eval ret={st['mean_return']:7.1f} velerr={st['mean_lin_vel_error']:.3f} "
          f"fall={st['fall_rate']:.2f} len={st['mean_episode_length']:.0f}")
