"""Probe: can reward shaping make forward motion emerge / differentiate configs?"""
import os, tempfile
import numpy as np
from stable_baselines3 import PPO
from src.training.train_ppo import train
from src.training.config import normalize
from src.envs.go2_env import Go2LocomotionEnv


def travel(model, cfg, n=4):
    env = Go2LocomotionEnv(cfg=cfg); dists = []; vels = []
    for ep in range(n):
        obs, _ = env.reset(seed=300 + ep); done = False; x0 = env.data.qpos[0]; vs = []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            vs.append(info["vx"]); done = term or trunc
        dists.append(env.data.qpos[0] - x0); vels.append(np.mean(vs))
    env.close()
    return float(np.mean(dists)), float(np.mean(vels))


CONFIGS = {
    "A": {"env": {"reward": {"alive": 0.0, "track_lin_vel": 4.0, "orientation": -0.5,
                             "action_rate": -0.001, "lin_vel_z": -0.5}}},
    "B": {"env": {"reward": {"alive": 0.0, "track_lin_vel": 8.0, "orientation": -0.2,
                             "action_rate": -0.0005, "ang_vel_xy": -0.02, "torques": -0.00005}}},
    "C": {"env": {"reward": {"alive": 0.2, "track_lin_vel": 8.0, "orientation": -0.5}}},
}

for name, extra in CONFIGS.items():
    cfg = {"timesteps": 60000, "ppo": {"n_steps": 2048, "batch_size": 512, "ent_coef": 0.03,
                                       "learning_rate": 5e-4},
           "env": {"init_vel_forward": 1.0, "action_scale": 0.5, **extra.get("env", {})}}
    out = os.path.join(tempfile.gettempdir(), "rw2", name)
    s = train(cfg, out, verbose=0)
    nc, _ = normalize(cfg)
    m = PPO.load(os.path.join(out, "policy.zip"))
    dist, v = travel(m, nc["env"])
    print(f"RESULT cfg={name} ret={s['mean_return']:7.1f} fall={s['fall_rate']:.2f} "
          f"travel={dist:6.2f}m vx={v:6.3f}")
