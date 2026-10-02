"""With reward FIXED, do PPO/PD hyperparameters still differentiate return @20k?"""
import os, tempfile, copy
import numpy as np
from src.training.train_ppo import train
from src.training.config import DEFAULT_CONFIG

FIXED_REWARD = copy.deepcopy(DEFAULT_CONFIG["env"]["reward"])  # canonical, held constant

VARIANTS = {
    "default": {},
    "lr_1e3": {"ppo": {"learning_rate": 1e-3}},
    "lr_1e5": {"ppo": {"learning_rate": 1e-5}},
    "ent_0": {"ppo": {"ent_coef": 0.0}},
    "ent_03": {"ppo": {"ent_coef": 0.03}},
    "nsteps_2048": {"ppo": {"n_steps": 2048, "batch_size": 512}},
    "kp_20": {"env": {"kp": 20.0, "kd": 0.5}},
    "kp_120": {"env": {"kp": 120.0, "kd": 2.0}},
    "ascale_1.2": {"env": {"action_scale": 1.2}},
    "ascale_0.15": {"env": {"action_scale": 0.15}},
}

for name, patch in VARIANTS.items():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["timesteps"] = 20000
    for sec, vals in patch.items():
        cfg[sec].update(vals)
    cfg["env"]["reward"] = copy.deepcopy(FIXED_REWARD)
    out = os.path.join(tempfile.gettempdir(), "fixedrw", name)
    s = train(cfg, out, verbose=0)
    print(f"RESULT {name:14s} ret={s['mean_return']:8.2f} fall={s['fall_rate']:.2f} "
          f"velerr={s['mean_lin_vel_error']:.3f} len={s['mean_episode_length']:.0f}")
