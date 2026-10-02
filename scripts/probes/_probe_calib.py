"""Calibrate the task goal: naive default vs a spread of random configs @20k."""
import os, tempfile, random, copy
import numpy as np
from src.training.train_ppo import train
from src.training.config import DEFAULT_CONFIG, SPEC


def run(cfg, tag):
    cfg = {"timesteps": 20000, **cfg}
    out = os.path.join(tempfile.gettempdir(), "calib", tag)
    s = train(cfg, out, verbose=0)
    print(f"RESULT {tag:16s} ret={s['mean_return']:8.2f} fall={s['fall_rate']:.2f} "
          f"velerr={s['mean_lin_vel_error']:.3f}")
    return s["mean_return"]


def setd(cfg, dotted, val):
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = val


# naive default
run({}, "naive_default")

# a spread of random configs over the spec
rng = random.Random(0)
rets = []
for i in range(6):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, (lo, hi) in SPEC.items():
        if key in ("seed", "timesteps"):
            continue
        setd(cfg, key, rng.uniform(lo, hi))
    if cfg["ppo"]["batch_size"] > cfg["ppo"]["n_steps"]:
        cfg["ppo"]["batch_size"] = cfg["ppo"]["n_steps"]
    cfg["seed"] = 0
    rets.append(run(cfg, f"random_{i}"))

print(f"RESULT SUMMARY random mean={np.mean(rets):.1f} min={min(rets):.1f} max={max(rets):.1f}")
