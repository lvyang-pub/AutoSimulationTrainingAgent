"""One-shot environment setup: fetch the Go2 model and verify it loads + steps."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    from src.envs.fetch_go2 import fetch
    import mujoco
    import numpy as np
    from src.envs.go2_env import Go2LocomotionEnv

    print("[setup] fetching Go2 model ...")
    scene = fetch()
    m = mujoco.MjModel.from_xml_path(scene)
    print(f"[setup] model OK: nq={m.nq} nu={m.nu} timestep={m.opt.timestep}")

    print("[setup] smoke-testing env ...")
    env = Go2LocomotionEnv(cfg={"episode_length": 100})
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(100):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            break
    print(f"[setup] env OK: obs={obs.shape} return(100 steps)={total:.1f} "
          f"final_height={info['height']:.3f}")
    env.close()
    print("[setup] done.")


if __name__ == "__main__":
    main()
