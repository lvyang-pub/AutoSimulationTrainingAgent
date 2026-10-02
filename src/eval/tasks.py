"""Reproducible tuning tasks for evaluation.

The agent tunes **training hyperparameters** (PPO + the PD/plant parameters) to
maximize the training return of a short PPO run. The reward function is fixed by
the task (carried in `start_config.env.reward`) so that returns are directly
comparable across trials — otherwise each run would be scoring itself with a
different yardstick.

Each task also pins the **starting config to a deliberately weak one** (low PD
gains / small action scale), which is what the naive baseline trains with. The
agent's job is to recover from that weak start. A fixed seed makes a task's
outcome reproducible, which is what lets the report compare agent versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Canonical reward, held fixed across a task's trials (see module docstring).
FIXED_REWARD = {
    "track_lin_vel": 2.0,
    "track_ang_vel": 0.2,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "action_rate": -0.005,
    "torques": -0.0001,
    "base_height": -2.0,
    "alive": 0.2,
}

# A weak plant: sagging PD gains and a small action scale. Naive training with
# this lands ~145-150 return; a well-tuned config reaches ~195+.
WEAK_START = {
    "env": {"action_scale": 0.3, "kp": 30.0, "kd": 0.6, "reward": dict(FIXED_REWARD)},
    "ppo": {"learning_rate": 3e-4},
}

# The tunable layer, for prompts / hints.
TUNING_HINT = (
    "Hold the reward fixed (do not edit env.reward). Tune training hyperparameters: "
    "ppo.{learning_rate,n_steps,batch_size,n_epochs,gamma,gae_lambda,clip_range,ent_coef,"
    "vf_coef,max_grad_norm}, net_arch, and the PD/plant env.{kp,kd,action_scale,"
    "max_torque,episode_length}. Known cliffs to avoid: learning_rate above ~5e-4 "
    "collapses the run; kp far above ~100 or below ~40, or action_scale above ~0.8, "
    "hurt the return badly."
)


@dataclass
class Task:
    name: str
    command: list[float]                 # [vx, vy, yaw_rate] (context; reward is fixed)
    goals: dict                          # success thresholds
    default_timesteps: int               # steps per trial
    max_trials: int                      # training runs allowed
    seed: int
    start_config: dict = field(default_factory=dict)
    hint: str = TUNING_HINT

    def command_text(self) -> str:
        return f"{self.command[0]:.2f} m/s forward"


TASKS: dict[str, Task] = {
    "go2_tune_v1": Task(
        name="go2_tune_v1",
        command=[1.0, 0.0, 0.0],
        goals={"mean_return_min": 190.0},
        default_timesteps=20000,
        max_trials=6,
        seed=0,
        start_config=WEAK_START,
    ),
    "go2_tune_hard": Task(
        name="go2_tune_hard",
        command=[1.0, 0.0, 0.0],
        goals={"mean_return_min": 192.0},
        default_timesteps=20000,
        max_trials=5,
        seed=1,
        start_config=WEAK_START,
    ),
    "go2_tune_tight": Task(
        name="go2_tune_tight",
        command=[1.0, 0.0, 0.0],
        goals={"mean_return_min": 195.0, "fall_rate_max": 0.1},
        default_timesteps=20000,
        max_trials=8,
        seed=2,
        start_config=WEAK_START,
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task '{name}'. available: {list(TASKS)}")
    return TASKS[name]
