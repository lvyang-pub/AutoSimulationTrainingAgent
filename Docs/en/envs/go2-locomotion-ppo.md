# Go2Locomotion-PPO

## Overview

A MuJoCo-based training environment that uses the PPO algorithm to train a Unitree Go2 robot dog to walk forward at 1 m/s on flat terrain.

## Environment Design

### Scene Design

Flat terrain with the following parameters:

| Parameter | Value |
|---|---|
| Gravity | 9.81 m/s² |
| Floor friction | 0.2 |
| Simulation timestep | 0.005 s (dt=0.02, frame_skip=4) |

### Plant

Standard Unitree Go2 quadruped robot, body mass 6.921 kg, initialized in the default standing pose.

### Training Algorithm

PPO algorithm with the following parameters:

| Parameter | Value |
|---|---|
| learning_rate | 3e-4 |
| batch_size | 256 |
| n_epochs | 5 |
| gamma | 0.99 |

Trains a two-layer feedforward network (FNN) with hidden size 64×64 and tanh activation.

### Interface Design

**Observation space**: 33-dimensional continuous vector

| Dims | Content |
|---|---|
| 3 | Base angular velocity (body frame) |
| 3 | Gravity vector projected into body frame |
| 12 | Joint angles |
| 12 | Joint velocities |
| 3 | Previous action |

**Action space**: 12-dimensional continuous vector in [-1, 1], mapped to target joint angles and converted to joint torques via a PD controller.



## API Design

**`get_trial_summary`**: Get core metrics for a completed training run

| Parameter | Type | Description |
|---|---|---|
| `runs_root` | str | Root directory of training results |
| `run_id` | str | Target run identifier |

| Return Field | Description |
|---|---|
| `mean_return` | Mean cumulative reward |
| `fall_rate` | Fall rate |
| `mean_lin_vel_error` | Mean linear velocity tracking error |

---

**`get_eval_curve`**: Get the evaluation curve and trend values for a run

| Parameter | Type | Description |
|---|---|---|
| `runs_root` | str | Root directory of training results |
| `run_id` | str | Target run identifier |

| Return Field | Description |
|---|---|
| (curve data) | Per-stage evaluation metrics and trend statistics |

---

**`list_trials`**: List all completed training runs with key metrics

| Parameter | Type | Description |
|---|---|---|
| `runs_root` | str | Root directory of training results |

| Return Field | Description |
|---|---|
| (trial list) | run_id and core metrics for all historical runs |

---

**`get_config`**: Get the actual configuration used in a training run

| Parameter | Type | Description |
|---|---|---|
| `runs_root` | str | Root directory of training results |
| `run_id` | str | Target run identifier |

| Return Field | Description |
|---|---|
| (config content) | Full hyperparameter configuration used in that run |

---

**`get_params`**: Read the full contents of the current config file

| Parameter | Description |
|---|---|
| — | No parameters required |

| Return Field | Description |
|---|---|
| (config content) | Full contents of parameters.json |

---

**`set_params`**: Merge new parameters into the config file

| Parameter | Type | Description |
|---|---|---|
| `params_dict` | dict | Key-value pairs to update |

| Return Field | Description |
|---|---|
| (merge result) | Updated parameter values and validation warnings |

---

**`get_param_spec`**: Get the valid range for each tunable parameter

| Parameter | Description |
|---|---|
| — | No parameters required |

| Return Field | Description |
|---|---|
| (param spec) | min / max range for each tunable parameter |

---

**`reset_params`**: Restore the config file to the default template

| Parameter | Description |
|---|---|
| — | No parameters required |

| Return Field | Description |
|---|---|
| (message) | Result of the reset operation |

## Config File

```json
{
  "parameters": {
    "timesteps": 200000,
    "seed": 0,
    "n_envs": 4,
    "net_arch": [64, 64],
    "ppo": {
      "learning_rate": 0.0003,
      "n_steps": 256,
      "batch_size": 256,
      "n_epochs": 5,
      "gamma": 0.99,
      "gae_lambda": 0.95,
      "clip_range": 0.2,
      "ent_coef": 0.005,
      "vf_coef": 0.5,
      "max_grad_norm": 0.5
    },
    "env": {
      "dt": 0.02,
      "frame_skip": 4,
      "episode_length": 256,
      "action_scale": 0.4,
      "kp": 80.0,
      "kd": 1.6,
      "max_torque": 23.7,
      "min_height": 0.18,
      "max_tilt": 0.7,
      "terminate_on_fall": true,
      "command": [1.0, 0.0, 0.0],
      "reward": {
        "track_lin_vel": 3.0,
        "track_ang_vel": -0.5,
        "lin_vel_z": -1.0,
        "ang_vel_xy": -0.05,
        "orientation": -1.0,
        "action_rate": -0.005,
        "torques": -0.0001,
        "base_height": -50.0,
        "alive": 0.0,
        "termination": -150.0
      }
    }
  },
  "spec": {
    "timesteps": {"min": 10000, "max": 2000000},
    "ppo.learning_rate": {"min": 1e-5, "max": 0.01},
    "ppo.n_epochs": {"min": 1, "max": 30},
    "env.action_scale": {"min": 0.1, "max": 1.5},
    "env.kp": {"min": 5.0, "max": 120.0},
    "env.kd": {"min": 0.1, "max": 5.0},
    "..."
  }
}
```

- `timesteps`: Total training steps; controls the length of a single training run
- `seed`: Random seed for reproducibility
- `n_envs`: Number of parallel environments; affects sampling throughput
- `net_arch`: Hidden layer sizes of the policy network
- `ppo.learning_rate`: Learning rate for the PPO policy network
- `ppo.n_steps`: Steps collected per environment before each PPO update
- `ppo.batch_size`: Mini-batch size for each gradient update
- `ppo.n_epochs`: Number of optimization epochs per data batch
- `ppo.gamma`: Discount factor; controls the weight of future rewards
- `ppo.gae_lambda`: GAE smoothing coefficient for advantage estimation
- `ppo.clip_range`: PPO policy update clipping range
- `ppo.ent_coef`: Entropy regularization coefficient; encourages exploration
- `ppo.vf_coef`: Weight of the value function loss
- `ppo.max_grad_norm`: Gradient clipping threshold
- `env.dt`: Physics simulation step size
- `env.frame_skip`: Number of physics steps skipped per control step
- `env.episode_length`: Maximum steps per episode
- `env.action_scale`: Action scaling factor; controls joint movement magnitude
- `env.kp`: PD controller proportional gain
- `env.kd`: PD controller derivative gain
- `env.max_torque`: Maximum joint output torque
- `env.min_height`: Minimum base height threshold; below this the robot is considered fallen
- `env.max_tilt`: Maximum tilt angle threshold; beyond this the robot is considered fallen
- `env.command`: Velocity command [vx, vy, yaw_rate]
- `env.reward.track_lin_vel`: Linear velocity tracking reward weight
- `env.reward.track_ang_vel`: Angular velocity tracking penalty weight
- `env.reward.lin_vel_z`: Vertical velocity penalty weight
- `env.reward.ang_vel_xy`: Roll/pitch angular velocity penalty weight
- `env.reward.orientation`: Body orientation deviation penalty weight
- `env.reward.action_rate`: Action rate penalty weight; suppresses jitter
- `env.reward.torques`: Joint torque penalty weight; encourages energy efficiency
- `env.reward.base_height`: Base height deviation penalty weight
- `env.reward.alive`: Survival reward weight
- `env.reward.termination`: Fall termination penalty
- `spec`: Valid range (min/max) for each tunable parameter, used by the agent for boundary checking during search
