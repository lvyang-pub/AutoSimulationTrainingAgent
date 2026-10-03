# Auto Simulation Training Agent (ASTA)

An agentic framework for autonomous robot training in simulation, built on MuJoCo.
Supports arbitrary robot environments and training configurations.

[中文文档](Docs/README.zh.md)

## Features

Once configured and launched, ASTA autonomously analyzes experiment results and adjusts
hyperparameters until the training goal is reached.


## Demo

In this example, we use a digital twin of the Unitree Go2 quadruped in a MuJoCo
environment, with the training goal of walking forward at 1 m/s.

ASTA analyzes multi-modal experiment outputs (charts, logs, etc.), tunes training
configurations, and achieves the goal in **6 trials** under a budget of at most
200,000 simulation steps per trial. Total API cost: ¥0.12 (DeepSeek API).


### Trial 1 vs Trial 6 (best)

Comparison of the first and final trials from run `Tasks/Go2Tune/Runs/20261003_183952`.

| Trial | Reward curve | Rollout |
| :---: | :---: | :---: |
| **Trial 1** | ![Trial 1 reward curve](assets/trials/curve_trial001.png) | ![Trial 1 rollout](assets/trials/go2_walk_trial001.gif) |
| **Trial 6** | ![Trial 6 reward curve](assets/trials/curve_trial006.png) | ![Trial 6 rollout](assets/trials/go2_walk_trial006.gif) |

The agent tuned three parameters — `kp=80, kd=2.0, action_scale=0.5` — lifting
mean return from −10 to 129, reducing velocity error from 1.0 to 0.18, and
bringing the fall rate to 0.


## Architecture

### Code layout

Each layer's responsibilities and call relationships. The Agent calls the Env
exclusively through the Harness; the Task's `schedular.py` wires them together
and manages run artifacts.

![Code Architecture](assets/arch_diagram.png)

### Runtime flow

End-to-end control flow from CLI invocation to `summary.json` on disk.

![Runtime Flow](assets/flow_diagram.png)


## Quick Start

### Setup

```bash
pip install -r requirements.txt

# Configure API key (either method)
#   a. Copy .env.example to .env and fill in your key
#   b. Export as an environment variable
export DEEPSEEK_API_KEY=sk-...
cp .env.example .env
```

### Run the agent

Launch a Task — the agent will autonomously loop through
train → analyze → tune → retrain until the goal is met:

```bash
python Tasks/Go2Tune/schedular.py
```

All run artifacts (agent memory, env observations, per-trial training outputs, logs)
land under `Tasks/Go2Tune/Runs/YYYYMMDD_HHMMSS/`. The `summary.json` at the run
root records success status, the best trial, and the final metrics.

Optional flags:
- `--no-viz` — skip post-training rollout rendering
- `--wall SECONDS` — total wall-clock budget
- `--timesteps N` — per-trial training step budget (default 200 000)

### Train once (no agent)

Run a single PPO trial directly, using parameters from
`Envs/Go2Locomotion-PPO/parameters.json`:

```bash
python Envs/Go2Locomotion-PPO/training/train_ppo.py \
    --out runs/demo --timesteps 200000
```

---

### Custom tasks and environments

The project is organized in three layers: **Env / Agent / Task**.
A Task's `schedular.py` wires one Agent to one Env for a single autonomous tuning
run. Layers communicate only through `Harness/exec.py` — the Agent never imports
Env code directly, so **swapping the Env requires no Agent changes, and vice versa**.

#### Add a new Task (reuse existing Env + Agent)

Copy `Tasks/Go2Tune/` to a new directory (e.g. `Tasks/MyTune/`) and edit
`schedular.py`:

| Field | Description |
|-------|-------------|
| `task_name` | Identifier used in memory and logs |
| `goals` | Success criteria, e.g. `{"mean_lin_vel_error_max": 0.25, "fall_rate_max": 0.2}` |
| `default_timesteps` | Per-trial step budget |
| `start_config` | Weak starting configuration the agent improves from |
| `AGENT_TEMPLATE` / `ENV_TEMPLATE` | Paths to the Agent / Env template directories |

`schedular.py` auto-copies templates into the Task's `Agents/` and `Envs/` on
first run — no manual copy needed.

```bash
python Tasks/MyTune/schedular.py
```

#### Add a new Env

Create a directory under `Envs/` following this layout:

```
Envs/MyEnv-PPO/
├── config.json          # path config, overwritten by schedular at runtime
├── parameters.json      # {"parameters": {...}, "spec": {key: {min, max}}}
├── env/                 # simulation assets and dynamics
├── training/
│   ├── config.py        # normalize(user_cfg) -> (cfg, warnings)
│   └── train_ppo.py     # train(...)  evaluate_from_zip(...)
├── observation/
│   ├── README.md
│   └── metrics.py       # get_trial_summary / get_eval_curve / list_trials
└── action/
    ├── README.md
    └── configure.py     # get_params / set_params / reset_params
```

The Agent's tool layer calls these **fixed function names** via the Harness;
new Envs must implement them with matching signatures.

`train()` must return a summary dict containing at minimum:

```
mean_return, std_return, fall_rate, mean_lin_vel_error,
mean_episode_length, convergence_delta, train_seconds, timesteps
```

> **Note:** `Agents/PPOTuner`'s prompts and `goal_met` logic are written for the
> Go2 forward-walking task (`mean_lin_vel_error` / `fall_rate`). For environments
> with different observation metrics, update `main.py` and `tools.py` accordingly,
> or write a new Agent that reuses the same Env interface.


## Visualization

After every training trial, visualizations are written to that trial's directory
(e.g. `Tasks/Go2Tune/Runs/<timestamp>/training_runs/trial_006/`):

- `rollout.mp4` — deterministic policy rollout, rendered off-screen via MuJoCo.
  Falls back to `rollout.gif` if OpenCV is unavailable — no extra dependencies needed.
- `curve.png` — episode return and periodic evaluation return vs. timesteps.
- `metrics.csv` / `summary.json` — per-episode metrics and run summary
  (read by `observation/metrics.py`).

Visualization is best-effort; a rendering failure never interrupts training.
Pass `--no-viz` to skip it entirely.


## Directory Structure

```
Agents/                 Agents (PPOTuner tuning agent, CurveAnalyst image analyzer)
Envs/Go2Locomotion-PPO/ Env: env/ (sim + reward), training/ (PPO),
                        observation/ (metrics API), action/ (configure API), parameters.json
Harness/exec.py         Isolation bridge — dynamically calls Env functions for the Agent
Tasks/Go2Tune/          Task: schedular.py entry point + runtime Agent/Env copies
  Runs/YYYYMMDD_HHMMSS/ Per-run artifacts (agent_memory / env_observations / training_runs / logs)
Docs/                   Design docs, requirements, dev log, reference material
assets/                 README assets (GIFs, diagrams)
```
