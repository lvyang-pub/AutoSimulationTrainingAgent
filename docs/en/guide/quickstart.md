# Quickstart

## Install ASTA

### Requirements

- Python 3.13 (verified on Windows + CPU; no GPU required)
- Dependencies: MuJoCo 3.10, Gymnasium 1.3, stable-baselines3 2.9, torch ≥ 2.14, numpy ≥ 2.0

### Get the code

```bash
git clone <repository-url> AutoSimulationTrainingAgent
cd AutoSimulationTrainingAgent
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure the API key

The agent needs to call an LLM. Put your API key in a `.env` file at the project root (recommended), or export it as an environment variable:

```bash
# Option 1: write to .env
cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY=sk-...

# Option 2: export as an environment variable
export DEEPSEEK_API_KEY=sk-...
```

## Run It

A single command launches the example task. The agent autonomously loops through train → analyze the curve → tune → retrain until the training goal is met:

```bash
python Tasks/Go2Tune/schedular.py
```

Optional flags:

| Flag | Description |
|---|---|
| `--timesteps N` | Per-trial training step budget |
| `--wall SECONDS` | Total wall-clock budget in seconds (0 = unlimited) |
| `--no-viz` | Skip post-training visualization rendering |

Run a single PPO training pass on its own (bypassing the agent, using the config in `parameters.json`):

```bash
python Envs/Go2Locomotion-PPO/training/train_ppo.py \
    --out runs/demo --timesteps 200000
```

## Get the Results

All artifacts from a run land under `Tasks/Go2Tune/Runs/<timestamp>/`, where the folder name is the run start time:

```
Tasks/Go2Tune/Runs/20261003_183952/
├── summary.json              # full run summary
├── agent_memory/
│   └── longterm.json         # agent long-term memory
├── logs/
└── training_runs/
    ├── trial_001/
    │   ├── config.json       # hyperparameters used for this trial
    │   ├── summary.json      # training metrics for this trial
    │   ├── metrics.csv       # full training log
    │   ├── policy.zip        # saved policy weights
    │   └── curve.png         # training reward curve
    └── ...
```

What each part records:

| Path | Contents |
|---|---|
| `summary.json` (run root) | Whether the goal was met, the best trial, final metrics, elapsed time, token usage — the global summary |
| `agent_memory/longterm.json` | The agent's long-term memory, written by the reflection mechanism after each trial for reuse in later sessions |
| `training_runs/trial_NNN/config.json` | The normalized hyperparameters actually applied for that trial |
| `training_runs/trial_NNN/summary.json` | Training result metrics for that trial (velocity error, fall rate, return, etc.) |
| `training_runs/trial_NNN/metrics.csv` | Full training log with per-evaluation records |
| `training_runs/trial_NNN/policy.zip` | Saved policy weights, reusable for evaluation or further training |
| `training_runs/trial_NNN/curve.png` | Training reward curve, for human review or the curve-analysis agent |
