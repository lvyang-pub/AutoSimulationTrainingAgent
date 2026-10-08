# PPOTuner

## Overview

An agent for fine-tuning reinforcement learning policies in a simulated environment using the PPO algorithm.

## Agent Features

### Main Loop

The main loop follows the ReAct (Reason-Act-Observe) pattern: the model first reasons, then selects a tool to call, then observes the result, and reasons again based on that result. This cycle may execute multiple times between individual training trials.

### Reflection

The reflection mechanism operates outside the main loop and is used purely for analysis. After each training run, it makes a single separate LLM call, combining training feedback (such as output logs and training results) to produce a summary description. This summary is then injected into the main loop as an observation.

### Tools

**`run_training`**: Run a PPO training session

| Parameter | Type | Description |
|---|---|---|
| `config` | dict | Hyperparameter configuration |
| `timesteps` | int | Training steps (free hyperparameter) |
| `seed` | int | Random seed |

| Return Field | Description |
|---|---|
| `run_id` | Unique identifier for this run |
| `summary` | Training summary |
| `warnings` | Warnings raised during training |

---

**`evaluate_policy`**: Evaluate a trained policy

| Parameter | Type | Description |
|---|---|---|
| `run_id` | str | Target run identifier |
| `n_episodes` | int | Number of evaluation episodes |
| `policy_mode` | str | Evaluation mode |

| Return Field | Description |
|---|---|
| `mean_return` | Mean cumulative reward |
| `fall_rate` | Fall rate |
| `mean_lin_vel_error` | Mean linear velocity error |

---

**`get_metrics`**: Retrieve learning curve summary for a run

| Parameter | Type | Description |
|---|---|---|
| `run_id` | str | Target run identifier |

| Return Field | Description |
|---|---|
| (curve summary) | Per-stage reward, loss, and other statistics |

---

**`list_trials`**: List all completed training runs

| Parameter | Description |
|---|---|
| — | No parameters required |

| Return Field | Description |
|---|---|
| (trial list) | run_id and basic info for all historical trials |

---

**`finish`**: Signal end of the tuning session

| Parameter | Type | Description |
|---|---|---|
| `success` | bool | Whether the tuning goal was achieved |
| `reasoning` | str | Explanation for termination |

| Return Field | Description |
|---|---|
| (terminal marker) | Notifies the Harness to exit the main loop |

### Two-Tier Memory

- **Short-term memory**: Historical information accumulated across multiple ReAct cycles within a single execution.
- **Long-term memory**: A summary written at the end of each session, persisted to disk as a file; used in subsequent runs to integrate past experience and avoid repeating failed attempts.
