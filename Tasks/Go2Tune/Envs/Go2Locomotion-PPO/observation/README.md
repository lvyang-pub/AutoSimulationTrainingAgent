# Observation — Go2Locomotion-PPO

## 概述

本目录下的函数用于向 agent 提供该环境完整运行后的可观测数据。这些不是单步 PPO 的即时反馈，而是一次完整训练运行结束后的汇总指标。

## 可用函数（`metrics.py`）

所有函数均在 `metrics.py` 中定义，可通过 `Harness/exec.py` 调用。

| 函数 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `get_trial_summary` | `runs_root, run_id` | dict | 返回一次训练运行的核心指标（mean_return, fall_rate, mean_lin_vel_error 等） |
| `get_eval_curve` | `runs_root, run_id` | dict | 返回完整 eval 曲线（每 100k 步的 mean_return）及趋势值 |
| `list_trials` | `runs_root` | dict | 列出 runs_root 下所有已完成试验及关键指标 |
| `get_config` | `runs_root, run_id` | dict | 返回某次运行实际使用的训练配置 |

## 调用示例

```python
# 通过 Harness 调用（agent 视角）
result = harness.call("observation/metrics.py", "get_trial_summary",
                      runs_root="/path/to/runs", run_id="trial_001")

# 直接调用（调试）
from Envs.Go2Locomotion-PPO.observation.metrics import get_trial_summary
summary = get_trial_summary(runs_root="runs", run_id="trial_001")
```

## 数据路径

- 训练产物存储在 `runs_root/<run_id>/`，路径在 `config.json` 的 `runs_root` 字段中指定
- `summary.json`：完整训练结果
- `metrics.csv`：逐步训练日志
- `config.json`：实际使用的训练配置
