# Action — Go2Locomotion-PPO

## 概述

本目录是环境的**全部对外接口**：既包含修改可配置参数的**写动作**，也包含读取运行结果的**读动作**、以及触发训练/评估的**执行动作**。读取观测本身就是对环境的一种动作，因此三者同处 `action/`，调用方式完全一致。

所有函数均为模块级公开函数，携带类型注解与 docstring。Agent 加载环境时会自动扫描本目录，将每个函数转换为一个工具：工具名即函数名，工具描述取自 docstring，参数 schema 由签名推导。

## 可用函数

### `config.py` — 读写参数

| 函数 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `get_params` | 无 | dict | 返回当前 parameters.json 全部内容 |
| `set_params` | `params_dict` | dict | 将 params_dict 合并写入 parameters.json['parameters']，返回更新后的值和警告 |
| `get_param_spec` | 无 | dict | 返回每个参数的取值范围 |
| `reset_params` | 无 | dict | 提示从模板恢复（不自动覆写） |

### `params.py` — 读取搜索空间

| 函数 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `search_space_description` | 无 | str | 返回可调搜索空间的人类可读描述 |
| `get_param_spec` | 无 | dict | 返回每个可调参数的范围（spec） |

### `train.py` — 训练与评估

| 函数 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `run_training` | `config`, `out_dir`, `verbose=0`, `visualize=True` | dict | 训练一次 PPO，返回汇总指标 |
| `normalize_config` | `user_cfg` | (dict, list) | 校验并裁剪用户配置，返回归一化配置与警告 |
| `evaluate_policy` | `policy_path`, `env_cfg`, `n_episodes=10`, `seed=31337`, `policy_mode="greedy"` | dict | 加载已保存的策略并评估，返回统计 |

### `metrics.py` — 读取训练结果

| 函数 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `get_trial_summary` | `runs_root`, `run_id` | dict | 返回一次训练运行的核心指标（mean_return, fall_rate, mean_lin_vel_error 等） |
| `get_eval_curve` | `runs_root`, `run_id` | dict | 返回完整 eval 曲线（每 100k 步的 mean_return）及趋势值 |
| `list_trials` | `runs_root` | dict | 列出 runs_root 下所有已完成试验及关键指标 |
| `get_config` | `runs_root`, `run_id` | dict | 返回某次运行实际使用的训练配置 |
| `get_curve_path` | `runs_root`, `run_id` | dict | 返回某次运行训练曲线 PNG 的路径（若存在） |

> 注：`runs_root`、`out_dir`、`policy_path` 等属于**框架注入参数**——由 Agent 在加载环境时给出当前会话的执行上下文，不出现在模型可见的工具 schema 中。

## 可配置参数

所有可配置参数定义在 `../parameters.json` 的 `parameters` 字段中，取值范围在 `spec` 字段中。

**PPO 超参**：`ppo.learning_rate`、`ppo.n_steps`、`ppo.batch_size`、`ppo.n_epochs`、`ppo.gamma`、`ppo.gae_lambda`、`ppo.clip_range`、`ppo.ent_coef`、`ppo.vf_coef`、`ppo.max_grad_norm`

**环境参数**：`env.action_scale`、`env.kp`、`env.kd`、`env.max_torque`、`env.min_height`、`env.max_tilt`、`env.episode_length`、`env.command`

**其他**：`timesteps`（训练步数）、`seed`、`n_envs`、`net_arch`

## 数据路径

- 训练产物存储在 `runs_root/<run_id>/`
- `summary.json`：完整训练结果
- `metrics.csv`：逐步训练日志
- `config.json`：实际使用的训练配置
- `curve.png`：训练曲线（`visualize=True` 时生成）

## 注意

- `env.reward` 的修改在某些 Task 中可能被 schedular 固定（fixed_reward 模式），修改无效
- `set_params` 的修改会立即写入 `parameters.json`，下一次训练运行将使用新值
