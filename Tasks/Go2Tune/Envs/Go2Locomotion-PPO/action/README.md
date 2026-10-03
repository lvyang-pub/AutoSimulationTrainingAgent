# Action — Go2Locomotion-PPO

## 概述

本目录下的函数用于对环境的可配置训练参数进行修改。所有修改均写入 `../parameters.json`，该文件是环境运行时参数的唯一来源。

## 可配置参数

所有可配置参数定义在 `../parameters.json` 的 `parameters` 字段中，取值范围在 `spec` 字段中。

主要可调参数：

**PPO 超参**：`ppo.learning_rate`、`ppo.n_steps`、`ppo.batch_size`、`ppo.n_epochs`、`ppo.gamma`、`ppo.gae_lambda`、`ppo.clip_range`、`ppo.ent_coef`、`ppo.vf_coef`、`ppo.max_grad_norm`

**环境参数**：`env.action_scale`、`env.kp`、`env.kd`、`env.max_torque`、`env.min_height`、`env.max_tilt`、`env.episode_length`、`env.command`

**其他**：`timesteps`（训练步数）、`seed`、`n_envs`、`net_arch`

## 可用函数（`configure.py`）

| 函数 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `get_params` | 无 | dict | 返回当前 parameters.json 全部内容 |
| `set_params` | `params_dict` | dict | 将 params_dict 合并写入 parameters.json，返回更新后的值和警告 |
| `get_param_spec` | 无 | dict | 返回每个参数的取值范围 |
| `reset_params` | 无 | dict | 提示从模板恢复（不自动覆写） |

## 调用示例

```python
# 通过 Harness 调用（agent 视角）
result = harness.call("action/configure.py", "set_params",
                      params_dict={"ppo": {"ent_coef": 0.01}, "env": {"kp": 80.0}})

# 直接调用（调试）
from Envs.Go2Locomotion-PPO.action.configure import set_params
result = set_params({"timesteps": 1000000, "ppo": {"learning_rate": 0.001}})
```

## 注意

- `env.reward` 的修改在某些 Task 中可能被 schedular 固定（fixed_reward 模式），修改无效
- 修改会立即写入 `parameters.json`，下一次训练运行将使用新值
