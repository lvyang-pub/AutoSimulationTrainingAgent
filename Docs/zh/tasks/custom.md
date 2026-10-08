# 自定义任务与环境

## 新增一个 Env

> 🚧 待填充：在 `Envs/` 下新建目录，参考格式（目录树）；说明必需实现的函数契约。

```
Envs/MyEnv-PPO/
├── config.json          # 路径配置，schedular 运行时覆写
├── parameters.json      # {"parameters": {...}, "spec": {key: {min, max}}}
├── env/                 # 环境资源和逻辑代码
├── training/            # 训练算法实现
├── observation/         # 提供给 Agent 观测的 API
└── action/              # 提供给 Agent 调参的 API
```

## 新增一个 Task（复用现有 Env + Agent）

> 🚧 待填充：
> ① 复制 `Tasks/Go2Tune/` 为新目录 ② 修改 schedular.py ③ 拷贝要用的 Agent 和 Env ④ 运行。

| 位置 | 说明 |
|---|---|
| `task_name` | 任务名，用于长期记忆与日志标识 |
| `goals` | 成功判据 |
| `default_timesteps` | 单次训练默认步数预算 |
| `start_config` | 初始训练配置 |

```bash
python Tasks/MyTune/schedular.py
```

## 注意事项

> 🚧 待填充：工具层的 run_training / evaluate 与 goal_met 目前针对 Go2 走路任务
> （mean_lin_vel_error / fall_rate）。换成观测指标不同的环境时，需相应修改
> Agent 的提示词与 main.py，或另写一个复用相同 Env 接口的新 Agent。
