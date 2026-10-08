# 三层设计 (Env / Agent / Task)

## 目录结构

> 🚧 待填充：顶层目录树（Agents/ Envs/ Harness/ Tasks/ Docs/ assets/ tmp/），逐目录一句话说明。

```
Agents/
Envs/
Harness/
Tasks/
Docs/
assets/
```

## Env 层

> 🚧 待填充：Env 目录各子目录职责。
> - `env/` 仿真资源与动力学
> - `training/` 训练算法（config.py / train_ppo.py / viz.py）
> - `observation/` 提供给 Agent 的观测 API
> - `action/` 提供给 Agent 的配置 API
> - `parameters.json` 可调参数唯一来源
> - `config.json` 路径占位（schedular 运行时覆写）

## Agent 层

> 🚧 待填充：Agents/ 下每个 agent 一个目录；PPOTuner 与 CurveAnalyst 的组成文件；
> 每个 agent 的 config.json 用于路径重定向。

## Task 层

> 🚧 待填充：Task 目录结构——schedular.py 入口、Agents/ 与 Envs/ 拷贝、Runs/ 产物；
> schedular 首次运行时从模板复制、之后重定向到最新时间戳目录。
