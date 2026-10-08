# Go2Tune

## 概述

单个 Task 包含了执行一次智能体自动实验所需的全部内容，该系统不会依赖任何外部文件夹下的代码，而是将所有代码收集到其内部进行运行。

一个 Task 文件夹的核心是 schedular，该文件通过编排智能体以及智能体和环境信息的整合，保证智能体能在其执行过程中顺利获取所需信息，并按其意图完成在环境中的尝试。

## 实现

### 定义目标

Go2Tune 的目标以指标阈值的形式在 schedular 中声明，并作为参数传入智能体：

- `mean_lin_vel_error <= 0.25`：机器人实际速度与命令速度的误差
- `fall_rate <= 0.2`：评估回合中的倒地比例

schedular 只负责声明这组阈值，目标是否达成由智能体内部的 `goal_met` 函数在每次 trial 后自动判定，判定结果记录在最终报告中。

### 组件装配

schedular 在启动时完成一次性的路径计算与组件初始化，之后将全部上下文注入智能体，自身不再参与训练过程。具体包括：

- 按当前时间戳在 `Runs/` 下创建本次运行目录及子目录（`agent_memory/`、`training_runs/`、`logs/`）
- 计算 Task 内部各组件目录（`Agents/PPOTuner`、`Envs/Go2Locomotion-PPO`、`Agents/CurveAnalyst`）的绝对路径，作为 `env_path`、`harness_path` 等参数传入 `TuningAgent`
- 从 `Agents/PPOTuner/config.json` 加载智能体配置（模型名称等）
- 注册 `on_trial_done` 回调：每次 trial 结束后，schedular 将该 trial 的 `curve.png` 交给 `CurveAnalyst` 进行视觉分析，将返回的文本作为补充观测注入智能体主循环

### 输出收集

每次运行在 `Runs/<timestamp>/` 下产生以下结构：

```
Runs/
└── 20261003_171150/
    ├── summary.json              # 整次运行汇总（成功与否、最优 trial、耗时等）
    ├── agent_memory/
    │   └── longterm.json         # 智能体长期记忆，由反思机制在每轮 trial 后写入
    ├── logs/
    └── training_runs/
        └── trial_001/
            ├── config.json       # 本次 trial 使用的规范化超参，由训练函数写入
            ├── summary.json      # 训练结果指标，由训练函数写入后作为返回值传回智能体
            ├── metrics.csv       # 完整训练过程日志，由训练回调逐步写入
            ├── policy.zip        # 保存的策略权重，由 SB3 训练结束后写入
            └── curve.png         # 训练奖励曲线图，由可视化模块在训练后生成
```

`summary.json`（运行根目录）由 schedular 在智能体退出后写入，汇总本次会话的全局结果。

