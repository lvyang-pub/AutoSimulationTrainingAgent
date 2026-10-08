# Go2Locomotion-PPO

## 概述

一个基于 MuJoCo 实现的训练环境，在平坦环境中使用 PPO 算法训练宇树 Go2 机器狗以 1m/s 的速度前进。

## 环境设计

### 场景设计

平面地形，具体参数如下：

| 参数 | 值 |
|---|---|
| 重力 | 9.81 m/s² |
| 地板摩擦系数 | 0.2 |
| 仿真步长 | 0.005 s（dt=0.02，跳过帧=4） |

### 受控体

标准的宇树 Go2 四足机器人，机体质量 6.921 kg，初始状态为默认站立姿态。

### 训练算法

PPO 算法，具体参数如下：

| 参数 | 值 |
|---|---|
| learning_rate | 3e-4 |
| batch_size | 256 |
| n_epochs | 5 |
| gamma | 0.99 |

训练一个 64×64 的双层前向传播网络（FNN），激活函数使用 tanh。

### 接口设计

**观测空间**：33 维连续向量

| 维度 | 内容 |
|---|---|
| 3 | 机体角速度（机体坐标系） |
| 3 | 重力向量在机体坐标系中的投影 |
| 12 | 关节角度 |
| 12 | 关节角速度 |
| 3 | 上一步动作 |

**动作空间**：12 维连续向量，取值范围 [-1, 1]，映射到目标关节角后经 PD 控制器转换为关节力矩。



## API 设计

**`get_trial_summary`**：获取一次训练运行的核心指标

| 参数 | 类型 | 说明 |
|---|---|---|
| `runs_root` | str | 训练结果根目录 |
| `run_id` | str | 目标训练标识 |

| 返回字段 | 说明 |
|---|---|
| `mean_return` | 平均累计奖励 |
| `fall_rate` | 跌倒率 |
| `mean_lin_vel_error` | 平均线速度误差 |

---

**`get_eval_curve`**：获取训练的 eval 曲线与趋势值

| 参数 | 类型 | 说明 |
|---|---|---|
| `runs_root` | str | 训练结果根目录 |
| `run_id` | str | 目标训练标识 |

| 返回字段 | 说明 |
|---|---|
| （曲线数据） | 各阶段评估指标及趋势统计 |

---

**`list_trials`**：列出所有已完成的训练及关键指标

| 参数 | 类型 | 说明 |
|---|---|---|
| `runs_root` | str | 训练结果根目录 |

| 返回字段 | 说明 |
|---|---|
| （trial 列表） | 所有历史训练的 run_id 与核心指标 |

---

**`get_config`**：获取某次训练实际使用的配置

| 参数 | 类型 | 说明 |
|---|---|---|
| `runs_root` | str | 训练结果根目录 |
| `run_id` | str | 目标训练标识 |

| 返回字段 | 说明 |
|---|---|
| （配置内容） | 该次训练实际使用的完整超参数配置 |

---

**`get_params`**：读取当前配置文件的全部内容

| 参数 | 说明 |
|---|---|
| — | 无需参数 |

| 返回字段 | 说明 |
|---|---|
| （配置内容） | parameters.json 的完整内容 |

---

**`set_params`**：将新参数合并写入配置文件

| 参数 | 类型 | 说明 |
|---|---|---|
| `params_dict` | dict | 需要更新的参数键值对 |

| 返回字段 | 说明 |
|---|---|
| （合并结果） | 写入后的参数值与校验警告 |

---

**`get_param_spec`**：获取每个参数的合法取值范围

| 参数 | 说明 |
|---|---|
| — | 无需参数 |

| 返回字段 | 说明 |
|---|---|
| （参数规格） | 每个可调参数的 min / max 范围 |

---

**`reset_params`**：将配置文件恢复为默认模板

| 参数 | 说明 |
|---|---|
| — | 无需参数 |

| 返回字段 | 说明 |
|---|---|
| （提示信息） | 恢复结果说明 |

## 配置文件

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

- `timesteps`：总训练步数，控制单次训练时长
- `seed`：随机种子，用于复现训练结果
- `n_envs`：并行环境数量，影响采样效率
- `net_arch`：策略网络隐藏层结构
- `ppo.learning_rate`：PPO 策略网络学习率
- `ppo.n_steps`：每次更新前每个环境采集的步数
- `ppo.batch_size`：每次梯度更新使用的样本批量大小
- `ppo.n_epochs`：每批数据的优化轮数
- `ppo.gamma`：折扣因子，控制未来奖励的权重
- `ppo.gae_lambda`：GAE 优势估计的平滑系数
- `ppo.clip_range`：PPO 策略更新的裁剪范围
- `ppo.ent_coef`：熵正则化系数，鼓励探索
- `ppo.vf_coef`：价值函数损失的权重系数
- `ppo.max_grad_norm`：梯度裁剪阈值
- `env.dt`：仿真物理步长
- `env.frame_skip`：每个控制步跳过的物理步数
- `env.episode_length`：每个回合的最大步数
- `env.action_scale`：动作缩放系数，控制关节运动幅度
- `env.kp`：PD 控制器比例增益
- `env.kd`：PD 控制器微分增益
- `env.max_torque`：关节最大输出力矩
- `env.min_height`：机体最低高度阈值，低于此值判定为跌倒
- `env.max_tilt`：机体最大倾斜角阈值，超出此值判定为跌倒
- `env.command`：速度指令 [vx, vy, yaw_rate]
- `env.reward.track_lin_vel`：线速度跟踪奖励权重
- `env.reward.track_ang_vel`：角速度跟踪惩罚权重
- `env.reward.lin_vel_z`：垂直方向速度惩罚权重
- `env.reward.ang_vel_xy`：横滚/俯仰角速度惩罚权重
- `env.reward.orientation`：机体姿态偏离惩罚权重
- `env.reward.action_rate`：动作变化率惩罚权重，抑制抖动
- `env.reward.torques`：关节力矩惩罚权重，鼓励节能
- `env.reward.base_height`：机体高度偏离惩罚权重
- `env.reward.alive`：存活奖励权重
- `env.reward.termination`：跌倒终止惩罚
- `spec`：每个可调参数的合法取值范围（min/max），供智能体搜索时做边界检查
