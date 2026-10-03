**中文** | [English](../README.md)

# Auto Simulation Training Agent (ASTA)
虚拟环境中自动训练机器人的智能体框架，底层基于MuJoCo，支持各种机器人和环境配置


## 功能

完成配置并启动后，ASTA自主分析实验结果，进行超参调节，直到达成预设目标


## 案例效果

在本案例中，我们使用了宇树Go2的数字孪生，在MuJoCo环境中进行强化学习训练，训练目标为1m/s速度前进。

ASTA通过自主分析实验结果中的图表，日志等多模态信息，调节训练配置，在单次最高模拟200000步的极低预算下，通过六次尝试即达成训练目标。总Token成本0.12元人民币（DeepSeek API）


### 调优过程对比：实验 1 vs 实验 6（最优）

`Tasks/Go2Tune/Runs/20261003_183952`中首轮与终止轮的对比。

| 实验轮次 | 训练奖励曲线 | 行走效果 |
| :---: | :---: | :---: |
| **实验1**<br> | ![实验1 奖励曲线](../assets/trials/curve_trial001.png) | ![实验1 行走](../assets/trials/go2_walk_trial001.gif) |
| **实验6**<br> | ![实验6 奖励曲线](../assets/trials/curve_trial006.png) | ![实验6 行走](../assets/trials/go2_walk_trial006.gif) |

在这次运行中，机器人调节了三个参数： `kp=80, kd=2.0, action_scale=0.5`，最终实现稳健的前进效果，奖励从第一次实验的最高为-10，提升至129，速度误差从1.0降低至0.18，摔倒次数降低为0，实现稳健前进。

## 系统架构

### 代码架构

各层职责与依赖关系：Agent 通过 Harness 调用 Env，Task 的 schedular 负责编排Agents和环境，并管理运行产物。

![代码架构](../assets/arch_diagram.png)

### 运行逻辑

一次完整调参运行的控制流，从单次任务启动到任务目标达成：

![运行逻辑](../assets/flow_diagram.png)

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt

# 配置 API key
#   方法1. 复制.env.example为.env 并填入自己的key
#   方法2. 导出环境变量
export DEEPSEEK_API_KEY=sk-...
cp .env.example .env   # 然后编辑.env填入你的key
```

### 启动训练

启动一个 Task，智能体将自主完成「训练 → 分析 → 调参 → 再训练」的循环，直到达成预设目标：
```bash
python Tasks/Go2Tune/schedular.py
```
单次运行的产物，包括agent记忆、环境观测记录、各轮训练配置和结果、日志都落在
`Tasks/Go2Tune/Runs/YYYYMMDD_HHMMSS/` 下，文件夹名称为当次运行启动的时间，`summary.json` 记录了是否成功、最优 trial和目标达成情况等信息。

可选参数：
- `--no-viz` 跳过训练后可视化渲染
- `--wall` 设置总时间预算(单位是秒)
- `--timesteps` 单次训练最高step预算

```bash
python Envs/Go2Locomotion-PPO/training/train_ppo.py \
    --out runs/demo --timesteps 200000
```

---

### 自定义的训练任务

项目按 **Env（环境）/ Agent（智能体）/ Task（任务）** 三层组织，Task内部用 `schedular.py`
编排单个或多个 Agent和Env跑一次自主调参。不同Agent之间或者Agent和环境之间只通过 `Harness/exec.py` 通信：


---

#### 新增一个 Env

在 `Envs/` 下新建一个目录，参考格式：

```
Envs/MyEnv-PPO/
├── config.json          # 路径配置，schedular 运行时覆写
├── parameters.json      # {"parameters": {...}, "spec": {key: {min,max}}}
├── env/                 # 环境资源和逻辑代码
├── training/            # 训练算法实现
│   ├── config.py        
│   └── train_ppo.py     
├── observation/         # 环境提供给Agent观测的API
│   ├── README.md
│   └── metrics.py      
└── action/              # 环境提供给Agent调整配置的API
    ├── README.md
    └── configure.py    reset_params / get_param_spec
```
---
#### 新增一个 Task（复用现有 Env + Agent，最常见）

1. 复制 `Tasks/Go2Tune/` 为一个新目录（如 `Tasks/MyTune/`）

2. 修改`schedular.py`：

| 位置 | 说明 |
|------|------|
| `task_name` | 任务名，用于长期记忆与日志标识 |
| `goals` | 成功判据，如 `{"mean_lin_vel_error_max": 0.25, "fall_rate_max": 0.2}` |
| `default_timesteps` | 单次训练默认步数预算 |
| `start_config` | 初始训练配置 |

3. 将你使用的Agent和Env拷贝到该Task目录下

4. 运行
```bash
python Tasks/MyTune/schedular.py
```



当前的实现仅针对训练宇树Go2机器人以特定速度前进，你可以按需创建新的环境，Agent和Task，从而满足你的任务需求




