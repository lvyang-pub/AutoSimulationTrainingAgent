# 快速上手

## 安装 ASTA

### 环境要求

- Python 3.13（已在 Windows + CPU 环境验证，无需 GPU）
- 依赖：MuJoCo 3.10、Gymnasium 1.3、stable-baselines3 2.9、torch ≥ 2.14、numpy ≥ 2.0

### 获取代码

```bash
git clone <仓库地址> AutoSimulationTrainingAgent
cd AutoSimulationTrainingAgent
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API Key

智能体需要调用大模型，将 API Key 写入项目根目录的 `.env` 文件（推荐），或直接导出为环境变量：

```bash
# 方式一：写入 .env
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-...

# 方式二：导出环境变量
export DEEPSEEK_API_KEY=sk-...
```

## 开始运行

一条命令启动示例任务。智能体会自主循环「训练 → 分析曲线 → 调参 → 再训练」，直到达成训练目标：

```bash
python Tasks/Go2Tune/schedular.py
```

可选参数：

| 参数 | 说明 |
|---|---|
| `--timesteps N` | 单次训练的最高 step 预算 |
| `--wall SECONDS` | 总运行时间预算（秒），0 表示不限 |
| `--no-viz` | 跳过训练后的可视化渲染 |

单独运行一次 PPO 训练（不经过智能体，仅用 `parameters.json` 中的配置）：

```bash
python Envs/Go2Locomotion-PPO/training/train_ppo.py \
    --out runs/demo --timesteps 200000
```

## 获得结果

每次运行的所有产物落在 `Tasks/Go2Tune/Runs/<时间戳>/` 下，文件夹名即运行启动时间：

```
Tasks/Go2Tune/Runs/20261003_183952/
├── summary.json              # 整次运行的汇总结果
├── agent_memory/
│   └── longterm.json         # 智能体长期记忆
├── logs/
└── training_runs/
    ├── trial_001/
    │   ├── config.json       # 本次 trial 使用的超参
    │   ├── summary.json      # 本次 trial 的训练指标
    │   ├── metrics.csv       # 完整训练过程日志
    │   ├── policy.zip        # 保存的策略权重
    │   └── curve.png         # 训练奖励曲线图
    └── ...
```

各部分记录的内容：

| 路径 | 内容 |
|---|---|
| `summary.json`（根目录） | 本次运行是否达成目标、最优 trial、最终指标、耗时、token 用量等全局汇总 |
| `agent_memory/longterm.json` | 智能体沉淀的长期记忆，每轮 trial 后由反思机制写入，供后续会话复用历史经验 |
| `training_runs/trial_NNN/config.json` | 该次 trial 实际生效的规范化超参 |
| `training_runs/trial_NNN/summary.json` | 该次 trial 的训练结果指标（速度误差、倒地率、回报等） |
| `training_runs/trial_NNN/metrics.csv` | 完整训练过程日志，逐次评估记录 |
| `training_runs/trial_NNN/policy.zip` | 训练完成后的策略权重，可用于复用或再评估 |
| `training_runs/trial_NNN/curve.png` | 训练奖励曲线图，供人工或曲线分析智能体查看 |
