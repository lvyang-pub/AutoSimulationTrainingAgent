# AutoSimulationTrainingAgent

一个自动调参智能体：分析 Unitree Go2 机器狗在 MuJoCo 中的强化学习训练结果，
自主调整其 **训练超参（PPO 超参 + PD/被控对象参数）**，目标是让短程训练拿到尽可能高的
回报（reward 由任务固定，保证各次试验的 return 可直接比较）。

底座模型为 `deepseek-flash`（OpenAI 兼容接口，支持 function calling）。

> 本仓库的**设计重点是 Agent 本身**（`src/agent/`、`src/eval/` 及 `doc/` 的文档）。
> `src/envs/` 与 `src/training/` 是让 Agent 有真实实验可做的**支撑物**，也必须能真实跑通。

## 快速开始

```bash
pip install -r requirements.txt

# 0) 配置 API key（二选一）
#    a. 复制 .env.example 为 .env 并填入自己的 key；或
#    b. 导出环境变量：export DEEPSEEK_API_KEY=sk-...
cp .env.example .env   # 然后编辑 .env 填入你的 key

# 1) 自动下载 Go2 模型并冒烟测试环境
python scripts/setup_env.py

# 2) 让智能体在一个任务上自主调参
python scripts/run_agent.py --task go2_tune_v1 --out runs/agent_v1

# 3) 跑评测：本智能体 vs 基线（naive / random），输出 Markdown 报告
python scripts/run_eval.py --tasks go2_tune_v1 go2_tune_hard --out eval/round1
```

API key 通过环境变量 `DEEPSEEK_API_KEY` 提供（或写入本地 `.env`，会被自动加载）。
仓库**不含任何密钥**；`.env` 已被 `.gitignore` 忽略。

## 目录

```
src/envs/       MuJoCo Go2 环境 + 可配置奖励（支撑物）
src/training/   PPO 训练脚本 + 配置 schema/校验
src/agent/      智能体：LLM 客户端 / 工具 / 记忆 / 反思 / ReAct 主循环
src/eval/       任务集、评测 runner、指标、报告
scripts/        setup_env / run_agent / run_eval（+ pytest.ini / conftest.py / scriptlog.py）
tests/          单元 + 集成测试（pytest）
eval/           各轮评测产物（outcomes.json / report.md）
log/            运行日志（脚本自动落盘，console 同步输出）
doc/文档/       需求文档 / 架构文档 / 测试结果文档 / 评测报告
doc/开发过程记录/ 开发过程记录
doc/需求文档.md   Lab B 作业要求原文
```

## 智能体如何工作（一句话）

`deepseek-flash` 作为推理内核，运行一个 **ReAct 循环**（Thought → 选择工具 → Observation），
通过**工具层**发起真实训练与评估，用**短期记忆**跟踪本次探索、用**反思**把每次实验的结果
沉淀成**长期记忆**中的可复用教训，并在**trial / 墙钟 / LLM 步数**三重预算内迭代。

详见 `doc/文档/架构文档.md`。

## 测试

配置文件在 `scripts/pytest.ini` 与 `scripts/conftest.py`；从仓库根目录直接运行即可
（conftest 会注册 `slow` 标记并把仓库根加入 `sys.path`）：

```bash
python -m pytest -m "not slow"   # 单元测试（秒级）
python -m pytest                 # 含 MuJoCo 集成测试
# 也可显式指定配置：
python -m pytest -c scripts/pytest.ini
```

## 日志

`run_agent.py` / `run_eval.py` 会把运行过程**自动落盘**到 `log/<name>_<时间戳>.log`
（同时照常打印到控制台），无需再手动重定向。可用 `--log-dir` 改目录。
