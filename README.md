# AutoSimulationTrainingAgent

本项目是一个在虚拟环境中，自动训练机器人的智能体：在本项目演示中，我们使用了一个 Unitree Go2 机器狗的数字孪生，在MuJoCo环境中做了强化学习训练

该智能体可以分析每次训练的结果，并根据结果调整超参数，以加速训练，改善训练结果。


## 快速开始

```bash
pip install -r requirements.txt

# 0) 配置 API key（两种方式）
#    a. 复制 .env.example 为 .env 并填入自己的 key；或
#    b. 导出环境变量：export DEEPSEEK_API_KEY=sk-...
cp .env.example .env   # 然后编辑 .env 填入你的 key

# 1) 下载 Go2 模型并测试环境
python scripts/setup_env.py

# 2) 开始自动参数调节
python scripts/run_agent.py --task go2_tune_v1 --out runs/agent_v1

# 3) 复现评测：本智能体 vs 基线（naive / random）
python scripts/run_eval.py --tasks go2_tune_v1 go2_tune_hard --out eval/round1
```

## 目录结构

```
src/envs/       训练环境
src/training/   训练程序
src/agent/      智能体
src/eval/       评测
scripts/        辅助脚本等
eval/           各轮评测产物（outcomes.json / report.md）
log/            运行日志（脚本自动落盘，console 同步输出）
doc/文档/       需求文档 / 架构文档 / 测试结果文档 / 评测报告
doc/开发过程记录/ 开发过程记录
```

## 工作机制

该系统以一个 **ReAct 循环**为核心，在每次循环，进行如下三步：思考 → 工具调用 → 观测，
- 通过**工具层**发起真实训练与评估
- 用**短期记忆**跟踪本次探索
- 通过**反思**把每次实验的结果沉淀成**长期记忆**中的经验

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
