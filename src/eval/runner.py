"""Run a tuning method over the task set and gather outcomes.

Methods:
  agent    - the ReAct tuning agent (optionally with a system-prompt override and
             a strategy hint, which is how the eval loop compares agent versions)
  naive    - train the default config once
  random   - random search over the spec bounds, same trial budget as the agent

Baselines share the agent's trial budget so the comparison is apples-to-apples.
"""
from __future__ import annotations

import copy
import json
import os
import random
import time

from ..agent.agent import TuningAgent, goal_met
from ..agent.llm import LLMClient
from ..training.config import DEFAULT_CONFIG, SPEC, normalize
from ..training.train_ppo import train
from .metrics import TaskOutcome
from .tasks import Task


def _best_from_trials(trials: list[dict]) -> dict:
    if not trials:
        return {}
    return max(trials, key=lambda t: t["metrics"].get("mean_return", float("-inf")))


def run_agent_on_task(task: Task, runs_root: str, longterm_path: str,
                      llm: LLMClient | None = None, system_prompt: str | None = None,
                      extra_hint: str = "", fault_after: int | None = None) -> TaskOutcome:
    goals = dict(task.goals)
    agent = TuningAgent(
        goals=goals, runs_root=runs_root, task_name=task.name,
        longterm_path=longterm_path, default_timesteps=task.default_timesteps,
        max_trials=task.max_trials, client=llm, system_prompt=system_prompt,
        extra_hint=extra_hint, fault_after=fault_after, command=task.command,
        start_config=task.start_config, visualize=False,
    )
    res = agent.run()
    m = res.best_metrics
    return TaskOutcome(
        task=task.name, method="agent", success=res.success,
        best_return=m.get("mean_return", float("-inf")),
        best_lin_vel_error=m.get("mean_lin_vel_error", 1e9),
        best_fall_rate=m.get("fall_rate", 1.0),
        n_trials=res.n_trials, tool_calls=res.tool_stats["total_calls"],
        tool_errors=res.tool_stats["errors"], wall_seconds=res.wall_seconds,
        total_tokens=res.usage["total_tokens"],
        native_tool_calls=res.tool_stats["native"],
        fallback_tool_calls=res.tool_stats["fallback"],
        fault_injected=fault_after is not None,
        recovered_from_fault=(fault_after is not None and res.success),
        notes=res.stopped_reason,
    )


def _train_and_score(cfg: dict, task: Task, out_dir: str) -> dict:
    cfg = dict(cfg)
    cfg.setdefault("env", {})["command"] = task.command
    summary = train(cfg, out_dir, verbose=0, visualize=False)
    return summary


def run_naive_on_task(task: Task, runs_root: str) -> TaskOutcome:
    """Baseline: train the task's suggested (weak) starting config exactly once."""
    t0 = time.time()
    out = os.path.join(runs_root, "naive")
    cfg = dict(DEFAULT_CONFIG)
    cfg["timesteps"] = task.default_timesteps
    cfg["seed"] = task.seed
    # apply the task's weak start (env/ppo), same premise the agent is handed
    for section, vals in task.start_config.items():
        if isinstance(vals, dict) and isinstance(cfg.get(section), dict):
            cfg[section] = {**cfg[section], **vals}
        else:
            cfg[section] = vals
    summary = _train_and_score(cfg, task, out)
    return TaskOutcome(
        task=task.name, method="naive", success=goal_met(summary, task.goals),
        best_return=summary["mean_return"], best_lin_vel_error=summary["mean_lin_vel_error"],
        best_fall_rate=summary["fall_rate"], n_trials=1, tool_calls=0, tool_errors=0,
        wall_seconds=time.time() - t0, total_tokens=0, notes="single default run",
    )


def _sample_config(rng: random.Random, fixed_reward: dict | None) -> dict:
    """Sample a config from the spec bounds, with a few structural fixes.

    The reward weights are NOT sampled — the task fixes the reward so trials are
    comparable; random search plays by the same rule and only samples the rest.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    def setd(dotted, val):
        parts = dotted.split(".")
        cur = cfg
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = val
    for key, (lo, hi) in SPEC.items():
        if key in ("seed", "timesteps") or key.startswith("env.reward."):
            continue
        setd(key, rng.uniform(lo, hi))
    # keep structural invariants the sampler might violate
    if cfg["ppo"]["batch_size"] > cfg["ppo"]["n_steps"]:
        cfg["ppo"]["batch_size"] = cfg["ppo"]["n_steps"]
    if fixed_reward:
        cfg["env"]["reward"] = copy.deepcopy(fixed_reward)
    return cfg


def run_random_on_task(task: Task, runs_root: str, seed: int = 0) -> TaskOutcome:
    """Baseline: random search with the same trial budget as the agent."""
    rng = random.Random(seed)
    t0 = time.time()
    fixed_reward = (task.start_config.get("env", {}) or {}).get("reward")
    best = None
    for i in range(task.max_trials):
        cfg = _sample_config(rng, fixed_reward)
        cfg["timesteps"] = task.default_timesteps
        cfg["seed"] = task.seed
        out = os.path.join(runs_root, f"random_{i:02d}")
        summary = _train_and_score(cfg, task, out)
        if best is None or summary["mean_return"] > best["mean_return"]:
            best = summary
    return TaskOutcome(
        task=task.name, method="random", success=goal_met(best, task.goals),
        best_return=best["mean_return"], best_lin_vel_error=best["mean_lin_vel_error"],
        best_fall_rate=best["fall_rate"], n_trials=task.max_trials, tool_calls=0,
        tool_errors=0, wall_seconds=time.time() - t0, total_tokens=0,
        notes="random search",
    )


def save_outcomes(outcomes: list[TaskOutcome], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([o.__dict__ for o in outcomes], fh, indent=2)


def load_outcomes(path: str) -> list[TaskOutcome]:
    with open(path, encoding="utf-8") as fh:
        return [TaskOutcome(**d) for d in json.load(fh)]
