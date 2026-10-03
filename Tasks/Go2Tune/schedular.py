"""Go2Tune Task Schedular

Orchestrates one agent run over the Go2Locomotion-PPO environment:
1. Creates a timestamped run directory under Runs/
2. Copies Agent and Env templates into this Task (once)
3. Patches config.json files to redirect all outputs into the run directory
4. Injects initial context from the Env into the Agent
5. Fires Agent.run() — the ReAct loop runs uninterrupted inside
6. Saves a run summary on completion

Usage:
    python Tasks/Go2Tune/schedular.py [--no-viz] [--wall SECONDS]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

# --------------------------------------------------------------------------- #
# Paths relative to this file
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

AGENT_TEMPLATE          = os.path.join(_ROOT, "Agents", "PPOTuner")
ENV_TEMPLATE            = os.path.join(_ROOT, "Envs", "Go2Locomotion-PPO")
CURVE_ANALYST_TEMPLATE  = os.path.join(_ROOT, "Agents", "CurveAnalyst")
HARNESS_PATH            = os.path.join(_ROOT, "Harness")

TASK_AGENTS = os.path.join(_HERE, "Agents")
TASK_ENVS   = os.path.join(_HERE, "Envs")
RUNS_ROOT   = os.path.join(_HERE, "Runs")


def _ensure_copy(src: str, dst: str) -> None:
    """Copy src tree into dst if dst does not yet exist."""
    if not os.path.exists(dst):
        shutil.copytree(src, dst)


def _patch_json(path: str, updates: dict) -> None:
    """Merge updates into a JSON file (top-level keys only)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run(wall_seconds: float | None = None, visualize: bool = True,
        default_timesteps: int = 200_000) -> dict:
    # ------------------------------------------------------------------ setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_ROOT, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    agent_memory_dir  = os.path.join(run_dir, "agent_memory")
    env_obs_dir       = os.path.join(run_dir, "env_observations")
    agent_runs_dir    = os.path.join(run_dir, "training_runs")
    logs_dir          = os.path.join(run_dir, "logs")
    for d in (agent_memory_dir, env_obs_dir, agent_runs_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    # copy templates into Task on first run
    task_agent_dir         = os.path.join(TASK_AGENTS, "PPOTuner")
    task_env_dir           = os.path.join(TASK_ENVS,   "Go2Locomotion-PPO")
    task_curve_analyst_dir = os.path.join(TASK_AGENTS, "CurveAnalyst")
    _ensure_copy(AGENT_TEMPLATE,         task_agent_dir)
    _ensure_copy(ENV_TEMPLATE,           task_env_dir)
    _ensure_copy(CURVE_ANALYST_TEMPLATE, task_curve_analyst_dir)

    curve_analyst_main = os.path.join(task_curve_analyst_dir, "main.py")

    # patch agent config.json to redirect outputs to this run
    agent_cfg_path = os.path.join(task_agent_dir, "config.json")
    _patch_json(agent_cfg_path, {
        "memory_dir":          agent_memory_dir,
        "log_dir":             logs_dir,
        "env_path":            task_env_dir,
        "harness_path":        HARNESS_PATH,
        "curve_analyst_path":  curve_analyst_main,
    })

    # patch env config.json to redirect outputs to this run
    env_cfg_path = os.path.join(task_env_dir, "config.json")
    _patch_json(env_cfg_path, {
        "runs_root":       agent_runs_dir,
        "observations_dir": env_obs_dir,
        "parameters_path": os.path.join(task_env_dir, "parameters.json"),
    })

    # ------------------------------------------------------------------ load agent config
    with open(agent_cfg_path, encoding="utf-8") as f:
        agent_cfg = json.load(f)

    longterm_path = os.path.join(agent_memory_dir, "longterm.json")

    # ------------------------------------------------------------------ inject agent into path
    sys.path.insert(0, task_agent_dir)
    sys.path.insert(0, _ROOT)

    from Agents.PPOTuner.main import TuningAgent      # noqa: E402
    from Agents.PPOTuner.llm import LLMClient         # noqa: E402
    from Agents.CurveAnalyst.main import CurveAnalyst  # noqa: E402

    # ------------------------------------------------------------------ task definition
    goals = {
        "mean_lin_vel_error_max": 0.25,
        "fall_rate_max": 0.2,
    }

    # load weak start config from env parameters
    with open(os.path.join(task_env_dir, "parameters.json"), encoding="utf-8") as f:
        params = json.load(f)
    start_config = {
        "env": {"kp": 40.0, "action_scale": 0.4},
    }

    print(f"[schedular] run_dir = {run_dir}", flush=True)
    print(f"[schedular] agent_memory = {agent_memory_dir}", flush=True)
    print(f"[schedular] training_runs = {agent_runs_dir}", flush=True)
    print(f"[schedular] curve_analyst = {curve_analyst_main}", flush=True)

    llm_model = agent_cfg.get("model", "deepseek-reasoner")
    print(f"[schedular] llm_model = {llm_model}", flush=True)

    # ------------------------------------------------------------------ curve analysis callback
    _curve_analyst = CurveAnalyst()

    def on_trial_done(run_id: str, run_dir_: str) -> str | None:
        """Called by TuningAgent after each successful trial.
        Runs CurveAnalyst on the trial's curve.png and returns the analysis text."""
        curve_path = os.path.join(run_dir_, "curve.png")
        if not os.path.exists(curve_path):
            return None
        try:
            result_ = _curve_analyst.analyze(
                curve_path,
                "请描述训练奖励曲线的整体趋势：是否在上升？是否已收敛？有无明显震荡或崩溃？"
                f"（本次 trial: {run_id}）"
            )
            print(f"[schedular] curve analysis for {run_id}: {result_.answer[:80]}...",
                  flush=True)
            return result_.answer
        except Exception as exc:  # noqa: BLE001
            print(f"[schedular] curve analysis failed for {run_id}: {exc}", flush=True)
            return None

    # ------------------------------------------------------------------ run agent
    agent = TuningAgent(
        goals=goals,
        runs_root=agent_runs_dir,
        task_name="go2_tune_v1",
        longterm_path=longterm_path,
        client=LLMClient(model=llm_model),
        default_timesteps=default_timesteps,
        max_trials=None,
        max_wall_seconds=wall_seconds,
        visualize=visualize,
        env_path=task_env_dir,
        harness_path=HARNESS_PATH,
        start_config=start_config,
    )

    t0 = time.time()
    result = agent.run(on_trial_done=on_trial_done)
    elapsed = time.time() - t0

    # ------------------------------------------------------------------ save summary
    summary = {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "wall_seconds": round(elapsed, 1),
        "success": result.success,
        "n_trials": result.n_trials,
        "best_run_id": result.best_run_id,
        "best_metrics": result.best_metrics,
        "stopped_reason": result.stopped_reason,
        "tool_stats": result.tool_stats,
        "usage": result.usage,
        "final_report": result.final_report,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[schedular] done in {elapsed:.0f}s | success={result.success} | "
          f"trials={result.n_trials} | stopped={result.stopped_reason}", flush=True)
    print(f"[schedular] summary -> {summary_path}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Go2Tune agent task")
    ap.add_argument("--wall", type=float, default=8 * 3600,
                    help="wall-clock budget in seconds (0 = unlimited)")
    ap.add_argument("--no-viz", action="store_true",
                    help="skip post-training visualization")
    ap.add_argument("--timesteps", type=int, default=200_000,
                    help="per-trial training step budget (default 200000)")
    args = ap.parse_args()
    wall = args.wall if args.wall and args.wall > 0 else None
    run(wall_seconds=wall, visualize=not args.no_viz,
        default_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
