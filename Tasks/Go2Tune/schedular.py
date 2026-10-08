"""Go2Tune Task Schedular.

Wires the task to StandardAgent-A. The agent is generic; everything Go2 lives
here and in this folder's `task.py` / `assembly.py` / `Agents/Go2Tuner/`.

1. Resolves paths and creates a timestamped run directory under Runs/.
2. Loads the StandardAgent-A package (folder name has a hyphen, so it is loaded
   explicitly and registered as the module `asta_agent`).
3. Assembles the agent's tools from the environment via `assembly` (which owns
   the env->Toolbox bridge and the injected execution paths), then builds the
   stopping rule and the briefing.
4. Constructs the agent -- here the task-local Go2Tuner, a copy of
   StandardAgent-A modified for tuning -- and runs it.
5. Saves a run summary on completion.

Usage:
    python Tasks/Go2Tune/schedular.py [--no-viz] [--wall SECONDS] [--timesteps N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

# --------------------------------------------------------------------------- #
# paths relative to this file
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

AGENT_PKG_DIR = os.path.join(_ROOT, "Agents", "StandardAgent-A")
TASK_ENVS = os.path.join(_HERE, "Envs")
TASK_AGENT_DIR = os.path.join(_HERE, "Agents")          # task-side helper agents
RUNS_ROOT = os.path.join(_HERE, "Runs")


def _load_agent_package(alias: str = "asta_agent"):
    """Load Agents/StandardAgent-A as an importable package under `alias`.

    The folder name contains a hyphen, which cannot be a Python identifier, so
    the package is loaded by path and registered in sys.modules *before* it is
    executed -- that lets its own relative imports (and `from asta_agent import
    ...` in task code) resolve.
    """
    if alias in sys.modules:
        return sys.modules[alias]
    init = os.path.join(AGENT_PKG_DIR, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        alias, init, submodule_search_locations=[AGENT_PKG_DIR])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def run(wall_seconds: float | None = None, visualize: bool = True,
        default_timesteps: int = 200_000, max_trials: int | None = None) -> dict:
    # ------------------------------------------------------------------ setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_ROOT, timestamp)
    agent_memory_dir = os.path.join(run_dir, "agent_memory")
    agent_runs_dir = os.path.join(run_dir, "training_runs")
    logs_dir = os.path.join(run_dir, "logs")
    for d in (agent_memory_dir, agent_runs_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    task_env_dir = os.path.join(TASK_ENVS, "Go2Locomotion-PPO")
    longterm_path = os.path.join(agent_memory_dir, "longterm.json")

    # ------------------------------------------------------------------ load pieces
    asta = _load_agent_package("asta_agent")
    sys.path.insert(0, _HERE)                    # so `import task` / `import assembly`
    sys.path.insert(0, TASK_AGENT_DIR)           # so `Agents.*` resolves

    import assembly as assembler
    import task as task_def
    from Agents.CurveAnalyst.main import CurveAnalyst
    from Agents.Go2Tuner.main import StandardAgent

    print(f"[schedular] run_dir       = {run_dir}", flush=True)
    print(f"[schedular] agent_memory  = {agent_memory_dir}", flush=True)
    print(f"[schedular] training_runs = {agent_runs_dir}", flush=True)
    print(f"[schedular] env           = {task_env_dir}", flush=True)

    llm_model = os.environ.get("AUTOSIM_MODEL", "deepseek-reasoner")
    print(f"[schedular] llm_model     = {llm_model}", flush=True)

    # ------------------------------------------------------------------ assembly: agent + environment
    # `assembly` is the schedular's seam between the generic agent and this
    # environment: it scans the env's action functions, turns them into agent
    # tools, and binds the execution paths (runs_root / out_dir / policy_path)
    # the framework owns so the model never writes them.
    exec_ctx = assembler.ExecutionContext(
        runs_root=agent_runs_dir,
        env_path=task_env_dir,
        visualize=visualize,
        default_timesteps=default_timesteps,
    )
    toolbox = assembler.compose(task_env_dir, exec_ctx,
                                curve_analyst=CurveAnalyst())

    # ------------------------------------------------------------------ stopping rule
    should_stop = task_def.make_should_stop(max_trials, wall_seconds)

    # ------------------------------------------------------------------ briefing
    client = asta.LLMClient(model=llm_model)

    # ------------------------------------------------------------------ agent (task-local copy of StandardAgent-A)
    # The copy is a verbatim base whose *prompts.py* carries the Go2 wording; the
    # base resolves its system and initial prompts from there (build_system_prompt
    # reads the env's search space), so nothing prompt-ish is injected here.
    agent = StandardAgent(
        toolbox=toolbox,
        objective_fn=task_def.objective,
        task_name=task_def.TASK_NAME,
        goal_text=task_def.goal_text(),
        metric_keys=task_def.METRIC_KEYS,
        longterm_path=longterm_path,
        client=client,
        should_stop=should_stop,
        env_path=task_env_dir,
        extra_hint=task_def.STRATEGY_HINT,
        max_trials=max_trials,
        max_wall_seconds=wall_seconds,
        default_timesteps=default_timesteps,
        command=task_def.COMMAND,
        start_config=task_def.START_CONFIG,
    )

    t0 = time.time()
    result = agent.run()
    elapsed = time.time() - t0

    # ------------------------------------------------------------------ summary
    extra = result.extra or {}
    summary = {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "wall_seconds": round(elapsed, 1),
        "stopped_reason": result.stopped_reason,
        "best_trial": extra.get("best_trial"),
        "best_metrics": extra.get("best_metrics"),
        "n_trials": extra.get("n_trials", 0),
        "objective_met": task_def.objective_from_summary(extra.get("best_metrics") or {}),
        "llm_finished_success": bool(result.finish_data.get("success")),
        "tool_stats": {
            "total_calls": result.tool_calls,
            "errors": result.tool_errors,
            "native": result.native_tool_calls,
            "fallback": result.fallback_tool_calls,
        },
        "usage": result.usage,
        "final_report": result.final_report,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[schedular] done in {elapsed:.0f}s | objective_met={summary['objective_met']} | "
          f"trials={summary['n_trials']} | stopped={result.stopped_reason}", flush=True)
    print(f"[schedular] summary -> {summary_path}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Go2Tune task")
    ap.add_argument("--wall", type=float, default=8 * 3600,
                    help="wall-clock budget in seconds (0 = unlimited)")
    ap.add_argument("--no-viz", action="store_true",
                    help="skip post-training visualization")
    ap.add_argument("--timesteps", type=int, default=200_000,
                    help="per-trial training step budget (default 200000)")
    ap.add_argument("--max-trials", type=int, default=None,
                    help="trial budget (default: unlimited)")
    args = ap.parse_args()
    wall = args.wall if args.wall and args.wall > 0 else None
    run(wall_seconds=wall, visualize=not args.no_viz,
        default_timesteps=args.timesteps, max_trials=args.max_trials)


if __name__ == "__main__":
    main()
