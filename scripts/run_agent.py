"""Run the tuning agent on a single task.

Example:
    python scripts/run_agent.py --task go2_tune_v1 --out runs/agent_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import TuningAgent
from src.eval.tasks import get_task
from src.training.config import normalize
from scriptlog import start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="go2_tune_v1")
    ap.add_argument("--out", default="runs/agent")
    ap.add_argument("--trials", type=int, help="override max trials")
    ap.add_argument("--timesteps", type=int, help="override steps per trial")
    ap.add_argument("--wall", type=float, default=1800, help="wall-clock budget (s)")
    ap.add_argument("--longterm", default="memory/longterm.json")
    ap.add_argument("--hint", default="", help="extra strategy note")
    ap.add_argument("--log-dir", default="log", help="where the run transcript is written")
    ap.add_argument("--no-viz", action="store_true",
                    help="skip the per-trial rollout video + curve PNG")
    args = ap.parse_args()

    log_path, _fh = start("agent", args.log_dir)
    print(f"[log] transcript -> {log_path}")

    task = get_task(args.task)
    os.makedirs(args.out, exist_ok=True)
    agent = TuningAgent(
        goals=dict(task.goals),
        runs_root=os.path.join(args.out, "runs"),
        task_name=task.name,
        longterm_path=args.longterm,
        default_timesteps=args.timesteps or task.default_timesteps,
        max_trials=args.trials or task.max_trials,
        max_wall_seconds=args.wall,
        extra_hint=(args.hint or task.hint),
        command=task.command,
        start_config=task.start_config,
        visualize=not args.no_viz,
    )
    result = agent.run()
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)

    print(f"\n=== Agent run: {task.name} ===")
    print(f"success (objective): {result.success}")
    print(f"llm claimed success : {result.llm_finished_success}")
    print(f"best run: {result.best_run_id}  metrics={result.best_metrics}")
    print(f"trials={result.n_trials} tool_calls={result.tool_stats} "
          f"tokens={result.usage['total_tokens']} wall={result.wall_seconds}s")
    print(f"stopped: {result.stopped_reason}")
    print("\n--- final report ---")
    print(result.final_report)


if __name__ == "__main__":
    main()
