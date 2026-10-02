"""Run the evaluation: agent vs baselines over the task set.

Example:
    python scripts/run_eval.py --tasks go2_tune_v1 go2_tune_hard --methods agent naive random
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.llm import LLMClient
from src.eval import report, runner
from src.eval.metrics import summarize
from src.eval.tasks import TASKS
from scriptlog import start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--methods", nargs="+", default=["agent", "naive", "random"])
    ap.add_argument("--out", default="eval/round1")
    ap.add_argument("--longterm", default="memory/longterm.json")
    ap.add_argument("--fault-task", default=None, help="inject a fault on this task")
    ap.add_argument("--hint", default="", help="strategy hint for the agent")
    ap.add_argument("--compare", default=None, help="previous outcomes.json to compare against")
    ap.add_argument("--title", default="智能体评测")
    ap.add_argument("--log-dir", default="log", help="where the run transcript is written")
    args = ap.parse_args()

    log_path, _fh = start("eval", args.log_dir)
    print(f"[log] transcript -> {log_path}")

    os.makedirs(args.out, exist_ok=True)
    llm = LLMClient()
    outcomes = []

    for tname in args.tasks:
        task = TASKS[tname]
        if "naive" in args.methods:
            print(f"[eval] {tname}: naive ...")
            outcomes.append(runner.run_naive_on_task(task, os.path.join(args.out, tname)))
        if "random" in args.methods:
            print(f"[eval] {tname}: random search ...")
            outcomes.append(runner.run_random_on_task(task, os.path.join(args.out, tname),
                                                      seed=task.seed))
        if "agent" in args.methods:
            print(f"[eval] {tname}: agent ...")
            fault = task.max_trials if args.fault_task == tname else None
            outcomes.append(runner.run_agent_on_task(
                task, os.path.join(args.out, tname, "agent"), args.longterm,
                llm=llm, extra_hint=(args.hint or task.hint), fault_after=fault))

    runner.save_outcomes(outcomes, os.path.join(args.out, "outcomes.json"))
    compare = runner.load_outcomes(args.compare) if args.compare else None
    md = report.render(outcomes, title=args.title, compare_with=compare)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)

    print("\n" + md)
    print("\nsummary:", summarize(outcomes))


if __name__ == "__main__":
    main()
