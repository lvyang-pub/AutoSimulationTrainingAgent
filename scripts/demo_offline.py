"""Offline demo of the agent loop: no API key, no long training.

Runs the full ReAct harness against a *scripted* LLM and a *stub* trainer, so a
grader can watch the control flow — trial -> reflection -> memory -> finish —
in a couple of seconds. Real runs use scripts/run_agent.py.

    python scripts/demo_offline.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.agent.tools as tools_mod
from src.agent.agent import TuningAgent
from src.agent.llm import LLMResponse, ToolCall
from src.eval.tasks import get_task


class ScriptedLLM:
    """Plays a fixed sequence of tool calls; answers reflection/report calls."""

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None):
        if tools is None:
            return LLMResponse(content="Diagnosis: the policy is standing (vx~0). "
                                       "Recommendation: drop `alive` and raise `track_lin_vel`.")
        if not self.script:
            return LLMResponse(content="(done)")
        name, args = self.script.pop(0)
        return LLMResponse(content="", tool_calls=[
            ToolCall(name=name, arguments=args, source="native")])

    def usage_summary(self):
        return {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# Fake summaries shaped like the calibrated hyperparameter sweep: a weak start
# (~150), a cliff config (~50), then a good config (~197).
SUMMARIES = [
    {"mean_return": 149.0, "std_return": 12.0, "fall_rate": 0.0, "mean_lin_vel_error": 0.98,
     "mean_episode_length": 500.0, "convergence_delta": 40.0, "train_seconds": 1.0},   # weak plant
    {"mean_return": 53.0, "std_return": 30.0, "fall_rate": 0.0, "mean_lin_vel_error": 0.99,
     "mean_episode_length": 500.0, "convergence_delta": -20.0, "train_seconds": 1.0},  # over-stiff PD
    {"mean_return": 197.0, "std_return": 6.0, "fall_rate": 0.0, "mean_lin_vel_error": 1.00,
     "mean_episode_length": 500.0, "convergence_delta": 55.0, "train_seconds": 1.0},   # tuned
]
_counter = {"i": 0}


def fake_train(config, out_dir, verbose=0, **kw):
    os.makedirs(out_dir, exist_ok=True)
    s = SUMMARIES[min(_counter["i"], len(SUMMARIES) - 1)]
    _counter["i"] += 1
    return dict(s)


def main() -> None:
    tools_mod.train = fake_train               # no real PPO
    task = get_task("go2_tune_v1")
    start_over = {**task.start_config, "timesteps": task.default_timesteps}
    script = [
        ("run_training", {"config": start_over}),
        ("get_metrics", {"run_id": "trial_001"}),
        ("run_training", {"config": {**start_over, "ppo": {"learning_rate": 1e-3}}}),
        ("run_training", {"config": {**start_over, "env": {"kp": 80.0, "kd": 1.6,
                                                           "action_scale": 0.4}}}),
        ("finish", {"success": True,
                    "reasoning": "config with kp=80/ascale=0.4 reached return 197 >= 190"}),
    ]
    llm = ScriptedLLM(script)
    out = os.path.join(tempfile.gettempdir(), "autosim_demo")
    agent = TuningAgent(
        goals=dict(task.goals),
        runs_root=out, task_name="demo",
        longterm_path=os.path.join(out, "longterm.json"),
        max_trials=5, max_llm_steps=12, client=llm,
        command=task.command,
        start_config=task.start_config,
    )
    res = agent.run()

    print("=" * 60)
    print("OFFLINE AGENT DEMO (scripted LLM + stub trainer)")
    print("=" * 60)
    for t in res.trials:
        m = t["metrics"]
        print(f"  {t['trial_id']}: return={m.get('mean_return')} "
              f"vel_err={m.get('mean_lin_vel_error')} fall={m.get('fall_rate')}")
    print(f"\nobjective success : {res.success}")
    print(f"model claimed     : {res.llm_finished_success}")
    print(f"best run          : {res.best_run_id}  metrics={res.best_metrics}")
    print(f"trials / tool calls / errors : {res.n_trials} / "
          f"{res.tool_stats['total_calls']} / {res.tool_stats['errors']}")
    print(f"stopped reason    : {res.stopped_reason}")
    print(f"long-term lessons : {os.path.join(out, 'longterm.json')}")
    print("\n(real run: python scripts/run_agent.py --task go2_tune_v1)")


if __name__ == "__main__":
    main()
