"""Go2Tune task definition: what success means, and how the agent is briefed.

Everything here is Go2-specific: the goal thresholds, the locomotion metric
names, the wording of the task, and the strategy hint. The agent receives all
of it as data at construction time and stays generic.
"""
from __future__ import annotations

import json

from asta_agent import TrialRecord

# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #
# A criterion is (metric_key, comparison, threshold). A trial is accepted when
# every criterion holds -- this is what replaces the old hardcoded `goal_met`.
GOALS = [
    ("mean_lin_vel_error", "<=", 0.25),
    ("fall_rate", "<=", 0.2),
]

METRIC_KEYS = ["mean_return", "std_return", "fall_rate", "mean_lin_vel_error",
               "mean_episode_length", "convergence_delta"]

TASK_NAME = "go2_locomotion_tuning"
COMMAND = [1.0, 0.0, 0.0]
START_CONFIG = {"env": {"kp": 40.0, "action_scale": 0.4}}
DEFAULT_TIMESTEPS = 200_000


def objective(rec: TrialRecord) -> bool:
    """Objective success check for one trial, independent of the LLM."""
    summary = rec.summary or {}
    if not summary:
        return False
    for key, op, threshold in GOALS:
        val = summary.get(key)
        if val is None:
            return False
        if op == "<=" and not val <= threshold:
            return False
        if op == ">=" and not val >= threshold:
            return False
        if op == "<" and not val < threshold:
            return False
        if op == ">" and not val > threshold:
            return False
    return True


def goal_text() -> str:
    parts = []
    for key, op, threshold in GOALS:
        parts.append(f"{key} {op} {threshold}")
    return ", ".join(parts) or "(no goal set)"


# --------------------------------------------------------------------------- #
# stopping rule -- implemented here in the schedular/task, passed to the agent
# --------------------------------------------------------------------------- #
def make_should_stop(max_trials: int | None, max_wall_seconds: float | None):
    """Build the stop function the agent calls once per loop iteration.

    Returns True when the total objective (the goal) is already met by some
    trial, when the trial budget is spent, or when the wall-clock budget is
    spent. The agent itself knows none of this; it only asks.
    """
    def should_stop(state) -> bool:
        if max_wall_seconds is not None and state.elapsed > max_wall_seconds:
            return True
        trial_results = [r for r in state.results_named("run_training")
                         if r.status == "ok"]
        if max_trials is not None and len(trial_results) >= max_trials:
            return True
        # goal reached: any trial's summary satisfies every criterion
        for r in trial_results:
            summary = (r.data or {}).get("summary") or {}
            if _summary_meets_goal(summary):
                return True
        return False
    return should_stop


def objective_from_summary(summary: dict) -> bool:
    """Public check: does a plain metrics dict meet every goal criterion?"""
    return _summary_meets_goal(summary)


def _summary_meets_goal(summary: dict) -> bool:
    if not summary:
        return False
    for key, op, threshold in GOALS:
        val = summary.get(key)
        if val is None:
            return False
        if op == "<=" and not val <= threshold:
            return False
        if op == ">=" and not val >= threshold:
            return False
        if op == "<" and not val < threshold:
            return False
        if op == ">" and not val > threshold:
            return False
    return True


# --------------------------------------------------------------------------- #
# initial briefing handed to the agent
# --------------------------------------------------------------------------- #
STRATEGY_HINT = (
    "Reward shaping is fixed; tune the plant gains (kp, kd, action_scale) and "
    "PPO hyperparameters. Raising kp without enough damping causes oscillation; "
    "lower action_scale with a moderate kp usually stabilises the gait first."
)


def build_initial_prompt(task_name: str, lessons: str,
                         max_trials: int | None, max_wall_seconds: float | None,
                         default_timesteps: int) -> str:
    if max_trials is None and max_wall_seconds is None:
        budget = ("Budget: none -- you are NOT step-limited. Keep running trials "
                  "until the goal is met; call finish only when success requires it "
                  "or you have exhausted sensible ideas. ")
    else:
        bits = []
        if max_trials is not None:
            bits.append(f"Trial budget: {max_trials} training runs.")
        if max_wall_seconds is not None:
            bits.append(f"Wall-clock budget: {max_wall_seconds:.0f}s.")
        budget = " ".join(bits) + " "

    start = {"timesteps": default_timesteps, "env": {"command": list(COMMAND)}}
    for section, vals in START_CONFIG.items():
        if isinstance(vals, dict) and isinstance(start.get(section), dict):
            start[section] = {**start[section], **vals}
        else:
            start[section] = vals

    return (
        f"Task: make the Go2 quadruped walk forward at {COMMAND[0]:.2f} m/s.\n"
        f"Success requires: {goal_text()}.\n"
        f"{budget}Each default run trains {default_timesteps} steps.\n\n"
        f"Lessons from previous sessions:\n{lessons}\n\n"
        f"Strategy note: {STRATEGY_HINT}\n\n"
        f"Suggested starting config (you may change it): "
        f"{json.dumps(start, separators=(',', ':'))}\n\n"
        "Begin. Think briefly, then call one or more tools."
    )
