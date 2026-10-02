"""Evaluation metrics for a tuning method on a task."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskOutcome:
    task: str
    method: str
    success: bool                 # objective goal met
    best_return: float
    best_lin_vel_error: float
    best_fall_rate: float
    n_trials: int
    tool_calls: int
    tool_errors: int
    wall_seconds: float
    total_tokens: int
    native_tool_calls: int = 0
    fallback_tool_calls: int = 0
    fault_injected: bool = False
    recovered_from_fault: bool = False
    notes: str = ""

    @property
    def tool_success_rate(self) -> float:
        return 1 - self.tool_errors / self.tool_calls if self.tool_calls else 0.0


def summarize(outcomes: list[TaskOutcome]) -> dict:
    """Aggregate a set of outcomes into the evaluation's headline metrics."""
    n = len(outcomes)
    if n == 0:
        return {}
    success = sum(o.success for o in outcomes)
    return {
        "n_tasks": n,
        "task_success_rate": round(success / n, 3),
        "mean_best_return": round(sum(o.best_return for o in outcomes) / n, 1),
        "mean_lin_vel_error": round(sum(o.best_lin_vel_error for o in outcomes) / n, 4),
        "mean_fall_rate": round(sum(o.best_fall_rate for o in outcomes) / n, 3),
        "mean_trials": round(sum(o.n_trials for o in outcomes) / n, 2),
        "mean_tool_calls": round(sum(o.tool_calls for o in outcomes) / n, 1),
        "tool_success_rate": round(sum(o.tool_success_rate for o in outcomes) / n, 3),
        "mean_wall_seconds": round(sum(o.wall_seconds for o in outcomes) / n, 1),
        "mean_tokens": round(sum(o.total_tokens for o in outcomes) / n, 0),
        "total_tokens": sum(o.total_tokens for o in outcomes),
        "native_tool_call_ratio": round(
            sum(o.native_tool_calls for o in outcomes) /
            max(1, sum(o.native_tool_calls + o.fallback_tool_calls for o in outcomes)), 3),
        "fault_recovery_rate": _fault_rate(outcomes),
    }


def _fault_rate(outcomes: list[TaskOutcome]) -> float | None:
    faulted = [o for o in outcomes if o.fault_injected]
    if not faulted:
        return None
    return round(sum(o.recovered_from_fault for o in faulted) / len(faulted), 3)
