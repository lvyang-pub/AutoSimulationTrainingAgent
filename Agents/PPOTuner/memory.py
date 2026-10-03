"""Agent memory: short-term trial trajectory + long-term lessons.

Short-term memory holds the current run's trials and renders them as compact
context for the LLM (keeping the window bounded). Long-term memory persists
across runs to a JSON file so a later run can reuse what earlier runs learned —
this is what separates the agent from a per-run random search.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class TrialRecord:
    trial_id: int
    config: dict
    summary: dict                 # mean_return, fall_rate, mean_lin_vel_error, ...
    command: dict = field(default_factory=dict)   # what "success" means for this run
    reflection: str = ""          # one-line diagnosis produced after the trial
    accepted: bool | None = None  # did the agent judge it as meeting the goal?

    def compact(self, max_metrics: int = 6) -> dict:
        keys = ["mean_return", "std_return", "fall_rate", "mean_lin_vel_error",
                "mean_episode_length", "convergence_delta"]
        metrics = {k: self.summary.get(k) for k in keys if k in self.summary}
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "metrics": metrics,
            "reflection": self.reflection,
        }


@dataclass
class Lesson:
    """A reusable finding, persisted across runs."""
    task: str
    diagnosis: str
    recommendation: str
    evidence: str = ""


class ShortTermMemory:
    def __init__(self):
        self.trials: list[TrialRecord] = []

    def add(self, rec: TrialRecord) -> None:
        self.trials.append(rec)

    def best(self) -> TrialRecord | None:
        if not self.trials:
            return None
        return max(self.trials, key=lambda t: t.summary.get("mean_return", float("-inf")))

    def to_context(self, max_trials: int = 8, include_config: bool = True) -> str:
        """Render recent trials as text for the LLM."""
        if not self.trials:
            return "No trials yet."
        recent = self.trials[-max_trials:]
        lines = []
        for t in recent:
            c = t.compact()
            m = c["metrics"]
            cfg = c["config"] if include_config else "(omitted)"
            lines.append(
                f"Trial {t.trial_id}: return={m.get('mean_return')} "
                f"fall_rate={m.get('fall_rate')} lin_vel_err={m.get('mean_lin_vel_error')} "
                f"conv={m.get('convergence_delta')}\n"
                f"    config={json.dumps(cfg, separators=(',', ':'))}\n"
                f"    reflection={t.reflection or '(none)'}"
            )
        return "\n".join(lines)


class LongTermMemory:
    def __init__(self, path: str):
        self.path = path
        self.lessons: list[Lesson] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    data = json.load(fh)
                self.lessons = [Lesson(**d) for d in data.get("lessons", [])]
            except (json.JSONDecodeError, TypeError, KeyError):
                self.lessons = []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"lessons": [asdict(x) for x in self.lessons]}, fh, indent=2)

    def add(self, lesson: Lesson) -> None:
        # de-duplicate on diagnosis text
        for existing in self.lessons:
            if existing.task == lesson.task and existing.diagnosis == lesson.diagnosis:
                existing.recommendation = lesson.recommendation
                existing.evidence = lesson.evidence
                self.save()
                return
        self.lessons.append(lesson)
        self.save()

    def for_task(self, task: str, limit: int = 10) -> list[Lesson]:
        return [x for x in self.lessons if x.task == task][-limit:]

    def to_context(self, task: str, limit: int = 10) -> str:
        lessons = self.for_task(task, limit)
        if not lessons:
            return "No accumulated lessons yet."
        return "\n".join(
            f"- [{x.task}] {x.diagnosis} -> {x.recommendation}" for x in lessons
        )
