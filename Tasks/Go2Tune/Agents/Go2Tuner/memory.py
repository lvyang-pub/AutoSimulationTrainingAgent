"""Agent memory: a per-run trial trajectory + cross-run lessons.

Short-term memory holds the current run's trials. Long-term memory persists to a
JSON file so a later run can reuse what earlier runs learned -- this is what
separates the agent from a per-run random search.

This module is task-agnostic: a trial is *any* structured record with an id, a
config and a metrics dict. Nothing here names a domain. Where a task wants a
specific render, it supplies its own `format_fn`; otherwise the generic layout
is used.

Both stores expose a `summarize(client, ...)` path, which compresses their
contents through the LLM using SHORT_MEMORY_PROMPT / LONG_MEMORY_PROMPT from
`prompts.py`. The raw accessors (`to_context`, `for_task`) remain available and
are what the default loop uses, so behaviour is identical when no summarizer is
wired in.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class TrialRecord:
    """One trial: a config that was run and the metrics it produced."""
    trial_id: str
    config: dict
    summary: dict                                   # metric name -> value
    reflection: str = ""                            # one-line diagnosis, if any
    accepted: bool | None = None                    # did it meet the goal?

    def compact(self, max_metrics: int = 6) -> dict:
        metrics = dict(list(self.summary.items())[:max_metrics])
        return {"trial_id": self.trial_id, "config": self.config,
                "metrics": metrics, "reflection": self.reflection}


@dataclass
class Lesson:
    """A reusable finding, persisted across runs."""
    task: str
    diagnosis: str
    recommendation: str
    evidence: str = ""


# --------------------------------------------------------------------------- #
# short-term memory (this run)
# --------------------------------------------------------------------------- #
class ShortTermMemory:
    def __init__(self, format_fn=None):
        # format_fn(records: list[dict]) -> str renders the generic record list.
        self.trials: list[TrialRecord] = []
        self._format_fn = format_fn
        self._compressed: str = ""

    def add(self, rec: TrialRecord) -> None:
        self.trials.append(rec)

    def best(self, key: str = "mean_return") -> TrialRecord | None:
        if not self.trials:
            return None
        return max(self.trials, key=lambda t: t.summary.get(key, float("-inf")))

    def records(self, max_trials: int = 8, include_config: bool = True) -> list[dict]:
        """The generic shape handed to `format_fn` (or the default renderer)."""
        out = []
        for t in self.trials[-max_trials:]:
            c = t.compact()
            if not include_config:
                c["config"] = "(omitted)"
            out.append(c)
        return out

    def to_context(self, max_trials: int = 8, include_config: bool = True) -> str:
        """Render recent trials as text for the LLM."""
        if self._compressed:
            return self._compressed
        if not self.trials:
            return "No trials yet."
        records = self.records(max_trials, include_config)
        if self._format_fn is not None:
            return self._format_fn(records)
        lines = []
        for c in records:
            m = c["metrics"]
            metrics = ", ".join(f"{k}={v}" for k, v in m.items())
            lines.append(f"Trial {c['trial_id']}: {metrics}\n"
                         f"    config={json.dumps(c['config'], separators=(',', ':'))}\n"
                         f"    reflection={c['reflection'] or '(none)'}")
        return "\n".join(lines)

    def summarize(self, client, prompt: str) -> str:
        """LLM-compress the trials; the result then feeds `to_context`."""
        if not self.trials:
            return ""
        text = self._format_fn(self.records()) if self._format_fn else self.to_context()
        try:
            resp = client.chat([
                {"role": "system", "content": "You compress experiment logs."},
                {"role": "user", "content": prompt.format(trials=text)},
            ])
            self._compressed = (resp.content or "").strip() or self._compressed
        except Exception as exc:  # noqa: BLE001 - summarization is best-effort
            print(f"[memory] short-term summarize failed: {exc}", flush=True)
        return self._compressed


# --------------------------------------------------------------------------- #
# long-term memory (across runs)
# --------------------------------------------------------------------------- #
class LongTermMemory:
    def __init__(self, path: str):
        self.path = path
        self.lessons: list[Lesson] = []
        self._compressed: str = ""
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
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"lessons": [asdict(x) for x in self.lessons]}, fh, indent=2,
                      ensure_ascii=False)

    def add(self, lesson: Lesson) -> None:
        for existing in self.lessons:                   # de-duplicate on diagnosis
            if existing.task == lesson.task and existing.diagnosis == lesson.diagnosis:
                existing.recommendation = lesson.recommendation
                existing.evidence = lesson.evidence
                self.save()
                return
        self.lessons.append(lesson)
        self.save()

    def for_task(self, task: str, limit: int = 10) -> list[Lesson]:
        return [x for x in self.lessons if x.task == task][-limit:]

    def to_context(self, task: str | None = None, limit: int = 10) -> str:
        if self._compressed:
            return self._compressed
        lessons = ([x for x in self.lessons] if task is None
                   else self.for_task(task))[-limit:]
        if not lessons:
            return "No accumulated lessons yet."
        return "\n".join(f"- [{x.task}] {x.diagnosis} -> {x.recommendation}"
                         for x in lessons)

    def summarize(self, client, prompt: str, task: str | None = None) -> str:
        """LLM-merge the lessons into durable rules; feeds `to_context`."""
        lessons = [x for x in self.lessons] if task is None else self.for_task(task)
        if not lessons:
            return ""
        text = "\n".join(f"[{x.task}] {x.diagnosis} -> {x.recommendation} "
                         f"(evidence: {x.evidence})" for x in lessons)
        try:
            resp = client.chat([
                {"role": "system", "content": "You consolidate lessons."},
                {"role": "user", "content": prompt.format(lessons=text)},
            ])
            self._compressed = (resp.content or "").strip() or self._compressed
        except Exception as exc:  # noqa: BLE001 - summarization is best-effort
            print(f"[memory] long-term summarize failed: {exc}", flush=True)
        return self._compressed
