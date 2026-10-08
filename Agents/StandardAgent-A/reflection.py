"""Post-trial reflection: turn a finished trial into a lesson.

After a trial completes, the agent asks the LLM to diagnose the result and name
one concrete next move. The free-text diagnosis goes into the short-term memory;
if it names a reusable rule, that is distilled into a `Lesson` for long-term
memory.

Task-agnostic: this module knows a trial has a config and a metrics dict, and
nothing else. The prompt text lives in `prompts.py` (REFLECTION_PROMPT).
"""
from __future__ import annotations

import json

from .memory import Lesson
from .prompts import REFLECTION_PROMPT


def reflect_on_trial(client, task: str, config: dict, metrics: dict,
                     lessons_context: str = "", prompt: str = REFLECTION_PROMPT):
    """Return (diagnosis_text, Lesson|None) for one finished trial.

    Never raises: if the LLM call or parsing fails, the diagnosis falls back to
    a plain summary line and no lesson is produced.
    """
    try:
        resp = client.chat([
            {"role": "system", "content": "You are diagnosing an experiment trial."},
            {"role": "user", "content": prompt.format(
                config=json.dumps(config, ensure_ascii=False),
                metrics=json.dumps(metrics, ensure_ascii=False),
                lessons=lessons_context or "(none)",
            )},
        ])
        text = (resp.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - reflection is best-effort
        return f"(reflection unavailable: {exc})", None

    if not text:
        return "(no reflection produced)", None

    recommendation = _extract_recommendation(text)
    lesson = None
    if recommendation:
        lesson = Lesson(task=task, diagnosis=text.splitlines()[0][:200],
                        recommendation=recommendation,
                        evidence=json.dumps(metrics, ensure_ascii=False)[:200])
    return text, lesson


def _extract_recommendation(text: str) -> str:
    """Pull the recommendation half out of the reflection, if it is separable."""
    low = text.lower()
    for marker in ("recommendation:", "recommend:", "next trial:", "next:"):
        idx = low.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip().split("\n")[0][:200]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1][:200] if len(lines) > 1 else ""
