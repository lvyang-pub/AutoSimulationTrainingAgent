"""Reflection: turn a finished trial into a diagnosis + a persisted lesson.

Reflection is what lets the agent improve its search over trials (and, via
long-term memory, across runs) instead of behaving like a fresh random search
each time. It is a separate LLM call with a focused prompt so it stays cheap.
"""
from __future__ import annotations

import json

from .llm import LLMClient
from .memory import Lesson
from .prompts import REFLECTION_PROMPT


def reflect_on_trial(client: LLMClient, task: str, config: dict, summary: dict,
                     goals: dict, lessons_context: str) -> tuple[str, Lesson | None]:
    """Return (reflection_text, lesson_or_None). Never raises."""
    metrics = {k: summary.get(k) for k in
               ("mean_return", "std_return", "fall_rate", "mean_lin_vel_error",
                "mean_episode_length", "convergence_delta")}
    prompt = REFLECTION_PROMPT.format(
        config=json.dumps(config, separators=(",", ":")),
        metrics=json.dumps(metrics, indent=2),
        lessons=lessons_context or "(none)",
    )
    try:
        resp = client.chat([
            {"role": "system", "content": "You are an RL debugging assistant."},
            {"role": "user", "content": prompt},
        ])
        text = (resp.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - reflection is best-effort
        return f"(reflection unavailable: {type(exc).__name__})", None

    lesson = Lesson(
        task=task,
        diagnosis=text[:280],
        recommendation=_extract_recommendation(text),
        evidence=json.dumps(metrics, separators=(",", ":")),
    )
    return text, lesson


def _extract_recommendation(text: str) -> str:
    """Pull the sentence containing a recommendation hint, else the last line."""
    for marker in ("recommend", "should", "next trial", "try "):
        idx = text.lower().find(marker)
        if idx >= 0:
            tail = text[idx:]
            end = tail.find(". ")
            return (tail[:end + 1] if end > 0 else tail)[:200]
    return text[-200:]
