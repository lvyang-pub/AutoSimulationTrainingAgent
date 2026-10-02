"""DeepSeek client wrapper.

Primary mode is OpenAI-style function calling. deepseek-flash does not always
emit clean tool_calls, so `chat_tool` falls back to parsing a JSON object out of
the message content. Token usage is accumulated across the session.

One design point matters for the harness: `ToolCall` carries a `source` field
("native" or "fallback") so the agent can report how reliably the model uses
tools — that is one of the evaluation metrics.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from openai import OpenAI

DEFAULT_MODEL = os.environ.get("AUTOSIM_MODEL", "deepseek-flash")
DEFAULT_BASE_URL = os.environ.get("AUTOSIM_BASE_URL", "https://api.deepseek.com")


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) so a local file works without deps."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


def _default_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Export it before running, e.g.\n"
            "  Windows (PowerShell): $env:DEEPSEEK_API_KEY=\"sk-...\"\n"
            "  bash:                  export DEEPSEEK_API_KEY=sk-...\n"
            "Or put it in a .env file (see .env.example)."
        )
    return key


@dataclass
class ToolCall:
    name: str
    arguments: dict
    source: str = "native"   # "native" | "fallback"
    call_id: str = "call_0"


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 base_url: str = DEFAULT_BASE_URL, temperature: float = 0.3,
                 timeout: float = 120.0, max_retries: int = 3):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key or _default_api_key(),
            base_url=base_url,
            timeout=timeout,
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Pull the first JSON object out of free text."""
        if not text:
            return None
        # strip code fences
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        candidates = []
        if fenced:
            candidates.append(fenced.group(1))
        # balance braces scan
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
        for c in candidates:
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                continue
        return None

    def _accumulate(self, resp) -> None:
        self.n_calls += 1
        usage = getattr(resp, "usage", None)
        if usage:
            self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    # -------------------------------------------------------------- main call
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """One chat completion. Retries transient failures. Returns LLMResponse."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(model=self.model, messages=messages,
                              temperature=self.temperature)
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                resp = self.client.chat.completions.create(**kwargs)
                self._accumulate(resp)
                msg = resp.choices[0].message
                content = msg.content or ""

                calls: list[ToolCall] = []
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        calls.append(ToolCall(
                            name=tc.function.name,
                            arguments=args,
                            source="native",
                            call_id=tc.id or "call_0",
                        ))
                else:
                    parsed = self._extract_json(content)
                    if parsed and isinstance(parsed, dict) and "tool" in parsed:
                        calls.append(ToolCall(
                            name=str(parsed["tool"]),
                            arguments=parsed.get("arguments", {}) or {},
                            source="fallback",
                        ))

                raw = {"role": "assistant", "content": content}
                if getattr(msg, "tool_calls", None):
                    raw["tool_calls"] = [
                        {"id": c.call_id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in calls
                    ]
                return LLMResponse(content=content, tool_calls=calls, raw_message=raw,
                                   prompt_tokens=getattr(getattr(resp, "usage", None),
                                                         "prompt_tokens", 0) or 0,
                                   completion_tokens=getattr(getattr(resp, "usage", None),
                                                             "completion_tokens", 0) or 0)
            except Exception as exc:  # noqa: BLE001 - surface as retry
                last_err = exc
                if attempt < self.max_retries - 1:
                    continue
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def usage_summary(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }
