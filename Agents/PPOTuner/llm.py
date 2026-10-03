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
import time
from dataclasses import dataclass, field

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI

DEFAULT_MODEL = os.environ.get("AUTOSIM_MODEL", "deepseek-reasoner")
DEFAULT_BASE_URL = os.environ.get("AUTOSIM_BASE_URL", "https://api.deepseek.com")

# Models that do not accept a `temperature` parameter
_NO_TEMPERATURE_MODELS = {"deepseek-reasoner"}

# Connection-level failures that are safe to retry on a fresh connection. A
# read timeout now means "the server never started responding" (connect+write
# budget), which is safe to retry — unlike an overall call timeout, which could
# fire mid-generation and re-bill a partial completion.
_RETRYABLE = (APIConnectionError, APITimeoutError, httpx.ConnectError, httpx.ReadTimeout)

# Segmented per-call budgets (seconds). connect/write bound "the server never
# answered"; read is the ceiling on a single generation. It is generous (a
# reasoner thinking for 10 minutes is not a fault) but FINITE, so no call can
# park forever. Override any of these via the matching env var.
DEFAULT_CONNECT_TIMEOUT = float(os.environ.get("AUTOSIM_LLM_CONNECT_TIMEOUT", "10"))
DEFAULT_WRITE_TIMEOUT = float(os.environ.get("AUTOSIM_LLM_WRITE_TIMEOUT", "15"))
DEFAULT_POOL_TIMEOUT = float(os.environ.get("AUTOSIM_LLM_POOL_TIMEOUT", "15"))
DEFAULT_READ_TIMEOUT = float(os.environ.get("AUTOSIM_LLM_READ_TIMEOUT", "600"))


def _make_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=DEFAULT_CONNECT_TIMEOUT,
        write=DEFAULT_WRITE_TIMEOUT,
        pool=DEFAULT_POOL_TIMEOUT,
        read=DEFAULT_READ_TIMEOUT,
    )


def _make_http_client() -> httpx.Client:
    """Own httpx client with connection reuse DISABLED.

    The old run hung for hours on a half-dead keep-alive socket in CloseWait:
    the pool holds a connection the server silently dropped, and the next
    request parks on it forever. `Connection: close` means every call opens a
    fresh socket, so a dead connection can never be reused.
    """
    return httpx.Client(
        timeout=_make_timeout(),
        headers={"Connection": "close"},
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
        trust_env=False,
    )


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
                 timeout: float | None = None, max_retries: int = 3):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._owns_http = _make_http_client()
        self.client = OpenAI(
            api_key=api_key or _default_api_key(),
            base_url=base_url,
            timeout=_make_timeout() if timeout is None else timeout,
            max_retries=0,          # retries are handled in chat() below
            http_client=self._owns_http,
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        try:
            self._owns_http.close()
        except Exception:
            pass

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
                kwargs = dict(model=self.model, messages=messages)
                if self.model not in _NO_TEMPERATURE_MODELS:
                    kwargs["temperature"] = self.temperature
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                resp = self.client.chat.completions.create(**kwargs)
                self._accumulate(resp)
                msg = resp.choices[0].message
                # log reasoning_content when present (deepseek-reasoner)
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    print(f"[llm] <think>{reasoning[:200]}{'...' if len(reasoning) > 200 else ''}</think>",
                          flush=True)
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
            except _RETRYABLE as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    print(f"[llm] transient {type(exc).__name__}; retry "
                          f"{attempt + 1}/{self.max_retries - 1} in {backoff}s", flush=True)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(
                    f"LLM call failed after {self.max_retries} attempts "
                    f"(connection/timeout): {last_err}") from last_err
            except Exception as exc:  # noqa: BLE001 - e.g. 4xx APIStatusError
                raise RuntimeError(f"LLM call failed: {exc}") from exc

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
