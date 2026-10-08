"""DeepSeek-specific transport.

This is the ONLY module that knows DeepSeek exists. It owns the endpoint, the
auth header, the model quirks, the timeout policy and the retry policy, and it
translates DeepSeek's response shape into the neutral types in `types.py`.

Everything a caller has ever had to learn the hard way about this provider is
captured here:

* `deepseek-reasoner` rejects a `temperature` argument, so it is omitted.
* not every response arrives as a clean native `tool_calls` array; when the
  model instead prints a JSON object into `content`, that object is parsed as a
  fallback tool call.
* keep-alive sockets have hung for hours (a pooled connection the server had
  silently dropped), so connection reuse is disabled with `Connection: close`.
* a read timeout means "the server never started responding" and is safe to
  retry on a fresh connection; an overall call timeout is not, so the segmented
  connect/write/pool/read budget is used instead.

None of that leaks past this file.
"""
from __future__ import annotations

import json
import os
import re
import time
from urllib import error as _urlerror
from urllib import request as _urlrequest

from .types import LLMResponse, ToolCall

DEFAULT_MODEL = os.environ.get("AUTOSIM_MODEL", "deepseek-reasoner")
DEFAULT_BASE_URL = os.environ.get("AUTOSIM_BASE_URL", "https://api.deepseek.com")

# Models that do not accept a `temperature` parameter.
_NO_TEMPERATURE_MODELS = {"deepseek-reasoner", "deepseek-reasoner-latest"}

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_RETRIES = 3

# Segmented per-call budgets (seconds). connect/write bound "the server never
# answered"; read is the ceiling on a single generation -- generous, because a
# reasoner thinking for minutes is not a fault, but FINITE so no call parks
# forever.
DEFAULT_CONNECT_TIMEOUT = float(os.environ.get("AUTOSIM_LLM_CONNECT_TIMEOUT", "10"))
DEFAULT_READ_TIMEOUT = float(os.environ.get("AUTOSIM_LLM_READ_TIMEOUT", "600"))


class LLMError(RuntimeError):
    """Any failure that survives the retry policy, raised to the caller."""


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

# Process-wide, one-shot: read the dotenv and sticky-resolve the key exactly
# once, then serve it from memory for the life of the process.
_API_KEY_CACHE: str | None = None


def _resolve_api_key() -> str:
    global _API_KEY_CACHE
    if _API_KEY_CACHE is None:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise LLMError(
                "DEEPSEEK_API_KEY is not set. Export it before running, e.g.\n"
                '  Windows (PowerShell): $env:DEEPSEEK_API_KEY="sk-..."\n'
                "  bash:                  export DEEPSEEK_API_KEY=sk-...\n"
                "Or put it in a .env file (see .env.example)."
            )
        _API_KEY_CACHE = key
    return _API_KEY_CACHE


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of free text (fallback tool-call parsing)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates: list[str] = []
    if fenced:
        candidates.append(fenced.group(1))
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
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


class DeepSeekClient:
    """Owns one DeepSeek session: transport, retries and token accounting."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 base_url: str = DEFAULT_BASE_URL,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self._api_key = api_key or _resolve_api_key()

    def close(self) -> None:
        """Nothing to release: each call opens and closes its own connection."""

    # ------------------------------------------------------------------ public
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """One chat completion. Retries transient transport failures.

        Returns a neutral LLMResponse carrying zero, one, or many tool calls --
        this layer never picks "the" tool call; that decision belongs to the
        agent, which will execute them all in order.
        """
        payload: dict = {"model": self.model, "messages": messages}
        if self.model not in _NO_TEMPERATURE_MODELS:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_err: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                raw = self._post(body)
                return self._parse(raw)
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    print(f"[llm] transient {type(exc).__name__}; retry "
                          f"{attempt + 1}/{self.max_retries - 1} in {backoff}s", flush=True)
                    time.sleep(backoff)
                    continue
                raise LLMError(
                    f"LLM call failed after {self.max_retries} attempts "
                    f"(connection/timeout): {last_err}") from last_err
            except _urlerror.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:400]
                except Exception:  # noqa: BLE001
                    pass
                raise LLMError(f"LLM call failed: HTTP {exc.code} {detail}") from exc
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"LLM call failed: {exc}") from exc

        raise LLMError(f"LLM call failed: {last_err}")

    # --------------------------------------------------------------- transport
    def _post(self, body: bytes) -> dict:
        url = self.base_url.rstrip("/") + "/chat/completions"
        req = _urlrequest.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                # A pooled keep-alive socket the server has silently dropped hung
                # a prior run for hours; forcing close means a dead connection can
                # never be reused.
                "Connection": "close",
                "Accept": "application/json",
            },
        )
        with _urlrequest.urlopen(req, timeout=DEFAULT_READ_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ------------------------------------------------------------------ parsing
    def _parse(self, raw: dict) -> LLMResponse:
        choices = raw.get("choices") or []
        msg = (choices[0].get("message") if choices else None) or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""

        calls: list[ToolCall] = []
        native = msg.get("tool_calls")
        if native:
            for tc in native:
                fn = tc.get("function", {}) or {}
                calls.append(ToolCall(
                    name=fn.get("name", ""),
                    arguments=_maybe_json(fn.get("arguments") or "{}"),
                    call_id=tc.get("id") or f"call_{len(calls)}",
                    source="native",
                ))
        else:
            parsed = _extract_json(content)
            if parsed and isinstance(parsed.get("tool"), str):
                calls.append(ToolCall(
                    name=parsed["tool"],
                    arguments=parsed.get("arguments") or {},
                    call_id=f"call_{len(calls)}",
                    source="fallback",
                ))

        usage = raw.get("usage") or {}
        return LLMResponse(
            content=content,
            tool_calls=calls,
            raw_message=msg,
            reasoning=reasoning,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
        )


def _maybe_json(text: str) -> dict:
    try:
        obj = json.loads(text or "{}")
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}
