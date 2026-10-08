"""Public facade for the LLM layer.

`LLMClient` is the single object the agent holds. It owns a provider backend,
accumulates session-wide token usage, and exposes one method -- `chat()` --
re-entering a provider and translating its reply into the neutral types.

Swapping DeepSeek for another provider is a change to `_make_backend` and the
matching module in this package; no agent code moves.
"""
from __future__ import annotations

import os

from .deepseek import DEFAULT_BASE_URL, DEFAULT_MODEL, DeepSeekClient
from .types import LLMResponse, ToolCall, Usage


def _make_backend(model: str, api_key: str | None, base_url: str,
                  temperature: float, max_retries: int):
    """Select the provider backend. Today: DeepSeek only."""
    return DeepSeekClient(model=model, api_key=api_key, base_url=base_url,
                          temperature=temperature, max_retries=max_retries)


class LLMClient:
    """Provider-neutral chat client with session-wide usage accounting."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, temperature: float = 0.3,
                 max_retries: int = 3):
        self.model = model or DEFAULT_MODEL
        self.usage = Usage()
        self._backend = _make_backend(
            model=self.model,
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            temperature=temperature,
            max_retries=max_retries,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """One assistant turn. May carry any number of tool calls."""
        resp = self._backend.chat(messages, tools)
        self.usage.add(resp)
        if resp.reasoning:
            preview = resp.reasoning[:200]
            more = "..." if len(resp.reasoning) > 200 else ""
            print(f"[llm] <think>{preview}{more}</think>", flush=True)
        return resp

    def close(self) -> None:
        self._backend.close()

    def usage_summary(self) -> dict:
        return self.usage.to_dict()


__all__ = ["LLMClient", "LLMResponse", "ToolCall", "Usage"]
