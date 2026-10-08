"""Provider-neutral data types for the LLM layer.

These are the only shapes the agent ever touches. Every provider-specific
detail -- endpoints, headers, how a tool call is encoded on the wire -- stays
behind the facade in this package, so swapping providers never reaches the
agent's code.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """One function call requested by the model in a single assistant turn."""
    name: str
    arguments: dict
    call_id: str
    source: str = "native"      # "native" | "fallback"


@dataclass
class LLMResponse:
    """A single assistant turn: free text, plus any tool calls to execute."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict | None = None
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Usage:
    """Session-wide token accounting, accumulated across every call."""
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, resp: LLMResponse) -> None:
        self.n_calls += 1
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
