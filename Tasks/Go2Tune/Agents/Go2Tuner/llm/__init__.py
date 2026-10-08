"""LLM layer: all provider-specific code, behind a small neutral interface.

The agent imports only `LLMClient` (and, when it needs the shapes, the types).
Nothing about DeepSeek -- its endpoint, its model quirks, its timeout policy --
is visible above this package.
"""
from .client import LLMClient
from .types import LLMResponse, ToolCall, Usage

__all__ = ["LLMClient", "LLMResponse", "ToolCall", "Usage"]
