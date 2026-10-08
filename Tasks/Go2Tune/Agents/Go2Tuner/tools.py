"""Generic tool layer.

A `Toolbox` is a registry of named tools. The agent does not know what any of
them do -- it lists their schemas to the model, dispatches the model's calls by
name, and appends whatever text comes back as an observation. The caller
populates the toolbox before the run, so no task-specific knowledge lives here.

A tool signals "the loop is done" by returning a result whose status is
`"terminal"`; the agent stops after that call. The canonical example is a
`finish` tool. Nothing else about a result is special: the agent never branches
on a tool's name, its data, or its meaning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

ToolHandler = Callable[[dict], "ToolResult"]


@dataclass
class ToolResult:
    """A tool's outcome, rendered into an observation for the model."""
    status: str                      # "ok" | "error" | "terminal"
    tool: str
    data: dict = field(default_factory=dict)
    message: str = ""
    error_type: str = ""

    @property
    def terminal(self) -> bool:
        return self.status == "terminal"

    def to_observation(self) -> str:
        if self.status in ("ok", "terminal"):
            body = self.message or ("finished" if self.terminal else "ok")
            if self.data:
                return f"{body}\n{json.dumps(self.data, ensure_ascii=False, indent=2)}"
            return body
        return f"ERROR ({self.error_type or 'tool_error'}): {self.message}"


@dataclass
class Tool:
    """A named capability: its model-facing schema and its handler."""
    name: str
    description: str
    parameters: dict
    handler: ToolHandler

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class Toolbox:
    """A registry of tools, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._order: list[str] = []

    def register(self, tool: Tool) -> None:
        if tool.name not in self._tools:
            self._order.append(tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._order)

    def schemas(self) -> list[dict]:
        return [self._tools[n].schema() for n in self._order]

    def dispatch(self, name: str, arguments: dict) -> ToolResult:
        """Run one tool by name. Never raises: failures become error results."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult("error", name, error_type="unknown_tool",
                              message=f"no tool named {name!r}; available: {self.names()}")
        try:
            result = tool.handler(arguments or {})
        except Exception as exc:  # noqa: BLE001 - tool faults are observations
            return ToolResult("error", name, error_type=type(exc).__name__,
                              message=str(exc))
        if not isinstance(result, ToolResult):
            result = ToolResult("ok", name, data={"result": result})
        return result
