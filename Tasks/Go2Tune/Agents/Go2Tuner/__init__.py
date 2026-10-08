"""StandardAgent-A: a full-featured ReAct agent, configured entirely from data.

The base ships every capability an experiment agent needs -- the loop, bounded
context, memory, reflection, objective checks, budgets, fault injection -- so a
task copy is produced by changing *only* `prompts.py`.

The agent does not load environment functions; building its Toolbox from an
env's `action/` surface is the schedular's job (see the task's `assembly.py`).
Here it exposes only the tool-injection API the schedular calls.

Public surface:
    StandardAgent, AgentResult, RunState        -- the agent and its results
    Tool, Toolbox, ToolResult                   -- the tool-injection API
    LLMClient, LLMResponse, ToolCall            -- the LLM facade
    TrialRecord, Lesson, ShortTermMemory, LongTermMemory -- memory
    reflect_on_trial                            -- post-trial reflection
"""
from .llm import LLMClient, LLMResponse, ToolCall
from .main import AgentResult, RunState, StandardAgent
from .memory import Lesson, LongTermMemory, ShortTermMemory, TrialRecord
from .reflection import reflect_on_trial
from .tools import Tool, Toolbox, ToolResult

__all__ = [
    "StandardAgent", "AgentResult", "RunState",
    "Toolbox", "Tool", "ToolResult",
    "LLMClient", "LLMResponse", "ToolCall",
    "TrialRecord", "Lesson", "ShortTermMemory", "LongTermMemory",
    "reflect_on_trial",
]
