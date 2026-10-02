"""Unit tests for the LLM JSON fallback parser.

deepseek-flash does not always emit native tool_calls, so the fallback parse of
a JSON object out of free text is a load-bearing path — it is what keeps the
ReAct loop alive when the model replies in prose.
"""
from __future__ import annotations

from src.agent.llm import LLMClient


def test_bare_json_object():
    got = LLMClient._extract_json('{"tool": "list_trials", "arguments": {}}')
    assert got["tool"] == "list_trials"


def test_json_inside_prose():
    text = 'Sure. I will inspect first.\n{"tool":"get_metrics","arguments":{"run_id":"r1"}}\nDone.'
    got = LLMClient._extract_json(text)
    assert got["arguments"]["run_id"] == "r1"


def test_json_in_code_fence():
    text = '```json\n{"tool": "finish", "arguments": {"success": true}}\n```'
    got = LLMClient._extract_json(text)
    assert got["tool"] == "finish" and got["arguments"]["success"] is True


def test_nested_braces_balanced():
    text = '{"tool":"run_training","arguments":{"config":{"ppo":{"n_steps":512}}}}'
    got = LLMClient._extract_json(text)
    assert got["arguments"]["config"]["ppo"]["n_steps"] == 512


def test_no_json_returns_none():
    assert LLMClient._extract_json("I think we should train a policy.") is None


def test_empty_returns_none():
    assert LLMClient._extract_json("") is None


def test_malformed_object_returns_none():
    assert LLMClient._extract_json('{"tool": "x", }') is None
