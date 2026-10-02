"""End-to-end test of the ReAct harness with a scripted, offline LLM.

This exercises the whole agent wiring — loop, tool dispatch, objective scoring,
reflection, and long-term memory — without any network calls or real training.
"""
from __future__ import annotations

import json

import pytest

import src.agent.agent as agent_mod
import src.agent.reflection as reflection_mod
from src.agent.agent import TuningAgent
from src.agent.llm import LLMResponse, ToolCall
from src.agent.memory import Lesson


class ScriptedLLM:
    """Returns a fixed script of tool calls, then a finish; or reflection text."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        if tools is None:                     # reflection / final-report call
            return LLMResponse(content="Diagnosis: raise track_lin_vel. "
                                       "Recommendation: increase the weight.")
        if not self.script:
            return LLMResponse(content="(done)", tool_calls=[])
        name, args = self.script.pop(0)
        return LLMResponse(content="", tool_calls=[
            ToolCall(name=name, arguments=args, source="native", call_id="c1")])

    def usage_summary(self):
        return {"n_calls": len(self.calls), "prompt_tokens": 1,
                "completion_tokens": 1, "total_tokens": 2}


@pytest.fixture
def stub_train(monkeypatch):
    """Replace training with a cheap fake that returns a controllable summary."""
    box = {"summary": None, "configs": []}

    def fake_train(config, out_dir, verbose=0, **kw):
        import os
        os.makedirs(out_dir, exist_ok=True)
        box["configs"].append(config)
        s = box["summary"] or {
            "mean_return": 10.0, "std_return": 1.0, "fall_rate": 1.0,
            "mean_lin_vel_error": 1.0, "mean_episode_length": 500.0,
            "convergence_delta": None, "train_seconds": 0.0,
        }
        return dict(s)

    import src.agent.tools as tools_mod
    monkeypatch.setattr(tools_mod, "train", fake_train)
    return box


def _agent(tmp_path, llm, **kw):
    return TuningAgent(
        goals={"mean_lin_vel_error_max": 0.3, "fall_rate_max": 0.2},
        runs_root=str(tmp_path / "runs"), task_name="t",
        longterm_path=str(tmp_path / "lt.json"), max_trials=3,
        max_llm_steps=10, client=llm, **kw)


def test_agent_runs_trial_then_finishes(tmp_path, stub_train):
    stub_train["summary"] = {"mean_return": 300.0, "std_return": 1.0, "fall_rate": 0.0,
                             "mean_lin_vel_error": 0.1, "mean_episode_length": 500.0,
                             "convergence_delta": None, "train_seconds": 0.0}
    llm = ScriptedLLM([
        ("run_training", {"config": {"ppo": {"learning_rate": 1e-3}}}),
        ("finish", {"success": True, "reasoning": "goal met"}),
    ])
    res = _agent(tmp_path, llm).run()
    assert res.n_trials == 1
    assert res.success is True                 # objective goal met
    assert res.llm_finished_success is True
    assert res.stopped_reason == "finish"
    assert res.best_run_id == "trial_001"


def test_agent_writes_lesson_to_longterm_memory(tmp_path, stub_train):
    stub_train["summary"] = {"mean_return": 5.0, "std_return": 1.0, "fall_rate": 1.0,
                             "mean_lin_vel_error": 1.0, "mean_episode_length": 500.0,
                             "convergence_delta": None, "train_seconds": 0.0}
    llm = ScriptedLLM([("run_training", {"config": {}}),
                       ("finish", {"success": False, "reasoning": "budget"})])
    _agent(tmp_path, llm).run()
    from src.agent.memory import LongTermMemory
    lt = LongTermMemory(str(tmp_path / "lt.json"))
    assert len(lt.lessons) >= 1
    assert lt.lessons[0].task == "t"


def test_agent_reports_failure_when_goal_unmet(tmp_path, stub_train):
    llm = ScriptedLLM([("run_training", {"config": {}}),
                       ("finish", {"success": True, "reasoning": "I think it is good"})])
    res = _agent(tmp_path, llm).run()          # default summary fails the goal
    assert res.success is False                # objective overrides self-report
    assert res.llm_finished_success is True


def test_agent_survives_injected_fault(tmp_path, stub_train):
    llm = ScriptedLLM([
        ("run_training", {"config": {}}),      # this one is fault-injected
        ("run_training", {"config": {}}),      # agent recovers and retries
        ("finish", {"success": False, "reasoning": "done"}),
    ])
    agent = _agent(tmp_path, llm)
    agent.fault_after = 1
    res = agent.run()
    assert res.tool_stats["errors"] >= 1       # the fault was observed as an error
    assert res.n_trials == 1                    # only the successful run counted


def test_agent_hits_trial_budget(tmp_path, stub_train):
    llm = ScriptedLLM([("run_training", {"config": {}}) for _ in range(10)])
    res = _agent(tmp_path, llm).run()
    assert res.n_trials == 3                    # capped at max_trials
    assert res.stopped_reason in ("trial_budget", "max_steps")


def test_command_and_start_config_reach_prompt(tmp_path, stub_train):
    llm = ScriptedLLM([("finish", {"success": False, "reasoning": "x"})])
    agent = _agent(tmp_path, llm, command=[0.6, 0.0, 0.0],
                   start_config={"env": {"kp": 30.0}})
    agent.run()
    first_user = llm.calls[0][1]["content"]
    assert "0.60 m/s" in first_user
    assert '"kp":30.0' in first_user.replace(" ", "")


def test_goal_met_signal_appended_to_observation(tmp_path, stub_train):
    """After a goal-meeting run, the agent sees an objective 'stop' nudge."""
    stub_train["summary"] = {"mean_return": 300.0, "std_return": 1.0, "fall_rate": 0.0,
                             "mean_lin_vel_error": 0.1, "mean_episode_length": 500.0,
                             "convergence_delta": None, "train_seconds": 0.0}
    llm = ScriptedLLM([
        ("run_training", {"config": {}}),
        ("finish", {"success": True, "reasoning": "goal met"}),
    ])
    _agent(tmp_path, llm).run()
    # the observation nudge must appear in some agent-loop call (not reflection)
    all_tool = [m for call in llm.calls for m in call if m.get("role") == "tool"]
    assert any("goal is now MET" in m["content"] for m in all_tool)


def test_no_signal_when_goal_unmet(tmp_path, stub_train):
    llm = ScriptedLLM([
        ("run_training", {"config": {}}),
        ("finish", {"success": False, "reasoning": "no"}),
    ])
    _agent(tmp_path, llm).run()                # default summary fails the goal
    all_tool = [m for call in llm.calls for m in call if m.get("role") == "tool"]
    assert not any("goal is now MET" in m["content"] for m in all_tool)
