"""Offline smoke test: no API calls, no training. Verifies the refactored wiring.

Two parts:
  1. `assembly.compose` over the real env -- the env's action functions become
     tools, framework paths are hidden from their schemas, and `finish` is added.
  2. the agent loop with stub tools -- multi-call turns execute and the stopping
     rule is honoured.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PKG = os.path.join(_ROOT, "Agents", "StandardAgent-A")
ENV = os.path.join(_HERE, "Envs", "Go2Locomotion-PPO")


def load_pkg(alias="asta_agent"):
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(PKG, "__init__.py"),
        submodule_search_locations=[PKG])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def test_assembly(asta):
    sys.path.insert(0, _HERE)
    import assembly as assembler

    ctx = assembler.ExecutionContext(
        runs_root=os.path.join(_HERE, "Runs", "_smoke"),
        env_path=ENV, visualize=False, default_timesteps=1000)
    box = assembler.compose(ENV, ctx)

    names = box.names()
    print("assembled tools:", names)
    for expected in ("run_training", "evaluate_policy", "get_trial_summary",
                     "list_trials", "get_eval_curve", "get_params",
                     "set_params", "finish"):
        assert expected in names, f"missing tool: {expected}"

    # framework paths must NOT appear in any model-facing schema
    leaked = []
    for schema in box.schemas():
        props = schema["function"]["parameters"].get("properties", {})
        for hidden in ("out_dir", "runs_root", "policy_path"):
            if hidden in props:
                leaked.append((schema["function"]["name"], hidden))
    assert not leaked, f"injected params leaked into schema: {leaked}"

    # run_training keeps a user-facing `config` with the schedular's guidance
    rt = box.get("run_training")
    assert "config" in rt.parameters["properties"], "run_training lost `config`"
    assert "timesteps" in rt.parameters["properties"]["config"].get("description", "")

    # finish ends the loop via a terminal result
    res = box.dispatch("finish", {"success": True, "reasoning": "done"})
    assert res.terminal, "finish must be terminal"
    print("OK: assembly composed", len(names), "tools; paths hidden; finish terminal")


def test_loop(asta):
    box = asta.Toolbox()
    calls = {}

    def make(name):
        def fn(a):
            calls[name] = calls.get(name, 0) + 1
            if name == "run_training":
                tid = f"trial_{calls[name]:03d}"
                return asta.ToolResult("ok", name, data={
                    "run_id": tid, "summary": {"mean_lin_vel_error": 0.30,
                                               "fall_rate": 0.10}})
            return asta.ToolResult("ok", name, data={"ok": True})
        return fn

    for nm in ("run_training", "get_metrics", "list_trials"):
        box.register(asta.Tool(nm, f"{nm} tool", {"type": "object",
                      "properties": {}}, make(nm)))
    box.register(asta.Tool("finish", "finish", {"type": "object", "properties": {
        "success": {"type": "boolean"}, "reasoning": {"type": "string"}},
        "required": ["success", "reasoning"]},
        lambda a: asta.ToolResult("terminal", "finish",
                                  data={"success": a.get("success")})))

    class StubClient(asta.LLMClient):
        """One tool call per turn: train, inspect, train again -> stop rule fires."""

        def __init__(self):
            self.turn = 0
            self.script = [("run_training", {}),
                           ("get_metrics", {"run_id": "trial_001"}),
                           ("run_training", {})]

        def chat(self, messages, tools=None):
            if not tools:                      # reflection / summarization turn
                return asta.LLMResponse(content="ok", tool_calls=[],
                                        raw_message={"role": "assistant",
                                                     "content": "ok"})
            self.turn += 1
            name, args = self.script[min(self.turn - 1, len(self.script) - 1)]
            cid = f"c{self.turn}"
            import json
            raw = {"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}]}
            return asta.LLMResponse(content="",
                                    tool_calls=[asta.ToolCall(name, args, cid)],
                                    raw_message=raw)

        def usage_summary(self):
            return {}

        def close(self):
            pass

    sys.path.insert(0, _HERE)
    import task as task_def
    should_stop = task_def.make_should_stop(max_trials=2, max_wall_seconds=None)
    agent = asta.StandardAgent(toolbox=box, client=StubClient(),
                               should_stop=should_stop)
    agent.initial_prompt = "go"
    res = agent.run()

    print("=== loop smoke ===")
    print("stopped_reason:", res.stopped_reason, "| calls:", calls)
    assert calls.get("run_training") == 2, "expected 2 training runs"
    assert calls.get("get_metrics") == 1, "expected 1 metrics call"
    assert res.stopped_reason == "stop_condition", res.stopped_reason
    print("OK: multi-call executed, stop_condition honoured")


def main():
    os.makedirs(os.path.join(_HERE, "Runs", "_smoke"), exist_ok=True)
    asta = load_pkg()
    test_assembly(asta)
    test_loop(asta)


if __name__ == "__main__":
    main()
