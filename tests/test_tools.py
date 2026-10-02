"""Unit tests for the tool layer's dispatch and error-as-observation contract.

These use the cheap tools (list_trials / get_metrics / finish) plus a stubbed
run_training so no real PPO training happens.
"""
from __future__ import annotations

import json

import src.agent.tools as tools_mod
from src.agent.tools import ToolContext, ToolResult, Tools, build_tool_schemas, dispatch


def _ctx(tmp_path, max_trials=3):
    return ToolContext(str(tmp_path / "runs"), "task", max_trials, 600.0, 1000)


def test_unknown_tool_is_error_not_exception(tmp_path):
    t = Tools(_ctx(tmp_path))
    r = dispatch(t, "does_not_exist", {})
    assert r.status == "error" and r.error_type == "unknown_tool"


def test_bad_arguments_are_reported(tmp_path):
    t = Tools(_ctx(tmp_path))
    r = dispatch(t, "evaluate_policy", {"wrong_arg": 1})
    assert r.status == "error" and r.error_type == "bad_arguments"


def test_non_dict_arguments_rejected(tmp_path):
    t = Tools(_ctx(tmp_path))
    r = dispatch(t, "list_trials", ["nope"])
    assert r.status == "error" and r.error_type == "bad_arguments"


def test_list_trials_empty(tmp_path):
    r = Tools(_ctx(tmp_path)).list_trials()
    assert r.status == "ok" and r.data["n_trials"] == 0


def test_unknown_run_id_is_actionable(tmp_path):
    r = Tools(_ctx(tmp_path)).get_metrics("trial_999")
    assert r.status == "error" and r.error_type == "unknown_run"
    assert "trial_999" in r.message


def test_finish_is_terminal(tmp_path):
    t = Tools(_ctx(tmp_path))
    r = t.finish(success=True, reasoning="done")
    assert r.status == "terminal" and r.data["success"] is True
    assert t.ctx.finished is True


def test_run_training_blocked_after_finish(tmp_path):
    t = Tools(_ctx(tmp_path))
    t.finish(success=False, reasoning="stop")
    r = t.run_training(config={})
    assert r.status == "error" and r.error_type == "session_finished"


def test_observation_is_valid_json(tmp_path):
    r = Tools(_ctx(tmp_path)).list_trials()
    parsed = json.loads(r.to_observation())
    assert parsed["tool"] == "list_trials" and parsed["status"] == "ok"


def test_error_observation_carries_error_type(tmp_path):
    r = Tools(_ctx(tmp_path)).get_metrics("nope")
    parsed = json.loads(r.to_observation())
    assert parsed["status"] == "error" and parsed["error_type"] == "unknown_run"


def test_budget_exhausted_blocks_training(tmp_path):
    ctx = _ctx(tmp_path, max_trials=1)
    ctx.trial_count = 1                      # pretend one trial already ran
    r = Tools(ctx).run_training(config={})
    assert r.status == "error" and r.error_type == "budget_exhausted"


def test_run_training_records_and_reports(tmp_path, monkeypatch):
    """run_training should normalize, call train, and surface a summary."""
    fake_summary = {
        "mean_return": 12.34, "std_return": 1.0, "fall_rate": 0.0,
        "mean_lin_vel_error": 0.11, "mean_episode_length": 500.0,
        "convergence_delta": 2.0, "train_seconds": 0.01,
    }

    def fake_train(config, out_dir, verbose=0, visualize=True):
        import os
        os.makedirs(out_dir, exist_ok=True)
        return dict(fake_summary)

    monkeypatch.setattr(tools_mod, "train", fake_train)
    t = Tools(_ctx(tmp_path))
    r = t.run_training(config={"ppo": {"learning_rate": 1e-3}})
    assert r.status == "ok"
    assert r.data["run_id"] == "trial_001"
    assert r.data["mean_return"] == 12.34
    assert "trial_001" in t.ctx.runs


def test_fixed_reward_overrides_agent_supplied_reward(tmp_path, monkeypatch):
    """When the task fixes the reward, the agent cannot change it."""
    fixed = {"alive": 0.2, "track_lin_vel": 2.0}
    seen = {}

    def fake_train(config, out_dir, verbose=0, visualize=True):
        import os
        os.makedirs(out_dir, exist_ok=True)
        seen["reward"] = config["env"]["reward"]
        return {"mean_return": 1.0, "std_return": 0.0, "fall_rate": 0.0,
                "mean_lin_vel_error": 1.0, "mean_episode_length": 500.0,
                "convergence_delta": 0.0, "train_seconds": 0.0}

    monkeypatch.setattr(tools_mod, "train", fake_train)
    ctx = ToolContext(str(tmp_path / "runs"), "task", 3, 600.0, 1000, fixed_reward=fixed)
    t = Tools(ctx)
    r = t.run_training(config={"env": {"reward": {"alive": 999.0}}})
    assert r.status == "ok"
    assert seen["reward"] == fixed              # agent's 999 ignored
    assert any("fixed" in w for w in r.data["config_warnings"])


def test_timesteps_clamped_to_task_budget(tmp_path, monkeypatch):
    """The agent cannot out-train the baselines by asking for more steps."""
    seen = {}

    def fake_train(config, out_dir, verbose=0, visualize=True):
        import os
        os.makedirs(out_dir, exist_ok=True)
        seen["timesteps"] = config["timesteps"]
        return {"mean_return": 1.0, "std_return": 0.0, "fall_rate": 0.0,
                "mean_lin_vel_error": 1.0, "mean_episode_length": 500.0,
                "convergence_delta": 0.0, "train_seconds": 0.0}

    monkeypatch.setattr(tools_mod, "train", fake_train)
    t = Tools(_ctx(tmp_path))                 # _ctx sets default_timesteps=1000
    t.run_training(config={"timesteps": 999999})
    assert seen["timesteps"] == 1000


def test_smaller_timesteps_allowed(tmp_path, monkeypatch):
    """A cheaper probe is fine — only the ceiling is enforced."""
    seen = {}
    def fake_train(config, out_dir, verbose=0, visualize=True):
        import os
        os.makedirs(out_dir, exist_ok=True)
        seen["timesteps"] = config["timesteps"]
        return {"mean_return": 1.0, "std_return": 0.0, "fall_rate": 0.0,
                "mean_lin_vel_error": 1.0, "mean_episode_length": 500.0,
                "convergence_delta": 0.0, "train_seconds": 0.0}
    monkeypatch.setattr(tools_mod, "train", fake_train)
    Tools(_ctx(tmp_path)).run_training(config={"timesteps": 500})
    assert seen["timesteps"] == 500


def test_no_fixed_reward_leaves_agent_reward_alone(tmp_path, monkeypatch):
    """Without pinning, an agent-supplied reward is honored (used by the demo)."""
    seen = {}

    def fake_train(config, out_dir, verbose=0, visualize=True):
        import os
        os.makedirs(out_dir, exist_ok=True)
        seen["reward"] = config["env"]["reward"]
        return {"mean_return": 1.0, "std_return": 0.0, "fall_rate": 0.0,
                "mean_lin_vel_error": 1.0, "mean_episode_length": 500.0,
                "convergence_delta": 0.0, "train_seconds": 0.0}

    monkeypatch.setattr(tools_mod, "train", fake_train)
    t = Tools(_ctx(tmp_path))
    t.run_training(config={"env": {"reward": {"alive": 7.0}}})
    assert seen["reward"]["alive"] == 7.0


def test_visualize_flag_threads_to_train(tmp_path, monkeypatch):
    """run_training must pass the context's visualize switch to train()."""
    seen = {}

    def fake_train(config, out_dir, verbose=0, visualize=True):
        import os
        os.makedirs(out_dir, exist_ok=True)
        seen["visualize"] = visualize
        return {"mean_return": 1.0, "std_return": 0.0, "fall_rate": 0.0,
                "mean_lin_vel_error": 1.0, "mean_episode_length": 500.0,
                "convergence_delta": 0.0, "train_seconds": 0.0}

    monkeypatch.setattr(tools_mod, "train", fake_train)
    ctx = ToolContext(str(tmp_path / "runs"), "task", 3, 600.0, 1000,
                      visualize=False)
    Tools(ctx).run_training(config={})
    assert seen["visualize"] is False


def test_visualize_defaults_on(tmp_path, monkeypatch):
    seen = {}

    def fake_train(config, out_dir, verbose=0, visualize=True):
        import os
        os.makedirs(out_dir, exist_ok=True)
        seen["visualize"] = visualize
        return {"mean_return": 1.0, "std_return": 0.0, "fall_rate": 0.0,
                "mean_lin_vel_error": 1.0, "mean_episode_length": 500.0,
                "convergence_delta": 0.0, "train_seconds": 0.0}

    monkeypatch.setattr(tools_mod, "train", fake_train)
    Tools(_ctx(tmp_path)).run_training(config={})
    assert seen["visualize"] is True


def test_schemas_shape(tmp_path):
    schemas = build_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"run_training", "evaluate_policy", "get_metrics",
                     "list_trials", "finish"}
    for s in schemas:
        assert s["type"] == "function"
        assert "parameters" in s["function"]
