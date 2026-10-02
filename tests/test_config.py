"""Unit tests for config normalization — the agent's guardrail against bad LLM output."""
from __future__ import annotations

import pytest

from src.training.config import DEFAULT_CONFIG, normalize


def test_empty_config_returns_defaults():
    cfg, warnings = normalize(None)
    assert cfg == DEFAULT_CONFIG
    assert warnings == []


def test_partial_merge_keeps_other_defaults():
    cfg, _ = normalize({"ppo": {"learning_rate": 1e-3}})
    assert cfg["ppo"]["learning_rate"] == 1e-3
    assert cfg["ppo"]["n_steps"] == DEFAULT_CONFIG["ppo"]["n_steps"]


def test_out_of_range_is_clamped_with_warning():
    cfg, warnings = normalize({"ppo": {"learning_rate": 5.0}})
    assert cfg["ppo"]["learning_rate"] == 1e-2
    assert any("learning_rate" in w for w in warnings)


def test_unknown_key_is_ignored_with_warning():
    cfg, warnings = normalize({"ppo": {"not_a_real_key": 1}})
    assert "not_a_real_key" not in cfg["ppo"]
    assert any("unknown key" in w for w in warnings)


def test_batch_larger_than_nsteps_is_repaired():
    cfg, warnings = normalize({"ppo": {"n_steps": 128, "batch_size": 512}})
    assert cfg["ppo"]["n_steps"] >= cfg["ppo"]["batch_size"]
    assert any("batch_size" in w for w in warnings)


def test_reward_weights_merge_and_clamp():
    cfg, warnings = normalize({"env": {"reward": {"track_lin_vel": 999.0, "alive": 0.0}}})
    assert cfg["env"]["reward"]["track_lin_vel"] == 5.0     # clamped to SPEC high
    assert cfg["env"]["reward"]["alive"] == 0.0
    assert cfg["env"]["reward"]["orientation"] == DEFAULT_CONFIG["env"]["reward"]["orientation"]
    assert any("track_lin_vel" in w for w in warnings)


def test_bad_net_arch_falls_back():
    cfg, warnings = normalize({"net_arch": "two layers"})
    assert cfg["net_arch"] == DEFAULT_CONFIG["net_arch"]
    assert any("net_arch" in w for w in warnings)


def test_non_dict_section_raises():
    with pytest.raises(ValueError):
        normalize({"ppo": [1, 2, 3]})


def test_int_leaves_stay_int():
    cfg, _ = normalize({"ppo": {"n_epochs": 7.6}, "env": {"frame_skip": 3.2}})
    assert isinstance(cfg["ppo"]["n_epochs"], int)
    assert isinstance(cfg["env"]["frame_skip"], int)
