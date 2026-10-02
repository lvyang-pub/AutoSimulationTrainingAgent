"""Unit tests for the agent's objective success decision (goal_met).

This is deliberately decoupled from the LLM's self-report: the evaluation only
trusts the objective check, so it must be correct in isolation.
"""
from __future__ import annotations

from src.agent.agent import goal_met

GOALS = {"mean_lin_vel_error_max": 0.3, "fall_rate_max": 0.2, "mean_return_min": 100.0}


def test_success_when_all_criteria_met():
    s = {"mean_lin_vel_error": 0.1, "fall_rate": 0.0, "mean_return": 200.0}
    assert goal_met(s, GOALS) is True


def test_fail_on_velocity_error():
    s = {"mean_lin_vel_error": 0.9, "fall_rate": 0.0, "mean_return": 500.0}
    assert goal_met(s, GOALS) is False


def test_fail_on_fall_rate():
    s = {"mean_lin_vel_error": 0.1, "fall_rate": 0.5, "mean_return": 500.0}
    assert goal_met(s, GOALS) is False


def test_fail_on_min_return():
    s = {"mean_lin_vel_error": 0.1, "fall_rate": 0.0, "mean_return": 50.0}
    assert goal_met(s, GOALS) is False


def test_empty_summary_is_failure():
    assert goal_met({}, GOALS) is False


def test_missing_return_goal_is_optional():
    s = {"mean_lin_vel_error": 0.1, "fall_rate": 0.0}
    assert goal_met(s, {"mean_lin_vel_error_max": 0.3, "fall_rate_max": 0.2}) is True


def test_return_only_goal_is_supported():
    # the calibrated tuning tasks grade on mean_return alone
    goals = {"mean_return_min": 190.0}
    assert goal_met({"mean_return": 195.0}, goals) is True
    assert goal_met({"mean_return": 180.0}, goals) is False


def test_return_and_fall_goal_together():
    goals = {"mean_return_min": 195.0, "fall_rate_max": 0.1}
    assert goal_met({"mean_return": 196.0, "fall_rate": 0.0}, goals) is True
    assert goal_met({"mean_return": 196.0, "fall_rate": 0.5}, goals) is False
