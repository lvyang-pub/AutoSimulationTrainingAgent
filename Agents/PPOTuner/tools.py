"""Agent tool layer.

Tools are the agent's only way to affect the world. Every tool returns a
structured ToolResult rather than raising, so a failure becomes an *observation*
the agent can reason about (this is what the robustness metric measures).

`run_training` is the expensive tool; `evaluate_policy`/`get_metrics` read back a
run's artifacts. `finish` is terminal.

Env code is called through Harness/exec.py — this file never imports from the
Envs/ directory directly.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    status: str          # "ok" | "error" | "terminal"
    tool: str
    data: dict = field(default_factory=dict)
    message: str = ""
    error_type: str = ""

    def to_observation(self) -> str:
        payload = {"tool": self.tool, "status": self.status}
        if self.status == "ok":
            data = dict(self.data)
            data.pop("config_warnings", None)   # purely informational; not needed by LLM
            data.pop("eval_curve", None)         # trend + final already capture the signal
            payload["data"] = data
        else:
            payload["error_type"] = self.error_type
            payload["message"] = self.message
        return json.dumps(payload, separators=(",", ":"))


@dataclass
class RunRecord:
    run_id: str
    dir: str
    config: dict
    summary: dict


class ToolContext:
    """State shared by all tools during one agent session."""

    def __init__(self, runs_root: str, task_name: str, max_trials: int,
                 max_wall_seconds: float, default_timesteps: int,
                 env_path: str = "", harness_path: str = "",
                 fixed_reward: dict | None = None, visualize: bool = True):
        self.runs_root = runs_root
        self.task_name = task_name
        self.max_trials = max_trials
        self.max_wall_seconds = max_wall_seconds
        self.default_timesteps = default_timesteps
        # absolute path to the env root (e.g. Envs/Go2Locomotion-PPO)
        self.env_path = os.path.abspath(env_path) if env_path else ""
        # absolute path to Harness/ directory
        self.harness_path = os.path.abspath(harness_path) if harness_path else ""
        # When set, every run_training forces this reward, so returns are
        # comparable across trials (the task fixes the yardstick).
        self.fixed_reward = fixed_reward
        # Render a rollout video + curve after each trial (best-effort).
        self.visualize = visualize
        self.runs: dict[str, RunRecord] = {}
        self.trial_count = 0
        self.finished = False
        os.makedirs(runs_root, exist_ok=True)

    def next_run_id(self) -> str:
        self.trial_count += 1
        return f"trial_{self.trial_count:03d}"

    def harness_call(self, module_rel: str, func_name: str, **kwargs):
        """Call a function in env_path/module_rel via the shared Harness."""
        exec_path = os.path.join(self.harness_path, "exec.py")
        if not os.path.exists(exec_path):
            raise FileNotFoundError(f"Harness exec.py not found at {exec_path}")
        spec = importlib.util.spec_from_file_location("harness_exec", exec_path)
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        return harness.call(self.env_path, module_rel, func_name, **kwargs)


class Tools:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    # ------------------------------------------------------------- run_training
    def run_training(self, config: dict | None = None, timesteps: int | None = None,
                     retries: int = 1) -> ToolResult:
        if self.ctx.finished:
            return ToolResult("error", "run_training", error_type="session_finished",
                              message="the session is already finished")
        if self.ctx.max_trials is not None and self.ctx.trial_count >= self.ctx.max_trials:
            return ToolResult("error", "run_training", error_type="budget_exhausted",
                              message=f"trial budget of {self.ctx.max_trials} reached")
        config = dict(config or {})
        if timesteps is not None:
            config["timesteps"] = timesteps                    # honor explicit arg
        elif "timesteps" not in config:
            config["timesteps"] = self.ctx.default_timesteps
        # timesteps is a free hyperparameter: the agent may train longer (or
        # shorter) than the task's default/baseline budget. normalize() still
        # clamps it to the global SPEC range.
        # pin the task's reward so the agent's runs stay comparable; it may
        # still tune PPO + plant hyperparameters freely.
        reward_forced = self.ctx.fixed_reward is not None
        if reward_forced:
            env_over = dict(config.get("env") or {})
            env_over.pop("reward", None)
            env_over["reward"] = dict(self.ctx.fixed_reward)
            config["env"] = env_over

        run_id = self.ctx.next_run_id()
        run_dir = os.path.join(self.ctx.runs_root, run_id)
        last_exc = None
        for attempt in range(retries + 1):
            try:
                summary = self.ctx.harness_call(
                    "training/train_ppo.py", "train",
                    config=config, out_dir=run_dir, verbose=0,
                    visualize=self.ctx.visualize)
                norm_cfg, warnings = self.ctx.harness_call(
                    "training/config.py", "normalize", user_cfg=config)
                if reward_forced:
                    warnings = warnings + ["env.reward is fixed by the task; "
                                           "only hyperparameters were applied"]
                rec = RunRecord(run_id, run_dir, norm_cfg, summary)
                self.ctx.runs[run_id] = rec
                return ToolResult("ok", "run_training", data={
                    "run_id": run_id,
                    "mean_return": round(summary["mean_return"], 2),
                    "std_return": round(summary["std_return"], 2),
                    "fall_rate": summary["fall_rate"],
                    "mean_lin_vel_error": round(summary["mean_lin_vel_error"], 4),
                    "mean_episode_length": summary["mean_episode_length"],
                    "convergence_delta": summary["convergence_delta"],
                    "train_seconds": summary["train_seconds"],
                    "config_warnings": warnings,
                })
            except Exception as exc:  # noqa: BLE001 - retry then surface
                last_exc = exc
        return ToolResult("error", "run_training", error_type="training_failed",
                          message=f"{type(last_exc).__name__}: {last_exc}")

    # ---------------------------------------------------------- evaluate_policy
    def evaluate_policy(self, run_id: str, n_episodes: int = 10) -> ToolResult:
        rec = self.ctx.runs.get(run_id)
        if rec is None:
            return ToolResult("error", "evaluate_policy", error_type="unknown_run",
                              message=f"no such run_id '{run_id}'. valid: {list(self.ctx.runs)}")
        try:
            n_episodes = int(n_episodes)
            if not (1 <= n_episodes <= 50):
                n_episodes = max(1, min(50, n_episodes))
            policy_path = os.path.join(rec.dir, "policy.zip")
            if not os.path.exists(policy_path):
                return ToolResult("error", "evaluate_policy", error_type="missing_policy",
                                  message=f"policy.zip not found for {run_id}")
            stats = self.ctx.harness_call(
                "training/train_ppo.py", "evaluate_from_zip",
                policy_path=policy_path, env_cfg=rec.config["env"],
                n_episodes=n_episodes, seed=31337, policy_mode="greedy")
            return ToolResult("ok", "evaluate_policy", data={"run_id": run_id, **{
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()
            }})
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", "evaluate_policy", error_type="eval_failed",
                              message=f"{type(exc).__name__}: {exc}")

    # -------------------------------------------------------------- get_metrics
    def get_metrics(self, run_id: str) -> ToolResult:
        rec = self.ctx.runs.get(run_id)
        if rec is None:
            return ToolResult("error", "get_metrics", error_type="unknown_run",
                              message=f"no such run_id '{run_id}'")
        path = os.path.join(rec.dir, "metrics.csv")
        try:
            evals = []
            with open(path, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row["kind"] == "eval":
                        evals.append({"t": int(row["timesteps"]),
                                      "mean_return": float(row["mean_return"]),
                                      "fall_rate": float(row["fall_rate"]),
                                      "lin_vel_err": float(row["mean_lin_vel_error"])})
            trend = None
            if len(evals) >= 2:
                trend = round(evals[-1]["mean_return"] - evals[0]["mean_return"], 2)
            return ToolResult("ok", "get_metrics", data={
                "run_id": run_id,
                "eval_curve": evals,   # stripped from observation; kept for internal use
                "return_trend": trend,
                "final": evals[-1] if evals else None,
            })
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", "get_metrics", error_type="metrics_unreadable",
                              message=str(exc))

    # --------------------------------------------------------------- list_trials
    def list_trials(self) -> ToolResult:
        rows = [{"run_id": r.run_id,
                 "mean_return": round(r.summary["mean_return"], 2),
                 "fall_rate": r.summary["fall_rate"],
                 "mean_lin_vel_error": round(r.summary["mean_lin_vel_error"], 4)}
                for r in self.ctx.runs.values()]
        return ToolResult("ok", "list_trials", data={"trials": rows,
                                                     "n_trials": len(rows)})

    # ---------------------------------------------------------------- finish
    def finish(self, success: bool, best_run_id: str = "", reasoning: str = "") -> ToolResult:
        self.ctx.finished = True
        return ToolResult("terminal", "finish", data={
            "success": bool(success),
            "best_run_id": best_run_id,
            "reasoning": reasoning,
        })


# --------------------------------------------------------------------------- #
# tool schemas handed to the LLM
# --------------------------------------------------------------------------- #
def build_tool_schemas() -> list[dict]:
    cfg_note = ("Partial run config; unspecified fields keep their defaults. "
                "Keys: timesteps, seed, net_arch, ppo.{learning_rate,n_steps,"
                "batch_size,n_epochs,gamma,gae_lambda,clip_range,ent_coef,vf_coef,"
                "max_grad_norm}, env.{action_scale,kp,kd,max_torque,min_height,"
                "max_tilt,episode_length,command}. Note: env.reward is fixed by the "
                "task and your changes to it are ignored; timesteps IS tunable — you "
                "may train longer or shorter than the default budget.")
    return [
        {"type": "function", "function": {
            "name": "run_training",
            "description": "Train one PPO run with the given config and return its summary "
                           "metrics. This is the primary experiment action.",
            "parameters": {"type": "object", "properties": {
                "config": {"type": "object", "description": cfg_note},
                "timesteps": {"type": "integer", "description": "override training steps"},
            }, "required": ["config"]}}},
        {"type": "function", "function": {
            "name": "evaluate_policy",
            "description": "Run deterministic evaluation of a trained run over N episodes.",
            "parameters": {"type": "object", "properties": {
                "run_id": {"type": "string"},
                "n_episodes": {"type": "integer", "default": 10},
            }, "required": ["run_id"]}}},
        {"type": "function", "function": {
            "name": "get_metrics",
            "description": "Read the evaluation curve of a run to judge convergence.",
            "parameters": {"type": "object", "properties": {
                "run_id": {"type": "string"},
            }, "required": ["run_id"]}}},
        {"type": "function", "function": {
            "name": "list_trials",
            "description": "List all trials run so far in this session with their scores.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "finish",
            "description": "End the session. Call when the goal is met or the budget is spent.",
            "parameters": {"type": "object", "properties": {
                "success": {"type": "boolean"},
                "best_run_id": {"type": "string"},
                "reasoning": {"type": "string"},
            }, "required": ["success", "reasoning"]}}},
    ]


def dispatch(tools: Tools, name: str, args: dict) -> ToolResult:
    """Route a tool call, guarding against malformed arguments."""
    handlers = {
        "run_training": tools.run_training,
        "evaluate_policy": tools.evaluate_policy,
        "get_metrics": tools.get_metrics,
        "list_trials": tools.list_trials,
        "finish": tools.finish,
    }
    fn = handlers.get(name)
    if fn is None:
        return ToolResult("error", name, error_type="unknown_tool",
                          message=f"no tool named '{name}'")
    if not isinstance(args, dict):
        return ToolResult("error", name, error_type="bad_arguments",
                          message="arguments must be an object")
    try:
        return fn(**args)
    except TypeError as exc:
        return ToolResult("error", name, error_type="bad_arguments", message=str(exc))
