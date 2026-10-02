"""The tuning agent: a ReAct harness over the tool layer.

Loop shape (ReAct):
    Thought + Action  -> one tool call chosen by the LLM
    Observation       -> the tool's structured result, appended to context
After a completed training trial, a reflection step diagnoses it and writes a
lesson to long-term memory. The loop stops on `finish`, on budget exhaustion, or
on repeated LLM failure.

Success is judged two ways: the LLM's own `finish(success=...)`, and an objective
`goal_met` computed from the best trial's metrics. The eval suite uses the latter
so scoring never depends on the model's self-assessment.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from .llm import LLMClient
from .memory import LongTermMemory, ShortTermMemory, TrialRecord
from .prompts import FINAL_PROMPT, build_system_prompt
from .reflection import reflect_on_trial
from .tools import ToolContext, ToolResult, Tools, build_tool_schemas, dispatch


@dataclass
class AgentResult:
    success: bool                 # objective: goal met by best trial
    llm_finished_success: bool    # what the model claimed
    best_run_id: str
    best_metrics: dict
    n_trials: int
    trials: list[dict]
    tool_stats: dict
    usage: dict
    wall_seconds: float
    final_report: str
    stopped_reason: str

    def to_dict(self) -> dict:
        return self.__dict__


def goal_met(summary: dict, goals: dict) -> bool:
    """Objective success check, independent of the LLM.

    Every criterion present in `goals` must hold; absent criteria are ignored,
    so a task can grade on return + fall rate without a velocity-error term.
    """
    if not summary:
        return False
    checks: list[bool] = []
    if "mean_lin_vel_error_max" in goals:
        e = summary.get("mean_lin_vel_error")
        checks.append(e is not None and e <= goals["mean_lin_vel_error_max"])
    if "fall_rate_max" in goals:
        f = summary.get("fall_rate")
        checks.append(f is not None and f <= goals["fall_rate_max"])
    if "mean_return_min" in goals:
        r = summary.get("mean_return")
        checks.append(r is not None and r >= goals["mean_return_min"])
    return bool(checks) and all(checks)


class TuningAgent:
    def __init__(self, goals: dict, runs_root: str, task_name: str,
                 longterm_path: str, default_timesteps: int = 20000,
                 max_trials: int = 6, max_wall_seconds: float = 1800,
                 max_llm_steps: int = 24, client: LLMClient | None = None,
                 system_prompt: str | None = None, extra_hint: str = "",
                 fault_after: int | None = None, command: list[float] | None = None,
                 start_config: dict | None = None):
        self.goals = goals
        self.task_name = task_name
        self.command = list(command) if command else [1.0, 0.0, 0.0]
        self.start_config = start_config or {}
        self.default_timesteps = default_timesteps
        self.max_wall_seconds = max_wall_seconds
        self.max_llm_steps = max_llm_steps
        self.client = client or LLMClient()
        self.system_prompt = system_prompt or build_system_prompt()
        self.extra_hint = extra_hint

        fixed_reward = (self.start_config.get("env", {}) or {}).get("reward")
        self.ctx = ToolContext(runs_root, task_name, max_trials, max_wall_seconds,
                               default_timesteps, fixed_reward=fixed_reward)
        self.tools = Tools(self.ctx)
        # fault injection: fail the run_training call whose 1-based index equals
        # fault_after, once. Used to measure robustness/recovery.
        self.fault_after = fault_after
        self._fault_hit = False
        self.schemas = build_tool_schemas()
        self.short = ShortTermMemory()
        self.long = LongTermMemory(longterm_path)

        self.tool_calls = 0
        self.tool_errors = 0
        self.native_tool_calls = 0
        self.fallback_tool_calls = 0

    # ------------------------------------------------------------------ helpers
    def _goal_text(self) -> str:
        parts = []
        if "mean_lin_vel_error_max" in self.goals:
            parts.append(f"mean_lin_vel_error <= {self.goals['mean_lin_vel_error_max']}")
        if "fall_rate_max" in self.goals:
            parts.append(f"fall_rate <= {self.goals['fall_rate_max']}")
        if "mean_return_min" in self.goals:
            parts.append(f"mean_return >= {self.goals['mean_return_min']}")
        return ", ".join(parts) or "(no goal set)"

    def _initial_user_msg(self) -> str:
        lessons = self.long.to_context(self.task_name)
        start = {"timesteps": self.default_timesteps}
        start["env"] = {"command": list(self.command)}
        # merge the task's weak start config over the base so the agent begins
        # from a known-poor baseline it must improve on
        for section, vals in self.start_config.items():
            if isinstance(vals, dict) and isinstance(start.get(section), dict):
                start[section] = {**start[section], **vals}
            else:
                start[section] = vals
        cmd_vx = self.command[0]
        hint = f"\nStrategy note: {self.extra_hint}\n" if self.extra_hint else ""
        return (
            hint +
            f"Task: make the Go2 walk forward at {cmd_vx:.2f} m/s.\n"
            f"Success requires: {self._goal_text()}.\n"
            f"Trial budget: {self.ctx.max_trials} training runs. "
            f"Wall-clock budget: {self.max_wall_seconds:.0f}s. "
            f"Each default run trains {self.default_timesteps} steps.\n\n"
            f"Lessons from previous sessions:\n{lessons}\n\n"
            f"Suggested starting config (you may change it): "
            f"{json.dumps(start, separators=(',', ':'))}\n\n"
            f"Begin. Think briefly, then call one tool."
        )

    def _cap_context(self, messages: list[dict], keep_system: int = 1,
                     max_msgs: int = 40) -> list[dict]:
        """Bound the conversation so token use stays predictable."""
        if len(messages) <= max_msgs:
            return messages
        head = messages[:keep_system]
        tail = messages[-(max_msgs - keep_system):]
        return head + [{"role": "user", "content": "(older observations trimmed)"}] + tail

    # --------------------------------------------------------------------- run
    def run(self) -> AgentResult:
        t0 = time.time()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._initial_user_msg()},
        ]
        llm_finished_success = False
        finish_reasoning = ""
        stopped_reason = "max_steps"
        step = 0

        while step < self.max_llm_steps:
            step += 1
            if time.time() - t0 > self.max_wall_seconds:
                stopped_reason = "wall_budget"
                break
            if self.ctx.trial_count >= self.ctx.max_trials:
                # let the model wrap up with one more call (e.g. finish/summary)
                stopped_reason = "trial_budget"

            try:
                resp = self.client.chat(self._cap_context(messages), self.schemas)
            except Exception as exc:  # noqa: BLE001
                stopped_reason = f"llm_error:{type(exc).__name__}"
                break

            messages.append(resp.raw_message or {"role": "assistant", "content": resp.content})

            if not resp.tool_calls:
                # no action: nudge once, then stop if it keeps happening
                messages.append({"role": "user",
                                 "content": "No tool call detected. Call exactly one tool "
                                            "as a JSON object: {\"tool\": name, \"arguments\": {...}}"})
                if step >= 3 and not resp.tool_calls:
                    continue
                continue

            call = resp.tool_calls[0]
            self.tool_calls += 1
            if call.source == "native":
                self.native_tool_calls += 1
            else:
                self.fallback_tool_calls += 1

            # fault injection: make one run_training fail to test recovery.
            # Injected before dispatch so a transient failure does not consume
            # the trial budget — the agent is expected to retry.
            if (call.name == "run_training" and not self._fault_hit
                    and self.fault_after is not None
                    and self.ctx.trial_count + 1 == self.fault_after):
                self._fault_hit = True
                result = ToolResult("error", "run_training", error_type="injected_fault",
                                    message="simulated tool failure (transient); retry")
            else:
                result = dispatch(self.tools, call.name, call.arguments)

            if result.status == "error":
                self.tool_errors += 1

            # after a training trial, reflect and remember
            goal_hit = False
            if call.name == "run_training" and result.status == "ok":
                rid = result.data["run_id"]
                rec = self.ctx.runs[rid]
                goal_hit = goal_met(rec.summary, self.goals)
                text, lesson = reflect_on_trial(
                    self.client, self.task_name, rec.config, rec.summary, self.goals,
                    self.long.to_context(self.task_name))
                if lesson:
                    self.long.add(lesson)
                self.short.add(TrialRecord(
                    trial_id=rec.run_id, config=rec.config, summary=rec.summary,
                    reflection=text, accepted=goal_hit))

            # append observation; add an objective goal-met signal so the agent
            # can stop early instead of spending the whole budget (efficiency).
            obs = result.to_observation()
            if goal_hit:
                obs += ("\n[objective check] the goal is now MET by this run "
                        f"({rid}). If you have no clearly better change to test, "
                        "call finish now rather than spending the remaining budget.")
            if call.source == "native":
                messages.append({"role": "tool", "tool_call_id": call.call_id,
                                 "content": obs})
            else:
                messages.append({"role": "user",
                                 "content": f"Observation: {obs}"})

            if result.status == "terminal":
                llm_finished_success = bool(result.data.get("success"))
                finish_reasoning = result.data.get("reasoning", "")
                stopped_reason = "finish"
                break

        # final report
        best = self.short.best()
        final_report = self._final_report(best, finish_reasoning, stopped_reason)
        wall = time.time() - t0

        best_metrics = best.summary if best else {}
        return AgentResult(
            success=goal_met(best_metrics, self.goals),
            llm_finished_success=llm_finished_success,
            best_run_id=best.trial_id if best else "",
            best_metrics={k: best_metrics.get(k) for k in
                          ("mean_return", "fall_rate", "mean_lin_vel_error",
                           "mean_episode_length", "convergence_delta")},
            n_trials=self.ctx.trial_count,
            trials=[t.compact() for t in self.short.trials],
            tool_stats={
                "total_calls": self.tool_calls,
                "errors": self.tool_errors,
                "success_rate": round(1 - self.tool_errors / self.tool_calls, 3)
                if self.tool_calls else 0.0,
                "native": self.native_tool_calls,
                "fallback": self.fallback_tool_calls,
            },
            usage=self.client.usage_summary(),
            wall_seconds=round(wall, 1),
            final_report=final_report,
            stopped_reason=stopped_reason,
        )

    def _final_report(self, best, finish_reasoning: str, stopped_reason: str) -> str:
        best_txt = json.dumps(best.compact()["metrics"], separators=(",", ":")) if best else "none"
        try:
            resp = self.client.chat([
                {"role": "system", "content": "You are an RL engineer writing a short report."},
                {"role": "user", "content": FINAL_PROMPT.format(
                    goal=self._goal_text(), n_trials=self.ctx.trial_count, best=best_txt)},
            ])
            report = (resp.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            report = f"(report unavailable: {exc})"
        header = f"Stopped: {stopped_reason}. "
        if finish_reasoning:
            header += f"Agent's closing note: {finish_reasoning} "
        return header + report
