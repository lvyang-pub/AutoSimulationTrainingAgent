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
                 max_trials: int | None = None, max_wall_seconds: float | None = None,
                 max_llm_steps: int | None = None, client: LLMClient | None = None,
                 system_prompt: str | None = None, extra_hint: str = "",
                 fault_after: int | None = None, command: list[float] | None = None,
                 start_config: dict | None = None, visualize: bool = True,
                 env_path: str = "", harness_path: str = ""):
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
                               default_timesteps, env_path=env_path,
                               harness_path=harness_path,
                               fixed_reward=fixed_reward, visualize=visualize)
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

    def inject_observation(self, text: str) -> None:
        """Deprecated: use the on_trial_done callback in run() instead."""
        pass

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
        if self.ctx.max_trials is None and self.max_wall_seconds is None:
            budget = ("Budget: none — you are NOT step-limited. Keep running trials and "
                      "iterating until the goal is met; call finish only when success "
                      "requires it or you have exhausted sensible ideas. ")
        else:
            parts = []
            if self.ctx.max_trials is not None:
                parts.append(f"Trial budget: {self.ctx.max_trials} training runs.")
            if self.max_wall_seconds is not None:
                parts.append(f"Wall-clock budget: {self.max_wall_seconds:.0f}s.")
            budget = " ".join(parts) + " "
        return (
            hint +
            f"Task: make the Go2 walk forward at {cmd_vx:.2f} m/s.\n"
            f"Success requires: {self._goal_text()}.\n"
            f"{budget}"
            f"Each default run trains {self.default_timesteps} steps.\n\n"
            f"Lessons from previous sessions:\n{lessons}\n\n"
            f"Suggested starting config (you may change it): "
            f"{json.dumps(start, separators=(',', ':'))}\n\n"
            f"Begin. Think briefly, then call one tool."
        )

    @staticmethod
    def _assistant_msg(resp) -> dict:
        """Assistant transcript message with a 1:1 tool pairing.

        The harness executes exactly one tool per step, but the model may emit
        several tool_calls in one message. Forwarding them all while answering
        only one leaves an assistant message whose tool_calls are not each
        followed by a tool message — DeepSeek rejects that with a 400. So keep
        only the call we will actually answer.
        """
        msg = dict(resp.raw_message or {"role": "assistant", "content": resp.content})
        calls = msg.get("tool_calls")
        if calls and resp.tool_calls:
            keep = resp.tool_calls[0].call_id
            kept = [c for c in calls if c.get("id") == keep]
            if kept:
                msg["tool_calls"] = kept
            else:
                msg.pop("tool_calls", None)
        return msg

    @staticmethod
    def _repair_pairs(messages: list[dict]) -> list[dict]:
        """Drop unpaired tool_calls/tool messages so the transcript stays valid.

        Trimming by count can strand an assistant tool_calls without its tool
        reply, or a tool message whose assistant was cut — DeepSeek 400s on
        both. Keep an assistant tool_calls only if every call id is answered by
        a tool message still present; keep a tool message only if its issuing
        assistant survived.
        """
        answered = {m.get("tool_call_id") for m in messages
                    if m.get("role") == "tool" and m.get("tool_call_id")}
        out: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                if any(r.get("role") == "assistant" and r.get("tool_calls")
                       and any(tc.get("id") == m.get("tool_call_id")
                               for tc in r["tool_calls"]) for r in out):
                    out.append(m)
                continue
            if role == "assistant" and m.get("tool_calls"):
                if all(tc.get("id") in answered for tc in m["tool_calls"]):
                    out.append(m)
                else:
                    stripped = {k: v for k, v in m.items() if k != "tool_calls"}
                    if stripped.get("content"):
                        out.append(stripped)
                continue
            out.append(m)
        return out

    def _cap_context(self, messages: list[dict], keep_system: int = 1,
                     max_msgs: int = 30) -> list[dict]:
        """Bound the conversation so token use stays predictable.

        Trimming must never split an assistant tool_calls from its tool reply,
        so the kept window is repaired for pairing before it is sent.
        """
        if len(messages) <= max_msgs:
            return messages
        head = messages[:keep_system]
        tail = messages[-(max_msgs - keep_system):]
        return self._repair_pairs(
            head + [{"role": "user", "content": "(older observations trimmed)"}] + tail)

    # --------------------------------------------------------------------- run
    def _observe(self, messages: list[dict], call, result, on_trial_done) -> None:
        """Record one tool result: reflect on trials, then append the observation."""
        if result.status == "error":
            self.tool_errors += 1

        rid = None
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

        obs = result.to_observation()
        if goal_hit:
            obs += ("\n[objective check] the goal is now MET by this run "
                    f"({rid}). If you have no clearly better change to test, "
                    "call finish now rather than spending the remaining budget.")
        if call.source == "native":
            messages.append({"role": "tool", "tool_call_id": call.call_id,
                             "content": obs})
        else:
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        if call.name == "run_training" and result.status == "ok" and on_trial_done:
            try:
                extra = on_trial_done(rid, rec.dir)
            except Exception:  # noqa: BLE001
                extra = None
            if extra and str(extra).strip():
                messages.append({"role": "user",
                                 "content": f"[Supplementary analysis]\n{extra.strip()}"})

    def run(self, on_trial_done=None) -> AgentResult:
        """Run the ReAct loop.

        on_trial_done: optional callable(run_id: str, run_dir: str) -> str | None
            Called by the agent after every successful run_training trial.
            The schedular passes this to invoke CurveAnalyst (or any other agent).
            The returned text (if any) is injected into the conversation as a
            supplementary observation before the next LLM step.
        """
        t0 = time.time()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._initial_user_msg()},
        ]
        llm_finished_success = False
        finish_reasoning = ""
        stopped_reason = "finish"
        step = 0

        while True:
            step += 1
            # Budgets are OPT-IN: `None` means unbounded, so the natural stopping
            # condition is the agent calling `finish` after the goal is met. A
            # wall-clock cap, if set, is only a safety valve.
            if self.max_wall_seconds is not None and time.time() - t0 > self.max_wall_seconds:
                stopped_reason = "wall_budget"
                break
            if self.ctx.max_trials is not None and self.ctx.trial_count >= self.ctx.max_trials:
                # let the model wrap up with one more call (e.g. finish/summary)
                stopped_reason = "trial_budget"

            try:
                resp = self.client.chat(self._cap_context(messages), self.schemas)
            except Exception as exc:  # noqa: BLE001
                stopped_reason = f"llm_error:{type(exc).__name__}"
                print(f"[agent] LLM error (stopping): {exc}", flush=True)
                break

            messages.append(self._assistant_msg(resp))

            if not resp.tool_calls:
                # no action: nudge once, then stop if it keeps happening
                messages.append({"role": "user",
                                 "content": "No tool call detected. Call exactly one tool "
                                            "as a JSON object: {\"tool\": name, \"arguments\": {...}}"})
                continue

            # One tool per step, matching the single call kept in _assistant_msg,
            # so every tool_calls is answered by exactly one tool message.
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

            self._observe(messages, call, result, on_trial_done)

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
