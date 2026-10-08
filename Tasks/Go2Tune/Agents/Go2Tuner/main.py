"""StandardAgent-A: a full-featured ReAct agent, configured entirely from data.

This is the *base* agent. It owns the whole experimental loop -- not just a
minimal ReAct cycle -- so that a task copy can be produced by changing nothing
but `prompts.py`. Every behaviour below is driven by constructor arguments (a
toolbox, an optional objective predicate, budgets, a memory path) and by the
text in `prompts.py`; none of it is hard-wired to a domain.

Features, all generic:

* **ReAct loop.** Hand the conversation + tool schemas to the LLM, execute the
  tool calls, append observations, repeat.
* **One-call-per-step.** The model may emit several tool_calls; only the first
  is executed and the transcript keeps a strict 1:1 call/answer pairing, which
  some providers (DeepSeek) require.
* **Bounded context.** The conversation is trimmed to `max_context_msgs` and
  repaired so no tool_call is left unanswered.
* **Trial detection + reflection.** When a tool result carries a `summary`
  mapping, it is recorded as a trial; an LLM call diagnoses it and may write a
  lesson to long-term memory.
* **Objective check.** An injected `objective_fn(trial) -> bool` scores each
  trial; a note is appended when the goal is met so the model can stop.
* **Budgets** (trials / wall-clock / LLM steps): all OPT-IN, `None` = unbounded.
  Only wall-clock and step budgets break the loop; the trial budget lets the
  model wrap up with a final call.
* **Fault injection** (`fault_after=N`): fail the Nth trial call once, to test
  recovery.
* **Supplementary observations.** Any string under a key named in
  `supplementary_keys` is surfaced as an extra observation.
* **Final report.** A closing LLM summary built from `prompts.FINAL_PROMPT`.

All prompt text comes from `prompts.py` via stable hooks; see that module for
the placeholder contract.
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Callable

from . import prompts as _prompts
from .llm import LLMClient, LLMResponse
from .memory import LongTermMemory, ShortTermMemory, TrialRecord
from .reflection import reflect_on_trial
from .tools import ToolResult, Toolbox

_DEFAULT_METRIC_KEYS = ("mean_return", "std_return", "fall_rate",
                        "mean_lin_vel_error", "mean_episode_length",
                        "convergence_delta")


@dataclass
class RunState:
    """Everything observable about a run so far, handed to the stop function."""
    messages: list[dict]
    start_time: float
    step: int = 0
    results: list[ToolResult] = field(default_factory=list)
    finished: bool = False

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def results_named(self, tool_name: str) -> list[ToolResult]:
        return [r for r in self.results if r.tool == tool_name]


@dataclass
class AgentResult:
    """What a finished run produced. No task-specific fields."""
    stopped_reason: str
    steps: int
    tool_calls: int
    tool_errors: int
    native_tool_calls: int
    fallback_tool_calls: int
    wall_seconds: float
    usage: dict
    final_report: str
    finish_data: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)   # task-side fields (best_trial, ...)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class StandardAgent:
    """A generic ReAct agent with the full experimental loop built in."""

    def __init__(self, toolbox: Toolbox,
                 client: LLMClient | None = None,
                 should_stop: Callable[[RunState], bool] | None = None,
                 system_prompt: str | None = None,
                 initial_prompt: str | None = None,
                 # ---- experiment configuration (all optional) ----
                 objective_fn: Callable[[TrialRecord], bool] | None = None,
                 task_name: str = "",
                 goal_text: str = "",
                 metric_keys: list[str] | None = None,
                 longterm_path: str | None = None,
                 env_path: str = "",
                 extra_hint: str = "",
                 max_trials: int | None = None,
                 max_wall_seconds: float | None = None,
                 max_llm_steps: int | None = None,
                 fault_after: int | None = None,
                 default_timesteps: int = 200_000,
                 command: list[float] | None = None,
                 start_config: dict | None = None,
                 max_context_msgs: int = 30,
                 supplementary_keys: tuple[str, ...] = ("curve_analysis",)):
        self.toolbox = toolbox
        self.client = client or LLMClient()
        self.should_stop = should_stop

        # prompt text is resolved later (see _build_system_prompt), so a copy can
        # swap prompts.py without touching this file.
        self.system_prompt = system_prompt
        self.initial_prompt = initial_prompt

        self.objective_fn = objective_fn
        self.task_name = task_name
        self.goal_text = goal_text
        self.metric_keys = list(metric_keys or _DEFAULT_METRIC_KEYS)
        self.env_path = env_path
        self.extra_hint = extra_hint
        self.max_trials = max_trials
        self.max_wall_seconds = max_wall_seconds
        self.max_llm_steps = max_llm_steps
        self.default_timesteps = default_timesteps
        self.command = list(command) if command else [1.0, 0.0, 0.0]
        self.start_config = start_config or {}
        self.max_context_msgs = max_context_msgs
        self.supplementary_keys = tuple(supplementary_keys)

        self.fault_after = fault_after
        self._fault_hit = False

        self.short = ShortTermMemory()
        self.long = LongTermMemory(longterm_path) if longterm_path else None

        self.tool_calls = 0
        self.tool_errors = 0
        self.native_tool_calls = 0
        self.fallback_tool_calls = 0

    # ------------------------------------------------------------- helpers
    def _tools_text(self) -> str:
        return "\n".join(f"- {name}: {self.toolbox.get(name).description}"
                         for name in self.toolbox.names()) or "(no tools registered)"

    def _call_hook(self, name: str, **kwargs):
        """Call prompts.<name>(**matching-kwargs) if it exists; else return None.

        Only the kwargs the hook actually accepts are forwarded, so a copy may
        narrow a hook's signature without breaking the base.
        """
        fn = getattr(_prompts, name, None)
        if not callable(fn):
            return None
        accepted = inspect.signature(fn).parameters
        if any(p.kind == p.VAR_KEYWORD for p in accepted.values()):
            return fn(**kwargs)
        return fn(**{k: v for k, v in kwargs.items() if k in accepted})

    # ------------------------------------------------------------- prompts
    def _build_system_prompt(self) -> str:
        systems = self._call_hook("build_system_prompt", env_path=self.env_path,
                                  tools_text=self._tools_text(),
                                  goal_text=self.goal_text)
        if systems is not None:
            return systems
        return _prompts.SYSTEM_PROMPT.format(tools=self._tools_text(), env_context="")

    def _budget_text(self) -> str:
        if self.max_trials is None and self.max_wall_seconds is None:
            return ("Budget: none -- you are NOT step-limited. Keep running trials "
                    "and iterating until the goal is met; call finish only when "
                    "success requires it or you have exhausted sensible ideas. ")
        parts = []
        if self.max_trials is not None:
            parts.append(f"Trial budget: {self.max_trials} training runs.")
        if self.max_wall_seconds is not None:
            parts.append(f"Wall-clock budget: {self.max_wall_seconds:.0f}s.")
        return " ".join(parts) + " "

    def _start_config(self) -> dict:
        start = {"timesteps": self.default_timesteps,
                 "env": {"command": list(self.command)}}
        for section, vals in self.start_config.items():
            if isinstance(vals, dict) and isinstance(start.get(section), dict):
                start[section] = {**start[section], **vals}
            else:
                start[section] = vals
        return start

    def _build_initial_prompt(self) -> str:
        lessons = self.long.to_context(self.task_name) if self.long else "(none)"
        built = self._call_hook(
            "build_initial_prompt", task_name=self.task_name, goal_text=self.goal_text,
            lessons=lessons, budget_text=self._budget_text(),
            start_config=self._start_config(), default_timesteps=self.default_timesteps,
            extra_hint=self.extra_hint, env_path=self.env_path)
        if built is not None:
            return built
        return _prompts.INITIAL_PROMPT.format(
            context="", goal=self.goal_text or "(none)", budget=self._budget_text(),
            start_config=json.dumps(self._start_config(), separators=(",", ":")),
            default_timesteps=self.default_timesteps, lessons=lessons)

    # ------------------------------------------------------------- transcript
    @staticmethod
    def _assistant_msg(resp: LLMResponse) -> dict:
        """Assistant turn with a 1:1 tool pairing.

        The loop runs exactly one tool per step, but the model may emit several
        tool_calls at once. Keeping all of them while answering one leaves
        tool_calls without a matching tool reply, which some providers reject, so
        keep only the call that will be answered.
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
        """Drop unpaired tool_calls/tool messages so the transcript stays valid."""
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

    def _cap_context(self, messages: list[dict], keep_system: int = 1) -> list[dict]:
        """Bound the conversation so token use stays predictable."""
        if self.max_context_msgs is None or len(messages) <= self.max_context_msgs:
            return messages
        head = messages[:keep_system]
        tail = messages[-(self.max_context_msgs - keep_system):]
        return self._repair_pairs(
            head + [{"role": "user", "content": "(older observations trimmed)"}] + tail)

    # -------------------------------------------------------------- observation
    def _render(self, result: ToolResult) -> str:
        """Render a result as a compact JSON observation.

        A trial result nests its metrics under "summary"; flatten the configured
        metric keys to the top level so the model reads them directly.
        """
        payload = {"tool": result.tool, "status": result.status}
        if result.status in ("ok", "terminal"):
            data = dict(result.data or {})
            data.pop("config_warnings", None)
            data.pop("eval_curve", None)
            if "summary" in data:
                summary = data.pop("summary") or {}
                for k in self.metric_keys:
                    if k in summary:
                        data[k] = summary[k]
                if "train_seconds" in summary:
                    data["train_seconds"] = summary["train_seconds"]
            payload.update(data)
        else:
            payload["error_type"] = result.error_type
            payload["message"] = result.message
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    def _is_trial(self, result: ToolResult) -> bool:
        """A trial is any successful result that carries a `summary` mapping."""
        return result.status == "ok" and isinstance((result.data or {}).get("summary"), dict)

    def _append_msg(self, messages: list[dict], call, content: str) -> None:
        if call.source == "native":
            messages.append({"role": "tool", "tool_call_id": call.call_id,
                             "content": content})
        else:
            messages.append({"role": "user", "content": f"Observation: {content}"})

    def _observe(self, messages: list[dict], call, result: ToolResult) -> None:
        """Record one result: reflect on trials, then append the observation."""
        if result.status == "error":
            self.tool_errors += 1

        goal_hit = False
        rid = None
        if self._is_trial(result):
            data = result.data or {}
            rid = data.get("run_id")
            rec = TrialRecord(trial_id=rid, config=data.get("config") or {},
                              summary=data.get("summary") or {})
            if self.objective_fn is not None:
                goal_hit = bool(self.objective_fn(rec))
            lessons_ctx = self.long.to_context(self.task_name) if self.long else ""
            text, lesson = reflect_on_trial(
                self.client, self.task_name, rec.config, rec.summary,
                prompt=_prompts.REFLECTION_PROMPT, lessons_context=lessons_ctx)
            if lesson and self.long is not None:
                self.long.add(lesson)
            rec.reflection = text
            rec.accepted = goal_hit
            self.short.add(rec)

        obs = self._render(result)
        if goal_hit:
            obs += (f"\n[objective check] the goal is now MET by this run ({rid}). "
                    "If you have no clearly better change to test, call finish now "
                    "rather than spending the remaining budget.")
        self._append_msg(messages, call, obs)

        # supplementary analysis the task folded into the result payload
        for key in self.supplementary_keys:
            extra = (result.data or {}).get(key)
            if extra and str(extra).strip():
                messages.append({"role": "user",
                                 "content": f"[Supplementary analysis]\n{str(extra).strip()}"})
                break

    # ------------------------------------------------------------------ run
    def run(self) -> AgentResult:
        """Run the ReAct loop to completion."""
        t0 = time.time()
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt or self._build_system_prompt()},
            {"role": "user", "content": self.initial_prompt or self._build_initial_prompt()},
        ]
        state = RunState(messages=messages, start_time=t0)

        stopped_reason = "finish"
        finish_data: dict = {}
        finish_reasoning = ""
        trial_count = 0

        while True:
            # Caller-supplied stop rule (optional).
            if self.should_stop is not None and self.should_stop(state):
                stopped_reason = "stop_condition"
                break
            # Budgets are OPT-IN: `None` means unbounded.
            if (self.max_wall_seconds is not None
                    and time.time() - t0 > self.max_wall_seconds):
                stopped_reason = "wall_budget"
                break
            if self.max_trials is not None and trial_count >= self.max_trials:
                # do not break: let the model wrap up with one final call
                stopped_reason = "trial_budget"
            if self.max_llm_steps is not None and state.step >= self.max_llm_steps:
                stopped_reason = "llm_step_budget"
                break

            state.step += 1
            try:
                resp = self.client.chat(self._cap_context(messages), self.toolbox.schemas())
            except Exception as exc:  # noqa: BLE001 - stop cleanly on LLM failure
                stopped_reason = f"llm_error:{type(exc).__name__}"
                print(f"[agent] LLM error (stopping): {exc}", flush=True)
                break

            messages.append(self._assistant_msg(resp))

            if not resp.tool_calls:
                messages.append({"role": "user", "content":
                                 "No tool call detected. Call exactly one tool as a JSON "
                                 "object: {\"tool\": name, \"arguments\": {...}}"})
                continue

            call = resp.tool_calls[0]          # one tool per step
            self.tool_calls += 1
            if call.source == "native":
                self.native_tool_calls += 1
            else:
                self.fallback_tool_calls += 1

            result = self.toolbox.dispatch(call.name, call.arguments)

            # trial detection + fault injection (fail once, then let the agent retry)
            if self._is_trial(result):
                trial_count += 1
                if (self.fault_after is not None and not self._fault_hit
                        and trial_count == self.fault_after):
                    self._fault_hit = True
                    result = ToolResult("error", call.name, error_type="injected_fault",
                                        message="simulated tool failure (transient); retry")

            state.results.append(result)
            self._observe(messages, call, result)

            if result.terminal:
                finish_data = dict(result.data)
                finish_reasoning = result.data.get("reasoning", "")
                stopped_reason = "finish"
                state.finished = True
                break

        wall = time.time() - t0
        final_report = self._final_report(state, stopped_reason, trial_count, finish_reasoning)

        best = self.short.best()
        best_metrics = best.summary if best else {}
        out = AgentResult(
            stopped_reason=stopped_reason,
            steps=state.step,
            tool_calls=self.tool_calls,
            tool_errors=self.tool_errors,
            native_tool_calls=self.native_tool_calls,
            fallback_tool_calls=self.fallback_tool_calls,
            wall_seconds=round(wall, 1),
            usage=self.client.usage_summary(),
            final_report=final_report,
            finish_data=finish_data,
        )
        out.extra = {
            "success": bool(self.objective_fn(best)) if (self.objective_fn and best) else False,
            "best_trial": best.trial_id if best else None,
            "best_metrics": {k: best_metrics.get(k) for k in self.metric_keys},
            "n_trials": trial_count,
            "trials": [t.compact() for t in self.short.trials],
        }
        return out

    def _final_report(self, state: RunState, stopped_reason: str, trial_count: int,
                      finish_reasoning: str = "") -> str:
        best = self.short.best()
        best_txt = (json.dumps(best.compact()["metrics"], separators=(",", ":"))
                    if best else "none")
        # a superset of placeholders is passed; str.format ignores any a given
        # prompt does not use, so base and copy prompts both work unchanged.
        try:
            resp = self.client.chat([
                {"role": "system",
                 "content": "You are writing a short technical report on an agent session."},
                {"role": "user", "content": _prompts.FINAL_PROMPT.format(
                    goal=self.goal_text or "(none)", steps=state.step,
                    stopped=stopped_reason, n_trials=trial_count, best=best_txt)},
            ])
            report = (resp.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            report = f"(report unavailable: {exc})"
        header = f"Stopped: {stopped_reason}. "
        if finish_reasoning:
            header += f"Agent's closing note: {finish_reasoning} "
        return header + report
