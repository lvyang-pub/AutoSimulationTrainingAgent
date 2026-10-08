"""Assembly: the seam where the agent meets the environment.

The schedular owns this file. It is the ONLY place that knows both sides:

  * the agent, which knows nothing about any environment and only dispatches
    tools it is handed (`asta_agent.Tool` / `Toolbox` / `ToolResult`), and
  * the environment, which declares what it can do as plain functions under
    `<env>/action/*.py` (training, evaluation, metrics -- reading a result is
    just another action).

Nothing here is task-prompt logic (that lives in `prompts.py`) or the loop
(that lives in the agent). It is the plumbing in between.

## What it does

1. Scans `<env>/action/*.py` and turns each public function into an agent
   `Tool` (name = function name, description = the FULL docstring, schema from
   the signature + type hints).
2. Hides the *framework's* parameters (`runs_root`, `out_dir`, `policy_path`,
   ...) from the model and binds them from the execution context at assembly
   time. A parameter is framework-injected when its default is the `INJECTED`
   sentinel or its name matches an attribute of the context.
3. Wraps each function in a small adapter that reshapes a bare env return into
   the observation shape the agent's trial detector expects -- a training run
   must yield `{run_id, summary, config}` so the run is recorded as a trial --
   and folds in schedular-side post-processing (the CurveAnalyst read of the
   training curve).
4. Adds the schedular-only control tools that are NOT environment actions, such
   as `finish` (which ends the loop by returning a `terminal` result).

There is no overlay and no per-call path lookup beyond a callable context value
being re-evaluated per call (so ids and fresh directories stay fresh). The agent
never imports the environment; it only ever sees the `Toolbox` this file returns.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import typing

from asta_agent import Tool, Toolbox, ToolResult


# --------------------------------------------------------------------------- #
# framework-injected parameter sentinel
# --------------------------------------------------------------------------- #
class _Injected:
    """Sentinel: a parameter the framework supplies, not the model."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "INJECTED"


INJECTED = _Injected()

_ACTION_DIR = "action"
_CURVE_ANALYST_QUESTION = (
    "请描述训练奖励曲线的整体趋势：是否在上升？是否已收敛？有无明显震荡或崩溃？")

# The env's `run_training` docstring is deliberately terse, so the schedular
# supplies the config-key guidance the model needs to shape a trial. This is
# task glue and lives here (not in the env, not in the agent).
_CONFIG_NOTE = (
    "Partial run config; unspecified fields keep their defaults. Keys: "
    "timesteps, seed, net_arch, ppo.{learning_rate,n_steps,batch_size,n_epochs,"
    "gamma,gae_lambda,clip_range,ent_coef,vf_coef,max_grad_norm}, "
    "env.{action_scale,kp,kd,max_torque,min_height,max_tilt,episode_length,"
    "command,n_envs}. env.reward is fixed by the task; timesteps IS tunable.")


# --------------------------------------------------------------------------- #
# execution context -- the values the framework owns during one session
# --------------------------------------------------------------------------- #
class ExecutionContext:
    """The execution paths a session injects into the env's action functions.

    Attributes are looked up by name by `compose`: any action parameter that
    matches one is filled from here and hidden from the model. `runs_root` and
    `env_path` are plain values; `visualize` / `default_timesteps` carry the
    schedular's run settings. `policy_path` is a method, so it re-reads the
    latest trial each call.
    """

    def __init__(self, runs_root: str, env_path: str, *,
                 visualize: bool = True, default_timesteps: int = 200_000):
        self.runs_root = os.path.abspath(runs_root)
        self.env_path = os.path.abspath(env_path)
        self.visualize = visualize
        self.default_timesteps = default_timesteps
        self.trial_count = 0
        self.runs: dict[str, dict] = {}

    def next_run_id(self) -> str:
        self.trial_count += 1
        return f"trial_{self.trial_count:03d}"

    def policy_path(self) -> str:
        """Path to the most recent trial's saved policy (re-evaluated per call)."""
        if not self.runs:
            return ""
        last = list(self.runs.values())[-1]
        return os.path.join(last["dir"], "policy.zip")


# --------------------------------------------------------------------------- #
# type hint -> JSON schema
# --------------------------------------------------------------------------- #
_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean",
               dict: "object", list: "array"}


def _json_type(annotation) -> dict:
    """Best-effort JSON-schema fragment for a Python annotation."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return {}
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}
    origin = typing.get_origin(annotation)
    if origin is typing.Union:            # Optional[X] / X | None
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else {}
    if origin in (list, typing.List):
        return {"type": "array"}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    return {}


def _schema(fn, injected_names: set[str]) -> dict:
    """JSON schema for `fn`, skipping every parameter the framework injects."""
    props: dict = {}
    required: list[str] = []
    for pname, p in inspect.signature(fn).parameters.items():
        if pname in injected_names or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        frag = _json_type(p.annotation)
        if p.default is not inspect.Parameter.empty:
            frag = {**frag, "default": p.default}
        else:
            required.append(pname)
        props[pname] = frag
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _description(fn) -> str:
    """The model sees the function's whole docstring, not a truncated summary."""
    doc = inspect.getdoc(fn)
    return doc.strip() if doc and doc.strip() else fn.__name__


# --------------------------------------------------------------------------- #
# context lookup + injection detection
# --------------------------------------------------------------------------- #
class _Missing(Exception):
    """No value for an injected parameter in the assembly context."""


_MISSING = object()          # a sentinel distinct from a legitimate None value


def _ctx_get(context, name):
    """Read an attribute (object) or key (mapping) from the context, or RAISE."""
    if context is None:
        raise _Missing(name)
    if isinstance(context, dict):
        if name in context:
            return context[name]
        raise _Missing(name)
    if hasattr(context, name):
        return getattr(context, name)
    raise _Missing(name)


def _has(context, name) -> bool:
    if context is None:
        return False
    if isinstance(context, dict):
        return name in context
    return hasattr(context, name)


def _injected_names(fn, context) -> set[str]:
    """Names the framework fills: INJECTED-sentinel defaults, or context attrs."""
    names: set[str] = set()
    for pname, p in inspect.signature(fn).parameters.items():
        if isinstance(p.default, _Injected):
            names.add(pname)
        elif p.default is inspect.Parameter.empty and _has(context, pname):
            names.add(pname)
    return names


# --------------------------------------------------------------------------- #
# tool factory (injection bound at assembly time)
# --------------------------------------------------------------------------- #
def _resolve(value):
    """A plain value is reused; a callable is evaluated per call (fresh dir, id)."""
    return value() if callable(value) else value


def _make_tool(fn, context, on_ok=None) -> Tool:
    """Wrap one env function as a Tool, binding injected params from `context`.

    on_ok(tool_name, args, data) -- optional schedular hook run after a
    successful call; it may mutate/extend `data` before it becomes the
    observation (used to fold in curve analysis).
    """
    injected = _injected_names(fn, context)
    # Bind every injected value NOW, at assembly time. A plain value is frozen
    # here; a callable is invoked per call so it can yield something fresh.
    bound: dict = {}
    for name in injected:
        try:
            bound[name] = _ctx_get(context, name)
        except _Missing:
            bound[name] = _MISSING        # left unbound; error surfaces on call

    def handler(args: dict) -> ToolResult:
        kwargs = dict(args or {})
        for name in injected:
            if bound[name] is _MISSING:
                return ToolResult("error", fn.__name__, error_type="missing_path",
                                  message=f"execution path for '{name}' was not "
                                          f"available when the env was assembled")
            kwargs[name] = _resolve(bound[name])
        try:
            out = fn(**kwargs)
        except TypeError as exc:
            return ToolResult("error", fn.__name__, error_type="bad_arguments",
                              message=str(exc))
        except Exception as exc:  # noqa: BLE001 - tool faults are observations
            return ToolResult("error", fn.__name__, error_type=type(exc).__name__,
                              message=str(exc))
        data = dict(out) if isinstance(out, dict) else {"result": out}
        if on_ok is not None:
            try:
                on_ok(fn.__name__, kwargs, data)
            except Exception as exc:  # noqa: BLE001 - hooks are best-effort
                print(f"[assembly] post-ok hook failed for {fn.__name__}: {exc}",
                      flush=True)
        return ToolResult("ok", fn.__name__, data=data)

    return Tool(fn.__name__, _description(fn), _schema(fn, injected), handler)


# --------------------------------------------------------------------------- #
# task shims: reshape the env's returns into the agent's trial shape
# --------------------------------------------------------------------------- #
def _make_run_tool(run_fn, norm_fn, ctx: ExecutionContext, hook) -> Tool:
    """Wrap `run_training` so its result is self-describing.

    The env's `run_training` returns a bare summary dict and receives its
    `out_dir` from the framework (so it does not know its own id). The agent,
    however, records a trial by `run_id` and needs the normalized config. This
    wrapper supplies both by minting the id/dir here and calling the env
    function with the config it validates via `normalize_config` (if present).
    """
    schema = _schema(run_fn, {"out_dir", "verbose", "visualize"})
    if "config" in schema.get("properties", {}):
        schema["properties"]["config"]["description"] = _CONFIG_NOTE

    def handler(args: dict) -> ToolResult:
        args = args or {}
        config = dict(args.get("config") or args.get("user_cfg") or {})
        if "timesteps" not in config:
            config["timesteps"] = ctx.default_timesteps
        run_id = ctx.next_run_id()
        run_dir = os.path.join(ctx.runs_root, run_id)
        os.makedirs(run_dir, exist_ok=True)
        try:
            summary = run_fn(config=config, out_dir=run_dir, verbose=0,
                             visualize=ctx.visualize)
        except Exception as exc:  # noqa: BLE001 - tool faults are observations
            return ToolResult("error", "run_training",
                              error_type=type(exc).__name__, message=str(exc))
        norm_cfg, warnings = config, []
        if norm_fn is not None:
            try:
                norm_cfg, warnings = norm_fn(user_cfg=config)
            except Exception:  # noqa: BLE001 - normalization is best-effort
                pass
        ctx.runs[run_id] = {"dir": run_dir, "config": norm_cfg, "summary": summary}
        data = {"run_id": run_id, "summary": summary, "config": norm_cfg,
                "config_warnings": warnings}
        if hook is not None:
            try:
                hook("run_training", {"out_dir": run_dir}, data)
            except Exception as exc:  # noqa: BLE001 - hooks are best-effort
                print(f"[assembly] curve hook failed for {run_id}: {exc}", flush=True)
        return ToolResult("ok", "run_training", data=data)

    return Tool("run_training", _description(run_fn), schema, handler)


def _make_curve_hook(curve_analyst):
    """A post-ok hook: after a training run, describe its curve with the analyst."""
    if curve_analyst is None:
        return None

    def hook(tool_name: str, kwargs: dict, data: dict) -> None:
        if tool_name != "run_training":
            return
        run_dir = kwargs.get("out_dir")
        if not run_dir:
            return
        curve_path = os.path.join(run_dir, "curve.png")
        if not os.path.exists(curve_path):
            return
        analysed = curve_analyst.analyze(
            curve_path,
            _CURVE_ANALYST_QUESTION + f"（本次 trial 目录: {os.path.basename(run_dir)}）")
        data["curve_analysis"] = analysed.answer

    return hook


# --------------------------------------------------------------------------- #
# module loading (self-contained: no harness, no task imports)
# --------------------------------------------------------------------------- #
def _load_module(path: str, env_root: str):
    name = "env_action_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # let the action module import its own siblings (training/, env/, ...)
    added = []
    for p in (env_root, os.path.dirname(env_root)):
        if p and p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    try:
        spec.loader.exec_module(mod)
    finally:
        for p in added:
            if p in sys.path:
                sys.path.remove(p)
    return mod


def _iter_action_functions(env_path: str):
    """Yield (function) for every public function declared under `<env>/action/`."""
    env_root, action_dir = _resolve_root(env_path)
    if not os.path.isdir(action_dir):
        return
    for fname in sorted(os.listdir(action_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        mod = _load_module(os.path.join(action_dir, fname), env_root)
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("_") or obj.__module__ != mod.__name__:
                continue
            yield obj


def _resolve_root(env_path: str) -> tuple[str, str]:
    """Given an env root (or its `action/` dir), return (env_root, action_dir)."""
    env_path = os.path.abspath(env_path)
    if os.path.basename(env_path) == _ACTION_DIR:
        env_path = os.path.dirname(env_path)
    return env_path, os.path.join(env_path, _ACTION_DIR)


def _exec_action(env_path: str, module_rel: str, func_name: str, **kwargs):
    """Call one env action directly (bypassing injection); used by the shims."""
    mod = _load_module(os.path.join(os.path.abspath(env_path), module_rel),
                       os.path.abspath(env_path))
    return getattr(mod, func_name)(**kwargs)


def list_env_functions(env_path: str) -> list[str]:
    """Names of the action functions a `compose` call would expose."""
    names: list[str] = []
    for fn in _iter_action_functions(env_path):
        if fn.__name__ not in names:
            names.append(fn.__name__)
    return names


# --------------------------------------------------------------------------- #
# schedular-only control tools (not environment actions)
# --------------------------------------------------------------------------- #
def _finish_tool() -> Tool:
    def handler(a: dict) -> ToolResult:
        return ToolResult("terminal", "finish", data={
            "success": bool(a.get("success")),
            "best_run_id": a.get("best_run_id", ""),
            "reasoning": a.get("reasoning", ""),
        })

    return Tool(
        "finish",
        "End the session. Call when the goal is met or the budget is spent.",
        {"type": "object", "properties": {
            "success": {"type": "boolean"},
            "best_run_id": {"type": "string"},
            "reasoning": {"type": "string"}},
         "required": ["success", "reasoning"]},
        handler)


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def compose(env_path: str, context=None, *,
            curve_analyst=None, include_finish: bool = True) -> Toolbox:
    """Build the agent's Toolbox from the env's actions, with paths injected.

    env_path:      the env root (e.g. Envs/Go2Locomotion-PPO) or its `action/`.
    context:       the session's `ExecutionContext` (or a dict). Any action
                   parameter whose name matches one of its attributes is filled
                   from it and hidden from the model.
    curve_analyst: optional CurveAnalyst; when given, each training run's curve
                   is described and folded into the observation.
    include_finish: add the schedular-only `finish` control tool.
    """
    box = Toolbox()
    hook = _make_curve_hook(curve_analyst)

    # `run_training` must yield {run_id, summary, config}; every other action
    # passes through as the env returns it.
    functions = list(_iter_action_functions(env_path))
    by_name = {fn.__name__: fn for fn in functions}
    run_fn = by_name.get("run_training")
    norm_fn = by_name.get("normalize_config")

    if run_fn is not None:
        box.register(_make_run_tool(run_fn, norm_fn, context, hook))

    for fn in functions:
        if run_fn is not None and fn.__name__ in ("run_training", "normalize_config"):
            continue                       # handled by the run tool above
        box.register(_make_tool(fn, context, on_ok=hook))

    if include_finish:
        box.register(_finish_tool())
    return box
