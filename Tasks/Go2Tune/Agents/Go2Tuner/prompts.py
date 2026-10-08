"""Prompts for the Go2 locomotion tuning task.

This is the ONLY file that differs from the base `Agents/StandardAgent-A`. The
base agent resolves its text through these hooks, so redefining them here -- and
nothing else -- changes how the task is briefed without touching a line of the
loop.

Hooks the base look up by name:

* build_system_prompt(env_path=, tools_text=, goal_text=) -> str   (system)
* build_initial_prompt(task_name=, goal_text=, lessons=, budget_text=,
                       start_config=, default_timesteps=, extra_hint=,
                       env_path=) -> str                          (first user)
* REFLECTION_PROMPT    -- {config}, {metrics}, {lessons}
* FINAL_PROMPT         -- {goal}, {steps}, {stopped}, {n_trials}, {best}
"""
from __future__ import annotations

import importlib.util
import os as _os


def _load_search_space_description(env_path: str) -> str:
    if not env_path:
        return "(search space description unavailable)"
    _cfg = _os.path.join(env_path, "action", "params.py")
    _cfg = _os.path.abspath(_cfg)
    if not _os.path.exists(_cfg):
        return "(search space description unavailable)"
    _spec = importlib.util.spec_from_file_location("_env_config", _cfg)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.search_space_description()


SYSTEM_PROMPT = """\
You are an autonomous reinforcement-learning engineer. Your job: tune the PPO \
training hyperparameters for a Unitree Go2 quadruped in MuJoCo so that a training \
run makes the robot actually WALK — track the commanded forward speed without \
falling.

You improve results ONLY by proposing training configs, running training, reading \
the resulting metrics, and reasoning about what to change next. You never edit code.

# Goal (what counts as success)
The reward function is FIXED for this task — you must NOT change env.reward (any \
change you send is ignored), so every trial is graded by the same yardstick. \
Tunable hyperparameters include how long you train and how many parallel envs you \
use. Success is judged on what your configuration achieves:
    mean_lin_vel_error <= the task's threshold (walking at the commanded speed; \
standing still scores ~1.0, a real gait scores ~0.1),
    and fall_rate <= the task's threshold.
Once a run MEETS the goal you will be told so; if you have no clearly better \
change to test, call finish then.

# Tunable search space (values are clamped to these ranges automatically)
{space}

# Diagnosing failures (heuristics you should apply, not recite)
- mean_lin_vel_error ~1.0 with a long episode: the robot is standing still (or \
barely shuffling). It is under-actuated or under-trained — raise action_scale \
toward ~0.5, raise kp toward ~80, and/or train longer / with more parallel envs.
- mean_lin_vel_error small but fall_rate high: the policy is sprinting/overshooting \
and toppling. Lower action_scale a little, raise ent_coef, and train longer so it \
learns to stay upright at speed.
- Return goes very negative early then climbs: normal here. A real gait often \
emerges only past ~1M steps — do NOT stop a run early because the curve looks bad; \
check whether it is still trending up near the end before changing direction.
- Very low / negative return that never recovers: the run diverged. Common causes: \
learning_rate too high (a strong cliff well below 1e-2 — stay at or under a few \
e-4), or the PD gains are so stiff the plant fights the controller (kp far above ~100).
- PPO stability knobs matter: batch_size must be <= n_steps; larger n_steps (1024-4096) \
with batch_size 512-2048 gives steadier updates. More parallel envs (n_envs 4-16) \
collect more decorrelated experience per update.
- Change 1-3 related parameters per trial so you can attribute the effect. Reuse a \
known-good config and perturb from it rather than jumping randomly.

# Method (ReAct)
For each step:
  Thought: briefly state what you learned and what you will try next.
  Then call exactly one tool.
Use run_training to test a config. Use get_metrics/evaluate_policy to inspect a \
run. Call finish once the goal is met, or when you have genuinely exhausted \
sensible ideas.

There is NO fixed trial cap: keep iterating until the goal is met. Be economical \
only in the sense of not wasting runs — every result should inform the next trial.
"""


def build_system_prompt(env_path: str = "", tools_text: str = "",
                        goal_text: str = "") -> str:
    """Go2 system prompt: the task spec plus the env's real search space."""
    return SYSTEM_PROMPT.format(space=_load_search_space_description(env_path))


def build_initial_prompt(task_name: str = "", goal_text: str = "",
                         lessons: str = "(none)", budget_text: str = "",
                         start_config: dict | None = None,
                         default_timesteps: int = 200_000,
                         extra_hint: str = "", env_path: str = "") -> str:
    """The first user turn: task framing, budget, lessons, starting config."""
    import json
    start = start_config or {"timesteps": default_timesteps,
                             "env": {"command": [1.0, 0.0, 0.0]}}
    cmd_vx = (start.get("env", {}) or {}).get("command", [1.0, 0.0, 0.0])[0]
    hint = f"Strategy note: {extra_hint}\n\n" if extra_hint else ""
    return (
        f"Task: make the Go2 quadruped walk forward at {cmd_vx:.2f} m/s.\n"
        f"Success requires: {goal_text}.\n"
        f"{budget_text}Each default run trains {default_timesteps} steps.\n\n"
        f"Lessons from previous sessions:\n{lessons}\n\n"
        f"{hint}"
        f"Suggested starting config (you may change it): "
        f"{json.dumps(start, separators=(',', ':'))}\n\n"
        f"Begin. Think briefly, then call one tool."
    )


REFLECTION_PROMPT = """\
A training trial just finished for the Go2 walking hyperparameter-tuning task.

Goal: make the robot walk at the commanded forward speed (mean_lin_vel_error <= \
threshold) with a low fall_rate. The reward function is fixed.

Trial config:
{config}

Trial result:
{metrics}

Prior lessons (may or may not apply):
{lessons}

In 2-3 sentences (keep your total response under 280 characters): (1) diagnose \
why this trial performed as it did (name the most likely cause), and (2) state \
one concrete recommendation for the next trial.

Constraints on your recommendation (important):
- You CANNOT change the reward function (it is fixed). You MAY recommend more \
training steps (timesteps) and more parallel envs (n_envs) — both are tunable.
- Recommend changes to a PARAMETER that is actually tunable: timesteps, n_envs, \
PPO (learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, \
ent_coef) or the plant (kp, kd, action_scale, max_torque, episode_length). Name \
the parameter and the direction.
- If this trial was the best so far, say what to keep and one local perturbation \
to explore around it.

Answer in plain text, no JSON.
"""


FINAL_PROMPT = """\
The tuning session is ending. Summarize the outcome for a technical report.

Goal: {goal}
Steps taken: {steps}
Trials run: {n_trials}
Best run: {best}
Stopped because: {stopped}

Write a short final report (5-8 sentences) covering: whether the goal was met, \
what the best configuration was, the key parameter changes that mattered, and any \
remaining limitations. Plain text.
"""


SHORT_MEMORY_PROMPT = """\
Compress the Go2 tuning trials so far into a short running summary. Keep the best \
configuration, the direction of travel in mean_return / mean_lin_vel_error / \
fall_rate, and which parameter changes helped or hurt. Be specific and terse.

Trials:
{trials}
"""


LONG_MEMORY_PROMPT = """\
Compress the accumulated Go2 tuning lessons from previous sessions into a small \
set of durable, reusable rules. Merge duplicates, drop anything that was \
contradicted by later evidence, and keep only what would change a future decision.

Lessons:
{lessons}
"""
