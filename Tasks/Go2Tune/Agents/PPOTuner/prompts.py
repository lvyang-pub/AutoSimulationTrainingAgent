"""Prompts for the tuning agent. The system prompt is the agent's task
specification; keeping it in one place makes the search strategy auditable and
tunable (and is itself something the eval loop can iterate on)."""
from __future__ import annotations
import importlib.util, os as _os

def _load_search_space_description():
    _cfg = _os.path.join(_os.path.dirname(__file__), "..", "..", "Envs",
                         "Go2Locomotion-PPO", "training", "config.py")
    _cfg = _os.path.abspath(_cfg)
    if not _os.path.exists(_cfg):
        return "(search space description unavailable)"
    _spec = importlib.util.spec_from_file_location("_env_config", _cfg)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.search_space_description()

_search_space = _load_search_space_description()

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


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(space=_search_space)


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

In 2-3 sentences: (1) diagnose why this trial performed as it did (name the most \
likely cause), and (2) state one concrete recommendation for the next trial.

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
Trials run: {n_trials}
Best run: {best}

Write a short final report (5-8 sentences) covering: whether the goal was met, \
what the best configuration was, the key parameter changes that mattered, and any \
remaining limitations. Plain text.
"""
