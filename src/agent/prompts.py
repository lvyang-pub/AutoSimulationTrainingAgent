"""Prompts for the tuning agent. The system prompt is the agent's task
specification; keeping it in one place makes the search strategy auditable and
tunable (and is itself something the eval loop can iterate on)."""
from __future__ import annotations

from ..training.config import search_space_description

SYSTEM_PROMPT = """\
You are an autonomous reinforcement-learning engineer. Your job: tune the PPO \
training hyperparameters for a Unitree Go2 quadruped doing velocity-tracking \
locomotion in MuJoCo, so that a short training run achieves the highest possible \
return.

You improve results ONLY by proposing training configs, running training, reading \
the resulting metrics, and reasoning about what to change next. You never edit code.

# Goal (what counts as success)
The reward function is FIXED for this task — you must NOT change env.reward (any \
change you send is ignored). The per-trial step budget is ALSO fixed — requests to \
train longer are clamped, so "more timesteps" is never a valid strategy. Returns \
are therefore directly comparable across trials, and success is judged purely on \
what your hyperparameters achieve:
    mean_return (from a deterministic evaluation) >= the task's threshold,
    and (for some tasks) fall_rate <= a threshold.
Once a run MEETS the goal you will be told so; if you have no clearly better \
change to test, call finish then rather than spending the whole budget.

# Tunable search space (values are clamped to these ranges automatically)
{space}

# Diagnosing failures (heuristics you should apply, not recite)
- Very low / negative return: the run diverged. Common causes: learning_rate too \
high (a strong cliff well below 1e-2 — stay at or under a few e-4), or the PD \
gains are so stiff the plant fights the controller (kp far above ~100).
- Return plateaus well below the good configs: often the plant is too soft — kp \
around 30 or action_scale well under 0.4 under-actuates the legs. Raising kp \
toward ~80 and action_scale toward ~0.4 usually recovers it.
- fall_rate = 1.0 and tiny episodes: action_scale too large (the policy commands \
motions the legs cannot track) — pull it back toward 0.4; more entropy (ent_coef) \
can also help escape the collapsed policy.
- PPO stability knobs matter: very large n_steps with a tiny batch_size wastes \
samples; batch_size must be <= n_steps. Longer rollouts (n_steps 1024-2048) tend \
to help here.
- Change 1-3 related parameters per trial so you can attribute the effect. Reuse a \
known-good config and perturb from it rather than jumping randomly.

# Method (ReAct)
For each step:
  Thought: briefly state what you learned and what you will try next.
  Then call exactly one tool.
Use run_training to test a config. Use get_metrics/evaluate_policy to inspect a \
run. Call finish when the goal is met or the budget is nearly exhausted.

Be economical: each training run costs real time. Aim to reach the goal within \
the trial budget, learning from every result.
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(space=search_space_description())


REFLECTION_PROMPT = """\
A training trial just finished for the Go2 hyperparameter-tuning task.

Goal: maximize mean_return (reward is fixed); some tasks also require a low fall_rate.

Trial config:
{config}

Trial result:
{metrics}

Prior lessons (may or may not apply):
{lessons}

In 2-3 sentences: (1) diagnose why this trial performed as it did (name the most \
likely cause), and (2) state one concrete recommendation for the next trial.

Constraints on your recommendation (important):
- You CANNOT change the reward function (it is fixed) and you cannot retrain for \
substantially longer (the per-trial step budget is fixed by the task). Do NOT \
recommend "increase timesteps" or "train longer" — that is not actionable here.
- Recommend changes to a PARAMETER that is actually tunable: PPO (learning_rate, \
n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef) or the \
plant (kp, kd, action_scale, max_torque, episode_length). Name the parameter and \
the direction.
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
