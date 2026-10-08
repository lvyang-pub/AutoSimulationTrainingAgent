"""Every prompt the agent sends to an LLM, in one place.

This module is the *only* file a task-specific copy of the agent is expected to
change. The loop in `main.py` imports these names and formats them with a fixed
set of keyword arguments; a copy redefines the same names (task-tuned wording)
and leaves every other file untouched.

Placeholder contract (the names and kwargs are stable -- main.py relies on them):

* SYSTEM_PROMPT         -- {tools}, {env_context}
* INITIAL_PROMPT        -- {goal}, {budget}, {lessons}, {start_config},
                           {default_timesteps}, {env_context}, {context}
* FINAL_PROMPT          -- {steps}, {stopped}, {goal}, {n_trials}, {best}
* REFLECTION_PROMPT     -- {config}, {metrics}, {lessons}
* SHORT_MEMORY_PROMPT   -- {trials}
* LONG_MEMORY_PROMPT    -- {lessons}

The defaults below name no task. A real task overrides them.
"""
from __future__ import annotations


SYSTEM_PROMPT = """\
You are an autonomous agent that works toward a goal by running tools.

You operate in a loop: you choose one or more tools to call, you receive their
results as observations, and you repeat until you decide you are done. Reason
briefly before each action.

Rules:
- Call tools using the provided function interface. Call one or more per turn.
- After each result, decide whether the goal is met. If so, call the finish
  tool rather than spending more effort. If not, adjust and try again.
- Learn from every result. Do not repeat an action that already failed.

Environment context:
{env_context}

Available tools:
{tools}
"""


INITIAL_PROMPT = """\
{context}Begin. Think briefly, then call one or more tools.

Goal: {goal}
{budget}Suggested starting configuration (you may change it): {start_config}
Each default run uses {default_timesteps} steps.

Lessons from previous sessions:
{lessons}
"""


FINAL_PROMPT = """\
Write a short closing report for this session.

Goal: {goal}
Steps taken: {steps}
Trials run: {n_trials}
Best run: {best}
Stopped because: {stopped}

In 3-5 sentences, summarize what you did and what you would do next.
"""


REFLECTION_PROMPT = """\
A trial just finished.

Trial config:
{config}

Trial result:
{metrics}

Prior lessons (may or may not apply):
{lessons}

In 2-3 sentences (under 280 characters): (1) diagnose why this trial performed
as it did, and (2) give one concrete recommendation for the next trial. Name a
parameter and a direction. Answer in plain text, no JSON.
"""


SHORT_MEMORY_PROMPT = """\
Compress the trials so far into a short running summary an agent can keep in a
bounded context window. Keep the best configuration, the direction of travel in
the key metrics, and which parameter changes helped or hurt. Be specific and
terse.

Trials:
{trials}
"""


LONG_MEMORY_PROMPT = """\
Compress the accumulated lessons from previous sessions into a small set of
durable, reusable rules. Merge duplicates, drop anything that was contradicted
by later evidence, and keep only what would change a future decision.

Lessons:
{lessons}
"""
