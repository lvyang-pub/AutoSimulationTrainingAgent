# Architecture Overview

## Design Goals

> 🚧 TODO: a general framework extensible to multi-Agent / multi-Env / multi-Task;
> Agent and Env decoupled, so adding a task does not touch the core loop.

## Component Diagram

> 🚧 TODO: responsibilities and call relationships per layer. The Agent calls the Env
> through the Harness; the Task's schedular orchestrates them.

![Architecture](../../assets/arch_diagram.png)

## Execution Flow

> 🚧 TODO: the control flow of one full tuning run, from task start to goal reached.

![Flow](../../assets/flow_diagram.png)

## Data Flow (one trial)

> 🚧 TODO: the 8-step process (from the architecture doc §4) —
> LLM emits run_training → config.normalize validates → training writes to disk →
> summary returns → CurveAnalyst analyses the curve → reflection writes long-term memory →
> short-term memory appends → budget / goal_met check.
