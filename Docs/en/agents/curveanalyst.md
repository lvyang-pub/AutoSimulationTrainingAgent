# CurveAnalyst

## Overview

A vision-model-based training curve analysis agent that receives training curve images and returns a textual analysis of the training trend.

## Agent Design

### Main Flow

Each invocation makes a single-turn vision model call: the image is base64-encoded and sent together with an analysis question to DeepSeek Vision, which returns a textual description of the training trend. Triggered by the `on_trial_done` callback registered in the schedular after each trial completes, the analysis is injected into PPOTuner's main loop context as a supplementary observation.
