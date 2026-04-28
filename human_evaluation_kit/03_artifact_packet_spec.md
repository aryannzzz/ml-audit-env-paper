# 03 Artifact Packet Specification

This file answers: "Which artifacts do I have to give participants?"

## Short answer
Give participants exactly what an agent gets:
- Initial observation after `reset`
- Artifact contents only when they call `inspect` or `compare`
- No hidden ground truth

## Per-episode packet (what facilitator prepares)
1. Episode metadata
- `participant_id`
- `task_tier` (easy/medium/hard)
- `seed`
- `step_budget`

2. Initial observation from API reset
- available artifacts list
- task description
- inspected artifacts (initially empty)
- step counters

3. Action output transcripts
- every observation returned after participant actions
- include error messages if action invalid

4. Final episode record
- submitted verdict
- summary text
- steps used
- score breakdown if returned

## Typical artifact names in episodes
Participants may request these (subset varies by episode):
- `dataset_info`
- `preprocessing`
- `split_config`
- `model_config`
- `training_logs`
- `eval_report`
- `experiment_notes`
- `validation_strategy`
- `run_history`
- (optional in some variants) `feature_engineering`

## What NOT to give participants
- Ground-truth violation labels
- Internal generator parameters
- Any hidden clean/violation toggle
- Prior participant answers
- Extra explanatory hints

## Delivery modes
- Preferred: live API-driven session (facilitator executes actions)
- Acceptable: pre-exported artifact JSON packets, but only reveal artifact text when participant requests inspect/compare.

## Consistency requirement
Humans must operate under exactly the same information and budget constraints as model baselines.
