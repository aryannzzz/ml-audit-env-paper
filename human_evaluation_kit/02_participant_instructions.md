# 02 Participant Instructions (Share This With Participants)

## Your task
You are auditing synthetic ML experiment artifacts for methodology violations.
For each episode, your goal is to inspect relevant artifacts, flag violations with textual evidence, and submit a final verdict.

## Allowed actions
1. `inspect(artifact)`
- Read one artifact fully.

2. `compare(artifact_a, artifact_b)`
- Read two artifacts side by side.

3. `flag(violation_type, evidence_artifact, evidence_quote, severity)`
- Raise a violation with exact evidence quote.

4. `unflag(flag_id)`
- Remove a previously raised flag.

5. `submit(verdict, summary)`
- End episode with `pass`, `revise`, or `reject`.

## Violation types
- V1: Preprocessing Leakage
- V2: Temporal Shuffle
- V3: Target Leakage
- V4: Train/Test Overlap
- V5: Cherry-Picking
- V6: Metric Shopping
- V7: Entity Leakage
- V8: Multi-Test Leakage

## Episode constraints
- You have a strict step budget:
  - Easy: 8 steps
  - Medium: 12 steps
  - Hard: 18 steps
- You may submit early.
- If budget runs out, episode ends.

## Grounding rule
Every flag must include evidence text that appears in the cited artifact.
Unsupported claims reduce score.

## Important restrictions
- Do not use internet or external notes.
- Do not ask facilitator for hints about correctness.
- Use only information shown through the official episode interface.

## Practical strategy (recommended)
- Start with likely high-yield artifacts (`preprocessing`, `split_config`, `model_config`).
- Use `compare` for cross-document checks (`validation_strategy` vs `eval_report`, `run_history` vs `experiment_notes`).
- Avoid over-flagging; clean episodes exist.
- If uncertain, submit with a conservative summary rather than guessing many unsupported flags.
