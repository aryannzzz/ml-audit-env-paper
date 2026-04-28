# 04 Scoring and Reporting Guide

## Scoring principle
Use the same grader and reward structure as model runs.
Do not create a separate manual scoring rule for humans.

## Minimum metrics to report
Per participant and per tier:
- Final score
- Steps used
- Number of flags
- Correct flags
- False positives
- Verdict correctness

Aggregate (paper-ready):
- Mean +- std score by tier
- Mean +- std score overall
- Mean steps by tier
- False positive rate by tier

## Suggested analysis table
- Human baseline vs model baseline (same tiers)
- Rows: Human, GPT-4.1-mini, Qwen72B, Llama70B
- Columns: Easy, Medium, Hard, Overall

## Qualitative error analysis (important)
Tag each failed episode with one dominant failure mode:
- Missed cross-artifact reasoning
- Red-herring false positive
- Evidence quote mismatch
- Ran out of steps before submit
- Wrong final verdict despite correct partial evidence

## Statistical guidance
- If n is small, report as exploratory baseline.
- Always show variance (std) and sample size.
- Keep prompt/instructions fixed across participants.

## Recommended paper phrasing
- "We report an ML-trained human baseline under the identical action schema, step budget, and grader used for LLM agents."
- "Human baseline results are intended as a capability anchor, not as a claim of exhaustive expert consensus."
