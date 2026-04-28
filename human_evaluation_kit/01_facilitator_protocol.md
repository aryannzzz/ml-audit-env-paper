# 01 Facilitator Protocol

## Objective
Measure human baseline performance on MLAuditBench using the same constraints as LLM agents.

## What you need before running
- Running API server (default `http://localhost:7860`)
- This kit folder
- A spreadsheet app for CSV templates
- 60-90 minutes per participant session (including briefing/debrief)

## Participant eligibility (recommended)
- Has completed at least one ML course/project
- Understands leakage, validation, and metrics
- Comfortable reading JSON/text artifacts

## Session structure
1. Consent and briefing (5 min)
2. Warmup (optional 1 easy practice episode, not scored)
3. Scored episodes (easy + medium + hard) (30-45 min)
4. Debrief questionnaire (5-10 min)

## Experimental controls (must keep fixed)
- Same step budget by tier: easy=8, medium=12, hard=18
- Same allowed actions: inspect, compare, flag, unflag, submit
- Same evidence requirement: quote must be grounded in viewed artifact text
- No assistance during the scored phase
- No external tools/internet

## Assignment strategy
- Use `templates/episode_assignment.csv`.
- Balance across tiers and seeds so participants do not all see identical episodes.
- Suggested seed set: 42, 43, 44.

## Running a scored episode
1. Call reset: `POST /reset?task=<tier>&seed=<seed>`.
2. Share returned observation with participant.
3. Participant declares each action.
4. Facilitator executes action through API and returns observation.
5. Repeat until participant submits or budget is exhausted.
6. Record final score, steps, flags, and verdict in `session_log_template.csv`.

## Data quality checks
- Ensure each scored episode has:
  - participant_id
  - tier
  - seed
  - steps_used
  - final_score
  - verdict_submitted
  - flags_count
  - correct_flags (if available)
  - false_positives (if available)
- Mark protocol violations (`used internet`, `facilitator hinted`, etc.).

## Recommended sample size for paper-ready reporting
- Minimum publishable internal baseline: 18 episodes total (6 participants x 3 tiers)
- Better: 30+ episodes total for more stable std estimates.

## Ethics / institutional note
- If this is coursework-style low-risk benchmarking with synthetic data, it is often exempt.
- Still document voluntary participation and anonymization.
- For submission quality, confirm local policy with your institution.
