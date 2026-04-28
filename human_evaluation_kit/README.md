# Human Evaluation Kit for MLAuditBench

This folder contains a complete, practical protocol to run a human baseline study for MLAuditBench.

Use this if you want ML-literate humans (including your BTech friends) to solve benchmark episodes under the same constraints as LLM agents.

## Who can participate?
- Yes, your BTech friends can be participants.
- Recommended minimum: basic understanding of train/test split, leakage, metrics, and model evaluation.
- In the paper, describe them as `ML-trained human participants` unless they are established domain experts.

## Folder contents
- `01_facilitator_protocol.md`: End-to-end study setup and execution plan.
- `02_participant_instructions.md`: The exact instructions to give participants.
- `03_artifact_packet_spec.md`: Exactly which artifacts to provide, when, and in what format.
- `04_scoring_and_reporting.md`: How to score, aggregate, and report results.
- `templates/consent_script.md`: Lightweight consent script.
- `templates/participant_registry.csv`: Participant metadata template.
- `templates/episode_assignment.csv`: Episode balancing/assignment template.
- `templates/session_log_template.csv`: Per-episode logging sheet.
- `templates/debrief_questions.md`: Post-session qualitative form.
- `scripts/run_human_session.sh`: Simple CLI flow for manual sessions via API.

## Minimum recommended study design
- Participants: 6 to 12
- Episodes per participant: 3 (one per tier: easy, medium, hard)
- Total episodes: 18 to 36
- Constraint matching: same step budgets, same action schema, same scoring function as model runs.

## Quick start
1. Start the environment server.
2. Read and follow `01_facilitator_protocol.md`.
3. Share only `02_participant_instructions.md` with participants.
4. Use `scripts/run_human_session.sh` and `templates/session_log_template.csv` for collection.
5. Aggregate using your paper metrics in `04_scoring_and_reporting.md`.

## Important fairness rule
Humans must use the same information channel as models:
- No hidden ground truth
- No external internet search
- No extra hints from facilitator
- Same inspect/compare/flag/unflag/submit action space
