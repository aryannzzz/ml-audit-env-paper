---
title: MLAuditBench
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - rl-environment
  - ml-benchmark
  - reproducibility
---

# ML Experiment Integrity Auditor

**Machine learning research faces a reproducibility crisis.** Kapoor & Narayanan (2023) found data leakage in **294 papers across 17 scientific fields**. This environment is the **first RL training ground for automated ML experiment auditing** — where AI agents learn to detect violations by reading experiment artifacts, citing specific evidence, and making grounded judgments.

## Overview

Prior tools for detecting ML methodology issues rely on static analysis or post-hoc checklists. This project frames leakage detection as an **interactive agent task** with sequential decision-making, evidence-grounded actions, and step-constrained scoring — creating a benchmark where agents must reason like human reviewers.

### Key Features

- **8 Violation Types**: Preprocessing leakage (V1), temporal shuffle (V2), target leakage (V3), train/test overlap (V4), cherry-picking (V5), metric shopping (V6), entity leakage (V7), multi-test leakage (V8)
- **Evidence-Grounded Reasoning**: Agents must cite exact quotes from artifacts when flagging violations
- **Progressive Difficulty**: Easy (1 violation), Medium (2 violations + red herrings), Hard (3 violations + 2 red herrings). Hard tasks additionally include compound violations where two distinct violation types co-occur and require sequential cross-artifact reasoning.
- **Anti-Gaming Design**: 50% runtime probability of drawing a clean experiment with no violations
- **56 Experiments v1.0 (50 standard + 6 compound hard-tier)** across 4 ML archetypes (tabular classification, time-series regression, multi-class classification, survival analysis). **v1.1 extends to 62 experiments** by adding 6 text classification (NLP) scenarios; the v1.0 pool is the default for backward-compatible seeded evaluation.

## Quick Start

### Docker

```bash
docker build -t ml-audit-bench .
docker run -p 7860:7860 ml-audit-bench
# Server runs at http://localhost:7860
```

### Local Python

```bash
pip install -r requirements.txt
uvicorn app:app --port 7860
```

### Verify It Works

```bash
curl http://localhost:7860/health
# Expected: {"status": "ok", "environment": "ml-audit-bench", "pool_size": 56}

curl -X POST "http://localhost:7860/reset?task=easy&seed=42"
# Returns initial observation JSON
```

### Run the Baseline Agent

```bash
export OPENAI_API_KEY=your_key_here
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export ENV_URL=http://localhost:7860

python inference.py
# Emits [START]/[STEP]/[END] formatted output
```

### Live Demo

```bash
curl https://huggingface.co/spaces/mlauditbench-anon/ml-audit-env/health
```

## Baseline Scores

All scores are mean ± std over seeds 42–51 (n=10). Primary results (seeds 45–46) use
the final `inference.py`; see paper Table 1 and supplementary Table S1 for per-seed breakdown.

| Agent | Easy | Medium | Hard | Average |
|-------|------|--------|------|------|
| Random (lower bound) | ~0.16 | ~0.16 | ~0.15 | ~0.16 |
| GPT-4.1-mini (fast) | 0.530 ± 0.390 | 0.571 ± 0.340 | 0.629 ± 0.352 | 0.577 |
| GPT-4.1 (balanced) | 0.592 ± 0.407 | 0.688 ± 0.365 | 0.662 ± 0.378 | 0.647 |
| o4-mini (reasoning) | 0.604 ± 0.396 | 0.626 ± 0.395 | 0.528 ± 0.449 | 0.586 |
| Qwen2.5-72B (open, easy only) | 0.530 ± 0.523 | — | — | — |

## Adversarial robustness

A random agent that raises flags with no reasoning scores **~0.16** on average across all
difficulty tiers (compared to **0.519** for GPT-4.1-mini, averaged over seeds 42–44).
The benchmark's discriminability was validated using adversarial baselines — a
pattern-matching regex agent, a keyword-counter agent, and a pure random agent — none
of which use LLM inference.

For hard tasks, explicitly using `compare()` for `run_history` vs `experiment_notes` (V5)
and `validation_strategy` vs `eval_report` (V6) is mandatory for stable performance.

### Testing with Open Models

The benchmark supports open-weight models via the HuggingFace Inference Router. Use the
multi-provider evaluation harness for token rotation and automated result collection:

```bash
# Run Qwen2.5-72B and Llama-3.1-70B (requires HF_TOKEN_1/2/3 in .env)
python scripts/run_hf_eval.py --models hf_qwen25 hf_llama --seeds 45 46

# Run all registered models including reasoning models
python scripts/run_hf_eval.py --models hf_qwen25 hf_llama hf_qwq hf_deepseek --seeds 45 46
```

Or run a single model directly:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export OPENAI_API_KEY="$HF_TOKEN"

python inference.py
```

**Note**: The system prompt is designed to work with various LLM architectures. If open models struggle with JSON output format, the `parse_action()` function handles markdown code fences and extracts JSON from mixed text.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/scoring` | GET | Scoring formula and breakdown |
| `/experiment/{task}` | GET | Sample experiment viewer (no ground truth) |
| `/reset` | POST | Start new episode (params: task, seed) |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/close` | POST | Close episode |
| `/tasks` | GET | List available tasks |
| `/baseline` | GET | Pre-computed baseline results |
| `/grader` | POST | Direct grader invocation |

## Action Space

**inspect** — Read a single artifact.
```json
{"type": "inspect", "artifact": "preprocessing"}
```

**compare** — Read two artifacts side-by-side.
```json
{"type": "compare", "artifact_a": "validation_strategy", "artifact_b": "eval_report"}
```

**flag** — Raise a violation with evidence.
```json
{"type": "flag", "violation_type": "V1", "evidence_artifact": "preprocessing", "evidence_quote": "scaler.fit_transform(X_all)", "severity": "high"}
```

**unflag** — Self-correct by removing a flag.
```json
{"type": "unflag", "flag_id": "f0"}
```

**submit** — End episode with verdict.
```json
{"type": "submit", "verdict": "reject", "summary": "Found V1 preprocessing leakage"}
```

## Violation Types

| ID | Name | Severity | Detection Pattern |
|----|------|----------|-------------------|
| V1 | Preprocessing Leakage | High | `fit_transform` on full data before split |
| V2 | Temporal Shuffle | High | Shuffled split on timeseries data |
| V3 | Target Leakage | High | Target column in feature list |
| V4 | Train/Test Overlap | High | Overlapping IDs in train/test samples |
| V5 | Cherry-Picking | Medium | Multiple runs, only best reported |
| V6 | Metric Shopping | Medium | Many metrics tracked, one reported |
| V7 | Entity Leakage | High | Entity-unaware splitting of grouped data |
| V8 | Multi-Test Leakage | High | Test set used for HPO and evaluation |

## Scoring

```
final_score = violation_score × 0.80 + efficiency_bonus × 0.10 + verdict_bonus × 0.10
```

- **violation_score**: fraction of true violations correctly flagged with valid evidence
- **efficiency_bonus**: `1 - steps_used / budget`
- **verdict_bonus**: 1.0 if correct verdict, 0.0 otherwise

### Flag Rewards
- **+0.15**: Correct violation type AND evidence quote found in artifact
- **-0.05**: Correct type but fabricated/missing evidence
- **-0.10**: Wrong violation type (false positive)

### Evidence Matching (3-layer)
1. Exact substring match
2. Whitespace-normalized match
3. Token overlap ≥80% (quotes with 3+ tokens)

## Tasks

| Task | Violations | Budget | Red Herrings | Pool Size |
|------|------------|--------|--------------|-----------|
| Easy | 1 | 8 steps | 0 | 10 + 20 clean |
| Medium | 2 | 12 steps | 1-2 | 10 + 20 clean |
| Hard | 3 | 18 steps | 2 | 16 + 20 clean |

Hard tasks additionally include compound violations — episodes where two distinct violation
types co-occur in the same experiment, requiring sequential cross-artifact reasoning to
detect both.

## Project Structure

```
ml-audit-env/
├── environment/
│   ├── env.py           # Core RL environment (reset/step/state)
│   ├── models.py        # Pydantic v2 typed models
│   ├── grader.py        # Evidence matching + scoring
│   └── generator.py     # Experiment pool builder (56 experiments: 50 standard + 6 compound)
├── experiments/templates/
│   └── *.json           # 4 base templates
├── tests/
│   └── test_grader.py   # Unit tests
├── paper/
│   ├── main.tex         # NeurIPS paper
│   ├── supplement.tex   # Technical supplement
│   └── references.bib   # Bibliography
├── app.py               # FastAPI HTTP server
├── inference.py          # Baseline inference (OpenAI client)
├── openenv.yaml         # OpenEnv manifest
├── Dockerfile           # Container definition
├── requirements.txt     # Dependencies
├── validate.sh          # Pre-submission validator
├── croissant.json       # ML metadata
└── README.md
```

## Research Background

- Kapoor & Narayanan (2023): Systematic audit — 294 papers with leakage across 17 fields
- Yang et al. (2022): Static detection of leakage in Jupyter notebooks (~30% of 100K notebooks affected)
- Lones (2021): Practical taxonomy of common ML research pitfalls
- Drobnjaković et al. (2024): NBLyzer — abstract interpretation for leakage detection

## Adding New Violation Types

1. Create `inject_V9(exp)` in `environment/generator.py`
2. Add `"V9"` to `VALID_VIOLATION_TYPES` in `environment/models.py`
3. Add `"V9"` to `VALID_VIOLATIONS` in `environment/grader.py`
4. Add experiments using V9 to `_build_pool()`
5. Add V9 to `openenv.yaml` violation taxonomy
6. Write tests in `tests/test_grader.py`

## Citation

```bibtex
@inproceedings{anonymous2026mlauditbench,
  title={{MLAuditBench}: An Interactive {RL} Environment for Evaluating {LLM} Agents on {ML} Experiment Integrity Auditing},
  author={Anonymous Authors},
  booktitle={NeurIPS 2026 Evaluations \& Datasets Track},
  year={2026}
}
```

## License

CC BY 4.0 — Anonymous Authors
