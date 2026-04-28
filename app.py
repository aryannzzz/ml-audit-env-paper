"""
FastAPI application for ML Experiment Integrity Auditor.

Endpoints:
- GET  /health          — Health check
- GET  /scoring         — Scoring formula and breakdown
- GET  /experiment/{task} — Sample experiment viewer
- POST /reset           — Start new episode
- POST /step            — Execute action
- GET  /state           — Get current episode state
- GET  /tasks           — List available tasks
- GET  /baseline        — Pre-computed baseline results
- POST /grader          — Direct grader invocation
"""
import json
import random
import threading
from pathlib import Path as FilePath
from copy import deepcopy
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import ValidationError

from environment.env import MLAuditEnv
from environment.models import Action, GraderRequest, GraderResponse
from environment.grader import grade, grade_single_flag
from environment.generator import POOL, get_pool_stats


# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ML Experiment Integrity Auditor",
    description="OpenEnv RL environment for auditing ML experiments for data leakage, "
                "methodology violations, and result manipulation.",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE: Single-session mode. _env is a process-level singleton protected
# by _env_lock. This is sufficient for sequential competition evaluation.
# For concurrent multi-agent evaluation (v2.0), replace with session-keyed
# env pool: Dict[str, MLAuditEnv].
_env: MLAuditEnv | None = None
_env_lock = threading.Lock()
_UI_FILE = FilePath(__file__).resolve().parent / "ui" / "audit_console.html"

# Pre-computed baseline scores from deterministic GPT-4.1-mini runs (seed=42)
# These are static reference values for quick comparisons.
_BASELINE_SCORES = {
    "easy": {
        "score": 0.95,
        "steps": 4,
        "violations_found": 1,
        "false_positives": 0,
        "verdict": "reject",
    },
    "medium": {
        "score": 0.95,
        "steps": 6,
        "violations_found": 2,
        "false_positives": 0,
        "verdict": "reject",
    },
    "hard": {
        "score": 0.3979,
        "steps": 11,
        "violations_found": 1,
        "false_positives": 0,
        "verdict": "reject",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """
    Human-friendly dashboard for manual episode inspection.
    """
    if not _UI_FILE.exists():
        raise HTTPException(status_code=500, detail=f"UI file missing: {_UI_FILE}")
    return FileResponse(str(_UI_FILE))


@app.get("/ui", include_in_schema=False)
def ui() -> FileResponse:
    """
    Alias for the dashboard route.
    """
    return home()


@app.get("/health")
def health() -> dict[str, Any]:
    """
    Health check endpoint.
    Returns environment status and pool statistics.
    """
    stats = get_pool_stats()
    return {
        "status": "ok",
        "environment": "ml-audit-bench",
        "version": "1.0.0",
        "pool_size": stats["total"],
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/scoring")
def scoring() -> dict[str, Any]:
    """
    Return the complete scoring formula and breakdown.
    Helps judges and developers understand how episodes are graded.
    """
    return {
        "formula": "final_score = violation_score × 0.80 + efficiency_bonus × 0.10 + verdict_bonus × 0.10",
        "components": {
            "violation_score": {
                "description": "Fraction of true violations correctly flagged with valid evidence",
                "formula": "correct_flags / total_violations",
                "range": [0.0, 1.0],
            },
            "efficiency_bonus": {
                "description": "Reward for completing the audit quickly",
                "formula": "(1 - steps_used / budget)",
                "range": [0.0, 1.0],
            },
            "verdict_bonus": {
                "description": "Bonus for correct final verdict (pass/reject)",
                "value": "0.10 if correct, 0.0 otherwise",
                "range": [0.0, 0.10],
            },
        },
        "flag_rewards": {
            "correct_with_evidence": "+0.15 (correct violation type AND valid evidence quote)",
            "correct_without_evidence": "-0.05 (correct type but fabricated/missing evidence)",
            "false_positive": "-0.10 (wrong violation type)",
            "uninspected_artifact": "-0.10 (evidence from artifact never inspected)",
        },
        "evidence_matching": {
            "layer_1": "Exact substring match",
            "layer_2": "Whitespace-normalized match",
            "layer_3": "Token overlap ≥80% for quotes with 3+ tokens",
        },
        "anti_gaming": {
            "clean_experiment_ratio": "50% of experiments have NO violations",
            "red_herrings": "Suspicious patterns that are NOT violations",
            "evidence_requirement": "Flags without valid quotes are penalized",
        },
    }


@app.get("/experiment/{task}")
def get_sample_experiment(
    task: str = Path(description="Task difficulty: easy | medium | hard"),
    seed: int = Query(default=42, description="Seed for deterministic experiment selection"),
) -> dict[str, Any]:
    """
    View a sample experiment structure (without ground truth).
    Useful for understanding what agents see during an episode.

    Returns experiment metadata and available artifacts for inspection.
    """
    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail=f"task must be 'easy', 'medium', or 'hard', got '{task}'"
        )

    if not POOL.get(task):
        raise HTTPException(status_code=500, detail=f"No experiments in pool for task '{task}'")

    # Deterministic selection
    rng = random.Random(seed)
    exp = deepcopy(rng.choice(POOL[task]))

    # Remove ground truth (agent shouldn't see this)
    ground_truth = exp.pop("ground_truth", {})
    is_clean = len(ground_truth.get("violations", [])) == 0

    # Build artifact previews (truncated)
    artifact_previews = {}
    for artifact_name in exp.get("available_artifacts", []):
        content = exp.get(artifact_name)
        if content is not None:
            if isinstance(content, str):
                preview = content[:500] + "..." if len(content) > 500 else content
            else:
                full = json.dumps(content, indent=2)
                preview = full[:500] + "..." if len(full) > 500 else full
            artifact_previews[artifact_name] = preview

    return {
        "experiment_id": exp.get("experiment_id"),
        "task": task,
        "archetype": exp.get("archetype"),
        "dataset_type": exp.get("dataset_info", {}).get("dataset_type", "unknown"),
        "task_description": exp.get("task_description"),
        "available_artifacts": exp.get("available_artifacts", []),
        "artifact_previews": artifact_previews,
        "step_budget": {"easy": 8, "medium": 12, "hard": 18}.get(task, 8),
        "hint": "Use /reset to start an episode, then /step with inspect actions to read full artifacts",
        "_meta": {
            "is_clean": is_clean,
            "num_violations": len(ground_truth.get("violations", [])),
            "note": "Ground truth hidden. This preview helps judges understand the environment structure.",
        },
    }


@app.post("/reset")
def reset(task: str = Query(default="easy", description="Task difficulty: easy | medium | hard"), seed: int | None = Query(default=None, description="Optional seed for deterministic experiment selection")) -> dict[str, Any]:
    """
    Start a new episode.

    Args:
        task: Task difficulty level
        seed: Optional random seed for deterministic experiment selection

    Returns:
        {
            "observation": {...},
            "reward": 0.0,
            "done": false,
            "info": {}
        }
    """
    global _env

    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail=f"task must be 'easy', 'medium', or 'hard', got '{task}'"
        )

    with _env_lock:
        _env = MLAuditEnv(task=task)
        obs = _env.reset(seed=seed)

    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {},
    }


@app.post("/step")
def step(body: dict[str, Any]) -> dict[str, Any]:
    """
    Execute an action and advance the episode.

    Request body should contain an "action" dict with fields:
    - type: "inspect" | "compare" | "flag" | "unflag" | "submit"
    - Additional fields depending on action type

    Returns:
        {
            "observation": {...},
            "reward": float,
            "done": bool,
            "info": {"score": float} if done else {}
        }
    """
    global _env

    # Extract action from body
    action_dict = body.get("action", body)

    try:
        action = Action(**action_dict)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action schema: {e}"
        )

    with _env_lock:
        if _env is None:
            raise HTTPException(
                status_code=400,
                detail="No active episode. Call /reset first."
            )

        try:
            obs, reward, done, info = _env.step(action)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    """
    Get current episode state for checkpointing/debugging.
    Does not include ground truth violations.
    """
    global _env

    with _env_lock:
        if _env is None:
            raise HTTPException(
                status_code=400,
                detail="No active episode. Call /reset first."
            )

        return _env.state().model_dump()


@app.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    """
    List available tasks with metadata.
    """
    return [
        {
            "id": "easy",
            "description": "Single violation, 8-step budget. No red herrings.",
            "difficulty": 1,
            "max_steps": 8,
            "expected_score_range": "0.82-0.95",
        },
        {
            "id": "medium",
            "description": "Two violations, cross-artifact reasoning required. 12-step budget.",
            "difficulty": 2,
            "max_steps": 12,
            "expected_score_range": "0.70-0.98",
        },
        {
            "id": "hard",
            "description": "Three violations + red herrings, compound violations. 18-step budget.",
            "difficulty": 3,
            "max_steps": 18,
            "expected_score_range": "0.25-0.42",
        },
    ]


@app.get("/baseline")
def baseline(task: str = Query(default="easy", description="Task difficulty")) -> dict[str, Any]:
    """
    Return pre-computed baseline results.
    This is NOT a live LLM call — returns static reference results.
    Used for automated validation and comparison.
    """
    if task not in _BASELINE_SCORES:
        raise HTTPException(
            status_code=400,
            detail=f"task must be 'easy', 'medium', or 'hard', got '{task}'"
        )

    result = _BASELINE_SCORES[task]
    return {
        "task": task,
        "agent_type": "gpt_4_1_mini_baseline",
        "agent_description": "Reference run from inference.py using GPT-4.1-mini with seed=42",
        **result,
        "note": "Pre-computed deterministic run. Use inference.py for live evaluation.",
    }


@app.post("/close")
def close() -> dict[str, Any]:
    """
    Close the current episode and release resources.
    Optional endpoint for clean shutdown between episodes.
    """
    global _env
    if _env is not None:
        _env.close()
        _env = None
    return {"status": "closed"}


@app.post("/grader")
def run_grader(req: GraderRequest) -> dict[str, Any]:
    """
    Run grader directly on provided flags.
    Used for external validation and testing.

    Request body:
        {
            "task": "easy",
            "experiment_id": "optional_exp_id",
            "flags": [...],
            "verdict": "reject",
            "steps_used": 5
        }
    """
    if req.task not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail=f"task must be 'easy', 'medium', or 'hard'"
        )

    # Find the correct experiment
    pool_key = req.task
    if not POOL.get(pool_key):
        raise HTTPException(status_code=500, detail=f"No experiments in pool for task '{req.task}'")

    # If experiment_id provided, find it; otherwise use first
    if req.experiment_id:
        exp = next((e for e in POOL[pool_key] if e.get("experiment_id") == req.experiment_id), None)
        if exp is None:
            raise HTTPException(status_code=404, detail=f"Experiment '{req.experiment_id}' not found in task '{req.task}'")
    else:
        exp = POOL[pool_key][0]

    gt = exp["ground_truth"]

    # Get budget for task
    budgets = {"easy": 8, "medium": 12, "hard": 18}
    budget = budgets.get(req.task, 8)

    # Build inspected dict with all artifacts (for grader testing)
    # In a real episode, only inspected artifacts would be available
    inspected = {}
    for artifact_name in exp.get("available_artifacts", []):
        key = artifact_name
        value = exp.get(key)
        if value is not None:
            if isinstance(value, str):
                inspected[artifact_name] = value
            else:
                inspected[artifact_name] = json.dumps(value, indent=2)

    # Run grader
    score, breakdown = grade(
        flags=req.flags,
        ground_truth=gt,
        steps_used=req.steps_used,
        budget=budget,
        verdict=req.verdict,
        inspected=inspected,
    )

    # Build per-flag results
    flag_results = []
    for f in req.flags:
        r, label = grade_single_flag(f, gt["violations"], inspected)
        flag_results.append({
            "flag_id": f.get("flag_id", "?"),
            "violation_type": f.get("violation_type", "?"),
            "result": label,
            "reward": r,
        })

    return {
        "score": score,
        "breakdown": breakdown,
        "flag_results": flag_results,
        "ground_truth_violations": len(gt.get("violations", [])),
        "expected_verdict": gt.get("expected_verdict", "pass"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)