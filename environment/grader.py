"""
ML Experiment Integrity Auditor — Grader Module.
Implements evidence-grounded scoring with three-layer matching.
"""
import re
from typing import Any


VALID_VIOLATIONS = {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"}


def clean_violation_score(n_false: int, verdict: str) -> float:
    """Score clean episodes with explicit verdict-sensitive credit."""
    if n_false == 0 and verdict == "pass":
        return 1.0
    if n_false == 0 and verdict == "revise":
        return 0.55
    if n_false == 0 and verdict == "reject":
        return 0.20
    if n_false == 1:
        return 0.30
    if n_false == 2:
        return 0.10
    return 0.0


def normalize_text(s: str) -> str:
    """
    Collapse all whitespace sequences to single spaces and strip edges.
    Used for fuzzy evidence matching.
    """
    if not s:
        return ""
    return " ".join(s.split()).lower()


def tokenize(s: str) -> set[str]:
    """Extract lowercase word tokens from string."""
    if not s:
        return set()
    return set(re.findall(r'\b\w+\b', s.lower()))


def evidence_found(quote: str, artifact_content: str) -> bool:
    """
    Three-layer evidence matching:

    Layer 1: Exact substring match
    Layer 2: Whitespace-normalized substring match
    Layer 3: Token overlap >= 80% (for quotes with 3+ tokens)

    Returns True if evidence is found via any layer.
    """
    if not quote or not artifact_content:
        return False

    # Layer 1: Exact match
    if quote in artifact_content:
        return True

    # Layer 2: Whitespace-normalized match
    norm_quote = normalize_text(quote)
    norm_artifact = normalize_text(artifact_content)
    if norm_quote and norm_quote in norm_artifact:
        return True

    # Layer 3: Token overlap match (requires >= 3 tokens, >= 80% overlap)
    quote_tokens = tokenize(quote)
    if len(quote_tokens) >= 3:
        artifact_tokens = tokenize(artifact_content)
        overlap = len(quote_tokens & artifact_tokens)
        if overlap / len(quote_tokens) >= 0.80:
            return True

    return False


def grade_single_flag(
    flag: dict[str, Any],
    ground_truth_violations: list,
    inspected: dict[str, str]
) -> tuple[float, str]:
    """
    Grade a single flag against ground truth.

    Returns (reward, result_label) where result_label is one of:
    - "correct": Correct violation type AND evidence found in artifact (+0.15)
    - "false_positive": Wrong violation type (-0.10)
    - "fabricated_evidence": Correct type but evidence not found (-0.05)
    - "not_inspected": Evidence artifact not yet inspected (-0.10)
    """
    # Extract ground truth violation types
    gt_types = set()
    for v in ground_truth_violations:
        if isinstance(v, dict):
            gt_types.add(v.get("type", v.get("violation_type", "")))
        else:
            gt_types.add(str(v))

    vtype = flag.get("violation_type", "")
    ev_artifact = flag.get("evidence_artifact", "")
    ev_quote = flag.get("evidence_quote", "")

    # Must have inspected the artifact to cite it
    if ev_artifact not in inspected:
        return -0.10, "not_inspected"

    # Check if violation type is in ground truth
    if vtype not in gt_types:
        return -0.10, "false_positive"

    # Check if evidence quote is found in artifact
    artifact_content = inspected.get(ev_artifact, "")
    if not evidence_found(ev_quote, artifact_content):
        return -0.05, "fabricated_evidence"

    return +0.15, "correct"


def grade(
    flags: list[dict],
    ground_truth: dict,
    steps_used: int,
    budget: int,
    verdict: str,
    inspected: dict[str, str]
) -> tuple[float, dict]:
    """
    Compute final episode score.

    Score composition:
    - 80%: Violation detection (precision + recall via flag rewards)
    - 10%: Efficiency (steps remaining / budget)
    - 10%: Correct verdict

    Args:
        flags: List of flags raised by agent
        ground_truth: {"violations": [...], "expected_verdict": "pass"|"revise"|"reject"}
        steps_used: Number of steps taken
        budget: Maximum allowed steps (step_budget)
        verdict: Agent's final verdict
        inspected: Dict mapping artifact names to content strings

    Returns:
        (final_score, breakdown_dict) where final_score is in [0.0, 1.0]
    """
    violations = ground_truth.get("violations", [])
    is_clean = len(violations) == 0
    expected_verdict = ground_truth.get("expected_verdict", "pass")

    # Grade each flag
    flag_results = []
    raw_reward = 0.0

    for flag in flags:
        reward, label = grade_single_flag(flag, violations, inspected)
        raw_reward += reward
        flag_results.append({
            "flag_id": flag.get("flag_id", "?"),
            "violation_type": flag.get("violation_type", "?"),
            "result": label,
            "reward": reward,
        })

    # Compute violation score
    if is_clean:
        # Clean experiments require an explicit correct verdict for full credit.
        n_false = len([fr for fr in flag_results if fr["reward"] < 0])
        violation_score = clean_violation_score(n_false=n_false, verdict=verdict)
    else:
        # Violated experiment: ratio of earned to possible
        max_possible = len(violations) * 0.15
        if max_possible > 0:
            violation_score = max(0.0, min(1.0, raw_reward / max_possible))
        else:
            violation_score = 0.0

    # Efficiency bonus: 0.0 to 0.10
    # More steps remaining = higher bonus
    if budget > 0:
        efficiency_bonus = round(max(0.0, 1.0 - steps_used / budget) * 0.10, 4)
    else:
        efficiency_bonus = 0.0

    # Verdict bonus: 0.10 if correct
    verdict_bonus = 0.10 if verdict == expected_verdict else 0.0

    # Final score: weighted combination
    final_score = round(
        min(1.0, violation_score * 0.80 + efficiency_bonus + verdict_bonus),
        4
    )

    breakdown = {
        "violation_score": round(violation_score, 4),
        "efficiency_bonus": efficiency_bonus,
        "verdict_bonus": verdict_bonus,
        "raw_flag_reward": round(raw_reward, 4),
        "max_possible_flag_reward": round(len(violations) * 0.15, 4),
        "flag_results": flag_results,
    }

    return final_score, breakdown


if __name__ == "__main__":
    assert clean_violation_score(n_false=0, verdict="pass") == 1.0
    assert clean_violation_score(n_false=0, verdict="reject") == 0.20
    assert clean_violation_score(n_false=2, verdict="reject") == 0.10
    print("clean scoring sanity checks passed")
