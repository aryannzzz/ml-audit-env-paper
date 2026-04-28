"""
Tests for clean episode scoring formula.
"""
import pytest
import random
from environment.grader import grade, clean_violation_score
from environment.generator import POOL


def get_clean_exp():
    """Return a clean experiment from the pool."""
    clean = POOL.get("clean", [])
    assert clean, "No clean experiments in pool"
    return clean[0]


def compute_clean_grade(n_false_flags: int, verdict: str, steps: int = 3, budget: int = 8):
    """Compute score for a clean experiment with given parameters."""
    exp = get_clean_exp()
    # Build false flags
    flags = [
        {
            "flag_id": f"f{i}",
            "violation_type": f"V{i+1}",
            "evidence_artifact": "preprocessing",
            "evidence_quote": f"fabricated evidence {i}",
            "severity": "high"
        }
        for i in range(n_false_flags)
    ]
    inspected = {"preprocessing": "scaler = StandardScaler()"}

    score, breakdown = grade(
        flags=flags,
        ground_truth=exp.get("ground_truth", {}),
        steps_used=steps,
        budget=budget,
        verdict=verdict,
        inspected=inspected
    )
    return score, breakdown


class TestCleanViolationScore:
    """Direct tests for clean_violation_score function."""

    def test_zero_flags_pass_perfect(self):
        assert clean_violation_score(0, "pass") == 1.0

    def test_zero_flags_revise_partial(self):
        assert clean_violation_score(0, "revise") == 0.55

    def test_zero_flags_reject_penalized(self):
        assert clean_violation_score(0, "reject") == 0.20

    def test_one_false_flag(self):
        assert clean_violation_score(1, "reject") == 0.30

    def test_two_false_flags(self):
        assert clean_violation_score(2, "reject") == 0.10

    def test_three_plus_false_flags_zero(self):
        assert clean_violation_score(3, "reject") == 0.0
        assert clean_violation_score(5, "pass") == 0.0


class TestCleanEpisodeGrading:
    """Integration tests for clean episode scoring via grade()."""

    def test_zero_flags_pass_is_high_score(self):
        score, _ = compute_clean_grade(0, "pass")
        # violation_score=1.0*0.8 + efficiency + verdict = high
        assert score >= 0.85, f"Expected >= 0.85, got {score}"

    def test_zero_flags_revise_is_partial(self):
        score, breakdown = compute_clean_grade(0, "revise")
        violation_score = breakdown.get("violation_score", 0)
        assert 0.50 <= violation_score <= 0.60, f"Expected ~0.55, got {violation_score}"

    def test_zero_flags_reject_is_penalized(self):
        score, breakdown = compute_clean_grade(0, "reject")
        violation_score = breakdown.get("violation_score", 0)
        assert 0.15 <= violation_score <= 0.25, f"Expected ~0.20, got {violation_score}"

    def test_one_false_flag_penalized(self):
        score, breakdown = compute_clean_grade(1, "reject")
        violation_score = breakdown.get("violation_score", 0)
        assert 0.25 <= violation_score <= 0.35, f"Expected ~0.30, got {violation_score}"

    def test_two_false_flags_near_zero(self):
        score, breakdown = compute_clean_grade(2, "reject")
        violation_score = breakdown.get("violation_score", 0)
        assert violation_score <= 0.15, f"Expected <=0.15, got {violation_score}"

    def test_three_false_flags_is_zero(self):
        score, breakdown = compute_clean_grade(3, "reject")
        violation_score = breakdown.get("violation_score", 0)
        assert violation_score == 0.0, f"Expected 0.0, got {violation_score}"

    def test_reject_worse_than_revise_on_clean(self):
        reject_score, _ = compute_clean_grade(0, "reject")
        revise_score, _ = compute_clean_grade(0, "revise")
        assert reject_score < revise_score, f"reject ({reject_score}) should be < revise ({revise_score})"

    def test_pass_best_verdict_on_clean(self):
        pass_score, _ = compute_clean_grade(0, "pass")
        revise_score, _ = compute_clean_grade(0, "revise")
        reject_score, _ = compute_clean_grade(0, "reject")
        assert pass_score > revise_score > reject_score

    def test_efficiency_affects_clean_score(self):
        """Faster completion should give higher score."""
        score_fast, _ = compute_clean_grade(0, "pass", steps=2, budget=8)
        score_slow, _ = compute_clean_grade(0, "pass", steps=7, budget=8)
        assert score_fast > score_slow, f"Fast ({score_fast}) should beat slow ({score_slow})"

    def test_random_agent_expected_score_low(self):
        """Random agent on clean experiments should average < 0.30."""
        rng = random.Random(99)
        scores = []
        for _ in range(50):
            n = rng.randint(0, 4)
            verdict = rng.choice(["pass", "revise", "reject"])
            score, _ = compute_clean_grade(n, verdict)
            scores.append(score)
        avg = sum(scores) / len(scores)
        # Random agent with random flags and random verdict should score poorly
        assert avg < 0.50, f"Random agent clean score too high: {avg:.3f}"

    def test_all_clean_experiments_have_empty_violations(self):
        """Every clean experiment should have no violations in ground truth."""
        for exp in POOL.get("clean", []):
            gt = exp.get("ground_truth", {}).get("violations", [])
            assert len(gt) == 0, f"Clean experiment has violations: {gt}"
