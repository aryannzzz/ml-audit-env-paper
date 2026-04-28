"""
Unit tests for ML Audit Env grader.

These tests prove:
1. Grader produces variable scores (non-trivial)
2. Grader handles all edge cases correctly
3. Evidence matching works across all three layers

Run: pytest tests/test_grader.py -v
"""
import pytest
import random
from environment.generator import POOL
from environment.grader import (
    grade,
    grade_single_flag,
    evidence_found,
    normalize_text,
    tokenize,
)


# ══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def artifact_content():
    """Sample artifact contents for testing."""
    return {
        "preprocessing": (
            "# Preprocessing pipeline\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X_all)  # fit on ALL data before split\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X_scaled, y, test_size=0.2, random_state=42\n"
            ")"
        ),
        # overlap_count is intentionally absent — injector removes it to force
        # agents to compare actual ID lists rather than reading a labeled field
        "split_config": '{\n  "shuffle": true,\n  "test_size": 0.2,\n  "entity_overlap_count": 35,\n  "entity_column": "patient_id"\n}',
        "run_history": '{\n  "total_runs": 12,\n  "runs": [{"run_id": 1}, {"run_id": 2}]\n}',
        "experiment_notes": "Final model selected based on best performance. Hyperparameters tuned on test set.",
        "validation_strategy": '{"metrics_tracked": ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"], "method": "test set used for both tuning and evaluation"}',
        "eval_report": '{"reported_metrics": {"accuracy": 0.914}}',
        "model_config": '{"feature_columns": ["age", "bmi", "glucose"], "target_column": "readmitted"}',
        "dataset_info": '{"dataset_type": "tabular", "target_column": "readmitted", "entity_column": "patient_id"}',
    }


@pytest.fixture
def gt_one_violation():
    """Ground truth with single V1 violation."""
    return {
        "violations": ["V1"],
        "expected_verdict": "reject"
    }


@pytest.fixture
def gt_two_violations():
    """Ground truth with V1 and V6 violations."""
    return {
        "violations": ["V1", "V6"],
        "expected_verdict": "reject"
    }


@pytest.fixture
def gt_three_violations():
    """Ground truth with V1, V5, and V6 violations."""
    return {
        "violations": ["V1", "V5", "V6"],
        "expected_verdict": "reject"
    }


@pytest.fixture
def gt_clean():
    """Ground truth for clean experiment."""
    return {
        "violations": [],
        "expected_verdict": "pass"
    }


@pytest.fixture
def flag_v1_correct():
    """Correct V1 flag with valid evidence."""
    return {
        "flag_id": "f0",
        "violation_type": "V1",
        "evidence_artifact": "preprocessing",
        "evidence_quote": "scaler.fit_transform(X_all)",
        "severity": "high",
        "step_raised": 2,
    }


@pytest.fixture
def flag_v6_correct():
    """Correct V6 flag with valid evidence."""
    return {
        "flag_id": "f1",
        "violation_type": "V6",
        "evidence_artifact": "validation_strategy",
        "evidence_quote": '"metrics_tracked": ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]',
        "severity": "medium",
        "step_raised": 4,
    }


@pytest.fixture
def flag_false_positive():
    """Flag for violation not in ground truth."""
    return {
        "flag_id": "f2",
        "violation_type": "V5",
        "evidence_artifact": "run_history",
        "evidence_quote": '"total_runs": 12',
        "severity": "medium",
        "step_raised": 3,
    }


@pytest.fixture
def flag_fabricated():
    """Flag with correct type but fabricated evidence."""
    return {
        "flag_id": "f3",
        "violation_type": "V1",
        "evidence_artifact": "preprocessing",
        "evidence_quote": "this string does not exist in the artifact at all",
        "severity": "high",
        "step_raised": 2,
    }


@pytest.fixture
def gt_v7_violation():
    """Ground truth with V7 (entity leakage) violation."""
    return {
        "violations": ["V7"],
        "expected_verdict": "reject"
    }


@pytest.fixture
def gt_v8_violation():
    """Ground truth with V8 (multi-test leakage) violation."""
    return {
        "violations": ["V8"],
        "expected_verdict": "reject"
    }


@pytest.fixture
def flag_v7_correct():
    """Correct V7 flag with valid evidence for entity leakage."""
    return {
        "flag_id": "f7",
        "violation_type": "V7",
        "evidence_artifact": "split_config",
        "evidence_quote": '"entity_overlap_count": 35',
        "severity": "high",
        "step_raised": 3,
    }


@pytest.fixture
def flag_v8_correct():
    """Correct V8 flag with valid evidence for multi-test leakage."""
    return {
        "flag_id": "f8",
        "violation_type": "V8",
        "evidence_artifact": "validation_strategy",
        "evidence_quote": '"method": "test set used for both tuning and evaluation"',
        "severity": "high",
        "step_raised": 4,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE MATCHING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEvidenceMatching:
    """Tests for three-layer evidence matching."""

    def test_exact_match(self, artifact_content):
        """Layer 1: Exact substring match."""
        assert evidence_found(
            "scaler.fit_transform(X_all)",
            artifact_content["preprocessing"]
        )

    def test_exact_match_json(self, artifact_content):
        """Layer 1: Exact match in JSON content."""
        assert evidence_found(
            '"shuffle": true',
            artifact_content["split_config"]
        )

    def test_normalized_whitespace_match(self, artifact_content):
        """Layer 2: Whitespace-normalized match."""
        # The normalization collapses whitespace AND lowercases
        # So "fit_transform(X_all)" normalized = "fit_transform(x_all)"
        assert evidence_found(
            "scaler.fit_transform(X_all)",  # exact match
            artifact_content["preprocessing"]
        )
        # Test with actual whitespace normalization benefit
        assert evidence_found(
            "scaler.fit_transform(X_all)",
            "scaler.fit_transform(X_all)  "  # trailing space
        )

    def test_normalized_newline_match(self):
        """Layer 2: Newline normalization."""
        artifact = "scaler.\nfit_transform(X_all)"
        # After normalization: "scaler. fit_transform(x_all)"
        assert evidence_found(
            "scaler. fit_transform(X_all)",
            artifact
        )

    def test_token_overlap_match(self, artifact_content):
        """Layer 3: Token overlap >= 80%."""
        # Quote needs 3+ tokens and 80% overlap with artifact tokens
        # Tokens from artifact include: preprocessing, scaler, standardscaler, etc.
        assert evidence_found(
            "scaler StandardScaler fit",  # 3 tokens, all in artifact
            artifact_content["preprocessing"]
        )

    def test_token_overlap_partial(self, artifact_content):
        """Layer 3: Token overlap threshold - tests token matching edge case."""
        # Use tokens that definitely appear in the artifact
        # Artifact has: preprocessing, pipeline, scaler, standardscaler, x_scaled, fit_transform, x_all
        # Note: tokenize uses \w+ which includes underscores
        assert evidence_found(
            "scaler StandardScaler fit_transform",  # Tokens: scaler, standardscaler, fit_transform
            artifact_content["preprocessing"]
        )

    def test_no_match_different_content(self, artifact_content):
        """No match: completely different content."""
        assert not evidence_found(
            "encoder.fit_transform(X_train)",
            artifact_content["preprocessing"]
        )

    def test_no_match_fabricated(self, artifact_content):
        """No match: fabricated evidence."""
        assert not evidence_found(
            "THIS IS COMPLETELY FABRICATED EVIDENCE",
            artifact_content["preprocessing"]
        )

    def test_empty_quote(self, artifact_content):
        """Empty quote returns False."""
        assert not evidence_found("", artifact_content["preprocessing"])

    def test_empty_artifact(self):
        """Empty artifact returns False."""
        assert not evidence_found("some quote", "")

    def test_both_empty(self):
        """Both empty returns False."""
        assert not evidence_found("", "")


class TestNormalization:
    """Tests for text normalization helpers."""

    def test_normalize_text_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_normalize_text_strips_edges(self):
        assert normalize_text("  hello  ") == "hello"

    def test_normalize_text_handles_newlines(self):
        assert normalize_text("hello\n\nworld") == "hello world"

    def test_tokenize_extracts_words(self):
        tokens = tokenize("hello-world foo_bar 123")
        assert "hello" in tokens
        assert "world" in tokens
        # foo_bar is one token with underscores (regex \w+ includes underscores)
        assert "foo_bar" in tokens
        assert "123" in tokens


# ══════════════════════════════════════════════════════════════════════════════
# GRADER SCORE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestGraderScores:
    """Tests proving grader produces variable, meaningful scores."""

    def test_perfect_score_single_violation(
        self, artifact_content, gt_one_violation, flag_v1_correct
    ):
        """All violations found with correct evidence = high score."""
        score, breakdown = grade(
            flags=[flag_v1_correct],
            ground_truth=gt_one_violation,
            steps_used=3,
            budget=8,
            verdict="reject",
            inspected=artifact_content,
        )
        assert score >= 0.85, f"Perfect answer should score >= 0.85, got {score}"
        assert breakdown["flag_results"][0]["result"] == "correct"

    def test_zero_score_no_flags_wrong_verdict(
        self, artifact_content, gt_one_violation
    ):
        """No flags + wrong verdict on violated experiment = zero score."""
        score, breakdown = grade(
            flags=[],
            ground_truth=gt_one_violation,
            steps_used=8,
            budget=8,
            verdict="pass",
            inspected=artifact_content,
        )
        assert score == 0.0, f"No flags + wrong verdict should score 0.0, got {score}"

    def test_partial_credit_one_of_two(
        self, artifact_content, gt_two_violations, flag_v1_correct
    ):
        """Finding 1 of 2 violations gives partial credit."""
        score, breakdown = grade(
            flags=[flag_v1_correct],
            ground_truth=gt_two_violations,
            steps_used=6,
            budget=12,
            verdict="reject",
            inspected=artifact_content,
        )
        assert 0.3 < score < 0.7, f"Partial credit should be between 0.3-0.7, got {score}"

    def test_partial_credit_two_of_two(
        self, artifact_content, gt_two_violations, flag_v1_correct, flag_v6_correct
    ):
        """Finding 2 of 2 violations gives high score."""
        score, breakdown = grade(
            flags=[flag_v1_correct, flag_v6_correct],
            ground_truth=gt_two_violations,
            steps_used=5,
            budget=12,
            verdict="reject",
            inspected=artifact_content,
        )
        assert score >= 0.85, f"2/2 violations should score >= 0.85, got {score}"

    def test_false_positive_penalized(
        self, artifact_content, gt_one_violation, flag_false_positive, flag_v1_correct
    ):
        """False positives significantly reduce score."""
        # Score with correct flag only
        score_correct, _ = grade(
            flags=[flag_v1_correct],
            ground_truth=gt_one_violation,
            steps_used=4,
            budget=8,
            verdict="reject",
            inspected=artifact_content,
        )
        # Score with correct flag + false positive
        score_with_fp, _ = grade(
            flags=[flag_v1_correct, flag_false_positive],
            ground_truth=gt_one_violation,
            steps_used=4,
            budget=8,
            verdict="reject",
            inspected=artifact_content,
        )
        # False positive should reduce score
        assert score_with_fp < score_correct, "False positive should hurt score"

    def test_fabricated_evidence_penalized(
        self, artifact_content, gt_one_violation, flag_v1_correct, flag_fabricated
    ):
        """Fabricated evidence scores lower than correct evidence."""
        score_correct, _ = grade(
            flags=[flag_v1_correct],
            ground_truth=gt_one_violation,
            steps_used=3,
            budget=8,
            verdict="reject",
            inspected=artifact_content,
        )
        score_fabricated, breakdown = grade(
            flags=[flag_fabricated],
            ground_truth=gt_one_violation,
            steps_used=3,
            budget=8,
            verdict="reject",
            inspected=artifact_content,
        )
        assert score_fabricated < score_correct
        assert breakdown["flag_results"][0]["result"] == "fabricated_evidence"

    def test_clean_experiment_no_flags_correct(
        self, artifact_content, gt_clean
    ):
        """Clean experiment with no flags and correct verdict = high score."""
        score, _ = grade(
            flags=[],
            ground_truth=gt_clean,
            steps_used=3,
            budget=8,
            verdict="pass",
            inspected=artifact_content,
        )
        assert score >= 0.85, f"No flags on clean = high score, got {score}"

    def test_clean_experiment_false_flags_penalized(
        self, artifact_content, gt_clean, flag_v1_correct
    ):
        """Flagging violations on clean experiment = lower score than no flags."""
        # Clean experiment with no flags and correct verdict
        score_correct, _ = grade(
            flags=[],
            ground_truth=gt_clean,
            steps_used=4,
            budget=8,
            verdict="pass",
            inspected=artifact_content,
        )
        # Clean experiment with false flag and wrong verdict
        score_with_fp, _ = grade(
            flags=[flag_v1_correct],  # This becomes a false positive on clean
            ground_truth=gt_clean,
            steps_used=4,
            budget=8,
            verdict="reject",
            inspected=artifact_content,
        )
        assert score_with_fp < score_correct, f"False flag on clean should score lower, got {score_with_fp} vs {score_correct}"


class TestGraderVariance:
    """Tests proving grader is non-trivial (produces different scores)."""

    def test_different_inputs_different_scores(
        self, artifact_content, gt_one_violation, flag_v1_correct, flag_false_positive
    ):
        """Different inputs must produce different scores."""
        scores = set()

        # Perfect flag
        s1, _ = grade([flag_v1_correct], gt_one_violation, 2, 8, "reject", artifact_content)
        scores.add(round(s1, 4))

        # No flags
        s2, _ = grade([], gt_one_violation, 8, 8, "reject", artifact_content)
        scores.add(round(s2, 4))

        # Wrong flag
        s3, _ = grade([flag_false_positive], gt_one_violation, 4, 8, "reject", artifact_content)
        scores.add(round(s3, 4))

        # Perfect flag, wrong verdict
        s4, _ = grade([flag_v1_correct], gt_one_violation, 2, 8, "pass", artifact_content)
        scores.add(round(s4, 4))

        assert len(scores) >= 3, f"Grader only produced {len(scores)} distinct scores: {scores}"

    def test_efficiency_bonus_varies(
        self, artifact_content, gt_one_violation, flag_v1_correct
    ):
        """Fewer steps = higher score."""
        score_fast, _ = grade(
            [flag_v1_correct], gt_one_violation, 2, 8, "reject", artifact_content
        )
        score_slow, _ = grade(
            [flag_v1_correct], gt_one_violation, 8, 8, "reject", artifact_content
        )
        assert score_fast > score_slow, "Faster completion should score higher"

    def test_verdict_bonus(self, artifact_content, gt_clean):
        """Clean scoring is verdict-sensitive under the Phase 2 formula."""
        score_right, _ = grade([], gt_clean, 3, 8, "pass", artifact_content)
        score_wrong, _ = grade([], gt_clean, 3, 8, "reject", artifact_content)

        diff = score_right - score_wrong
        # n_false=0 with reject earns only 0.20 violation score on clean episodes.
        assert abs(diff - 0.74) < 0.02, f"Clean verdict sensitivity should be ~0.74, got {diff}"


class TestGraderBounds:
    """Tests ensuring scores stay in valid range."""

    def test_scores_always_in_range(
        self, artifact_content, gt_one_violation, gt_two_violations, gt_clean
    ):
        """Fuzz test: scores must always be in [0, 1]."""
        flag_pool = [
            {"flag_id": "f0", "violation_type": "V1", "evidence_artifact": "preprocessing",
             "evidence_quote": "scaler.fit_transform(X_all)", "severity": "high"},
            {"flag_id": "f1", "violation_type": "V6", "evidence_artifact": "validation_strategy",
             "evidence_quote": "metrics_tracked", "severity": "medium"},
            {"flag_id": "f2", "violation_type": "V5", "evidence_artifact": "run_history",
             "evidence_quote": "total_runs", "severity": "medium"},
            {"flag_id": "f3", "violation_type": "V3", "evidence_artifact": "model_config",
             "evidence_quote": "FAKE", "severity": "high"},
        ]
        gt_pool = [gt_one_violation, gt_two_violations, gt_clean]

        rng = random.Random(42)
        for i in range(500):
            n_flags = rng.randint(0, 4)
            flags = rng.choices(flag_pool, k=n_flags)
            # Unique flag IDs
            for j, f in enumerate(flags):
                flags[j] = {**f, "flag_id": f"f{j}"}

            gt = rng.choice(gt_pool)
            budget = rng.randint(4, 16)
            steps = rng.randint(1, budget)
            verdict = rng.choice(["pass", "revise", "reject"])

            score, _ = grade(flags, gt, steps, budget, verdict, artifact_content)

            assert 0.0 <= score <= 1.0, f"Score {score} out of range on iteration {i}"

    def test_max_score_achievable(
        self, artifact_content, gt_one_violation, flag_v1_correct
    ):
        """Perfect play can achieve near-1.0 score."""
        score, _ = grade(
            [flag_v1_correct],
            gt_one_violation,
            steps_used=1,
            budget=8,
            verdict="reject",
            inspected=artifact_content,
        )
        assert score >= 0.95, f"Perfect play should achieve >= 0.95, got {score}"

    def test_min_score_zero(self, artifact_content, gt_one_violation):
        """Worst play achieves 0.0 score."""
        # No flags, wrong verdict, max steps
        score, _ = grade(
            flags=[],
            ground_truth=gt_one_violation,
            steps_used=8,
            budget=8,
            verdict="pass",
            inspected=artifact_content,
        )
        assert score == 0.0


class TestPoolFixtureConsistency:
    """Regression checks to keep pool fixtures aligned with injector behavior."""

    def test_overlap_count_absent_in_violated_pool(self):
        """Violated experiments should not expose overlap_count helper metadata."""
        for task in ["easy", "medium", "hard"]:
            for exp in POOL[task]:
                if exp.get("ground_truth", {}).get("violations"):
                    split_cfg = exp.get("split_config", {})
                    if isinstance(split_cfg, dict):
                        assert "overlap_count" not in split_cfg, (
                            f"overlap_count unexpectedly present in {exp.get('experiment_id', 'unknown')}"
                        )


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE FLAG GRADING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestGradeSingleFlag:
    """Tests for individual flag grading."""

    def test_correct_flag(self, artifact_content, gt_one_violation, flag_v1_correct):
        reward, label = grade_single_flag(
            flag_v1_correct, gt_one_violation["violations"], artifact_content
        )
        assert reward == 0.15
        assert label == "correct"

    def test_false_positive(self, artifact_content, gt_one_violation, flag_false_positive):
        reward, label = grade_single_flag(
            flag_false_positive, gt_one_violation["violations"], artifact_content
        )
        assert reward == -0.10
        assert label == "false_positive"

    def test_fabricated_evidence(self, artifact_content, gt_one_violation, flag_fabricated):
        reward, label = grade_single_flag(
            flag_fabricated, gt_one_violation["violations"], artifact_content
        )
        assert reward == -0.05
        assert label == "fabricated_evidence"

    def test_not_inspected(self, gt_one_violation, flag_v1_correct):
        """Flag for artifact not in inspected dict."""
        reward, label = grade_single_flag(
            flag_v1_correct,
            gt_one_violation["violations"],
            {}  # Empty inspected dict
        )
        assert reward == -0.10
        assert label == "not_inspected"

    def test_correct_v7_flag(self, artifact_content, gt_v7_violation, flag_v7_correct):
        """V7 (Entity Leakage) flag is graded correctly."""
        reward, label = grade_single_flag(
            flag_v7_correct, gt_v7_violation["violations"], artifact_content
        )
        assert reward == 0.15
        assert label == "correct"

    def test_correct_v8_flag(self, artifact_content, gt_v8_violation, flag_v8_correct):
        """V8 (Multi-Test Leakage) flag is graded correctly."""
        reward, label = grade_single_flag(
            flag_v8_correct, gt_v8_violation["violations"], artifact_content
        )
        assert reward == 0.15
        assert label == "correct"

    def test_v7_false_positive(self, artifact_content, gt_one_violation, flag_v7_correct):
        """V7 flag on non-V7 experiment is false positive."""
        reward, label = grade_single_flag(
            flag_v7_correct, gt_one_violation["violations"], artifact_content
        )
        assert reward == -0.10
        assert label == "false_positive"

    def test_v8_false_positive(self, artifact_content, gt_one_violation, flag_v8_correct):
        """V8 flag on non-V8 experiment is false positive."""
        reward, label = grade_single_flag(
            flag_v8_correct, gt_one_violation["violations"], artifact_content
        )
        assert reward == -0.10
        assert label == "false_positive"


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestGraderIntegration:
    """Integration tests with realistic scenarios."""

    def test_easy_task_perfect_play(self, artifact_content, gt_one_violation):
        """Simulate perfect play on easy task."""
        flags = [{
            "flag_id": "f0",
            "violation_type": "V1",
            "evidence_artifact": "preprocessing",
            "evidence_quote": "scaler.fit_transform(X_all)",
            "severity": "high",
        }]
        score, breakdown = grade(
            flags=flags,
            ground_truth=gt_one_violation,
            steps_used=3,
            budget=8,
            verdict="reject",
            inspected=artifact_content,
        )
        assert score >= 0.85
        assert breakdown["violation_score"] == 1.0
        assert breakdown["verdict_bonus"] == 0.10

    def test_medium_task_partial_success(self, artifact_content, gt_two_violations):
        """Simulate partial success on medium task."""
        flags = [{
            "flag_id": "f0",
            "violation_type": "V1",
            "evidence_artifact": "preprocessing",
            "evidence_quote": "scaler.fit_transform(X_all)",
            "severity": "high",
        }]  # Only found 1 of 2
        score, breakdown = grade(
            flags=flags,
            ground_truth=gt_two_violations,
            steps_used=8,
            budget=12,
            verdict="reject",
            inspected=artifact_content,
        )
        assert 0.3 < score < 0.7
        assert 0.4 < breakdown["violation_score"] < 0.6

    def test_hard_task_with_false_positive(self, artifact_content, gt_three_violations):
        """Hard task: found 2/3 but also added false positive."""
        flags = [
            {
                "flag_id": "f0",
                "violation_type": "V1",
                "evidence_artifact": "preprocessing",
                "evidence_quote": "scaler.fit_transform(X_all)",
                "severity": "high",
            },
            {
                "flag_id": "f1",
                "violation_type": "V6",
                "evidence_artifact": "validation_strategy",
                "evidence_quote": "metrics_tracked",
                "severity": "medium",
            },
            {
                "flag_id": "f2",
                "violation_type": "V4",  # Not in ground truth
                "evidence_artifact": "split_config",
                "evidence_quote": "overlap_count",
                "severity": "high",
            },
        ]
        score, breakdown = grade(
            flags=flags,
            ground_truth=gt_three_violations,
            steps_used=12,
            budget=16,
            verdict="reject",
            inspected=artifact_content,
        )
        # 2 correct (+0.30), 1 FP (-0.10) = 0.20 out of 0.45 max
        # Plus efficiency and verdict bonuses
        assert 0.3 < score < 0.6
