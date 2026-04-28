"""
Tests for compound violation experiments.
"""
import pytest
from environment.generator import POOL, inject_compound, _TEMPLATES
from environment.grader import grade
import copy


def get_all_experiments():
    """Flatten POOL into a single list."""
    all_exp = []
    for tier in ["easy", "medium", "hard", "clean"]:
        all_exp.extend(POOL.get(tier, []))
    return all_exp


class TestCompoundExperiments:
    def test_compound_experiments_count(self):
        """Verify we have 6 compound experiments."""
        all_exp = get_all_experiments()
        compound = [e for e in all_exp if e.get("_is_compound")]
        assert len(compound) == 6, f"Expected 6 compound experiments, got {len(compound)}"

    def test_compound_ground_truth_has_two_violations(self):
        """Each compound experiment should have exactly 2 violations."""
        all_exp = get_all_experiments()
        compound = [e for e in all_exp if e.get("_is_compound")]
        for exp in compound:
            gt = exp.get("ground_truth", {}).get("violations", [])
            assert len(gt) == 2, f"{exp.get('experiment_id')}: expected 2 GT violations, got {len(gt)}: {gt}"

    def test_compound_types_recorded(self):
        """Compound experiments should have _compound_types metadata."""
        all_exp = get_all_experiments()
        compound = [e for e in all_exp if e.get("_is_compound")]
        for exp in compound:
            types = exp.get("_compound_types", [])
            assert len(types) == 2, f"Expected 2 compound types, got {types}"

    def test_compound_in_hard_tier(self):
        """All compound experiments should be in the hard tier."""
        all_exp = get_all_experiments()
        compound = [e for e in all_exp if e.get("_is_compound")]
        for exp in compound:
            assert exp.get("difficulty") == "hard" or exp in POOL.get("hard", []), \
                f"Compound {exp.get('experiment_id')} not in hard tier"

    def test_inject_compound_v1_v5(self):
        """Test inject_compound creates V1+V5 correctly."""
        base = copy.deepcopy(_TEMPLATES.get("tabular_clf", {}))
        base["ground_truth"] = {"violations": [], "expected_verdict": "pass"}
        compound = inject_compound(base, "V1", "V5")
        gt = compound.get("ground_truth", {}).get("violations", [])
        assert "V1" in gt, "V1 not in compound ground truth"
        assert "V5" in gt, "V5 not in compound ground truth"
        assert compound.get("_is_compound") is True

    def test_inject_compound_v3_v6(self):
        """Test inject_compound creates V3+V6 correctly."""
        base = copy.deepcopy(_TEMPLATES.get("tabular_multi", {}))
        base["ground_truth"] = {"violations": [], "expected_verdict": "pass"}
        compound = inject_compound(base, "V3", "V6")
        gt = compound.get("ground_truth", {}).get("violations", [])
        assert "V3" in gt
        assert "V6" in gt

    def test_compound_partial_credit(self):
        """Flagging only one violation in a compound gives partial credit."""
        all_exp = get_all_experiments()
        v1v5 = [e for e in all_exp if e.get("_is_compound") and
                set(e.get("_compound_types", [])) == {"V1", "V5"}]
        if not v1v5:
            pytest.skip("No V1+V5 compound experiments in pool")

        exp = v1v5[0]
        # Simulate finding only V1
        flags = [{
            "flag_id": "f0",
            "violation_type": "V1",
            "evidence_artifact": "preprocessing",
            "evidence_quote": "fit_transform",
            "severity": "high"
        }]
        inspected = {"preprocessing": "scaler.fit_transform(X_all)"}

        score, breakdown = grade(
            flags=flags,
            ground_truth=exp.get("ground_truth", {}),
            steps_used=5,
            budget=18,
            verdict="reject",
            inspected=inspected
        )

        violation_score = breakdown.get("violation_score", 0)
        # With 1 of 2 violations found, should be ~0.5
        assert 0.3 <= violation_score <= 0.7, f"Partial credit should be ~0.5, got {violation_score}"

    def test_compound_full_credit(self):
        """Flagging both violations gives full credit."""
        all_exp = get_all_experiments()
        v1v5 = [e for e in all_exp if e.get("_is_compound") and
                set(e.get("_compound_types", [])) == {"V1", "V5"}]
        if not v1v5:
            pytest.skip("No V1+V5 compound experiments in pool")

        exp = v1v5[0]
        flags = [
            {
                "flag_id": "f0",
                "violation_type": "V1",
                "evidence_artifact": "preprocessing",
                "evidence_quote": "fit_transform",
                "severity": "high"
            },
            {
                "flag_id": "f1",
                "violation_type": "V5",
                "evidence_artifact": "run_history",
                "evidence_quote": "total_runs",
                "severity": "medium"
            }
        ]
        inspected = {
            "preprocessing": "scaler.fit_transform(X_all)",
            "run_history": '{"total_runs": 15}'
        }

        score, breakdown = grade(
            flags=flags,
            ground_truth=exp.get("ground_truth", {}),
            steps_used=8,
            budget=18,
            verdict="reject",
            inspected=inspected
        )

        violation_score = breakdown.get("violation_score", 0)
        assert violation_score >= 0.9, f"Full credit should be >=0.9, got {violation_score}"

    def test_compound_expected_verdict_is_reject(self):
        """Compound violations should have expected_verdict='reject'."""
        all_exp = get_all_experiments()
        compound = [e for e in all_exp if e.get("_is_compound")]
        for exp in compound:
            expected = exp.get("ground_truth", {}).get("expected_verdict", "")
            # Most compounds have severe violations
            assert expected in ("reject", "revise"), \
                f"{exp.get('experiment_id')}: expected verdict {expected}"
