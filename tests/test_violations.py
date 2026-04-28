"""
Comprehensive tests for all violation types V1-V8.
Tests injection correctness, grader detection, false positive penalty, and fabricated evidence.
"""
import pytest
import copy
from environment.generator import (
    POOL, _TEMPLATES, inject_V1, inject_V2, inject_V3, inject_V4,
    inject_V5, inject_V6, inject_V7, inject_V8
)
from environment.grader import grade_single_flag, evidence_found


def get_base_template(archetype="tabular_clf"):
    """Get a fresh copy of a base template."""
    base = copy.deepcopy(_TEMPLATES.get(archetype, _TEMPLATES["tabular_clf"]))
    base["ground_truth"] = {"violations": [], "expected_verdict": "pass"}
    return base


def get_artifact_content(exp, artifact_name):
    """Extract artifact content as string from experiment."""
    value = exp.get(artifact_name, {})
    if isinstance(value, dict):
        code = value.get("code_snippet", "")
        if code:
            return code
        import json
        return json.dumps(value, indent=2)
    return str(value)


# ══════════════════════════════════════════════════════════════════════════════
# V1: Preprocessing Leakage
# ══════════════════════════════════════════════════════════════════════════════

class TestV1Injection:
    def test_v1_injected_correctly(self):
        exp = inject_V1(get_base_template())
        gt = exp.get("ground_truth", {}).get("violations", [])
        assert "V1" in gt, f"V1 not in ground truth: {gt}"

    def test_v1_evidence_in_preprocessing(self):
        exp = inject_V1(get_base_template())
        code = get_artifact_content(exp, "preprocessing")
        assert "fit_transform" in code or "fit(" in code, f"V1 evidence not in preprocessing: {code[:200]}"

    def test_v1_detected_by_grader(self):
        exp = inject_V1(get_base_template())
        code = get_artifact_content(exp, "preprocessing")
        # Use exact evidence from the artifact
        evidence = "scaler.fit_transform(X_all)"
        flag = {
            "violation_type": "V1",
            "evidence_artifact": "preprocessing",
            "evidence_quote": evidence,
            "severity": "high"
        }
        inspected = {"preprocessing": code}
        reward, label = grade_single_flag(flag, ["V1"], inspected)
        assert label == "correct", f"Expected correct, got {label}"
        assert reward == pytest.approx(0.15, abs=0.01)

    def test_v1_false_positive_penalized(self):
        exp = inject_V1(get_base_template())
        code = get_artifact_content(exp, "preprocessing")
        flag = {
            "violation_type": "V2",  # Wrong type
            "evidence_artifact": "preprocessing",
            "evidence_quote": "fit_transform",
            "severity": "high"
        }
        inspected = {"preprocessing": code}
        reward, label = grade_single_flag(flag, ["V1"], inspected)
        assert label == "false_positive", f"Expected false_positive, got {label}"
        assert reward < 0

    def test_v1_fabricated_evidence_penalized(self):
        exp = inject_V1(get_base_template())
        code = get_artifact_content(exp, "preprocessing")
        flag = {
            "violation_type": "V1",
            "evidence_artifact": "preprocessing",
            "evidence_quote": "completely fabricated evidence xyz123",
            "severity": "high"
        }
        inspected = {"preprocessing": code}
        reward, label = grade_single_flag(flag, ["V1"], inspected)
        assert label == "fabricated_evidence", f"Expected fabricated_evidence, got {label}"
        assert reward < 0


# ══════════════════════════════════════════════════════════════════════════════
# V2: Temporal Shuffle
# ══════════════════════════════════════════════════════════════════════════════

class TestV2Injection:
    def test_v2_injected_correctly(self):
        exp = inject_V2(get_base_template("timeseries_reg"))
        gt = exp.get("ground_truth", {}).get("violations", [])
        assert "V2" in gt, f"V2 not in ground truth: {gt}"

    def test_v2_evidence_in_split_config(self):
        exp = inject_V2(get_base_template("timeseries_reg"))
        split = exp.get("split_config", {})
        assert split.get("shuffle") is True, f"V2 shuffle not True: {split}"

    def test_v2_detected_by_grader(self):
        exp = inject_V2(get_base_template("timeseries_reg"))
        code = get_artifact_content(exp, "preprocessing")
        evidence = "shuffle=True"
        flag = {
            "violation_type": "V2",
            "evidence_artifact": "preprocessing",
            "evidence_quote": evidence,
            "severity": "high"
        }
        inspected = {"preprocessing": code}
        reward, label = grade_single_flag(flag, ["V2"], inspected)
        assert label == "correct", f"Expected correct, got {label}"

    def test_v2_false_positive_penalized(self):
        exp = inject_V2(get_base_template("timeseries_reg"))
        code = get_artifact_content(exp, "preprocessing")
        flag = {
            "violation_type": "V3",  # Wrong type
            "evidence_artifact": "preprocessing",
            "evidence_quote": "shuffle",
            "severity": "high"
        }
        inspected = {"preprocessing": code}
        reward, label = grade_single_flag(flag, ["V2"], inspected)
        assert label == "false_positive"
        assert reward < 0


# ══════════════════════════════════════════════════════════════════════════════
# V3: Target Leakage
# ══════════════════════════════════════════════════════════════════════════════

class TestV3Injection:
    def test_v3_injected_correctly(self):
        exp = inject_V3(get_base_template())
        gt = exp.get("ground_truth", {}).get("violations", [])
        assert "V3" in gt, f"V3 not in ground truth: {gt}"

    def test_v3_evidence_in_model_config(self):
        exp = inject_V3(get_base_template())
        model_cfg = exp.get("model_config", {})
        features = model_cfg.get("feature_columns", [])
        target = exp.get("dataset_info", {}).get("target_column", "target")
        assert target in features, f"Target '{target}' not in features: {features}"

    def test_v3_detected_by_grader(self):
        exp = inject_V3(get_base_template())
        import json
        content = json.dumps(exp.get("model_config", {}), indent=2)
        target = exp.get("dataset_info", {}).get("target_column", "target")
        flag = {
            "violation_type": "V3",
            "evidence_artifact": "model_config",
            "evidence_quote": target,
            "severity": "high"
        }
        inspected = {"model_config": content}
        reward, label = grade_single_flag(flag, ["V3"], inspected)
        assert label == "correct", f"Expected correct, got {label}"

    def test_v3_fabricated_evidence_penalized(self):
        exp = inject_V3(get_base_template())
        import json
        content = json.dumps(exp.get("model_config", {}), indent=2)
        flag = {
            "violation_type": "V3",
            "evidence_artifact": "model_config",
            "evidence_quote": "nonexistent_column_xyz",
            "severity": "high"
        }
        inspected = {"model_config": content}
        reward, label = grade_single_flag(flag, ["V3"], inspected)
        assert label == "fabricated_evidence"
        assert reward < 0


# ══════════════════════════════════════════════════════════════════════════════
# V4: Train/Test Overlap
# ══════════════════════════════════════════════════════════════════════════════

class TestV4Injection:
    def test_v4_injected_correctly(self):
        exp = inject_V4(get_base_template(), seed=42)
        gt = exp.get("ground_truth", {}).get("violations", [])
        assert "V4" in gt, f"V4 not in ground truth: {gt}"

    def test_v4_creates_overlapping_ids(self):
        exp = inject_V4(get_base_template(), seed=42)
        split = exp.get("split_config", {})
        train_ids = set(split.get("train_ids_sample", []))
        test_ids = set(split.get("test_ids_sample", []))
        overlap = train_ids & test_ids
        assert len(overlap) > 0, f"No overlapping IDs found"

    def test_v4_detected_by_grader(self):
        exp = inject_V4(get_base_template(), seed=42)
        import json
        content = json.dumps(exp.get("split_config", {}), indent=2)
        # Use an overlapping ID as evidence
        split = exp.get("split_config", {})
        train_ids = set(split.get("train_ids_sample", []))
        test_ids = set(split.get("test_ids_sample", []))
        overlap = list(train_ids & test_ids)
        evidence = str(overlap[0]) if overlap else "1"
        flag = {
            "violation_type": "V4",
            "evidence_artifact": "split_config",
            "evidence_quote": evidence,
            "severity": "high"
        }
        inspected = {"split_config": content}
        reward, label = grade_single_flag(flag, ["V4"], inspected)
        assert label == "correct"

    def test_v4_no_overlap_count_metadata(self):
        exp = inject_V4(get_base_template(), seed=42)
        split = exp.get("split_config", {})
        assert "overlap_count" not in split, "overlap_count should be removed"


# ══════════════════════════════════════════════════════════════════════════════
# V5: Cherry-Picking
# ══════════════════════════════════════════════════════════════════════════════

class TestV5Injection:
    def test_v5_injected_correctly(self):
        exp = inject_V5(get_base_template(), seed=42)
        gt = exp.get("ground_truth", {}).get("violations", [])
        assert "V5" in gt, f"V5 not in ground truth: {gt}"

    def test_v5_creates_multiple_runs(self):
        exp = inject_V5(get_base_template(), seed=42)
        run_history = exp.get("run_history", {})
        total_runs = run_history.get("total_runs", 0)
        assert total_runs > 1, f"Expected multiple runs, got {total_runs}"

    def test_v5_detected_by_grader(self):
        exp = inject_V5(get_base_template(), seed=42)
        import json
        content = json.dumps(exp.get("run_history", {}), indent=2)
        total = exp.get("run_history", {}).get("total_runs", 15)
        flag = {
            "violation_type": "V5",
            "evidence_artifact": "run_history",
            "evidence_quote": f"total_runs\": {total}",
            "severity": "medium"
        }
        inspected = {"run_history": content}
        reward, label = grade_single_flag(flag, ["V5"], inspected)
        assert label == "correct"

    def test_v5_false_positive_penalized(self):
        exp = inject_V5(get_base_template(), seed=42)
        import json
        content = json.dumps(exp.get("run_history", {}), indent=2)
        flag = {
            "violation_type": "V8",  # Wrong type
            "evidence_artifact": "run_history",
            "evidence_quote": "total_runs",
            "severity": "high"
        }
        inspected = {"run_history": content}
        reward, label = grade_single_flag(flag, ["V5"], inspected)
        assert label == "false_positive"


# ══════════════════════════════════════════════════════════════════════════════
# V6: Metric Shopping
# ══════════════════════════════════════════════════════════════════════════════

class TestV6Injection:
    def test_v6_injected_correctly(self):
        exp = inject_V6(get_base_template())
        gt = exp.get("ground_truth", {}).get("violations", [])
        assert "V6" in gt, f"V6 not in ground truth: {gt}"

    def test_v6_creates_metric_mismatch(self):
        exp = inject_V6(get_base_template())
        tracked = exp.get("validation_strategy", {}).get("metrics_tracked", [])
        reported = exp.get("eval_report", {}).get("reported_metrics", {})
        assert len(tracked) > len(reported), f"tracked={len(tracked)}, reported={len(reported)}"

    def test_v6_detected_by_grader(self):
        exp = inject_V6(get_base_template())
        import json
        content = json.dumps(exp.get("validation_strategy", {}), indent=2)
        flag = {
            "violation_type": "V6",
            "evidence_artifact": "validation_strategy",
            "evidence_quote": "metrics_tracked",
            "severity": "medium"
        }
        inspected = {"validation_strategy": content}
        reward, label = grade_single_flag(flag, ["V6"], inspected)
        assert label == "correct"


# ══════════════════════════════════════════════════════════════════════════════
# V7: Entity Leakage
# ══════════════════════════════════════════════════════════════════════════════

class TestV7Injection:
    def test_v7_injected_correctly(self):
        exp = inject_V7(get_base_template(), seed=42)
        gt = exp.get("ground_truth", {}).get("violations", [])
        assert "V7" in gt, f"V7 not in ground truth: {gt}"

    def test_v7_creates_entity_overlap(self):
        exp = inject_V7(get_base_template(), seed=42)
        split = exp.get("split_config", {})
        train_ents = set(split.get("train_entities_sample", []))
        test_ents = set(split.get("test_entities_sample", []))
        overlap = train_ents & test_ents
        assert len(overlap) > 0, f"No entity overlap found"

    def test_v7_detected_by_grader(self):
        exp = inject_V7(get_base_template(), seed=42)
        import json
        content = json.dumps(exp.get("split_config", {}), indent=2)
        split = exp.get("split_config", {})
        train_ents = set(split.get("train_entities_sample", []))
        test_ents = set(split.get("test_entities_sample", []))
        overlap = list(train_ents & test_ents)
        evidence = overlap[0] if overlap else "P0001"
        flag = {
            "violation_type": "V7",
            "evidence_artifact": "split_config",
            "evidence_quote": evidence,
            "severity": "high"
        }
        inspected = {"split_config": content}
        reward, label = grade_single_flag(flag, ["V7"], inspected)
        assert label == "correct"

    def test_v7_fabricated_evidence_penalized(self):
        exp = inject_V7(get_base_template(), seed=42)
        import json
        content = json.dumps(exp.get("split_config", {}), indent=2)
        flag = {
            "violation_type": "V7",
            "evidence_artifact": "split_config",
            "evidence_quote": "NONEXISTENT_ENTITY_XYZ",
            "severity": "high"
        }
        inspected = {"split_config": content}
        reward, label = grade_single_flag(flag, ["V7"], inspected)
        assert label == "fabricated_evidence"


# ══════════════════════════════════════════════════════════════════════════════
# V8: Multi-Test Leakage
# ══════════════════════════════════════════════════════════════════════════════

class TestV8Injection:
    def test_v8_injected_correctly(self):
        exp = inject_V8(get_base_template())
        gt = exp.get("ground_truth", {}).get("violations", [])
        assert "V8" in gt, f"V8 not in ground truth: {gt}"

    def test_v8_evidence_in_validation_strategy(self):
        exp = inject_V8(get_base_template())
        val_strat = exp.get("validation_strategy", {})
        method = val_strat.get("method", "")
        assert "test" in method.lower(), f"V8 evidence not found: {val_strat}"

    def test_v8_detected_by_grader(self):
        exp = inject_V8(get_base_template())
        import json
        content = json.dumps(exp.get("validation_strategy", {}), indent=2)
        flag = {
            "violation_type": "V8",
            "evidence_artifact": "validation_strategy",
            "evidence_quote": "test set used for both tuning and evaluation",
            "severity": "high"
        }
        inspected = {"validation_strategy": content}
        reward, label = grade_single_flag(flag, ["V8"], inspected)
        assert label == "correct"

    def test_v8_false_positive_penalized(self):
        exp = inject_V8(get_base_template())
        import json
        content = json.dumps(exp.get("validation_strategy", {}), indent=2)
        flag = {
            "violation_type": "V1",  # Wrong type
            "evidence_artifact": "validation_strategy",
            "evidence_quote": "tuning",
            "severity": "high"
        }
        inspected = {"validation_strategy": content}
        reward, label = grade_single_flag(flag, ["V8"], inspected)
        assert label == "false_positive"

    def test_v8_fabricated_evidence_penalized(self):
        exp = inject_V8(get_base_template())
        import json
        content = json.dumps(exp.get("validation_strategy", {}), indent=2)
        flag = {
            "violation_type": "V8",
            "evidence_artifact": "validation_strategy",
            "evidence_quote": "completely made up evidence xyz987",
            "severity": "high"
        }
        inspected = {"validation_strategy": content}
        reward, label = grade_single_flag(flag, ["V8"], inspected)
        assert label == "fabricated_evidence"
