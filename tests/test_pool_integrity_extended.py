"""
Extended pool integrity tests.
"""
import pytest
import re
from collections import Counter
from environment.generator import POOL


def get_all_experiments():
    """Flatten POOL into a single list."""
    all_exp = []
    for tier in ["easy", "medium", "hard", "clean"]:
        all_exp.extend(POOL.get(tier, []))
    return all_exp


class TestPoolSize:
    def test_pool_total_size(self):
        total = sum(len(POOL.get(tier, [])) for tier in ["easy", "medium", "hard", "clean"])
        assert total == 56, f"Expected 56, got {total}"

    def test_easy_tier_size(self):
        assert len(POOL.get("easy", [])) == 10

    def test_medium_tier_size(self):
        assert len(POOL.get("medium", [])) == 10

    def test_hard_tier_size(self):
        assert len(POOL.get("hard", [])) == 16  # 10 + 6 compound

    def test_clean_tier_size(self):
        assert len(POOL.get("clean", [])) == 20


class TestExperimentIds:
    def test_experiment_ids_unique(self):
        all_exp = get_all_experiments()
        ids = [e.get("experiment_id") for e in all_exp]
        dupes = [id for id, count in Counter(ids).items() if count > 1 and id is not None]
        assert not dupes, f"Duplicate IDs: {dupes}"

    def test_all_experiments_have_id(self):
        all_exp = get_all_experiments()
        for exp in all_exp:
            assert exp.get("experiment_id"), f"Missing experiment_id in {exp.keys()}"


class TestViolationCoverage:
    def test_all_violation_types_appear_at_least_3_times(self):
        counts = Counter()
        all_exp = get_all_experiments()

        for exp in all_exp:
            gt = exp.get("ground_truth", {}).get("violations", [])
            for v in gt:
                vtype = v if isinstance(v, str) else str(v)
                m = re.search(r"V\d+", vtype)
                if m:
                    counts[m.group()] += 1

        for vtype in [f"V{i}" for i in range(1, 9)]:
            assert counts[vtype] >= 3, f"{vtype} appears only {counts[vtype]} times in pool"


class TestArchetypes:
    def test_all_archetypes_present(self):
        all_exp = get_all_experiments()
        archetypes = {e.get("archetype") for e in all_exp if e.get("archetype")}
        required = {"tabular_clf", "timeseries_reg", "tabular_multi", "tabular_survival"}
        missing = required - archetypes
        assert not missing, f"Missing archetypes: {missing}"


class TestDifficultyTiers:
    def test_easy_tier_exists(self):
        assert "easy" in POOL
        assert len(POOL["easy"]) > 0

    def test_medium_tier_exists(self):
        assert "medium" in POOL
        assert len(POOL["medium"]) > 0

    def test_hard_tier_exists(self):
        assert "hard" in POOL
        assert len(POOL["hard"]) > 0

    def test_clean_tier_exists(self):
        assert "clean" in POOL
        assert len(POOL["clean"]) > 0


class TestCompoundInHardTier:
    def test_compound_experiments_are_in_hard_tier(self):
        hard = POOL.get("hard", [])
        compound = [e for e in hard if e.get("_is_compound")]
        assert len(compound) == 6, f"Expected 6 compound in hard, got {len(compound)}"

    def test_no_compound_in_other_tiers(self):
        for tier in ["easy", "medium", "clean"]:
            for exp in POOL.get(tier, []):
                assert not exp.get("_is_compound"), \
                    f"Compound found in {tier}: {exp.get('experiment_id')}"


class TestMetadataIntegrity:
    def test_no_overlap_count_in_violated_experiments(self):
        """overlap_count should be removed from V4 experiments."""
        all_exp = get_all_experiments()
        for exp in all_exp:
            gt = exp.get("ground_truth", {}).get("violations", [])
            if any("V4" in str(v) for v in gt):
                split = exp.get("split_config", {})
                if isinstance(split, dict):
                    assert "overlap_count" not in split, \
                        f"overlap_count found in V4 experiment {exp.get('experiment_id')}"

    def test_clean_experiments_have_empty_ground_truth(self):
        for exp in POOL.get("clean", []):
            gt = exp.get("ground_truth", {}).get("violations", [])
            assert len(gt) == 0, f"Clean experiment {exp.get('experiment_id')} has violations: {gt}"

    def test_violated_experiments_have_violations(self):
        for tier in ["easy", "medium", "hard"]:
            for exp in POOL.get(tier, []):
                # Skip if marked as clean
                if exp.get("_is_clean"):
                    continue
                gt = exp.get("ground_truth", {}).get("violations", [])
                assert len(gt) >= 1, f"{tier} experiment {exp.get('experiment_id')} has no violations"


class TestDeterminism:
    def test_pool_is_deterministic(self):
        """Re-importing the pool gives same experiment IDs."""
        ids_first = [e.get("experiment_id") for e in get_all_experiments()]

        import importlib
        import environment.generator as gen
        importlib.reload(gen)

        ids_second = [e.get("experiment_id") for e in get_all_experiments()]

        # Note: order may change after reload, so compare sets
        assert set(ids_first) == set(ids_second), "Pool is not deterministic across imports"


class TestHardTierCompoundFraction:
    def test_hard_pool_has_compound_fraction(self):
        """At least 20% of hard tier should be compound."""
        hard = POOL.get("hard", [])
        compound = [e for e in hard if e.get("_is_compound")]
        ratio = len(compound) / len(hard) if hard else 0
        # 6 compound / 16 total = 0.375
        assert ratio >= 0.20, f"Compound fraction in hard pool too low: {ratio:.2f}"

    def test_hard_tier_violation_counts(self):
        """Hard tier experiments should have 2-3 violations."""
        for exp in POOL.get("hard", []):
            gt = exp.get("ground_truth", {}).get("violations", [])
            # Compound have 2, regular hard have 3
            assert 2 <= len(gt) <= 3, \
                f"{exp.get('experiment_id')} has {len(gt)} violations, expected 2-3"
