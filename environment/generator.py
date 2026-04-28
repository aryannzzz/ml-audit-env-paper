"""
Experiment pool generator for ML Audit Env.
Builds all 56 experiments at import time:
- 36 violated (10 easy, 10 medium, 16 hard including 6 compound)
- 20 adversarially clean

Experiments are generated programmatically from 4 base templates
with 8 injector functions (V1-V8). Ground truth is 100% programmatic.

Violation Types:
- V1: Preprocessing Leakage (scaler fit on full data before split)
- V2: Temporal Shuffle (time-series shuffled before split)
- V3: Target Leakage (target in features)
- V4: Train/Test Overlap (same sample IDs)
- V5: Cherry-Picking (multiple runs, best reported)
- V6: Metric Shopping (multiple metrics, best reported)
- V7: Entity Leakage (same entities in train/test)
- V8: Multi-Test Leakage (test set used for HPO)
"""
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any


# Directory containing base templates
TEMPLATES_DIR = Path(__file__).parent.parent / "experiments" / "templates"
VALID_ARCHETYPES = ["tabular_clf", "timeseries_reg", "tabular_multi", "tabular_survival", "text_clf"]


def _load_templates() -> dict[str, dict]:
    """Load all base templates from disk."""
    templates = {}
    for arch in VALID_ARCHETYPES:
        path = TEMPLATES_DIR / f"{arch}_base.json"
        if path.exists():
            with open(path, "r") as f:
                templates[arch] = json.load(f)
        else:
            # Fallback: create minimal template
            templates[arch] = {
                "archetype": arch,
                "experiment_id": f"{arch}_base",
                "task_description": f"Audit this {arch} experiment for violations.",
                "dataset_info": {"dataset_type": "tabular" if "tabular" in arch else "timeseries"},
                "preprocessing": {"fit_scope": "train_only", "code_snippet": ""},
                "split_config": {"shuffle": False, "overlap_count": 0},
                "feature_engineering": {"code_snippet": ""},
                "model_config": {"feature_columns": [], "target_column": "target"},
                "training_logs": "",
                "eval_report": {"reported_metrics": {}, "best_score": 0.9},
                "experiment_notes": "",
                "validation_strategy": {"metrics_tracked": []},
                "run_history": {"total_runs": 1, "runs": []},
                "ground_truth": {"violations": [], "expected_verdict": "pass"},
            }
    return templates


# Load templates at module import time
_TEMPLATES = _load_templates()


# ══════════════════════════════════════════════════════════════════════════════
# VIOLATION INJECTORS
# ══════════════════════════════════════════════════════════════════════════════


def inject_V1(exp: dict) -> dict:
    """
    V1: Scaler/encoder fit on full dataset BEFORE train-test split.
    For text_clf: TfidfVectorizer.fit_transform called on full corpus before split.
    """
    exp["preprocessing"].pop("fit_scope", None)
    if exp.get("archetype") == "text_clf":
        exp["preprocessing"]["code_snippet"] = (
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "# Fit vectorizer on ALL documents before splitting (leakage!)\n"
            "vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)\n"
            "X_tfidf = vectorizer.fit_transform(df['article_text'])  # fit on full corpus\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X_tfidf, df['category'], test_size=0.2, random_state=42, stratify=df['category']\n"
            ")"
        )
    else:
        exp["preprocessing"]["code_snippet"] = (
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X_all)  # scale all features\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X_scaled, y, test_size=0.2, random_state=42\n"
            ")"
        )
    exp["ground_truth"]["violations"].append("V1")
    return exp


def inject_V2(exp: dict) -> dict:
    """
    V2: Temporal shuffle — time-series data shuffled before split.
    Only valid for timeseries_reg experiments.
    """
    if exp.get("archetype") != "timeseries_reg":
        return exp  # Skip if not time-series

    exp["split_config"]["shuffle"] = True
    exp["split_config"]["method"] = "random"
    exp["split_config"].pop("split_date", None)

    exp["preprocessing"]["code_snippet"] = (
        "# Split data\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, shuffle=True, random_state=42\n"
        ")\n"
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        "X_test_scaled  = scaler.transform(X_test)"
    )
    exp["ground_truth"]["violations"].append("V2")
    return exp


def inject_V2_feature_engineering(exp: dict) -> dict:
    """
    V2f: Lag features computed on full dataset before split.
    This is a time-series variant of preprocessing leakage.
    """
    exp["feature_engineering"]["code_snippet"] = (
        "# Compute lag features on the full sequence BEFORE splitting (leakage!)\n"
        "df['temp_lag_7'] = df['temperature_c'].shift(7)\n"
        "df['temp_ma_7']  = df['temperature_c'].rolling(7).mean()\n"
        "# Then split AFTER computing features (test data leaks into training lags)\n"
        "X_train, X_test = train_test_split(df[features], test_size=0.2)"
    )
    # Keep V1 evidence in preprocessing artifact to match agent/test expectations.
    exp["preprocessing"]["code_snippet"] = (
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(X_all)  # scale all features\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X_scaled, y, test_size=0.2, random_state=42\n"
        ")"
    )
    exp["ground_truth"]["violations"].append("V1")  # Still counts as preprocessing leakage
    return exp


def inject_V3(exp: dict) -> dict:
    """
    V3: Target variable included in feature columns (target leakage).
    For text_clf: encoded label column appended as a hand-crafted feature.
    """
    target = exp.get("dataset_info", {}).get("target_column", "target")
    feature_cols = exp.get("model_config", {}).get("feature_columns", [])

    if target not in feature_cols:
        exp["model_config"]["feature_columns"] = feature_cols + [target]

    if exp.get("archetype") == "text_clf":
        # Add an explicit leaky feature_engineering snippet showing label encoding
        existing_fe = exp.get("feature_engineering", {}).get("code_snippet", "")
        exp["feature_engineering"]["code_snippet"] = (
            existing_fe.rstrip() + "\n\n"
            "# Encode category labels as an integer feature (leakage!)\n"
            "from sklearn.preprocessing import LabelEncoder\n"
            "le = LabelEncoder()\n"
            "# Append encoded 'category' column to feature matrix — target leakage\n"
            "category_encoded = le.fit_transform(df['category']).reshape(-1, 1)\n"
            "X_train_full = sp.hstack([X_train_full, category_encoded[train_idx]])\n"
            "X_test_full  = sp.hstack([X_test_full,  category_encoded[test_idx]])"
        )

    exp["ground_truth"]["violations"].append("V3")
    return exp


def inject_V4(exp: dict, seed: int = 0) -> dict:
    """
    V4: Overlapping sample IDs in train and test splits.
    """
    rng = random.Random(seed + 100)
    n_overlap = rng.randint(30, 60)

    # Create overlapping IDs and ensure both train/test lists exist.
    overlap_ids = list(range(1, n_overlap + 1))
    split_config = exp.get("split_config", {})

    train_ids = list(split_config.get("train_ids_sample", list(range(1, 6401))))
    test_ids = list(split_config.get("test_ids_sample", list(range(6401, 6421))))

    train_ids[:n_overlap] = overlap_ids
    test_ids[:n_overlap] = overlap_ids

    split_config["train_ids_sample"] = train_ids
    split_config["test_ids_sample"] = test_ids
    split_config.pop("overlap_count", None)
    exp["split_config"] = split_config

    exp["ground_truth"]["violations"].append("V4")
    return exp


def inject_V5(exp: dict, seed: int = 0) -> dict:
    """
    V5: Multiple runs performed, only best reported without disclosure.
    Cherry-picking results.
    """
    rng = random.Random(seed + 200)
    n_runs = rng.randint(10, 15)
    base_score = exp.get("eval_report", {}).get("best_score", 0.9)

    runs = []
    for i in range(n_runs):
        # Generate score variations
        variation = rng.uniform(-0.12, 0.02)
        score = round(max(0.5, min(0.999, base_score + variation)), 3)
        runs.append({
            "run_id": i + 1,
            "seed": i,
            "score": score,
        })

    # Make the best run the reported one
    best_idx = max(range(n_runs), key=lambda i: runs[i]["score"])
    runs[best_idx]["score"] = base_score

    exp["run_history"] = {
        "total_runs": n_runs,
        "runs": runs
    }

    # Modify experiment notes to NOT mention multiple runs
    notes = exp.get("experiment_notes", "")
    if isinstance(notes, str):
        notes = notes.replace("Single experiment", "Final model selected")
        notes = notes.replace("Single run", "Final model selected")
        if "multiple" not in notes.lower() and "runs" not in notes.lower():
            notes = notes  # Keep as-is, missing disclosure is the violation
    exp["experiment_notes"] = notes

    exp["ground_truth"]["violations"].append("V5")
    return exp


def inject_V6(exp: dict) -> dict:
    """
    V6: Multiple metrics tracked, only best one reported (metric shopping).
    """
    all_metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    exp["validation_strategy"]["metrics_tracked"] = all_metrics

    # Only keep the single best metric in eval_report
    reported = exp.get("eval_report", {}).get("reported_metrics", {})
    if reported:
        best_metric = list(reported.keys())[0]
        best_value = reported[best_metric]
        exp["eval_report"]["reported_metrics"] = {best_metric: best_value}

    exp["ground_truth"]["violations"].append("V6")
    return exp


def inject_V7(exp: dict, seed: int = 0) -> dict:
    """
    V7: Non-Independence Leakage (Entity Leakage).
    Same entities (patients/products/sensors) appear in both train and test splits.
    Different from V4: V7 is about real-world entities having multiple rows,
    not just sample ID overlap.

    Per Kapoor & Narayanan (2023), this is one of the top 8 leakage types in ML science.
    """
    rng = random.Random(seed + 700)
    n_overlap = rng.randint(25, 50)

    # Add entity column info to dataset_info
    dataset_info = exp.get("dataset_info", {})
    entity_col = dataset_info.get("entity_column", "patient_id")
    if "entity_column" not in dataset_info:
        dataset_info["entity_column"] = entity_col
        dataset_info["n_entities"] = rng.randint(1500, 3000)
        dataset_info["samples_per_entity_avg"] = round(rng.uniform(1.5, 3.0), 1)
    exp["dataset_info"] = dataset_info

    # Inject into split_config
    split_config = exp.get("split_config", {})
    train_entities = list(split_config.get(
        "train_entities_sample",
        [f"P{str(i).zfill(4)}" for i in range(1, 11)]
    ))
    test_entities = list(split_config.get(
        "test_entities_sample",
        [f"P{str(i+1000).zfill(4)}" for i in range(1, 11)]
    ))

    # Ensure there are enough train entities to draw explicit overlap from.
    if len(train_entities) < n_overlap:
        for i in range(len(train_entities) + 1, n_overlap + 1):
            train_entities.append(f"P{str(i).zfill(4)}")
    overlap_entities = train_entities[:n_overlap]

    # Make them overlap
    test_entities[:n_overlap] = overlap_entities

    split_config["train_entities_sample"] = train_entities
    split_config["test_entities_sample"] = test_entities
    split_config.pop("entity_overlap_count", None)
    split_config["entity_column"] = entity_col
    exp["split_config"] = split_config

    # Update preprocessing to show entity-unaware split while preserving V1 code when present.
    v7_snippet = (
        "# Entity-unaware split\n"
        f"# Same {entity_col}s can appear in both train and test\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42\n"
        ")  # standard random split"
    )
    existing_code = exp.get("preprocessing", {}).get("code_snippet", "")
    has_v1_pattern = ("fit_transform" in existing_code) or ("fit(" in existing_code)
    if existing_code and has_v1_pattern:
        exp["preprocessing"]["code_snippet"] = f"{existing_code}\n\n# V7: entity split issue\n{v7_snippet}"
    else:
        exp["preprocessing"]["code_snippet"] = v7_snippet

    exp["ground_truth"]["violations"].append("V7")
    return exp


def inject_V8(exp: dict) -> dict:
    """
    V8: Multi-Test Leakage.
    Same test set used for model selection AND final evaluation with no holdout.
    Hyperparameters tuned on test set, then reported on same test set.
    """
    # Update validation_strategy to show test set misuse
    exp["validation_strategy"].update({
        "method": "test set used for both tuning and evaluation",
        "holdout_set": "none",
        "hyperparameter_search": "grid_search_on_test",
    })

    # Add incriminating note to experiment_notes
    existing_notes = exp.get("experiment_notes", "")
    exp["experiment_notes"] = (
        f"{existing_notes}\n\n"
        "Hyperparameters tuned on test set. Best model selected based on test accuracy. "
        "Final metrics reported on same test set used for hyperparameter selection."
    ).strip()

    # Keep eval_report free of explicit warning labels.
    exp["eval_report"].pop("_note", None)

    exp["ground_truth"]["violations"].append("V8")
    return exp


def inject_compound(exp: dict, v_type_a: str, v_type_b: str) -> dict:
    """
    Apply two violation injectors to the same experiment and merge GT labels.

    Supported ordered pairs:
      ("V1", "V5"), ("V3", "V6"), ("V2", "V7")
    """
    injector_map = {
        "V1": inject_V1, "V2": inject_V2, "V3": inject_V3,
        "V4": inject_V4, "V5": inject_V5, "V6": inject_V6,
        "V7": inject_V7, "V8": inject_V8,
    }
    if v_type_a not in injector_map or v_type_b not in injector_map:
        raise ValueError(f"Unknown violation type: {v_type_a} or {v_type_b}")

    clean_exp = deepcopy(exp)
    exp_a = injector_map[v_type_a](deepcopy(clean_exp))
    gt_a = list(exp_a.get("ground_truth", {}).get("violations", []))

    exp_ab = injector_map[v_type_b](exp_a)
    gt_after = list(exp_ab.get("ground_truth", {}).get("violations", []))

    merged_gt = []
    for v in gt_a + gt_after:
        if v not in merged_gt:
            merged_gt.append(v)

    exp_ab["ground_truth"]["violations"] = merged_gt
    exp_ab["_ground_truth"] = merged_gt
    exp_ab["_is_compound"] = True
    exp_ab["_compound_types"] = [v_type_a, v_type_b]
    return exp_ab


# ══════════════════════════════════════════════════════════════════════════════
# RED HERRING HELPERS (suspicious but not violations)
# ══════════════════════════════════════════════════════════════════════════════


def add_red_herring_lr(exp: dict) -> dict:
    """Unusual learning rate — looks suspicious but is not a violation."""
    hp = exp.get("model_config", {}).get("hyperparams", {})
    hp["learning_rate"] = 0.001
    hp["_note"] = "Low LR used due to noisy gradients"
    exp["model_config"]["hyperparams"] = hp
    return exp


def add_red_herring_early_overfit(exp: dict) -> dict:
    """Overfitting in early training that resolves — not a violation."""
    exp["training_logs"] = (
        "Iteration 10:  train RMSE = 142.1,  val RMSE = 298.4  [appears to be overfitting]\n"
        "Iteration 50:  train RMSE = 158.3,  val RMSE = 201.7\n"
        "Iteration 100: train RMSE = 165.4,  val RMSE = 183.2\n"
        "Iteration 200: train RMSE = 158.2,  val RMSE = 166.3\n"
        "Note: Early overfitting resolved after LR warmup (epochs 1-15). Final model generalizes well."
    )
    return exp


def add_red_herring_high_accuracy(exp: dict) -> dict:
    """Suspiciously high accuracy that is actually genuine."""
    reported = exp.get("eval_report", {}).get("reported_metrics", {})
    if "accuracy" in reported:
        reported["accuracy"] = 0.982
        exp["eval_report"]["reported_metrics"] = reported
        exp["eval_report"]["best_score"] = 0.982

    dataset_info = exp.get("dataset_info", {})
    dataset_info["_note"] = "Well-separated classes; high accuracy is expected on this dataset"
    exp["dataset_info"] = dataset_info
    return exp


def add_red_herring_entity_grouping(exp: dict) -> dict:
    """
    Entities are grouped but CORRECTLY stratified (NOT a violation).
    Looks suspicious but split is done properly by entity.
    """
    # Never overwrite V7/V1 evidence in violated experiments.
    if any(v in exp.get("ground_truth", {}).get("violations", []) for v in ["V7", "V1"]):
        exp["experiment_notes"] = (
            exp.get("experiment_notes", "") +
            "\n\nAnalyst note: grouped entities exist; verify split strategy carefully."
        ).strip()
        return exp

    dataset_info = exp.get("dataset_info", {})
    entity_col = "patient_id"
    dataset_info["entity_column"] = entity_col
    dataset_info["n_entities"] = 2000
    dataset_info["samples_per_entity_avg"] = 2.5
    exp["dataset_info"] = dataset_info

    # Show proper entity-aware splitting
    split_config = exp.get("split_config", {})
    split_config["entity_column"] = entity_col
    split_config["entity_overlap_count"] = 0  # Correctly zero overlap
    split_config["train_entities_sample"] = ["P0001", "P0002", "P0003", "P0004", "P0005"]
    split_config["test_entities_sample"] = ["P1501", "P1502", "P1503", "P1504", "P1505"]  # Non-overlapping
    split_config["groupby_split"] = True
    exp["split_config"] = split_config

    # Show correct GroupKFold/GroupShuffleSplit usage
    exp["preprocessing"]["code_snippet"] = (
        "# Entity-aware split using GroupShuffleSplit\n"
        "from sklearn.model_selection import GroupShuffleSplit\n"
        "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n"
        f"train_idx, test_idx = next(gss.split(X, y, groups=df['{entity_col}']))\n"
        "# Ensures no entity appears in both train and test"
    )
    return exp


def add_red_herring_test_size(exp: dict) -> dict:
    """
    Very small test set that looks suspicious but is valid.
    Sometimes small test sets are justified (rare disease studies, etc.).
    """
    split_config = exp.get("split_config", {})
    split_config["test_size"] = 0.05  # Only 5%
    split_config["test_size_actual"] = exp.get("dataset_info", {}).get("n_samples", 8000) // 20
    split_config["_justification"] = "Small test set justified due to limited rare disease samples"
    exp["split_config"] = split_config

    # Add justification in notes
    exp["experiment_notes"] = (
        exp.get("experiment_notes", "") +
        "\n\nSmall test set (5%) was used due to limited availability of rare disease samples. "
        "Statistical power analysis confirmed sufficient samples for reliable evaluation."
    ).strip()
    return exp


def add_red_herring_validation_tuning(exp: dict) -> dict:
    """
    Hyperparameters tuned on VALIDATION set (not test set) — this is correct.
    Looks like V8 but is actually proper methodology.
    """
    # Never overwrite V8 evidence in violated experiments.
    if "V8" in exp.get("ground_truth", {}).get("violations", []):
        exp["experiment_notes"] = (
            exp.get("experiment_notes", "") +
            "\n\nTeam discussed CV-based tuning as a future improvement."
        ).strip()
        return exp

    exp["validation_strategy"].update({
        "method": "cross_validation + separate holdout test",
        "holdout_set": "20% held out for final evaluation",
        "hyperparameter_search": "grid_search_on_cv",
        "cv_folds": 5
    })

    exp["experiment_notes"] = (
        exp.get("experiment_notes", "") +
        "\n\nHyperparameters tuned using 5-fold cross-validation on training set only. "
        "Test set kept completely isolated until final evaluation."
    ).strip()
    return exp


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


def generate(
    archetype: str,
    violations: list[str],
    seed: int,
    red_herrings: list[str] | None = None
) -> dict[str, Any]:
    """
    Generate one experiment by applying violations to the base template.

    Args:
        archetype: One of tabular_clf, timeseries_reg, tabular_multi, tabular_survival
        violations: List of violation codes to inject (V1-V8)
        seed: Random seed for reproducibility
        red_herrings: Optional list of red herring types

    Returns:
        Complete experiment dict with ground truth
    """
    if archetype not in _TEMPLATES:
        raise ValueError(f"Unknown archetype: {archetype}")

    exp = deepcopy(_TEMPLATES[archetype])

    # Set experiment ID — opaque to prevent agents from reading violation type
    # from the ID. The original debug ID is stored in _debug_id (never sent to agents).
    violation_str = "_".join(violations) if violations else "clean"
    exp["_debug_id"] = f"{archetype}_seed{seed}_{violation_str}"
    exp["experiment_id"] = f"{archetype}_{seed:03d}"
    exp["_seed"] = seed
    exp["_is_clean"] = len(violations) == 0

    # Initialize ground truth
    exp["ground_truth"] = {"violations": [], "expected_verdict": "pass"}

    # Define available artifacts based on archetype
    base_artifacts = [
        "dataset_info",
        "preprocessing",
        "split_config",
        "model_config",
        "training_logs",
        "eval_report",
        "experiment_notes"
    ]
    extra_artifacts = {
        # timeseries episodes can include V5; expose run_history/validation_strategy
        # so cross-artifact checks are possible instead of impossible-by-design.
        "timeseries_reg": ["feature_engineering", "validation_strategy", "run_history"],
        "tabular_multi": ["validation_strategy", "run_history"],
        "tabular_clf": ["validation_strategy", "run_history"],
        "tabular_survival": ["validation_strategy", "run_history"],
        "text_clf": ["feature_engineering", "validation_strategy", "run_history"],
    }
    exp["available_artifacts"] = base_artifacts + extra_artifacts.get(archetype, [])

    # Apply violations
    for v in violations:
        if v == "V1":
            exp = inject_V1(exp)
        elif v == "V2":
            exp = inject_V2(exp)
        elif v == "V2f":
            exp = inject_V2_feature_engineering(exp)
        elif v == "V3":
            exp = inject_V3(exp)
        elif v == "V4":
            exp = inject_V4(exp, seed)
        elif v == "V5":
            exp = inject_V5(exp, seed)
        elif v == "V6":
            exp = inject_V6(exp)
        elif v == "V7":
            exp = inject_V7(exp, seed)
        elif v == "V8":
            exp = inject_V8(exp)

    # Apply red herrings
    for rh in (red_herrings or []):
        if rh == "lr":
            exp = add_red_herring_lr(exp)
        elif rh == "overfit":
            exp = add_red_herring_early_overfit(exp)
        elif rh == "high_acc":
            exp = add_red_herring_high_accuracy(exp)
        elif rh == "entity_grouping":
            exp = add_red_herring_entity_grouping(exp)
        elif rh == "test_size":
            exp = add_red_herring_test_size(exp)
        elif rh == "validation_tuning":
            exp = add_red_herring_validation_tuning(exp)

    # Keep violated experiments free of helper overlap labels so agents must
    # compare ID/entity lists directly instead of reading metadata shortcuts.
    if exp["ground_truth"]["violations"] and isinstance(exp.get("split_config"), dict):
        exp["split_config"].pop("overlap_count", None)

    # Set expected verdict based on violations
    if exp["ground_truth"]["violations"]:
        severe_violations = {"V1", "V2", "V3", "V4", "V7", "V8"}
        if any(v in severe_violations for v in exp["ground_truth"]["violations"]):
            exp["ground_truth"]["expected_verdict"] = "reject"
        else:
            exp["ground_truth"]["expected_verdict"] = "revise"

    return exp


# ══════════════════════════════════════════════════════════════════════════════
# BUILD EXPERIMENT POOL AT MODULE LOAD TIME
# ══════════════════════════════════════════════════════════════════════════════


def _build_pool() -> dict[str, list[dict]]:
    """
    Build the original 56-experiment pool (v1.0).
    Called once at module import time — no cold-start delay.

    Pool structure (56 total):
    - Easy (10): single violation, no red herrings
    - Medium (10): two violations, 1-2 red herrings
    - Hard (16): three violations + 2 red herrings (10) plus 6 compound pairs
    - Clean (20): no violations, various red herrings

    Every violation type (V1-V8) appears in at least 3 experiments per tier.
    Every archetype (tabular_clf, timeseries_reg, tabular_multi, tabular_survival) appears in every tier.
    """
    pool: dict[str, list[dict]] = {
        "easy": [],
        "medium": [],
        "hard": [],
        "clean": [],
    }

    # ── Easy: single violation, no red herrings ──────────────────────────────
    pool["easy"].append(generate("tabular_clf", ["V1"], seed=0))
    pool["easy"].append(generate("tabular_clf", ["V3"], seed=1))
    pool["easy"].append(generate("tabular_multi", ["V4"], seed=2))
    pool["easy"].append(generate("tabular_survival", ["V7"], seed=3))
    pool["easy"].append(generate("tabular_clf", ["V8"], seed=4))
    pool["easy"].append(generate("tabular_survival", ["V1"], seed=5))
    pool["easy"].append(generate("timeseries_reg", ["V2"], seed=6))
    pool["easy"].append(generate("tabular_multi", ["V6"], seed=7))
    pool["easy"].append(generate("tabular_multi", ["V5"], seed=8))
    pool["easy"].append(generate("tabular_survival", ["V3"], seed=9))

    # ── Medium: 2 violations, cross-artifact reasoning, 1-2 red herrings ─────
    pool["medium"].append(generate("timeseries_reg", ["V2", "V1"], seed=10, red_herrings=["lr"]))
    pool["medium"].append(generate("tabular_clf", ["V1", "V6"], seed=11, red_herrings=["high_acc"]))
    pool["medium"].append(generate("tabular_multi", ["V4", "V6"], seed=12, red_herrings=["lr"]))
    pool["medium"].append(generate("tabular_survival", ["V7", "V6"], seed=13, red_herrings=["entity_grouping"]))
    pool["medium"].append(generate("tabular_clf", ["V1", "V8"], seed=14, red_herrings=["validation_tuning"]))
    pool["medium"].append(generate("tabular_survival", ["V4", "V8"], seed=15, red_herrings=["test_size"]))
    pool["medium"].append(generate("timeseries_reg", ["V2f", "V5"], seed=16, red_herrings=["overfit"]))
    pool["medium"].append(generate("tabular_clf", ["V3", "V6"], seed=17, red_herrings=["high_acc"]))
    pool["medium"].append(generate("tabular_multi", ["V1", "V7"], seed=18, red_herrings=["entity_grouping"]))
    pool["medium"].append(generate("tabular_survival", ["V3", "V5"], seed=19, red_herrings=["lr"]))

    # ── Hard: 3 violations, all multi-artifact, 2 red herrings ───────────────
    pool["hard"].append(generate("tabular_multi", ["V4", "V5", "V6"], seed=20,
                                  red_herrings=["lr", "overfit"]))
    pool["hard"].append(generate("tabular_clf", ["V1", "V5", "V6"], seed=21,
                                  red_herrings=["high_acc", "overfit"]))
    pool["hard"].append(generate("tabular_survival", ["V7", "V5", "V8"], seed=22,
                                  red_herrings=["entity_grouping", "validation_tuning"]))
    pool["hard"].append(generate("timeseries_reg", ["V2", "V4", "V5"], seed=23,
                                  red_herrings=["overfit", "lr"]))
    pool["hard"].append(generate("tabular_clf", ["V3", "V7", "V8"], seed=24,
                                  red_herrings=["test_size", "high_acc"]))
    pool["hard"].append(generate("tabular_survival", ["V1", "V4", "V6"], seed=25,
                                  red_herrings=["entity_grouping", "lr"]))
    pool["hard"].append(generate("timeseries_reg", ["V2", "V1", "V5"], seed=26,
                                  red_herrings=["lr", "overfit"]))
    pool["hard"].append(generate("tabular_multi", ["V1", "V7", "V6"], seed=27,
                                  red_herrings=["entity_grouping", "high_acc"]))
    pool["hard"].append(generate("tabular_clf", ["V4", "V8", "V5"], seed=28,
                                  red_herrings=["validation_tuning", "overfit"]))
    pool["hard"].append(generate("tabular_survival", ["V7", "V3", "V6"], seed=29,
                                  red_herrings=["test_size", "lr"]))

    # ── Compound violations: hard tier only ──────────────────────────────────
    COMPOUND_PAIRS = [
        ("V1", "V5"),
        ("V1", "V5"),
        ("V3", "V6"),
        ("V3", "V6"),
        ("V2", "V7"),
        ("V2", "V7"),
    ]
    COMPOUND_SEEDS = [701, 702, 703, 704, 705, 706]

    for (va, vb), seed in zip(COMPOUND_PAIRS, COMPOUND_SEEDS):
        if "V2" in (va, vb) or "V7" in (va, vb):
            archetype = "timeseries_reg"
        elif "V6" in (va, vb):
            archetype = "tabular_multi"
        else:
            archetype = "tabular_clf"

        base = deepcopy(_TEMPLATES[archetype])
        base["ground_truth"] = {"violations": [], "expected_verdict": "pass"}
        base["difficulty"] = "hard"
        base["archetype"] = archetype
        debug_id = f"compound_{va}_{vb}_{seed}"
        base["_debug_id"] = debug_id
        base["experiment_id"] = f"compound_{seed:03d}"
        base["id"] = f"compound_{seed:03d}"
        base["_seed"] = seed
        base["_is_clean"] = False

        base_artifacts = [
            "dataset_info",
            "preprocessing",
            "split_config",
            "model_config",
            "training_logs",
            "eval_report",
            "experiment_notes",
        ]
        extra_artifacts = {
            "timeseries_reg": ["feature_engineering", "validation_strategy", "run_history"],
            "tabular_multi": ["validation_strategy", "run_history"],
            "tabular_clf": ["validation_strategy", "run_history"],
            "tabular_survival": ["validation_strategy", "run_history"],
        }
        base["available_artifacts"] = base_artifacts + extra_artifacts.get(archetype, [])

        compound_exp = inject_compound(base, va, vb)
        severe_violations = {"V1", "V2", "V3", "V4", "V7", "V8"}
        if any(v in severe_violations for v in compound_exp["ground_truth"]["violations"]):
            compound_exp["ground_truth"]["expected_verdict"] = "reject"
        else:
            compound_exp["ground_truth"]["expected_verdict"] = "revise"
        if isinstance(compound_exp.get("split_config"), dict):
            compound_exp["split_config"].pop("overlap_count", None)
        pool["hard"].append(compound_exp)

    # ── Adversarially clean: no violations, suspicious surface features ──────
    pool["clean"].append(generate("tabular_clf", [], seed=100, red_herrings=["high_acc"]))
    pool["clean"].append(generate("tabular_clf", [], seed=101, red_herrings=["entity_grouping"]))
    pool["clean"].append(generate("tabular_clf", [], seed=102, red_herrings=["validation_tuning"]))
    pool["clean"].append(generate("tabular_clf", [], seed=103, red_herrings=["test_size"]))
    pool["clean"].append(generate("timeseries_reg", [], seed=104, red_herrings=["overfit"]))
    pool["clean"].append(generate("timeseries_reg", [], seed=105, red_herrings=["lr"]))
    pool["clean"].append(generate("timeseries_reg", [], seed=106, red_herrings=["lr", "overfit"]))
    pool["clean"].append(generate("tabular_multi", [], seed=107, red_herrings=["high_acc"]))
    pool["clean"].append(generate("tabular_multi", [], seed=108, red_herrings=["validation_tuning"]))
    pool["clean"].append(generate("tabular_survival", [], seed=109, red_herrings=["entity_grouping"]))
    pool["clean"].append(generate("tabular_survival", [], seed=110, red_herrings=["test_size"]))
    pool["clean"].append(generate("tabular_survival", [], seed=111, red_herrings=["high_acc", "lr"]))
    pool["clean"].append(generate("tabular_clf", [], seed=112, red_herrings=["lr", "entity_grouping"]))
    pool["clean"].append(generate("tabular_clf", [], seed=113, red_herrings=["overfit", "test_size"]))
    pool["clean"].append(generate("timeseries_reg", [], seed=114, red_herrings=["high_acc", "overfit"]))
    pool["clean"].append(generate("timeseries_reg", [], seed=115, red_herrings=["entity_grouping"]))
    pool["clean"].append(generate("tabular_multi", [], seed=116, red_herrings=["lr", "overfit"]))
    pool["clean"].append(generate("tabular_multi", [], seed=117, red_herrings=["entity_grouping", "test_size"]))
    pool["clean"].append(generate("tabular_survival", [], seed=118, red_herrings=["validation_tuning", "overfit"]))
    pool["clean"].append(generate("tabular_survival", [], seed=119, red_herrings=["high_acc", "entity_grouping"]))

    return pool


def _build_text_clf_extension() -> dict[str, list[dict]]:
    """
    Build the text_clf NLP archetype extension (6 experiments: 2 easy, 2 medium, 2 hard).
    These are kept separate from the main POOL to preserve backward compatibility
    with published evaluation results (seeds 42-48 on the 56-experiment pool).

    Access via POOL_EXTENDED for the full 62-experiment pool.
    """
    ext: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}

    ext["easy"].append(generate("text_clf", ["V1"], seed=200))
    ext["easy"].append(generate("text_clf", ["V3"], seed=201))

    ext["medium"].append(generate("text_clf", ["V1", "V6"], seed=210, red_herrings=["high_acc"]))
    ext["medium"].append(generate("text_clf", ["V3", "V5"], seed=211, red_herrings=["lr"]))

    ext["hard"].append(generate("text_clf", ["V1", "V3", "V6"], seed=220,
                                 red_herrings=["high_acc", "overfit"]))
    ext["hard"].append(generate("text_clf", ["V1", "V5", "V8"], seed=221,
                                 red_herrings=["validation_tuning", "lr"]))
    return ext


# ── Pool instances ────────────────────────────────────────────────────────────

# Main pool (56 experiments) — used by the live environment for reproducible evals
POOL = _build_pool()

# text_clf extension (6 experiments, NLP archetype)
TEXT_CLF_EXT = _build_text_clf_extension()

# Extended pool (62 experiments = POOL + text_clf) for v1.1 evaluation
POOL_EXTENDED = {
    tier: POOL[tier] + TEXT_CLF_EXT.get(tier, [])
    for tier in POOL
}


def get_pool_stats() -> dict:
    """Get statistics about the experiment pool."""
    return {
        "easy_count": len(POOL["easy"]),
        "medium_count": len(POOL["medium"]),
        "hard_count": len(POOL["hard"]),
        "clean_count": len(POOL["clean"]),
        "total": sum(len(v) for v in POOL.values()),
        "total_extended": sum(len(POOL_EXTENDED[t]) for t in POOL_EXTENDED),
        "text_clf_experiments": sum(len(TEXT_CLF_EXT[t]) for t in TEXT_CLF_EXT),
    }