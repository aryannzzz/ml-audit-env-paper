#!/usr/bin/env python3
"""
Competition-safe inference script for ML Audit Bench.

Stdout emits only mandatory protocol lines:
- [START] once per episode
- [STEP] once per successful /step call
- [END] once per episode (always, even on exceptions)
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

# Load .env defaults but do not override explicit process env values.
load_dotenv()

try:
    import requests
except Exception as exc:  # pragma: no cover - defensive for validator runtime variance
    requests = None
    REQUESTS_IMPORT_ERROR = exc
else:
    REQUESTS_IMPORT_ERROR = None

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - defensive for validator runtime variance
    OpenAI = None
    OPENAI_IMPORT_ERROR = exc
else:
    OPENAI_IMPORT_ERROR = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid integer for {name}: {raw!r}. Using {default}.", file=sys.stderr, flush=True)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid float for {name}: {raw!r}. Using {default}.", file=sys.stderr, flush=True)
        return default


# Mandatory defaults per competition specification
API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
ENV_URL = os.getenv("ENV_URL") or "http://localhost:7860"

BENCHMARK = "ml-audit-bench"
SEED = _env_int("SEED", 42)
MAX_STEPS = _env_int("MAX_STEPS", 18)
MAX_EPISODES = _env_int("MAX_EPISODES", 1)
MAX_TOKENS = _env_int("MAX_TOKENS", 2048)
MAX_TOKENS_REASONING = _env_int("MAX_TOKENS_REASONING", 4000)
TEMPERATURE = _env_float("TEMPERATURE", 0.0)
REQUEST_TIMEOUT = _env_int("REQUEST_TIMEOUT", 30)
TASK_FILTER = (os.getenv("TASK_FILTER") or "").strip().lower()
DRY_RUN = (os.getenv("DRY_RUN") or "0").strip() == "1"
RETRY_DELAYS = [0.5, 1.0, 2.0]
ENFORCE_COMPARE = (os.getenv("ENFORCE_COMPARE") or "1").strip() == "1"
REASONING_MODELS = {"o3", "o4-mini", "o3-mini", "o1", "o1-mini", "o1-preview"}

SYSTEM_PROMPT = """
You are an ML experiment auditor. Your job: inspect artifacts systematically, flag every real violation you find, then submit the correct verdict.

━━━ VIOLATION TYPES ━━━
Flag ONLY when you have clear positive evidence from an inspected artifact.

V1 Preprocessing Leakage
  WHERE: preprocessing artifact (code_snippet)
  WHAT: scaler/encoder .fit() or .fit_transform() called on full dataset BEFORE train_test_split().
  Example signal: "scaler.fit_transform(X_all)" or "scaler.fit(X)" before "train_test_split".

V2 Temporal Shuffle
  WHERE: dataset_info (dataset_type == "timeseries") AND split_config (shuffle == true). BOTH required.
  WHAT: time-series data shuffled before splitting — destroys temporal ordering.

V3 Target Leakage
  WHERE: model_config (feature_columns list)
  WHAT: the exact string value of target_column appears as an element in feature_columns.
  NOTE: Lag features named differently (e.g. "temp_lag_7") are NOT V3 even if derived from the target.

V4 Train/Test Overlap
  WHERE: split_config (train_ids_sample, test_ids_sample)
  WHAT: one or more specific IDs appear in BOTH lists. Manually check for shared integers.

V5 Cherry-Picking
  WHERE: run_history (total_runs) AND experiment_notes (disclosure check)
  WHAT: total_runs > 1 AND experiment_notes does NOT disclose BOTH the run count AND the selection metric.
  HOW TO DETECT:
    (1) inspect run_history → if total_runs > 1, you have a cherry-picking candidate.
    (2) inspect experiment_notes → proper disclosure requires BOTH:
          - the exact number of runs (e.g. "ran 11 experiments" or "total_runs = 11"), AND
          - the selection criterion (e.g. "selected best by validation RMSE").
        "Final model selected with fixed seed" → NOT proper disclosure (no run count, no metric) → V5.
        "Single experiment run" or "total_runs=1" → only one run → NOT V5.
  VERDICT: revise (not reject).

V6 Metric Shopping
  WHERE: validation_strategy (metrics_tracked list) AND eval_report (reported_metrics)
  WHAT: multiple metrics tracked but only ONE (the best) reported in eval_report.
  HOW TO DETECT: (1) inspect validation_strategy → count metrics_tracked;
                 (2) inspect eval_report → count reported_metrics keys;
                 (3) if tracked > reported, that's V6.
  VERDICT: revise (not reject).

V7 Entity Leakage
  WHERE: split_config (train_entities_sample AND test_entities_sample lists)
  WHAT: same entity IDs appear in BOTH train_entities_sample AND test_entities_sample.
  HOW TO DETECT: look at train_entities_sample and test_entities_sample in split_config.
                 If any entity ID (e.g. "P0001") appears in BOTH lists, that is V7.
  NOTE: If entity lists are disjoint (zero overlap), NOT V7. GroupShuffleSplit in preprocessing = clean.

V8 Multi-Test Leakage
  WHERE: validation_strategy AND experiment_notes
  WHAT: test set used for BOTH hyperparameter tuning AND final evaluation (explicit evidence required).

━━━ INSPECTION STRATEGY ━━━
Inspect artifacts in this priority order. FLAG IMMEDIATELY when you find clear evidence —
do NOT defer flagging to the end. After flagging, continue inspecting remaining artifacts.

Priority: dataset_info → preprocessing → split_config → model_config →
          validation_strategy → eval_report → run_history → experiment_notes →
          training_logs → feature_engineering

KEY RULES:
- Flag as soon as you see evidence. Evidence is in last_artifact_content_returned and inspected_artifact_cache.
- After flagging, keep inspecting — medium experiments have 2 violations, hard have 3.
- Do NOT submit until you have inspected all available artifacts OR steps_remaining == 0.
- When steps_remaining == 1, flag any remaining violations you've found before submitting.
- When steps_remaining == 0, this is your FINAL step — submit immediately.

━━━ EVIDENCE QUOTING ━━━
evidence_quote MUST be verbatim text that appears exactly in the artifact content.
- Copy 10-60 characters of actual code or a field value verbatim. Do NOT paraphrase.
- For code violations: quote the exact offending line, e.g. "fit_transform(X_all)"
- For field violations: quote the exact field content, e.g. "\"shuffle\": true"
- Quote from inspected_artifact_cache (shows your last 8 inspected artifacts) or last_artifact_content_returned.
- The quote must be an exact substring of the artifact — if in doubt, copy more chars.
- Good: "scaler.fit_transform(X_all)"  |  Bad: "scaler was fit on all data before split"
- Good: "\"shuffle\": true"             |  Bad: "shuffle is true"
- Good: "\"total_runs\": 12"           |  Bad: "run_history shows multiple runs"
- Good: "readmitted"  (as a feature_columns entry)  |  Bad: "target appears in features"
If your quote is not an exact substring, the grader rejects it (−0.05 penalty).

━━━ CLEAN EXPERIMENTS ━━━
30-50% of episodes have NO violations. After inspecting artifacts with no clear evidence:
submit verdict=pass. Submitting reject/revise on a clean experiment costs −0.80 in score.

CRITICAL: Do NOT submit verdict=reject or verdict=revise unless you have raised at
least one flag. If you have not flagged anything, submit verdict=pass.
Suspicious-looking experiments (small test sets, multiple metrics) may simply be
well-justified experiments — absence of a flag means absence of proof.

Common red herrings (do NOT flag these):
- Small test set size (e.g. 5%) justified by rare-disease context → NOT a violation
- Multiple metrics tracked AND all reported → only V6 if a tracked metric is MISSING from eval_report
- GroupShuffleSplit used correctly with disjoint entity lists → NOT V7
- Hyperparameters tuned on cross-validation (NOT test set) → NOT V8

━━━ ACTIONS ━━━
{"type":"inspect","artifact":"NAME"}
{"type":"compare","artifact_a":"NAME","artifact_b":"NAME"}
{"type":"flag","violation_type":"V1-V8","evidence_artifact":"NAME","evidence_quote":"EXACT SHORT TEXT","severity":"high|medium|low"}
{"type":"unflag","flag_id":"ID"}
{"type":"submit","verdict":"pass|revise|reject","summary":"one-line reason"}

━━━ RULES ━━━
1. Inspect an artifact before citing evidence from it.
2. Do NOT flag the same violation type twice.
3. False flag penalty: −0.10. Fabricated evidence: −0.05. Only flag with certainty.
4. Verdict: reject for V1/V2/V3/V4/V7/V8 | revise for V5/V6 only | pass if clean.
5. When steps_remaining == 1, flag any remaining violations. When steps_remaining == 0, submit immediately.
6. Use unflag to retract an incorrect flag before submitting.
""".strip()

ACTION_PRIORITY = [
    "dataset_info",
    "preprocessing",
    "split_config",
    "model_config",
    "validation_strategy",
    "eval_report",
    "run_history",
    "experiment_notes",
    "training_logs",
    "feature_engineering",
]

FALLBACK_ACTIONS = [
    {"type": "inspect", "artifact": "dataset_info"},
    {"type": "inspect", "artifact": "preprocessing"},
    {"type": "inspect", "artifact": "split_config"},
    {"type": "inspect", "artifact": "model_config"},
    {"type": "compare", "artifact_a": "validation_strategy", "artifact_b": "eval_report"},
    {"type": "inspect", "artifact": "training_logs"},
]

# Compare hints used for pre-LLM nudges and unit-test helper coverage.
COMPARE_HINTS: Dict[str, Dict[str, Any]] = {
    "V5": {
        "requires": {"run_history", "experiment_notes"},
        "message": "V5 check ready: compare run_history vs experiment_notes for cherry-picking disclosure mismatch.",
    },
    "V6": {
        "requires": {"validation_strategy", "eval_report"},
        "message": "V6 check ready: compare validation_strategy vs eval_report for metric-shopping mismatch.",
    },
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def maybe_add_compare_hint(
    inspected_set: set[str],
    compare_hints: Dict[str, Dict[str, Any]],
    already_hinted: set[str],
) -> Optional[str]:
    """Return one compare hint when prerequisites are met, at most once per violation key."""
    for violation_key, hint in compare_hints.items():
        if violation_key in already_hinted:
            continue
        required = set(hint.get("requires", set()))
        if required and required.issubset(inspected_set):
            already_hinted.add(violation_key)
            return str(hint.get("message", ""))
    return None


def _one_line(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text if text else "null"


def _truncate_text(value: Any, limit: int = 800) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 14)] + "...<truncated>"


def _resolve_api_base_url(api_base_url: str, api_key: str) -> str:
    base = (api_base_url or "").strip()

    if not base:
        return "https://api.openai.com/v1"

    return base


def _canonical_compare_pair(artifact_a: str, artifact_b: str) -> Tuple[str, str]:
    a = str(artifact_a)
    b = str(artifact_b)
    return (a, b) if a <= b else (b, a)


def _summarize_flags(flags: Any) -> List[Dict[str, str]]:
    if not isinstance(flags, list):
        return []

    summaries: List[Dict[str, str]] = []
    for flag in flags:
        if not isinstance(flag, dict):
            continue
        summaries.append(
            {
                "flag_id": str(flag.get("flag_id", "")),
                "violation_type": str(flag.get("violation_type", "")),
                "evidence_artifact": str(flag.get("evidence_artifact", "")),
                "evidence_quote": _truncate_text(flag.get("evidence_quote", ""), 220),
            }
        )
    return summaries


def _extract_compare_sections(result_text: str) -> Dict[str, str]:
    text = (result_text or "").strip()
    if not text:
        return {}

    sections: Dict[str, str] = {}
    markers = list(re.finditer(r"===\s*([A-Za-z0-9_]+)\s*===\n", text))
    if not markers:
        return sections

    for idx, marker in enumerate(markers):
        artifact = marker.group(1).strip()
        content_start = marker.end()
        content_end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        sections[artifact] = text[content_start:content_end].strip()

    return sections


def _update_artifact_cache(action: Dict[str, Any], observation: Dict[str, Any], artifact_cache: Dict[str, str]) -> None:
    if observation.get("last_action_error"):
        return

    result_text = observation.get("last_action_result")
    if not isinstance(result_text, str) or not result_text.strip():
        return

    action_type = str(action.get("type", "")).strip().lower()
    if action_type == "inspect":
        artifact = action.get("artifact")
        if artifact:
            artifact_cache[str(artifact)] = result_text
        return

    if action_type == "compare":
        artifact_a = str(action.get("artifact_a", ""))
        artifact_b = str(action.get("artifact_b", ""))
        sections = _extract_compare_sections(result_text)

        if sections:
            for artifact, content in sections.items():
                artifact_cache[str(artifact)] = content
            return

        # Fallback if compare text does not include section markers.
        if artifact_a:
            artifact_cache[artifact_a] = result_text
        if artifact_b:
            artifact_cache[artifact_b] = result_text


def _required_compare_action(
    observation: Dict[str, Any],
    called_compares: Set[Tuple[str, str]],
    step_index: int,
    budget: int,
) -> Optional[Dict[str, str]]:
    if (step_index + 1) <= (budget / 2):
        return None
    # Reserve last 2 steps for flag + submit; don't consume them with forced compares
    if step_index >= budget - 2:
        return None

    inspected = {str(a) for a in (observation.get("inspected_artifacts") or [])}
    available = {str(a) for a in (observation.get("available_artifacts") or [])}

    required_pairs = [
        ("run_history", "experiment_notes"),
        ("validation_strategy", "eval_report"),
    ]
    for artifact_a, artifact_b in required_pairs:
        pair = _canonical_compare_pair(artifact_a, artifact_b)
        if (
            artifact_a in inspected
            and artifact_b in inspected
            and artifact_a in available
            and artifact_b in available
            and pair not in called_compares
        ):
            return {"type": "compare", "artifact_a": artifact_a, "artifact_b": artifact_b}

    return None


def _recent_repeated_inspect_artifact(last_actions: List[Tuple[str, str]], n: int = 3) -> Optional[str]:
    if len(last_actions) < n:
        return None

    recent = last_actions[-n:]
    first_type, first_artifact = recent[0]
    if first_type != "inspect" or not first_artifact:
        return None

    if all(action_type == "inspect" and artifact == first_artifact for action_type, artifact in recent):
        return first_artifact

    return None


def _loop_break_action(
    observation: Dict[str, Any],
    repeated_artifact: str,
    flagged: Optional[Set[str]] = None,
) -> Dict[str, str]:
    available = [str(a) for a in (observation.get("available_artifacts") or [])]
    inspected = {str(a) for a in (observation.get("inspected_artifacts") or [])}

    for artifact in ACTION_PRIORITY:
        if artifact in available and artifact not in inspected and artifact != repeated_artifact:
            return {"type": "inspect", "artifact": artifact}

    for artifact in available:
        if artifact not in inspected and artifact != repeated_artifact:
            return {"type": "inspect", "artifact": artifact}

    verdict = _smart_verdict(flagged)
    return {
        "type": "submit",
        "verdict": verdict,
        "summary": f"Loop guard: all artifacts inspected. Flags: {sorted(flagged) if flagged else 'none'}",
    }


def _http_request(
    method: str,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if requests is None:
        if REQUESTS_IMPORT_ERROR is not None:
            print(f"requests import unavailable: {REQUESTS_IMPORT_ERROR}", file=sys.stderr, flush=True)
        return None

    url = f"{ENV_URL.rstrip('/')}{endpoint}"
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            else:
                response = requests.post(url, params=params, json=json_body, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception as exc:
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
            else:
                print(f"ENV request failed for {endpoint}: {exc}", file=sys.stderr, flush=True)
    return None


def _parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    raw = (response_text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "type" in parsed:
                return parsed
            nested = parsed.get("action")
            if isinstance(nested, dict) and "type" in nested:
                return nested
    except json.JSONDecodeError:
        pass

    if "```" in raw:
        for index, chunk in enumerate(raw.split("```")):
            if index % 2 == 0:
                continue
            block = chunk.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "type" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

    for candidate in re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw, flags=re.DOTALL):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "type" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, dict) and "type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _smart_verdict(flagged: Optional[Set[str]]) -> str:
    """Determine submit verdict from flagged violation types."""
    if not flagged:
        return "pass"
    reject_types = {"V1", "V2", "V3", "V4", "V7", "V8"}
    revise_types = {"V5", "V6"}
    if flagged & reject_types:
        return "reject"
    if flagged & revise_types:
        return "revise"
    return "pass"


def _fallback_action(
    observation: Dict[str, Any],
    step_index: int,
    budget: int,
    flagged: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    available = [str(a) for a in (observation.get("available_artifacts") or [])]
    inspected = {str(a) for a in (observation.get("inspected_artifacts") or [])}

    # At the very last step, always submit (prevents auto-terminate with forced "reject")
    if step_index >= max(0, budget - 1):
        verdict = _smart_verdict(flagged)
        flagged_str = sorted(flagged) if flagged else "none"
        return {"type": "submit", "verdict": verdict, "summary": f"No additional evidence. Flags: {flagged_str}"}

    for artifact in ACTION_PRIORITY:
        if artifact in available and artifact not in inspected:
            return {"type": "inspect", "artifact": artifact}

    for artifact in available:
        if artifact not in inspected:
            return {"type": "inspect", "artifact": artifact}

    # All inspected, submit early
    verdict = _smart_verdict(flagged)
    flagged_str = sorted(flagged) if flagged else "none"
    return {"type": "submit", "verdict": verdict, "summary": f"All artifacts inspected. Flags: {flagged_str}"}


def _normalize_action(action: Dict[str, Any], observation: Dict[str, Any], step_index: int, budget: int) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return _fallback_action(observation, step_index, budget)

    action_type = str(action.get("type", "")).strip().lower()
    if not action_type:
        return _fallback_action(observation, step_index, budget)

    if action_type in {"load_artifact", "inspect_artifact", "read_artifact"}:
        action_type = "inspect"

    if action_type == "inspect":
        artifact = action.get("artifact") or action.get("artifact_name") or action.get("name")
        if artifact:
            return {"type": "inspect", "artifact": str(artifact)}
        return _fallback_action(observation, step_index, budget)

    if action_type == "compare":
        artifact_a = action.get("artifact_a")
        artifact_b = action.get("artifact_b")
        if artifact_a and artifact_b and str(artifact_a) != str(artifact_b):
            return {"type": "compare", "artifact_a": str(artifact_a), "artifact_b": str(artifact_b)}
        return _fallback_action(observation, step_index, budget)

    if action_type == "flag":
        violation_type = action.get("violation_type")
        evidence_artifact = action.get("evidence_artifact")
        evidence_quote = action.get("evidence_quote")
        severity = action.get("severity") or "medium"
        # Validate violation_type before sending to server to prevent 422 errors.
        # The server's Pydantic Action model rejects any value outside V1-V8,
        # which causes a connection abort that forfeits the entire episode score.
        _VALID_FLAG_TYPES = {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"}
        if violation_type and str(violation_type) not in _VALID_FLAG_TYPES:
            # Invalid violation type: treat as no action and fall back
            return _fallback_action(observation, step_index, budget)
        if violation_type and evidence_artifact and evidence_quote:
            return {
                "type": "flag",
                "violation_type": str(violation_type),
                "evidence_artifact": str(evidence_artifact),
                "evidence_quote": str(evidence_quote),
                "severity": str(severity),
            }
        return _fallback_action(observation, step_index, budget)

    if action_type == "unflag":
        flag_id = action.get("flag_id")
        if flag_id:
            return {"type": "unflag", "flag_id": str(flag_id)}
        return _fallback_action(observation, step_index, budget)

    if action_type == "submit":
        verdict = str(action.get("verdict") or "reject")
        summary = str(action.get("summary") or "Submitting audit result")
        flags_in_obs = observation.get("flags_raised") or []
        if verdict in {"reject", "revise"} and not flags_in_obs:
            # No flags → must be a clean episode
            verdict = "pass"
            summary = f"No flags raised — submitting pass (was: {verdict}). {summary}"
        elif flags_in_obs:
            # Override verdict based on the actual violation types flagged
            flagged_types = {
                f.get("violation_type", "") if isinstance(f, dict) else str(f)
                for f in flags_in_obs
            }
            computed = _smart_verdict(flagged_types)
            if verdict != computed:
                verdict = computed
        return {"type": "submit", "verdict": verdict, "summary": summary}

    return _fallback_action(observation, step_index, budget)


def _build_messages(
    observation: Dict[str, Any],
    step_index: int,
    budget: int,
    artifact_cache: Dict[str, str],
    called_compares: Set[Tuple[str, str]],
    history: List[Tuple[str, str]],
) -> List[Dict[str, str]]:
    inspected_artifacts = [str(a) for a in (observation.get("inspected_artifacts") or [])]
    available_artifacts = [str(a) for a in (observation.get("available_artifacts") or [])]

    # Show full content for the 8 most recently inspected artifacts
    ordered_cache: Dict[str, str] = {}
    for artifact in inspected_artifacts[-8:]:
        if artifact in artifact_cache:
            ordered_cache[artifact] = _truncate_text(artifact_cache[artifact], 900)

    # Track which artifacts still need inspection (helps model plan remaining steps)
    uninspected = [a for a in available_artifacts if a not in set(inspected_artifacts)]

    steps_remaining = max(0, budget - (step_index + 1))
    last_content = _truncate_text(observation.get("last_action_result", ""), 2000)

    # Override the goal field: the env's goal presupposes violation count which misleads
    # the model on clean episodes. Use a neutral directive instead.
    neutral_goal = (
        "This experiment may have 0, 1, 2, or 3 violations. "
        "Inspect artifacts in priority order. FLAG each violation immediately when you find evidence — "
        "then keep inspecting remaining artifacts. "
        "Submit pass if no violations; reject for V1/V2/V3/V4/V7/V8; revise for V5/V6 only."
    )

    content = {
        "step": step_index + 1,
        "budget": budget,
        "steps_remaining": steps_remaining,
        "uninspected_artifacts": uninspected,
        "task_description": observation.get("task_description") or "",
        "goal": neutral_goal,
        "dataset_type": observation.get("dataset_type") or "unknown",
        "available_artifacts": available_artifacts,
        "inspected_artifacts": inspected_artifacts,
        "flags_raised": _summarize_flags(observation.get("flags_raised", [])),
        "compares_already_called": [f"{a}<->{b}" for a, b in sorted(called_compares)],
        "inspected_artifact_cache": ordered_cache,
        "last_artifact_content_returned": last_content,
        "last_action_error": observation.get("last_action_error") or "none",
    }

    # Reminder injected when uninspected artifacts remain and model has been flagging
    flags_so_far = observation.get("flags_raised") or []
    reminder = ""
    if flags_so_far and uninspected and steps_remaining >= 3:
        reminder = (
            f" REMINDER: You flagged {len(flags_so_far)} violation(s) but "
            f"{len(uninspected)} artifact(s) are still uninspected "
            f"({', '.join(uninspected[:4])}). Keep inspecting — there may be more violations."
        )

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history[-3:]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append(
        {
            "role": "user",
            "content": (
                "Choose the next action as one JSON object only. "
                "last_artifact_content_returned is the EXACT TEXT from the most recent inspect/compare — "
                "use it verbatim for evidence_quote (short exact substring only)."
                + reminder
                + "\nCurrent state:\n"
                + json.dumps(content, ensure_ascii=True)
            ),
        }
    )
    return messages


def _llm_call(client: Any, messages: List[Dict[str, str]]) -> str:
    if client is None:
        return ""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
            params: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "timeout": REQUEST_TIMEOUT,
            }
            if model_name in REASONING_MODELS:
                params["max_completion_tokens"] = _env_int("MAX_TOKENS_REASONING", MAX_TOKENS_REASONING)
            else:
                params["max_tokens"] = _env_int("MAX_TOKENS", MAX_TOKENS)
                params["temperature"] = _env_float("TEMPERATURE", 0.0)

            completion = client.chat.completions.create(**params)
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
            else:
                print(f"LLM request failed: {exc}", file=sys.stderr, flush=True)
    return ""


def run_episode(client: Any, task: str, seed: int = 42) -> float:
    step_rewards: List[float] = []
    final_score = 0.0
    total_steps = 0
    response_text = ""
    artifact_cache: Dict[str, str] = {}
    last_actions: List[Tuple[str, str]] = []
    called_compares: Set[Tuple[str, str]] = set()
    dialog_history: List[Tuple[str, str]] = []
    consecutive_failures = 0
    fallback_index = 0
    flagged_violation_types: Set[str] = set()  # prevent duplicate violation type flags

    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    print(f"[START] task={task} env={BENCHMARK} model={model_name}", flush=True)

    try:
        reset_data = _http_request("POST", "/reset", params={"task": task, "seed": seed})
        if reset_data is None:
            # Backward-compatible fallback for environments that do not accept seed.
            reset_data = _http_request("POST", "/reset", params={"task": task})
        if reset_data is None:
            return 0.0

        observation = reset_data.get("observation", reset_data)
        if not isinstance(observation, dict):
            return 0.0

        budget = _env_int("MAX_STEPS", MAX_STEPS)
        try:
            budget = min(budget, int(observation.get("step_budget", budget)))
        except (TypeError, ValueError):
            pass
        if budget <= 0:
            budget = MAX_STEPS

        done = False
        connection_failed = False

        for step_index in range(budget):
            response_text = ""
            llm_messages: List[Dict[str, str]] = []
            forced_compare = _required_compare_action(observation, called_compares, step_index, budget) if ENFORCE_COMPARE else None
            if forced_compare is not None:
                action = forced_compare
            elif DRY_RUN:
                action = _fallback_action(observation, step_index, budget, flagged_violation_types)
            else:
                llm_messages = _build_messages(
                    observation,
                    step_index,
                    budget,
                    artifact_cache,
                    called_compares,
                    dialog_history,
                )
                response_text = _llm_call(client, llm_messages)
                parsed = _parse_action(response_text)
                if parsed is not None:
                    consecutive_failures = 0
                    action = parsed
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 3 and fallback_index < len(FALLBACK_ACTIONS):
                        action = dict(FALLBACK_ACTIONS[fallback_index])
                        fallback_index += 1
                    else:
                        action = _fallback_action(observation, step_index, budget, flagged_violation_types)

                user_msg = llm_messages[-1]["content"] if llm_messages else ""
                assistant_msg = response_text.strip() or json.dumps(action, ensure_ascii=True)
                dialog_history.append((user_msg, assistant_msg))
                if len(dialog_history) > 6:
                    dialog_history = dialog_history[-6:]

            action = _normalize_action(action, observation, step_index, budget)

            # Prevent flagging the same violation type twice (avoids -0.10 re-flag penalty)
            if (
                action.get("type") == "flag"
                and str(action.get("violation_type", "")) in flagged_violation_types
            ):
                action = _fallback_action(observation, step_index, budget, flagged_violation_types)

            repeated_artifact = _recent_repeated_inspect_artifact(last_actions, n=3)
            if (
                repeated_artifact is not None
                and action.get("type") == "inspect"
                and str(action.get("artifact", "")) == repeated_artifact
            ):
                action = _loop_break_action(observation, repeated_artifact, flagged_violation_types)

            # CRITICAL: On the last available step, block inspect AND compare actions.
            # Without this guard, the post-loop submit becomes step budget+1, triggering
            # env auto-terminate with verdict="reject" — which misscores clean episodes.
            # ENFORCE_COMPARE can fire a compare on the last step, so block that too.
            # We allow flag/submit on the last step.
            if step_index >= budget - 1 and action.get("type") in {"inspect", "compare"}:
                action = _fallback_action(observation, step_index, budget, flagged_violation_types)

            result = _http_request("POST", "/step", json_body={"action": action})
            if result is None:
                connection_failed = True
                break

            observation = result.get("observation", result)
            if not isinstance(observation, dict):
                connection_failed = True
                break

            reward = _to_float(result.get("reward", 0.0), default=0.0)
            done = bool(result.get("done", False))
            error_value = observation.get("last_action_error")
            error_text = _one_line(error_value)

            total_steps = step_index + 1
            step_rewards.append(reward)

            _update_artifact_cache(action, observation, artifact_cache)

            action_type = str(action.get("type", "")).strip().lower()
            # Track violation types that have been flagged (only on success, reward > 0)
            if action_type == "flag" and reward > 0:
                vt = str(action.get("violation_type", "")).strip()
                if vt:
                    flagged_violation_types.add(vt)

            if action_type == "compare":
                artifact_a = str(action.get("artifact_a", "")).strip()
                artifact_b = str(action.get("artifact_b", "")).strip()
                if artifact_a and artifact_b:
                    called_compares.add(_canonical_compare_pair(artifact_a, artifact_b))

            if action_type == "inspect":
                last_actions.append(("inspect", str(action.get("artifact", ""))))
            elif action_type == "compare":
                pair_label = f"{action.get('artifact_a', '')}|{action.get('artifact_b', '')}"
                last_actions.append(("compare", pair_label))
            elif action_type == "flag":
                last_actions.append(("flag", str(action.get("violation_type", ""))))
            elif action_type == "submit":
                last_actions.append(("submit", str(action.get("verdict", ""))))
            else:
                last_actions.append((action_type, ""))

            if len(last_actions) > 12:
                last_actions = last_actions[-12:]

            action_str = json.dumps(action, separators=(",", ":"), ensure_ascii=True)
            print(
                f"[STEP] step={total_steps} action={action_str} reward={reward:.2f} "
                f"done={str(done).lower()} error={error_text}",
                flush=True,
            )

            if done:
                info = result.get("info", {})
                if isinstance(info, dict):
                    final_score = _to_float(info.get("score", 0.0), default=0.0)
                break

        if not done and not connection_failed:
            # Choose verdict based on what was flagged (revise-only violations vs reject-severity)
            reject_types = {"V1", "V2", "V3", "V4", "V7", "V8"}
            revise_types = {"V5", "V6"}
            if flagged_violation_types & reject_types:
                budget_verdict = "reject"
            elif flagged_violation_types & revise_types:
                budget_verdict = "revise"
            else:
                budget_verdict = "pass"
            submit_action = {
                "type": "submit",
                "verdict": budget_verdict,
                "summary": f"Reached step budget. Flags raised: {sorted(flagged_violation_types) or 'none'}",
            }
            submit_result = _http_request("POST", "/step", json_body={"action": submit_action})
            if isinstance(submit_result, dict):
                submit_obs = submit_result.get("observation", submit_result)
                submit_reward = _to_float(submit_result.get("reward", 0.0), default=0.0)
                submit_done = bool(submit_result.get("done", False))
                submit_error = _one_line(
                    submit_obs.get("last_action_error") if isinstance(submit_obs, dict) else None
                )

                total_steps += 1
                step_rewards.append(submit_reward)
                submit_str = json.dumps(submit_action, separators=(",", ":"), ensure_ascii=True)
                print(
                    f"[STEP] step={total_steps} action={submit_str} reward={submit_reward:.2f} "
                    f"done={str(submit_done).lower()} error={submit_error}",
                    flush=True,
                )

                info = submit_result.get("info", {})
                if isinstance(info, dict):
                    final_score = _to_float(info.get("score", 0.0), default=0.0)

        return final_score

    except Exception as exc:
        print(f"Episode error for task={task}: {exc}", file=sys.stderr, flush=True)
        if response_text:
            snippet = response_text[:200].replace("\n", " ")
            print(f"Last model response: {snippet}", file=sys.stderr, flush=True)
        return 0.0

    finally:
        try:
            success = (final_score or 0.0) > 0.0
            rewards_str = ",".join(f"{float(r):.2f}" for r in (step_rewards or [])) or "0.00"
            print(
                f"[END] success={str(success).lower()} "
                f"steps={int(total_steps or 0)} "
                f"score={float(final_score or 0.0):.3f} "
                f"rewards={rewards_str}",
                flush=True,
            )
        except Exception:
            print("[END] success=false steps=0 score=0.000 rewards=0.00", flush=True)


def _resolve_tasks() -> List[str]:
    tasks = ["easy", "medium", "hard"]
    task_filter = (os.getenv("TASK_FILTER") or "").strip().lower()
    if not task_filter:
        return tasks
    if task_filter in tasks:
        return [task_filter]
    print(
        f"Invalid TASK_FILTER={task_filter!r}; expected one of easy|medium|hard. Running all tasks.",
        file=sys.stderr,
        flush=True,
    )
    return tasks


_PROVIDER_BASE_URLS = {
    "openai":    "https://api.openai.com/v1",
    "hf":        "https://router.huggingface.co/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "google":    "https://generativelanguage.googleapis.com/v1beta/openai/",
}

_PROVIDER_KEY_ENV = {
    "openai":    "OPENAI_API_KEY",
    "hf":        "HF_TOKEN",
    "deepinfra": "DEEPINFRA_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "google":    "GOOGLE_API_KEY",
}


def _apply_cli_overrides() -> Optional[List[int]]:
    """Parse CLI args and apply overrides to os.environ. Returns seed list if provided."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--provider", default=None, choices=list(_PROVIDER_BASE_URLS),
                        help="Provider: openai|hf|deepinfra|fireworks|google")
    parser.add_argument("--api-base-url", dest="api_base_url", default=None)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--seed-list", dest="seed_list", default=None,
                        help="Comma-separated seeds, e.g. 42,43,44,45,46")
    cli, _ = parser.parse_known_args()

    if cli.model:
        os.environ["MODEL_NAME"] = cli.model
    if cli.provider:
        url = _PROVIDER_BASE_URLS.get(cli.provider, "")
        if url:
            os.environ["API_BASE_URL"] = url
        key_env = _PROVIDER_KEY_ENV.get(cli.provider, "")
        if key_env and not cli.api_key:
            token = os.getenv(key_env) or os.getenv("OPENAI_API_KEY") or ""
            if token:
                os.environ["OPENAI_API_KEY"] = token
    if cli.api_base_url:
        os.environ["API_BASE_URL"] = cli.api_base_url
    if cli.api_key:
        os.environ["OPENAI_API_KEY"] = cli.api_key

    if cli.seed_list:
        try:
            return [int(s.strip()) for s in cli.seed_list.split(",") if s.strip()]
        except ValueError:
            print(f"Invalid --seed-list: {cli.seed_list!r}", file=sys.stderr, flush=True)
    return None


def main() -> int:
    seed_list = _apply_cli_overrides()

    print("Starting ML Audit inference...", file=sys.stderr, flush=True)

    # Re-read after CLI overrides
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
    api_base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    resolved_api_base_url = _resolve_api_base_url(api_base_url, api_key)

    client: Any = None
    if OpenAI is None:
        if OPENAI_IMPORT_ERROR is not None:
            print(f"WARN: OpenAI import unavailable: {OPENAI_IMPORT_ERROR}", file=sys.stderr, flush=True)
    elif not DRY_RUN and api_key:
        try:
            client = OpenAI(base_url=resolved_api_base_url, api_key=api_key)
        except Exception as exc:
            print(f"Failed to initialize OpenAI client: {exc}", file=sys.stderr, flush=True)
            client = None
    elif not DRY_RUN and not api_key:
        print("No API key provided; using fallback policy without LLM calls.", file=sys.stderr, flush=True)

    if seed_list is not None:
        for task in _resolve_tasks():
            for seed in seed_list:
                try:
                    run_episode(client, task=task, seed=seed)
                except Exception as exc:
                    print(f"Unhandled run_episode failure: {exc}", file=sys.stderr, flush=True)
                    continue
    else:
        for task in _resolve_tasks():
            for episode_index in range(MAX_EPISODES):
                try:
                    run_episode(client, task=task, seed=SEED + episode_index)
                except Exception as exc:
                    print(f"Unhandled run_episode failure: {exc}", file=sys.stderr, flush=True)
                    continue

    return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"Fatal main error: {exc}", file=sys.stderr, flush=True)
    sys.exit(0)