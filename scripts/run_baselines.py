#!/usr/bin/env python3
"""Run multi-seed MLAuditBench baselines and aggregate NeurIPS error bars."""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
RUN_LOG_DIR = LOG_DIR / "runs"
RESULTS_JSONL = LOG_DIR / "baseline_results.jsonl"
SUMMARY_JSON = LOG_DIR / "baseline_summary.json"
ENV_FILE = ROOT / ".env"
ENV_BAK = ROOT / ".env.bak"
INFER_FILE = ROOT / "inference.py"
INFER_BAK = ROOT / "inference.py.bak"

SEEDS = [42, 43, 44]
TIERS = ["easy", "medium", "hard"]

KEYS_TO_OVERRIDE = {
    "API_BASE_URL",
    "OPENAI_API_KEY",
    "MODEL_NAME",
    "SEED",
    "TASK_FILTER",
    "MAX_EPISODES",
    "TEMPERATURE",
    "MAX_TOKENS",
}

REASONING_ALIASES = {"o4-mini", "o3", "qwen-qwq-32b"}
END_RE = re.compile(r"\[END\]\s+success=(\w+)\s+steps=(\d+)\s+score=([\d.]+)")


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    alias: str
    model_name: str
    api_base_url: str
    key_env: str


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("openai", "gpt-4.1-mini", "gpt-4.1-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ModelSpec("openai", "gpt-4.1", "gpt-4.1", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ModelSpec("openai", "o4-mini", "o4-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ModelSpec("openai", "o3", "o3", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ModelSpec(
        "gemini",
        "gemini-2.0-flash",
        "gemini-2.0-flash",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
    ),
    ModelSpec(
        "gemini",
        "gemini-2.5-pro",
        "gemini-2.5-pro",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
    ),
    ModelSpec(
        "hf",
        "qwen-2.5-72b",
        "Qwen/Qwen2.5-72B-Instruct",
        "https://api-inference.huggingface.co/v1",
        "HF_API_KEY",
    ),
    ModelSpec(
        "hf",
        "qwen-qwq-32b",
        "Qwen/QwQ-32B",
        "https://api-inference.huggingface.co/v1",
        "HF_API_KEY",
    ),
]


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_backups() -> None:
    shutil.copy2(ENV_FILE, ENV_BAK)
    shutil.copy2(INFER_FILE, INFER_BAK)


def restore_from_backups() -> None:
    if ENV_BAK.exists():
        shutil.copy2(ENV_BAK, ENV_FILE)
    if INFER_BAK.exists():
        shutil.copy2(INFER_BAK, INFER_FILE)


def read_key_from_env_file(env_path: Path, key: str) -> str:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        if k.strip() == key:
            raw = v.strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
                return raw[1:-1]
            return raw
    return ""


def write_env(overrides: Dict[str, str]) -> None:
    lines = ENV_BAK.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines: List[str] = []
    written = set()

    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in KEYS_TO_OVERRIDE and key in overrides:
                new_lines.append(f'{key}="{overrides[key]}"\n')
                written.add(key)
                continue
        new_lines.append(line)

    for key in KEYS_TO_OVERRIDE:
        if key in overrides and key not in written:
            new_lines.append(f'{key}="{overrides[key]}"\n')

    ENV_FILE.write_text("".join(new_lines), encoding="utf-8")


def patch_inference_for_reasoning(enable: bool) -> None:
    if not enable:
        shutil.copy2(INFER_BAK, INFER_FILE)
        return

    src = INFER_BAK.read_text(encoding="utf-8")
    target = "                temperature=TEMPERATURE,\n"
    replacement = ""
    if target not in src:
        raise RuntimeError("Could not find temperature kwarg in inference.py backup")
    patched = src.replace(target, replacement, 1)
    INFER_FILE.write_text(patched, encoding="utf-8")


def parse_end_line(stdout: str) -> Dict[str, object]:
    matches = list(END_RE.finditer(stdout))
    if not matches:
        return {"success": False, "steps": 0, "score": 0.0}
    m = matches[-1]
    return {
        "success": m.group(1).lower() == "true",
        "steps": int(m.group(2)),
        "score": float(m.group(3)),
    }


def run_inference(spec: ModelSpec, tier: str, seed: int, run_index: int, total_runs: int) -> Dict[str, object]:
    key_value = read_key_from_env_file(ENV_BAK, spec.key_env)
    if not key_value:
        raise RuntimeError(f"Missing required key in .env.bak: {spec.key_env}")

    reasoning = spec.alias in REASONING_ALIASES
    max_tokens = "5000" if reasoning else "600"

    overrides = {
        "API_BASE_URL": spec.api_base_url,
        "OPENAI_API_KEY": key_value,
        "MODEL_NAME": spec.model_name,
        "SEED": str(seed),
        "TASK_FILTER": tier,
        "MAX_EPISODES": "1",
        "TEMPERATURE": "0.0",
        "MAX_TOKENS": max_tokens,
    }

    write_env(overrides)
    patch_inference_for_reasoning(enable=reasoning)

    run_name = f"{spec.provider}_{spec.alias}_{tier}_seed{seed}.log"
    log_path = RUN_LOG_DIR / run_name

    if spec.alias == "o3":
        time.sleep(10)
    else:
        time.sleep(2)

    start_ts = now_iso()
    timeout_seconds = 600
    timed_out = False

    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\n# TIMEOUT\n"
        return_code = 124

    end_info = parse_end_line(stdout)

    header = [
        f"# Run: provider={spec.provider} model_alias={spec.alias} model_name={spec.model_name} tier={tier} seed={seed}",
        f"# API_BASE_URL: {spec.api_base_url}",
        f"# Timestamp: {start_ts}",
        f"# Reasoning mode: {reasoning}",
        f"# Return code: {return_code}",
        f"# Timeout: {timed_out}",
        "# -----------------------------------------------",
        "=== STDOUT ===",
        stdout,
        "\n=== STDERR ===",
        stderr,
    ]
    log_path.write_text("\n".join(header), encoding="utf-8")

    if spec.alias == "o3":
        time.sleep(10)
    else:
        time.sleep(2)

    record = {
        "provider": spec.provider,
        "model_alias": spec.alias,
        "model_name": spec.model_name,
        "tier": tier,
        "seed": seed,
        "score": end_info["score"],
        "steps": end_info["steps"],
        "success": end_info["success"],
        "timestamp": now_iso(),
        "log_file": str(log_path.relative_to(ROOT)),
        "returncode": return_code,
        "timeout": timed_out,
    }

    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(
        f"[{run_index}/{total_runs}] {spec.provider}/{spec.alias} {tier} seed={seed} -> score={record['score']:.3f}",
        flush=True,
    )

    return record


def compute_summary(records: List[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for spec in MODEL_SPECS:
        grouped.setdefault(spec.alias, {"easy": [], "medium": [], "hard": []})

    for rec in records:
        grouped[str(rec["model_alias"])][str(rec["tier"])].append(float(rec["score"]))

    out: Dict[str, object] = {"generated_at": now_iso(), "models": {}}
    for alias, tier_map in grouped.items():
        out["models"][alias] = {}
        for tier in TIERS:
            scores = tier_map.get(tier, [])
            mean_v = statistics.mean(scores) if scores else 0.0
            std_v = statistics.stdev(scores) if len(scores) > 1 else 0.0
            out["models"][alias][tier] = {
                "mean": round(mean_v, 4),
                "std": round(std_v, 4),
                "seeds": [round(s, 4) for s in scores],
                "n": len(scores),
            }
    return out


def print_summary_table(summary: Dict[str, object]) -> None:
    print("\n===== BASELINE RESULTS (mean +- std, n=3 seeds) =====")
    print("Model                  | Easy          | Medium        | Hard")
    print("-----------------------|---------------|---------------|---------------")

    models = summary.get("models", {})
    for spec in MODEL_SPECS:
        tiers = models.get(spec.alias, {})
        row = [spec.alias.ljust(23)]
        for tier in TIERS:
            cell = tiers.get(tier, {"mean": 0.0, "std": 0.0})
            row.append(f"{cell['mean']:.3f} +- {cell['std']:.3f}".ljust(13))
        print(f"{row[0]}| {row[1]} | {row[2]} | {row[3]}")


def validate_outputs() -> None:
    line_count = 0
    if RESULTS_JSONL.exists():
        with RESULTS_JSONL.open("r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)

    run_files = sorted(RUN_LOG_DIR.glob("*.log"))
    print(f"\nresults_jsonl_lines={line_count}")
    print(f"run_log_files={len(run_files)}")


def main() -> int:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ensure_backups()

    if RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()

    records: List[Dict[str, object]] = []
    total_runs = len(MODEL_SPECS) * len(TIERS) * len(SEEDS)
    run_index = 0

    # Safety gate for o3: only proceed to seeds 43/44 per tier if seed42 succeeds with a valid END line.
    o3_seed42_ok: Dict[str, bool] = {tier: False for tier in TIERS}

    try:
        for spec in MODEL_SPECS:
            for tier in TIERS:
                for seed in SEEDS:
                    if spec.alias == "o3" and seed in (43, 44) and not o3_seed42_ok[tier]:
                        run_index += 1
                        skipped_log = RUN_LOG_DIR / f"{spec.provider}_{spec.alias}_{tier}_seed{seed}.log"
                        skipped_log.write_text(
                            "\n".join(
                                [
                                    f"# Run skipped due to o3 seed42 failure for tier={tier}",
                                    f"# Timestamp: {now_iso()}",
                                ]
                            ),
                            encoding="utf-8",
                        )
                        record = {
                            "provider": spec.provider,
                            "model_alias": spec.alias,
                            "model_name": spec.model_name,
                            "tier": tier,
                            "seed": seed,
                            "score": 0.0,
                            "steps": 0,
                            "success": False,
                            "timestamp": now_iso(),
                            "log_file": str(skipped_log.relative_to(ROOT)),
                            "returncode": -1,
                            "timeout": False,
                            "skipped": True,
                        }
                        records.append(record)
                        with RESULTS_JSONL.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=True) + "\n")
                        print(
                            f"[{run_index}/{total_runs}] {spec.provider}/{spec.alias} {tier} seed={seed} -> score=0.000 (skipped)",
                            flush=True,
                        )
                        continue

                    run_index += 1
                    rec = run_inference(spec=spec, tier=tier, seed=seed, run_index=run_index, total_runs=total_runs)
                    records.append(rec)

                    if spec.alias == "o3" and seed == 42:
                        o3_seed42_ok[tier] = bool(rec["success"]) or float(rec["score"]) > 0.0

        summary = compute_summary(records)
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print_summary_table(summary)
        validate_outputs()
        return 0
    finally:
        restore_from_backups()


if __name__ == "__main__":
    raise SystemExit(main())
