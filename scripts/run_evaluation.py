#!/usr/bin/env python3
"""
Full multi-model evaluation harness for MLAuditBench.

Runs 3 models × 3 tiers × 3 seeds = 27 episodes.
Writes per-episode logs to eval_logs/ and summary to eval_results.json.

Usage:
    # Start server first: uvicorn app:app --port 7860
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --models gpt-4.1-mini gpt-4.1 o4-mini
    python scripts/run_evaluation.py --seeds 42 43 44 --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
import statistics
from pathlib import Path
from typing import Optional

# Add parent to path so we can import parse_logs
sys.path.insert(0, str(Path(__file__).parent))
from parse_logs import parse_log_file, compute_stats, compute_violation_freq, print_stats_table, print_violation_table

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODELS = ["gpt-4.1-mini", "gpt-4.1", "o4-mini"]

# Fallback chains: if primary not found, try these in order
MODEL_FALLBACKS = {
    "gpt-4.1-mini": ["gpt-4.1-mini", "gpt-4o-mini", "gpt-3.5-turbo-0125"],
    "gpt-4.1":      ["gpt-4.1", "gpt-4o", "gpt-4-turbo"],
    "o4-mini":      ["o4-mini", "o3-mini", "o1-mini", "gpt-4o"],
}

# Models that do not support temperature=0 (reasoning models use their own temp)
REASONING_MODELS = {"o4-mini", "o3-mini", "o1-mini", "o1", "o1-preview", "o3"}

DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_TIERS = ["easy", "medium", "hard"]
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
LOG_DIR = Path("eval_logs")
RESULTS_FILE = Path("eval_results.json")

# ── Helpers ──────────────────────────────────────────────────────────────────

def check_server(env_url: str) -> bool:
    """Return True if the benchmark server is reachable."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{env_url}/health", timeout=5) as r:
            data = json.loads(r.read())
            return data.get("status") == "ok"
    except Exception as e:
        print(f"  Server not reachable at {env_url}/health: {e}", file=sys.stderr)
        return False


def get_available_models(api_key: str) -> set:
    """Fetch available model IDs from OpenAI API."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            return {m["id"] for m in data.get("data", [])}
    except Exception as e:
        print(f"  Could not fetch model list: {e}", file=sys.stderr)
        return set()


def resolve_model(desired: str, available: set) -> str:
    """
    Return the best available model for a desired model name.
    If the desired model is available, return it directly.
    Otherwise walk the fallback chain.
    """
    chain = MODEL_FALLBACKS.get(desired, [desired])
    for candidate in chain:
        if candidate in available:
            if candidate != desired:
                print(f"  Model '{desired}' not found. Using fallback: '{candidate}'")
            return candidate
    # Last resort: return the desired name and let the API error naturally
    print(f"  WARNING: No fallback found for '{desired}'. Trying anyway.")
    return desired


def build_env(model: str, seed: int, max_episodes: int = 1) -> dict:
    """Build environment variables for one inference.py run."""
    env = os.environ.copy()
    env["MODEL_NAME"] = model
    env["SEED"] = str(seed)
    env["MAX_EPISODES"] = str(max_episodes)
    env["TASK_FILTER"] = ""     # run all 3 tiers in one call — handled externally
    env["ENV_URL"] = ENV_URL
    # Reasoning models don't support temperature=0
    if model in REASONING_MODELS:
        # Reasoning models manage their own temperature; do not set TEMPERATURE.
        env.pop("TEMPERATURE", None)
    else:
        env["TEMPERATURE"] = "0.0"
    return env


def run_single_episode(model: str, tier: str, seed: int, log_path: Path,
                       dry_run: bool = False) -> Optional[float]:
    """
    Run one episode (one model, one tier, one seed) via inference.py.
    Captures [START]/[STEP]/[END] output to log_path.
    Returns the final score from the [END] line, or None on failure.
    """
    env = os.environ.copy()
    env["MODEL_NAME"] = model
    env["SEED"] = str(seed)
    env["MAX_EPISODES"] = "1"
    env["TASK_FILTER"] = tier
    env["ENV_URL"] = ENV_URL
    if model in REASONING_MODELS:
        env.pop("TEMPERATURE", None)
    else:
        env["TEMPERATURE"] = "0.0"
    if dry_run:
        env["DRY_RUN"] = "1"

    cmd = [sys.executable, "inference.py"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{tier.upper():6}] seed={seed} model={model.split('/')[-1]}", end="", flush=True)
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max per episode
        )
        stdout = result.stdout
        stderr = result.stderr

        # Write combined log
        with open(log_path, "w") as f:
            f.write(stdout)
            if stderr.strip():
                f.write("\n--- STDERR ---\n")
                f.write(stderr)

        # Extract score from [END] line
        for line in stdout.splitlines():
            if line.startswith("[END]"):
                import re
                m = re.search(r"score=([\d.]+)", line)
                if m:
                    score = float(m.group(1))
                    elapsed = time.time() - t0
                    print(f"  → score={score:.3f}  ({elapsed:.0f}s)")
                    return score

        print(f"  → no [END] found  ({time.time()-t0:.0f}s)")
        return None

    except subprocess.TimeoutExpired:
        print(f"  → TIMEOUT after 300s")
        return None
    except Exception as e:
        print(f"  → ERROR: {e}")
        return None


def generate_latex_table(results: dict) -> str:
    """
    Generate a LaTeX table from results dict.
    results format:
      {model_display_name: {tier: [score, score, score]}}
    """
    models = list(results.keys())
    tiers = ["easy", "medium", "hard"]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-model baseline scores on \textsc{MLAuditBench}")
    lines.append(r"(mean $\pm$ std over seeds 42--44). Higher is better.}")
    lines.append(r"\label{tab:baseline}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Agent} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} & \textbf{Average} \\")
    lines.append(r"\midrule")

    # Adversarial baseline row
    lines.append(r"Random agent & $\sim$0.16 & $\sim$0.16 & $\sim$0.15 & $\sim$0.16 \\")

    for model_name, tier_scores in results.items():
        row_parts = [f"{model_name}"]
        tier_avgs = []
        for tier in tiers:
            scores = tier_scores.get(tier, [])
            if scores:
                mean = statistics.mean(scores)
                std = statistics.stdev(scores) if len(scores) > 1 else 0.0
                tier_avgs.append(mean)
                if std > 0.001:
                    row_parts.append(f"${mean:.3f}\\pm{std:.3f}$")
                else:
                    row_parts.append(f"${mean:.3f}$")
            else:
                row_parts.append("--")
        avg = statistics.mean(tier_avgs) if tier_avgs else 0.0
        row_parts.append(f"${avg:.3f}$")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Model display names for paper ────────────────────────────────────────────

MODEL_DISPLAY_NAMES = {
    "gpt-4.1-mini":      "GPT-4.1-mini (fast)",
    "gpt-4o-mini":       "GPT-4o-mini (fast)",
    "gpt-4.1":           "GPT-4.1 (balanced)",
    "gpt-4o":            "GPT-4o (balanced)",
    "o4-mini":           "o4-mini (reasoning)",
    "o3-mini":           "o3-mini (reasoning)",
    "o1-mini":           "o1-mini (reasoning)",
    "gpt-3.5-turbo-0125": "GPT-3.5-turbo (fast)",
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run MLAuditBench multi-model evaluation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model names to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Seeds to run per tier")
    parser.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS,
                        help="Tiers to evaluate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use heuristic agent (no API calls)")
    parser.add_argument("--skip-server-check", action="store_true",
                        help="Skip server health check")
    args = parser.parse_args()

    print("=" * 65)
    print("MLAuditBench — Multi-Model Evaluation Harness")
    print("=" * 65)
    print(f"Models:  {args.models}")
    print(f"Seeds:   {args.seeds}")
    print(f"Tiers:   {args.tiers}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Server check
    if not args.skip_server_check:
        print("Checking server...")
        if not check_server(ENV_URL):
            print(f"\nERROR: Server not running at {ENV_URL}")
            print("Start it with:  uvicorn app:app --port 7860")
            print("Then re-run this script.")
            sys.exit(1)
        print("  Server OK")
    print()

    # Model resolution
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    print("Resolving models...")
    available_models = get_available_models(api_key) if api_key else set()
    resolved_models = {}
    for m in args.models:
        resolved = resolve_model(m, available_models)
        resolved_models[m] = resolved
        print(f"  {m:20} → {resolved}")
    print()

    # Run all episodes
    total_episodes = len(resolved_models) * len(args.tiers) * len(args.seeds)
    print(f"Running {total_episodes} episodes ({len(resolved_models)} models × "
          f"{len(args.tiers)} tiers × {len(args.seeds)} seeds)...")
    print()

    # results[model_display][tier] = [score, score, ...]
    results = {}
    all_log_files = []

    for desired_model, actual_model in resolved_models.items():
        display = MODEL_DISPLAY_NAMES.get(actual_model, actual_model.split("/")[-1])
        results[display] = {tier: [] for tier in args.tiers}
        print(f"Model: {display} ({actual_model})")

        for tier in args.tiers:
            for seed in args.seeds:
                log_name = f"{actual_model.replace('/', '_')}_{tier}_seed{seed}.log"
                log_path = LOG_DIR / log_name
                score = run_single_episode(
                    actual_model, tier, seed, log_path, dry_run=args.dry_run
                )
                if score is not None:
                    results[display][tier].append(score)
                all_log_files.append(log_path)
        print()

    # Parse logs and compute stats
    print("Computing statistics...")
    all_episodes = []
    for lf in all_log_files:
        if lf.exists():
            eps = parse_log_file(lf)
            all_episodes.extend(eps)

    if all_episodes:
        stats = compute_stats(all_episodes)
        freq = compute_violation_freq(all_episodes)
        print_stats_table(stats)
        print_violation_table(freq)
    else:
        print("No episodes parsed from logs.")

    # Generate LaTeX table
    print("\n" + "=" * 65)
    print("LATEX TABLE (paste into paper/main.tex replacing existing Table 1):")
    print("=" * 65)
    latex = generate_latex_table(results)
    print(latex)

    # Save results JSON
    serialisable_results = {}
    for model_disp, tier_scores in results.items():
        serialisable_results[model_disp] = {}
        for tier, scores in tier_scores.items():
            if scores:
                serialisable_results[model_disp][tier] = {
                    "scores": scores,
                    "mean": round(statistics.mean(scores), 4),
                    "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
                    "n": len(scores),
                }
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "models": args.models,
            "resolved_models": resolved_models,
            "seeds": args.seeds,
            "tiers": args.tiers,
            "results": serialisable_results,
            "latex_table": latex,
        }, f, indent=2)
    print(f"\nSaved: {RESULTS_FILE}")
    print(f"Logs:  {LOG_DIR}/")
    print("\nDone. Paste the LaTeX table above into paper/main.tex (Table 1).")
    print("Then update checklist.tex Q7 to \\answerYes{} with the actual ± values.")


if __name__ == "__main__":
    main()
