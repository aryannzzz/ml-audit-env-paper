#!/usr/bin/env python3
"""
Multi-provider evaluation harness with HF token rotation.

Runs open-weight models via HuggingFace Inference Router and OpenAI models.
Handles 429 quota errors by rotating through HF_TOKEN_1/2/3.

Usage:
    # Start server first: uvicorn app:app --port 7860
    python scripts/run_hf_eval.py --seeds 45 46
    python scripts/run_hf_eval.py --seeds 42 43 44 45 46 47 48 49 50 51
    python scripts/run_hf_eval.py --seeds 45 46 --models hf_qwen25 hf_llama
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Ensure we run from the neurips_submission root
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# Load .env for API keys
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

# ── Model definitions ─────────────────────────────────────────────────────────

HF_BASE_URL = "https://router.huggingface.co/v1"

MODEL_REGISTRY = {
    # OpenAI (use existing OPENAI_API_KEY)
    "gpt-4.1-mini": {
        "model_name": "gpt-4.1-mini",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "display": "GPT-4.1-mini (fast)",
        "is_reasoning": False,
    },
    "gpt-4.1": {
        "model_name": "gpt-4.1",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "display": "GPT-4.1 (balanced)",
        "is_reasoning": False,
    },
    "o4-mini": {
        "model_name": "o4-mini",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "display": "o4-mini (reasoning)",
        "is_reasoning": True,
    },
    # Open-weight via HF Inference Router
    "hf_qwen25": {
        "model_name": "Qwen/Qwen2.5-72B-Instruct",
        "provider": "hf",
        "base_url": HF_BASE_URL,
        "key_env": "HF_TOKEN_1",
        "display": "Qwen2.5-72B-Instruct (open)",
        "is_reasoning": False,
    },
    "hf_llama": {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "provider": "hf",
        "base_url": HF_BASE_URL,
        "key_env": "HF_TOKEN_1",
        "display": "Llama-3.1-70B-Instruct (open)",
        "is_reasoning": False,
    },
    "hf_qwq": {
        "model_name": "Qwen/QwQ-32B",
        "provider": "hf",
        "base_url": HF_BASE_URL,
        "key_env": "HF_TOKEN_1",
        "display": "QwQ-32B (open reasoning)",
        "is_reasoning": True,
    },
    "hf_deepseek": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "provider": "hf",
        "base_url": HF_BASE_URL,
        "key_env": "HF_TOKEN_1",
        "display": "DeepSeek-R1-Distill-32B (open reasoning)",
        "is_reasoning": True,
    },
}

DEFAULT_MODELS = ["hf_qwen25", "hf_llama"]
DEFAULT_SEEDS = [45, 46]
TIERS = ["easy", "medium", "hard"]
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
LOG_DIR = ROOT_DIR / "logs" / "runs"
RESULTS_FILE = ROOT_DIR / "eval_results.json"


# ── HF token rotation ─────────────────────────────────────────────────────────

HF_TOKENS = [
    os.getenv("HF_TOKEN_1", ""),
    os.getenv("HF_TOKEN_2", ""),
    os.getenv("HF_TOKEN_3", ""),
]
HF_TOKENS = [t for t in HF_TOKENS if t]
_hf_token_idx = 1  # Start from token 2 (index 1) — token 1 is depleted


def get_hf_token() -> str:
    """Return the current active HF token."""
    global _hf_token_idx
    if not HF_TOKENS:
        return ""
    return HF_TOKENS[_hf_token_idx % len(HF_TOKENS)]


def rotate_hf_token():
    """Advance to the next HF token (called on 429 quota errors)."""
    global _hf_token_idx
    _hf_token_idx += 1
    if _hf_token_idx >= len(HF_TOKENS):
        print("  WARNING: All HF tokens exhausted — no more rotation available.",
              file=sys.stderr)
        _hf_token_idx = len(HF_TOKENS) - 1
    else:
        print(f"  Rotated to HF token {_hf_token_idx + 1}/{len(HF_TOKENS)}", file=sys.stderr)


# ── Server check ──────────────────────────────────────────────────────────────

def check_server() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{ENV_URL}/health", timeout=5) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception as e:
        print(f"  Server not reachable: {e}", file=sys.stderr)
        return False


# ── Run one episode ───────────────────────────────────────────────────────────

def run_episode(model_key: str, tier: str, seed: int,
                dry_run: bool = False, max_retries: int = 2) -> tuple[float | None, str]:
    """
    Run one episode. Returns (score, log_text).
    Handles HF 429 errors by rotating tokens and retrying.
    """
    cfg = MODEL_REGISTRY[model_key]
    model_name = cfg["model_name"]
    base_url = cfg["base_url"]
    provider = cfg["provider"]

    for attempt in range(max_retries + 1):
        # Pick API key
        if provider == "hf":
            api_key = get_hf_token()
        else:
            api_key = os.getenv(cfg["key_env"], "")

        if not api_key and not dry_run:
            print(f"  No API key for {model_key}", file=sys.stderr)
            return None, ""

        env = os.environ.copy()
        env["MODEL_NAME"] = model_name
        env["API_BASE_URL"] = base_url
        env["OPENAI_API_KEY"] = api_key
        env["SEED"] = str(seed)
        env["MAX_EPISODES"] = "1"
        env["TASK_FILTER"] = tier
        env["ENV_URL"] = ENV_URL
        env["REQUEST_TIMEOUT"] = "60"
        if cfg["is_reasoning"]:
            env.pop("TEMPERATURE", None)
        else:
            env["TEMPERATURE"] = "0.0"
        if dry_run:
            env["DRY_RUN"] = "1"

        cmd = [sys.executable, str(ROOT_DIR / "inference.py")]
        try:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=360,
            )
            stdout = result.stdout
            stderr = result.stderr

            # Detect quota / credit-depleted / rate-limit errors
            if provider == "hf" and (
                "429" in stderr or "402" in stderr or
                "quota" in stderr.lower() or "rate limit" in stderr.lower() or
                "depleted" in stderr.lower() or "RESOURCE_EXHAUSTED" in stderr or
                "credits" in stderr.lower()
            ):
                print(f"  Credit/quota error (attempt {attempt+1}/{max_retries+1}), rotating token...",
                      file=sys.stderr)
                rotate_hf_token()
                time.sleep(3)
                continue

            # Extract score from [END] line
            for line in stdout.splitlines():
                if line.startswith("[END]"):
                    m = re.search(r"score=([\d.]+)", line)
                    if m:
                        return float(m.group(1)), stdout + "\n--- STDERR ---\n" + stderr

            # No [END] line found
            return None, stdout + "\n--- STDERR ---\n" + stderr

        except subprocess.TimeoutExpired:
            print(f"  Timeout on attempt {attempt+1}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(10)
            else:
                return None, "TIMEOUT"
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            return None, str(e)

    return None, "MAX_RETRIES_EXCEEDED"


# ── Main ──────────────────────────────────────────────────────────────────────

def load_existing_results() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(existing: dict, new_results: dict, seeds: list[int]):
    """Merge new_results into existing eval_results.json."""
    data = dict(existing)

    # Ensure top-level fields exist
    if "models" not in data:
        data["models"] = []
    if "seeds" not in data:
        data["seeds"] = list(range(42, 49))
    if "results" not in data:
        data["results"] = {}

    # Merge new results
    for display_name, tier_data in new_results.items():
        data["results"][display_name] = tier_data

    # Update seed list to union of existing + new
    all_seeds = sorted(set(data.get("seeds", []) + seeds))
    data["seeds"] = all_seeds
    data["notes"] = data.get("notes", "") + f"\nOpen-weight models added {time.strftime('%Y-%m-%d')}."

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Multi-provider MLAuditBench evaluation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=list(MODEL_REGISTRY), help="Model keys to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Seeds to evaluate")
    parser.add_argument("--tiers", nargs="+", default=TIERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-server-check", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Don't update eval_results.json")
    args = parser.parse_args()

    print("=" * 68)
    print("MLAuditBench — Multi-Provider Evaluation Harness")
    print("=" * 68)
    print(f"Models:  {args.models}")
    print(f"Seeds:   {args.seeds}")
    print(f"Tiers:   {args.tiers}")
    print(f"HF tokens available: {len(HF_TOKENS)}")
    print()

    if not args.skip_server_check:
        print("Checking server...", end=" ", flush=True)
        if not check_server():
            print(f"\nERROR: Server not running at {ENV_URL}")
            print("Start with: cd neurips_submission && uvicorn app:app --port 7860")
            sys.exit(1)
        print("OK")
    print()

    # results[display_name][tier] = {"scores": [...], "mean": x, "std": x, "n": n}
    results: dict = {}
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total = len(args.models) * len(args.tiers) * len(args.seeds)
    done = 0

    for model_key in args.models:
        cfg = MODEL_REGISTRY[model_key]
        display = cfg["display"]
        print(f"\nModel: {display} ({cfg['model_name']})")
        results[display] = {t: {"scores": [], "mean": 0.0, "std": 0.0, "n": 0}
                            for t in args.tiers}

        for tier in args.tiers:
            for seed in args.seeds:
                done += 1
                model_slug = cfg["model_name"].replace("/", "_").replace(".", "-")
                provider = cfg["provider"]
                log_name = f"{provider}_{model_slug}_{tier}_seed{seed}.log"
                log_path = LOG_DIR / log_name

                print(f"  [{done:3}/{total}] {tier:6} seed={seed} ...", end="", flush=True)
                t0 = time.time()

                score, log_text = run_episode(model_key, tier, seed, dry_run=args.dry_run)
                elapsed = time.time() - t0

                if score is not None:
                    results[display][tier]["scores"].append(score)
                    print(f" score={score:.3f}  ({elapsed:.0f}s)")
                else:
                    print(f" FAILED  ({elapsed:.0f}s)")

                # Write log
                with open(log_path, "w") as f:
                    f.write(f"# Model: {cfg['model_name']}\n")
                    f.write(f"# Provider: {provider}\n")
                    f.write(f"# Tier: {tier}  Seed: {seed}\n")
                    f.write(f"# Score: {score}\n\n")
                    f.write(log_text)

            # Compute stats for this tier
            scores = results[display][tier]["scores"]
            if scores:
                mean = statistics.mean(scores)
                std = statistics.stdev(scores) if len(scores) > 1 else 0.0
                results[display][tier].update({
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "n": len(scores),
                })
                print(f"    {tier}: mean={mean:.3f} ± {std:.3f}  (n={len(scores)})")

    # Print summary
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    for display, tier_data in results.items():
        avgs = [tier_data[t]["mean"] for t in args.tiers if tier_data[t]["scores"]]
        overall = statistics.mean(avgs) if avgs else 0.0
        print(f"\n{display}")
        for tier in args.tiers:
            d = tier_data[tier]
            if d["scores"]:
                print(f"  {tier:8}: {d['mean']:.3f} ± {d['std']:.3f}")
        print(f"  {'avg':8}: {overall:.3f}")

    # Save
    if not args.no_save:
        existing = load_existing_results()
        save_results(existing, results, args.seeds)

    print("\nDone.")


if __name__ == "__main__":
    main()
