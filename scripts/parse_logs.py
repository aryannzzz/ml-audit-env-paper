#!/usr/bin/env python3
"""
Parse MLAuditBench inference logs and compute baseline statistics.

Parses [START]/[STEP]/[END] log lines produced by inference.py.

Usage:
    python scripts/parse_logs.py [log_dir] [--output output_dir]

Output:
    - Formatted table to stdout
    - baseline_stats.json in output_dir
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_log_file(path: Path) -> list:
    episodes = []
    current = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[START]"):
                task_m = re.search(r"task=(\S+)", line)
                model_m = re.search(r"model=(\S+)", line)
                current = {
                    "task": task_m.group(1) if task_m else "unknown",
                    "model": model_m.group(1) if model_m else "unknown",
                    "score": None,
                    "steps": None,
                    "violations_flagged": [],
                    "source_file": path.name,
                }
            elif line.startswith("[STEP]") and current is not None:
                if "violation_type" in line:
                    vtype_m = re.search(r'"violation_type"\s*:\s*"(V\d+)"', line)
                    if vtype_m:
                        current["violations_flagged"].append(vtype_m.group(1))
            elif line.startswith("[END]") and current is not None:
                score_m = re.search(r"score=([\d.]+)", line)
                steps_m = re.search(r"steps=(\d+)", line)
                if score_m:
                    current["score"] = float(score_m.group(1))
                if steps_m:
                    current["steps"] = int(steps_m.group(1))
                if current["score"] is not None:
                    episodes.append(current)
                current = None
    return episodes


def compute_stats(episodes: list) -> dict:
    groups = defaultdict(list)
    for ep in episodes:
        groups[(ep["model"], ep["task"])].append(ep)
    stats = {}
    for (model, task), eps in groups.items():
        scores = [e["score"] for e in eps if e["score"] is not None]
        steps_list = [e["steps"] for e in eps if e["steps"] is not None]
        stats[(model, task)] = {
            "n": len(scores),
            "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
            "std_score": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "min_score": round(min(scores), 4) if scores else 0.0,
            "max_score": round(max(scores), 4) if scores else 0.0,
            "mean_steps": round(statistics.mean(steps_list), 1) if steps_list else 0.0,
        }
    return stats


def compute_violation_freq(episodes: list) -> dict:
    freq = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(int)
    for ep in episodes:
        counts[ep["model"]] += 1
        for v in ep["violations_flagged"]:
            freq[ep["model"]][v] += 1
    return {
        model: {v: round(freq[model][v] / counts[model], 3) for v in freq[model]}
        for model in freq
    }


def print_stats_table(stats: dict):
    models = sorted({k[0] for k in stats})
    tasks = ["easy", "medium", "hard"]
    print("\n" + "=" * 85)
    print("BASELINE STATISTICS  (mean ± std across seeds)")
    print("=" * 85)
    print(f"{'Model':<40} {'Tier':<8} {'N':>3} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    print("-" * 85)
    for model in models:
        short = model.split("/")[-1][:39]
        for task in tasks:
            key = (model, task)
            if key in stats:
                s = stats[key]
                print(f"{short:<40} {task:<8} {s['n']:>3} "
                      f"{s['mean_score']:>7.3f} {s['std_score']:>7.3f} "
                      f"{s['min_score']:>7.3f} {s['max_score']:>7.3f}")
        # Per-model average across tiers
        tier_means = [stats[(model, t)]["mean_score"] for t in tasks if (model, t) in stats]
        if tier_means:
            avg = round(statistics.mean(tier_means), 3)
            print(f"{'  → Average':<40} {'all':<8} {'':>3} {avg:>7.3f}")
        print()
    print("=" * 85)


def print_violation_table(freq: dict):
    violation_types = [f"V{i}" for i in range(1, 9)]
    models = sorted(freq.keys())
    print("\n" + "=" * 85)
    print("VIOLATION DETECTION FREQUENCY  (fraction of episodes per violation type)")
    print("=" * 85)
    header = f"{'Model':<40} " + " ".join(f"{v:>5}" for v in violation_types)
    print(header)
    print("-" * 85)
    for model in models:
        short = model.split("/")[-1][:39]
        row = f"{short:<40} "
        row += " ".join(f"{freq[model].get(v, 0.0):>5.2f}" for v in violation_types)
        print(row)
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Parse MLAuditBench inference logs")
    parser.add_argument("log_dir", nargs="?", default=".", help="Directory with .log files")
    parser.add_argument("--output", default=None, help="Output dir for baseline_stats.json")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output) if args.output else log_dir
    log_files = sorted(log_dir.rglob("*.log"))
    if not log_files:
        print(f"No .log files found in {log_dir}")
        return

    print(f"Found {len(log_files)} log files")
    all_episodes = []
    for lf in log_files:
        eps = parse_log_file(lf)
        if eps:
            all_episodes.extend(eps)
            print(f"  {lf.name}: {len(eps)} episode(s)")

    print(f"\nTotal episodes parsed: {len(all_episodes)}")
    if not all_episodes:
        print("No valid episodes found.")
        return

    stats = compute_stats(all_episodes)
    freq = compute_violation_freq(all_episodes)
    print_stats_table(stats)
    print_violation_table(freq)

    output = {
        "total_episodes": len(all_episodes),
        "per_model_tier": {
            f"{model}|{task}": v
            for (model, task), v in stats.items()
        },
        "violation_frequency": {model: d for model, d in freq.items()},
    }
    out_path = output_dir / "baseline_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
