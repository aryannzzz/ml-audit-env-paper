#!/usr/bin/env python3
"""
parse_v2_results.py — Parse v2 evaluation log files and append results to
all_results.txt under Sections 11 and 12.

Usage:
    python3 scripts/parse_v2_results.py logs/v2_gpt41_<ts>.txt logs/v2_o4mini_<ts>.txt
"""
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).parent.parent
RESULTS_FILE = REPO / "all_results.txt"

GT = {
    "easy":   {42:["V1"],43:[],44:[],45:[],46:["V2"],47:[],48:["V4"],49:[],50:[],51:[]},
    "medium": {42:["V1","V5"],43:[],44:[],45:[],46:[],47:["V1","V8"],48:["V3","V6"],49:[],50:[],51:["V3","V6"]},
    "hard":   {42:[],43:[],44:["V1","V4","V6"],45:[],46:[],47:["V2","V4","V5"],48:["V3","V6"],49:["V4","V5","V6"],50:[],51:["V4","V5","V6"]},
}

def parse_log(path: str):
    """Return {(model, seed, task): {'score': float, 'flags': [str]}}."""
    results = {}
    with open(path) as f:
        content = f.read()

    # Split into per-seed blocks
    blocks = re.split(r'=== MODEL=(\S+) SEED=(\d+) ATTEMPT=\d+ CODE=\d+ ===', content)
    i = 1
    while i + 2 < len(blocks):
        model = blocks[i]; seed = int(blocks[i+1]); body = blocks[i+2]; i += 3
        task_sections = re.split(r'\[START\] task=(\S+) env=', body)
        j = 1
        while j + 1 < len(task_sections):
            task = task_sections[j]; task_body = task_sections[j+1]; j += 2
            flags = re.findall(r'"violation_type":"([^"]+)".*?reward=([\d.]+)', task_body)
            valid_flags = [vt for vt, rew in flags if float(rew) > 0]
            score_m = re.search(r'\[END\].*?score=([\d.]+)', task_body)
            score = float(score_m.group(1)) if score_m else None
            key = (model, seed, task)
            if key not in results or (results[key]['score'] or 0) < (score or 0):
                results[key] = {'score': score, 'flags': valid_flags}
    return results


def summarise(results, model):
    tiers = ['easy', 'medium', 'hard']
    lines = []
    for tier in tiers:
        scores = []
        for seed in range(42, 52):
            key = (model, seed, tier)
            r = results.get(key, {})
            scores.append(r.get('score') or 0.0)
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        lines.append((tier, scores, mean, std))
    return lines


def format_section(section_num, model_label, model_id, results, logfile):
    lines = summarise(results, model_id)
    today = date.today().isoformat()
    out = []
    out.append("=" * 80)
    out.append(f"{section_num}. {model_label} v2 RESULTS  ({today})")
    out.append(f"    Model: {model_id} | Seeds 42-51 | inference.py v2 | Sequential (temperature=0)")
    out.append("=" * 80)
    for tier, scores, mean, std in lines:
        gt = GT[tier]
        out.append("")
        out.append(f"{tier.upper()} TIER:")
        out.append(f"  seed | score | flags raised         | ground truth")
        for i, seed in enumerate(range(42, 52)):
            key = (model_id, seed, tier)
            r = results.get(key, {})
            score = r.get('score') or 0.0
            flags = r.get('flags') or []
            gt_v = gt[seed]
            out.append(f"  {seed:4d} | {score:.3f} | {str(flags):<20s} | {str(gt_v)}")
        out.append(f"  Mean: {mean:.4f}  Std: {std:.4f}")
        if tier in ('easy', 'medium'):
            out.append(f"  (v1 baseline for reference — see Section 1)")
    out.append(f"\n  Source log: {logfile}")
    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_v2_results.py <gpt41_log> [<o4mini_log>]")
        sys.exit(1)

    model_map = {
        'gpt-4.1': ('GPT-4.1 (balanced)', 'gpt-4.1', 11),
        'o4-mini':  ('o4-mini (reasoning)', 'o4-mini', 12),
    }

    content = RESULTS_FILE.read_text()
    # Remove existing END marker to append
    content = content.rstrip()
    if content.endswith("END OF AGGREGATED RESULTS"):
        content = content[: content.rfind("=" * 10)].rstrip()

    appended = []
    for logfile in sys.argv[1:]:
        results = parse_log(logfile)
        # Detect which model this log is for
        models_found = {k[0] for k in results}
        for model_id in models_found:
            if model_id not in model_map:
                print(f"Unknown model {model_id!r} in {logfile} — skipping.")
                continue
            label, mid, sec = model_map[model_id]
            section_text = format_section(sec, label, mid, results, logfile)
            appended.append(section_text)
            print(f"Prepared Section {sec}: {label}")

    new_content = content + "\n\n" + "\n\n".join(appended) + "\n\n" + \
        "=" * 80 + "\nEND OF AGGREGATED RESULTS\n" + "=" * 80 + "\n"
    RESULTS_FILE.write_text(new_content)
    print(f"\nUpdated {RESULTS_FILE}")


if __name__ == "__main__":
    main()
