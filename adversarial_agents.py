"""
adversarial_agents.py
=====================
Three adversarial baseline agents for MLAuditBench red-team evaluation.
None of these agents make LLM API calls. They represent the ceiling that
pure pattern-matching can achieve, establishing a lower bound on benchmark
discriminability.

Usage:
    python adversarial_agents.py           # runs all 3 agents, prints scores
    python adversarial_agents.py --agent pattern_matcher
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from typing import Any

from environment.env import MLAuditEnv
from environment.models import Action

SUSPICIOUS_KEYWORDS = {
    "V1": ["fit_transform", "StandardScaler", "fit(X)"],
    "V2": ["shuffle", "random_state", "TimeSeriesSplit"],
    "V3": ["target", "label", "y_col", "outcome"],
    "V4": ["overlap", "duplicate", "train_ids", "test_ids"],
    "V5": ["run", "attempt", "trial", "experiment"],
    "V6": ["metric", "accuracy", "auc", "f1", "precision", "recall"],
    "V7": ["patient", "entity", "group", "subject"],
    "V8": ["test set", "holdout", "tuning", "grid search"],
}

KNOWN_TARGET_NAMES = ["readmission_30d", "price", "survival_months", "label"]
VIOLATIONS = [f"V{i}" for i in range(1, 9)]


@dataclass
class FlagDecision:
    violation_type: str
    evidence_artifact: str
    evidence_quote: str
    severity: str = "medium"


class BaseAgent:
    name = "base"

    def run_episode(self, task: str, seed: int) -> float:
        env = MLAuditEnv(task=task)
        obs = env.reset(seed=seed)
        available = list(obs.available_artifacts)
        cache: dict[str, str] = {}
        max_inspects = max(1, obs.step_budget - 4)

        for artifact in self.inspect_order(available)[:max_inspects]:
            step_obs, _, done, info = env.step(Action(type="inspect", artifact=artifact))
            if step_obs.last_action_result:
                cache[artifact] = str(step_obs.last_action_result)
            if done:
                return float(info.get("score", 0.0))

        flags = self.detect(cache)
        for f in flags:
            _, _, done, info = env.step(
                Action(
                    type="flag",
                    violation_type=f.violation_type,
                    evidence_artifact=f.evidence_artifact,
                    evidence_quote=f.evidence_quote,
                    severity=f.severity,
                )
            )
            if done:
                return float(info.get("score", 0.0))

        _, _, done, info = env.step(Action(type="submit", verdict="reject", summary=f"{self.name} baseline"))
        if done:
            return float(info.get("score", 0.0))
        return float(info.get("score", 0.0))

    def inspect_order(self, available: list[str]) -> list[str]:
        return available

    def detect(self, cache: dict[str, str]) -> list[FlagDecision]:
        return []


class PatternMatcherAgent(BaseAgent):
    name = "PatternMatcherAgent"

    def inspect_order(self, available: list[str]) -> list[str]:
        priority = ["preprocessing", "split_config"]
        rest = [a for a in available if a not in priority]
        return [a for a in priority if a in available] + rest

    @staticmethod
    def _extract_quote(text: str, pattern: str) -> str | None:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        return text[m.start() : min(len(text), m.end() + 10)].strip()[:80]

    @staticmethod
    def _extract_first_list_overlap(text: str, train_key: str, test_key: str) -> str | None:
        train_match = re.search(rf'"{train_key}"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
        test_match = re.search(rf'"{test_key}"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
        if not train_match or not test_match:
            return None

        train_blob = train_match.group(1)
        test_blob = test_match.group(1)
        candidates = re.findall(r'"[^"]+"|\b\d+\b', train_blob)[:5]
        for c in candidates:
            token = c.strip()
            if token and token in test_blob:
                return token[:80]
        return None

    def detect(self, cache: dict[str, str]) -> list[FlagDecision]:
        flags: list[FlagDecision] = []
        pre = cache.get("preprocessing", "")
        split_cfg = cache.get("split_config", "")
        model_cfg = cache.get("model_config", "")
        dataset_info = cache.get("dataset_info", "")
        run_history = cache.get("run_history", "")
        notes = cache.get("experiment_notes", "")
        val = cache.get("validation_strategy", "")
        report = cache.get("eval_report", "")
        everything = "\n".join(cache.values())

        if pre:
            idx_fit = pre.lower().find("fit_transform")
            idx_split = pre.lower().find("train_test_split")
            if idx_fit != -1 and idx_split != -1 and idx_fit < idx_split:
                quote = self._extract_quote(pre, r"fit_transform\s*\(") or "fit_transform"
                flags.append(FlagDecision("V1", "preprocessing", quote, "high"))

        if split_cfg and re.search(r'"shuffle"\s*:\s*true', split_cfg, flags=re.IGNORECASE):
            quote = self._extract_quote(split_cfg, r'"shuffle"\s*:\s*true') or '"shuffle": true'
            flags.append(FlagDecision("V2", "split_config", quote, "high"))

        if model_cfg and dataset_info:
            target_match = re.search(r'"target_column"\s*:\s*"([^"]+)"', dataset_info)
            candidate_targets = KNOWN_TARGET_NAMES[:]
            if target_match:
                candidate_targets.insert(0, target_match.group(1))
            for target in candidate_targets:
                if target and target in model_cfg:
                    flags.append(FlagDecision("V3", "model_config", target[:80], "high"))
                    break

        v4_quote = self._extract_first_list_overlap(split_cfg, "train_ids_sample", "test_ids_sample")
        if v4_quote is not None:
            flags.append(FlagDecision("V4", "split_config", v4_quote, "high"))

        if run_history and notes:
            total_runs = re.search(r'"total_runs"\s*:\s*(\d+)', run_history)
            one_run_claim = re.search(r'\b(one run|single run|single experiment)\b', notes, flags=re.IGNORECASE)
            if total_runs and int(total_runs.group(1)) > 1 and one_run_claim:
                quote = total_runs.group(0)[:80]
                flags.append(FlagDecision("V5", "run_history", quote, "medium"))

        if val and report:
            tracked = re.search(r'"metrics_tracked"\s*:\s*\[(.*?)\]', val, flags=re.DOTALL)
            reported = re.search(r'"reported_metrics"\s*:\s*\{(.*?)\}', report, flags=re.DOTALL)
            if tracked and reported:
                tracked_count = len(re.findall(r'"[^"]+"', tracked.group(1)))
                reported_count = len(re.findall(r'"[^"]+"\s*:', reported.group(1)))
                if tracked_count >= 3 and reported_count < tracked_count:
                    quote = tracked.group(0)[:80]
                    flags.append(FlagDecision("V6", "validation_strategy", quote, "medium"))

        v7_quote = self._extract_first_list_overlap(split_cfg, "train_entities_sample", "test_entities_sample")
        if v7_quote is not None:
            flags.append(FlagDecision("V7", "split_config", v7_quote, "high"))

        if re.search(r'(test|holdout).{0,40}(hyperparameter|tuning)|'
                     r'(hyperparameter|tuning).{0,40}(test|holdout)', everything, flags=re.IGNORECASE):
            quote = self._extract_quote(everything, r'(test|holdout).{0,40}(hyperparameter|tuning)|'
                                                  r'(hyperparameter|tuning).{0,40}(test|holdout)')
            flags.append(FlagDecision("V8", "experiment_notes", quote or "test set tuning", "high"))

        dedup: dict[str, FlagDecision] = {}
        for f in flags:
            dedup.setdefault(f.violation_type, f)
        return list(dedup.values())


class KeywordCounterAgent(BaseAgent):
    name = "KeywordCounterAgent"
    threshold = 3

    def detect(self, cache: dict[str, str]) -> list[FlagDecision]:
        text = "\n".join(cache.values()).lower()
        by_violation: list[tuple[str, int]] = []
        for v, kws in SUSPICIOUS_KEYWORDS.items():
            count = 0
            for kw in kws:
                count += text.count(kw.lower())
            by_violation.append((v, count))

        chosen = [v for v, c in sorted(by_violation, key=lambda x: x[1], reverse=True) if c > self.threshold]
        if not chosen:
            chosen = [v for v, c in sorted(by_violation, key=lambda x: x[1], reverse=True)[:2] if c > 0]

        default_artifact = next(iter(cache.keys()), "dataset_info")
        return [
            FlagDecision(v, default_artifact, f"keyword-count signal for {v}"[:80], "medium")
            for v in chosen
        ]


class RandomFlagAgent(BaseAgent):
    name = "RandomFlagAgent"

    def __init__(self, seed_offset: int = 0):
        self.seed_offset = seed_offset

    def detect(self, cache: dict[str, str]) -> list[FlagDecision]:
        rng = random.Random(self.seed_offset + len(cache))
        k = rng.randint(0, 3)
        chosen = rng.sample(VIOLATIONS, k=k)
        artifacts = list(cache.keys()) or ["dataset_info"]
        flags = []
        for v in chosen:
            art = rng.choice(artifacts)
            blob = cache.get(art, "")
            if blob:
                start = rng.randint(0, max(0, len(blob) - 1))
                quote = blob[start : min(len(blob), start + 80)]
            else:
                quote = "random evidence"
            flags.append(FlagDecision(v, art, quote or "random evidence", "medium"))
        return flags


def run_benchmark(selected: str | None = None) -> dict[str, dict[str, float]]:
    agents: dict[str, BaseAgent] = {
        "pattern_matcher": PatternMatcherAgent(),
        "keyword_counter": KeywordCounterAgent(),
        "random_flag": RandomFlagAgent(seed_offset=1234),
    }
    if selected:
        if selected not in agents:
            raise ValueError(f"Unknown agent '{selected}'. Choose from: {', '.join(agents)}")
        agents = {selected: agents[selected]}

    results: dict[str, dict[str, float]] = {}
    tasks = ["easy", "medium", "hard"]

    for key, agent in agents.items():
        scores: dict[str, float] = {}
        for task in tasks:
            task_scores = []
            for seed in range(5):
                task_scores.append(agent.run_episode(task=task, seed=seed))
            scores[task] = sum(task_scores) / len(task_scores)
        scores["average"] = sum(scores[t] for t in tasks) / len(tasks)
        results[agent.name] = scores

    return results


def print_table(results: dict[str, dict[str, float]]) -> None:
    print("Agent                | Easy  | Medium | Hard  | Average")
    print("---------------------|-------|--------|-------|--------")
    for agent_name, score_map in results.items():
        print(
            f"{agent_name:<20} | "
            f"{score_map['easy']:.2f}  | "
            f"{score_map['medium']:.2f}   | "
            f"{score_map['hard']:.2f}  | "
            f"{score_map['average']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial no-LLM baseline agents.")
    parser.add_argument(
        "--agent",
        default=None,
        choices=["pattern_matcher", "keyword_counter", "random_flag"],
        help="Run a single agent instead of all agents.",
    )
    args = parser.parse_args()

    results = run_benchmark(args.agent)
    print_table(results)


if __name__ == "__main__":
    main()
