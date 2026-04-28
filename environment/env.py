"""
MLAuditEnv — Main environment class.
Implements reset() / step() / state() per OpenEnv spec.
"""
import json
import random
from copy import deepcopy
from typing import Any

from environment.models import Action, Observation, FlagRecord, EpisodeState
from environment.grader import grade, grade_single_flag
from environment.generator import POOL


# Step budgets per task difficulty
# Hard increased to 18 to handle compound violations (require 4+ artifacts)
STEP_BUDGETS = {
    "easy": 8,
    "medium": 12,
    "hard": 18,
}

# Probability of drawing a clean experiment by task (anti-exploit)
CLEAN_RATIO = {
    "easy": 0.50,
    "medium": 0.35,
    "hard": 0.20,
}

# Per-tier seed offsets so that the same integer seed produces INDEPENDENT
# episode draws for easy vs medium vs hard.  Without this, random.seed(N)
# produces identical clean/violated outcomes across all three tiers, inflating
# cross-tier correlation and overall score variance.
_TIER_SEED_OFFSETS = {"easy": 0, "medium": 10_000, "hard": 20_000}

STEP_PENALTY = {
    "easy": 0.005,
    "medium": 0.010,
    "hard": 0.015,
}
PENALTY_THRESHOLD_RATIO = 0.65


class MLAuditEnv:
    """
    ML Experiment Integrity Auditor environment.

    Episode lifecycle:
    1. reset(task) — start new episode with random experiment
    2. step(action) — execute action, receive observation and reward
    3. Episode ends when agent calls submit() or exceeds step budget
    """

    def __init__(self, task: str = "easy"):
        """
        Initialize environment with task difficulty.

        Args:
            task: "easy" | "medium" | "hard"
        """
        if task not in STEP_BUDGETS:
            raise ValueError(f"task must be one of {list(STEP_BUDGETS.keys())}, got '{task}'")

        self.task = task
        self._state: dict[str, Any] | None = None

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def reset(self, task: str | None = None, seed: int | None = None) -> Observation:
        """
        Start a new episode.

        Returns initial observation with artifact list but no content.
        50% chance of drawing a clean experiment to prevent exploit.

        Args:
            task: Optional task override for this episode.
            seed: Optional random seed for deterministic experiment selection.
                  If provided, selects experiment deterministically.
        """
        if task is not None:
            if task not in STEP_BUDGETS:
                raise ValueError(f"task must be one of {list(STEP_BUDGETS.keys())}, got '{task}'")
            self.task = task

        # Save original random state if seed provided
        if seed is not None:
            random_state = random.getstate()
            # Include tier offset so different tiers get independent episode draws
            # for the same seed value, preventing cross-tier correlation.
            random.seed(seed + _TIER_SEED_OFFSETS[self.task])

        # Task-specific chance of drawing clean experiment
        if random.random() < CLEAN_RATIO[self.task] and POOL["clean"]:
            exp = deepcopy(random.choice(POOL["clean"]))
            is_clean = True
        else:
            exp = deepcopy(random.choice(POOL[self.task]))
            is_clean = exp.get("_is_clean", False)

        # Restore random state if seed was used
        if seed is not None:
            random.setstate(random_state)

        self._state = {
            "exp": exp,
            "inspected": {},           # artifact_name -> content
            "flags": [],               # list of flag dicts
            "steps_used": 0,
            "step_budget": STEP_BUDGETS[self.task],
            "cumulative_reward": 0.0,
            "done": False,
            "is_clean": is_clean,
            "final_score": None,
        }

        return self._build_obs(
            result=None,
            reward=0.0,
            msg="New experiment loaded. Inspect artifacts to begin audit.",
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one action.

        Args:
            action: Action model with type and relevant fields

        Returns:
            (observation, reward, done, info)
            info contains {"score": float} when done=True
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state["done"]:
            raise RuntimeError("Episode finished. Call reset() to start new episode.")

        s = self._state
        s["steps_used"] += 1
        reward = 0.0
        step_penalty = 0.0
        result_text = ""
        error_text = None

        # Step penalty after configured threshold (encourages efficiency)
        penalty_threshold = int(s["step_budget"] * PENALTY_THRESHOLD_RATIO)
        if s["steps_used"] > penalty_threshold:
            step_penalty = -STEP_PENALTY[self.task]

        # Check if budget exhausted BEFORE processing action
        if s["steps_used"] > s["step_budget"]:
            # Auto-terminate episode: compute final score based on current flags
            gt = s["exp"]["ground_truth"]
            score, breakdown = grade(
                flags=s["flags"],
                ground_truth=gt,
                steps_used=s["steps_used"],
                budget=s["step_budget"],
                verdict="reject",  # Default to reject when budget exhausted
                inspected=s["inspected"],
            )
            s["final_score"] = score
            s["done"] = True

            done = True
            info = {"score": score}
            msg = f"Budget exhausted. Auto-terminating with score: {score:.4f}"
            obs = self._build_obs("", 0.0, msg)
            return obs, 0.0, done, info

        # Dispatch to action handler
        if action.type == "inspect":
            reward, result_text, error_text = self._handle_inspect(action)

        elif action.type == "compare":
            reward, result_text, error_text = self._handle_compare(action)

        elif action.type == "flag":
            reward, result_text, error_text = self._handle_flag(action)

        elif action.type == "unflag":
            reward, result_text, error_text = self._handle_unflag(action)

        elif action.type == "submit":
            reward, result_text, error_text, s["done"] = self._handle_submit(action)

        else:
            error_text = f"Unknown action type: '{action.type}'"
            reward -= 0.01

        reward += step_penalty

        s["cumulative_reward"] += reward

        done = s["done"]
        info = {"score": s["final_score"]} if done and s["final_score"] is not None else {}

        msg = f"{s['step_budget'] - s['steps_used']} steps remaining."
        if done:
            msg = f"Episode complete. Final score: {s['final_score']:.4f}"

        obs = self._build_obs(result_text, reward, msg, error=error_text)
        return obs, round(reward, 4), done, info

    def state(self) -> EpisodeState:
        """
        Return current episode state for checkpointing/debugging.
        Does NOT include ground truth violations.
        """
        if self._state is None:
            raise RuntimeError("Call reset() first")

        s = self._state
        exp = s["exp"]

        return EpisodeState(
            experiment_id=exp["experiment_id"],
            archetype=exp.get("archetype", "unknown"),
            task_difficulty=self.task,
            dataset_type=exp.get("dataset_info", {}).get("dataset_type", "unknown"),
            steps_used=s["steps_used"],
            step_budget=s["step_budget"],
            inspected_artifacts=list(s["inspected"].keys()),
            flags_raised=list(s["flags"]),
            cumulative_reward=round(s["cumulative_reward"], 4),
            done=s["done"],
            is_clean_experiment=s["is_clean"],
        )

    def close(self):
        """Clean up environment state."""
        self._state = None

    # ══════════════════════════════════════════════════════════════════════════
    # ACTION HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_inspect(self, action: Action) -> tuple[float, str, str | None]:
        """Handle inspect action: read single artifact."""
        s = self._state
        artifact_name = action.artifact
        available = s["exp"].get("available_artifacts", [])

        if artifact_name not in available:
            return -0.01, "", f"'{artifact_name}' not in available_artifacts: {available}"

        content = self._get_artifact_content(artifact_name)

        if artifact_name in s["inspected"]:
            # Re-inspect penalty
            reward = -0.01
        else:
            # First inspect reward
            reward = 0.02
            s["inspected"][artifact_name] = content

        return reward, content, None

    def _handle_compare(self, action: Action) -> tuple[float, str, str | None]:
        """Handle compare action: read two artifacts side-by-side."""
        s = self._state
        available = s["exp"].get("available_artifacts", [])
        parts = []
        reward = 0.0
        errors = []

        for artifact_name in [action.artifact_a, action.artifact_b]:
            if artifact_name not in available:
                errors.append(f"'{artifact_name}' not available")
                continue

            content = self._get_artifact_content(artifact_name)

            if artifact_name not in s["inspected"]:
                reward += 0.015  # Half of inspect reward per artifact
                s["inspected"][artifact_name] = content

            parts.append(f"=== {artifact_name} ===\n{content}")

        if errors:
            return -0.01, "\n\n".join(parts), "; ".join(errors)

        return round(reward, 4), "\n\n".join(parts), None

    def _handle_flag(self, action: Action) -> tuple[float, str, str | None]:
        """Handle flag action: raise a violation flag with evidence."""
        s = self._state
        gt = s["exp"]["ground_truth"]

        flag_dict = {
            "flag_id": f"f{len(s['flags'])}",
            "violation_type": action.violation_type,
            "evidence_artifact": action.evidence_artifact,
            "evidence_quote": action.evidence_quote,
            "severity": action.severity or "medium",
            "step_raised": s["steps_used"],
        }

        # Grade the flag immediately for step reward
        reward, label = grade_single_flag(flag_dict, gt["violations"], s["inspected"])
        s["flags"].append(flag_dict)

        msg = f"Flag {flag_dict['flag_id']} raised ({action.violation_type})."
        return reward, msg, None

    def _handle_unflag(self, action: Action) -> tuple[float, str, str | None]:
        """Handle unflag action: remove a previously raised flag."""
        s = self._state
        gt = s["exp"]["ground_truth"]

        # Find the flag to remove
        target = next((f for f in s["flags"] if f["flag_id"] == action.flag_id), None)
        if target is None:
            return -0.01, "", f"Flag '{action.flag_id}' not found."

        # Was this flag correct?
        original_reward, label = grade_single_flag(target, gt["violations"], s["inspected"])

        if label == "correct":
            # Removing a correct flag is bad
            reward = -0.10
        else:
            # Self-correcting a false positive is good
            reward = 0.05

        s["flags"] = [f for f in s["flags"] if f["flag_id"] != action.flag_id]
        return reward, f"Flag {action.flag_id} removed.", None

    def _handle_submit(self, action: Action) -> tuple[float, str, str | None, bool]:
        """Handle submit action: end episode and compute final score."""
        s = self._state
        gt = s["exp"]["ground_truth"]

        score, breakdown = grade(
            flags=s["flags"],
            ground_truth=gt,
            steps_used=s["steps_used"],
            budget=s["step_budget"],
            verdict=action.verdict,
            inspected=s["inspected"],
        )

        s["final_score"] = score

        # Final reward is the score itself
        reward = score
        msg = f"Episode complete. Final score: {score:.4f}. {len(s['flags'])} flags submitted."

        return reward, msg, None, True  # done=True

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _get_artifact_content(self, name: str) -> str:
        """
        Get artifact content as string.
        Converts dicts to pretty-printed JSON.
        """
        exp = self._state["exp"]

        # Map artifact name to experiment key
        key_map = {
            "dataset_info": "dataset_info",
            "preprocessing": "preprocessing",
            "split_config": "split_config",
            "feature_engineering": "feature_engineering",
            "model_config": "model_config",
            "training_logs": "training_logs",
            "eval_report": "eval_report",
            "experiment_notes": "experiment_notes",
            "validation_strategy": "validation_strategy",
            "run_history": "run_history",
        }

        key = key_map.get(name, name)
        value = exp.get(key)

        if value is None:
            return f"[Artifact '{name}' not found in this experiment]"

        if isinstance(value, str):
            return value
        else:
            return json.dumps(value, indent=2)

    def _build_obs(
        self,
        result: str | None,
        reward: float,
        msg: str,
        error: str | None = None
    ) -> Observation:
        """Build observation from current state."""
        s = self._state
        exp = s["exp"]

        # Goal is specific to the task difficulty, not the same as task_description
        goal_map = {
            "easy": "Find the single violation in this experiment and submit a verdict with evidence.",
            "medium": "Find all two violations (may span multiple artifacts) and submit with evidence.",
            "hard": "Find all three violations while ignoring red herrings. Submit with evidence.",
        }

        return Observation(
            experiment_id=exp["experiment_id"],
            task_description=exp.get("task_description", ""),
            goal=goal_map.get(self.task, "Audit the experiment for violations."),
            dataset_type=exp.get("dataset_info", {}).get("dataset_type", "unknown"),
            available_artifacts=exp.get("available_artifacts", []),
            inspected_artifacts=list(s["inspected"].keys()),
            last_action_result=result,
            last_action_error=error,
            flags_raised=[FlagRecord(**f) for f in s["flags"]],
            steps_used=s["steps_used"],
            step_budget=s["step_budget"],
            step_reward=round(reward, 4),
            cumulative_reward=round(s["cumulative_reward"], 4),
            message=msg,
        )
