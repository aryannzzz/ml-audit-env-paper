"""
Tests for step budgets and penalty system.
"""
import pytest
from environment.env import MLAuditEnv, STEP_BUDGETS, STEP_PENALTY, PENALTY_THRESHOLD_RATIO
from environment.models import Action


class TestStepBudgets:
    @pytest.mark.parametrize("task,expected_budget", [
        ("easy", 8),
        ("medium", 12),
        ("hard", 18),
    ])
    def test_budget_value(self, task, expected_budget):
        env = MLAuditEnv(task=task)
        env.reset(task=task, seed=42)
        actual = env._state["step_budget"]
        assert actual == expected_budget, f"{task}: expected {expected_budget}, got {actual}"

    def test_budget_constants_match(self):
        assert STEP_BUDGETS["easy"] == 8
        assert STEP_BUDGETS["medium"] == 12
        assert STEP_BUDGETS["hard"] == 18


class TestBudgetExhaustion:
    def test_budget_exhaustion_terminates_episode(self):
        env = MLAuditEnv(task="easy")
        env.reset(task="easy", seed=42)
        budget = env._state["step_budget"]

        done = False
        for _ in range(budget + 5):
            _, _, done, _ = env.step(Action(type="inspect", artifact="preprocessing"))
            if done:
                break

        assert done, "Episode must terminate when budget exhausted"

    def test_auto_submit_on_exhaustion_gives_score(self):
        env = MLAuditEnv(task="easy")
        env.reset(task="easy", seed=42)
        budget = env._state["step_budget"]

        info = {}
        for _ in range(budget + 5):
            _, _, done, info = env.step(Action(type="inspect", artifact="preprocessing"))
            if done:
                break

        assert "score" in info, "No score in info after budget exhaustion"
        assert 0.0 <= info["score"] <= 1.0


class TestStepPenalty:
    def test_penalty_threshold_is_65_percent(self):
        assert PENALTY_THRESHOLD_RATIO == 0.65

    def test_penalty_values_per_difficulty(self):
        assert STEP_PENALTY["easy"] == 0.005
        assert STEP_PENALTY["medium"] == 0.010
        assert STEP_PENALTY["hard"] == 0.015

    def test_no_penalty_before_threshold(self):
        env = MLAuditEnv(task="easy")
        env.reset(task="easy", seed=42)
        budget = env._state["step_budget"]
        threshold = int(budget * PENALTY_THRESHOLD_RATIO)  # 5 for easy

        rewards = []
        for i in range(threshold):
            _, reward, done, _ = env.step(Action(type="inspect", artifact="split_config"))
            rewards.append((i + 1, reward))
            if done:
                break

        # Before threshold, inspect reward should be pure +0.02 or -0.01
        for step, reward in rewards:
            if step == 1:
                assert reward == pytest.approx(0.02, abs=0.001), f"Step {step}: expected +0.02, got {reward}"

    def test_penalty_after_threshold(self):
        env = MLAuditEnv(task="easy")
        env.reset(task="easy", seed=42)
        budget = env._state["step_budget"]
        threshold = int(budget * PENALTY_THRESHOLD_RATIO)  # 5 for easy

        # Use up steps to reach threshold
        for _ in range(threshold):
            env.step(Action(type="inspect", artifact="split_config"))

        # Next step should have penalty
        _, reward, _, _ = env.step(Action(type="inspect", artifact="model_config"))
        # Re-inspect gives -0.01, plus penalty -0.005 = -0.015
        # First inspect gives +0.02, minus penalty 0.005 = +0.015
        assert reward < 0.02, f"Expected penalty applied, got {reward}"


class TestStepCounting:
    def test_step_count_increases(self):
        env = MLAuditEnv(task="easy")
        env.reset(task="easy", seed=42)

        assert env._state["steps_used"] == 0
        env.step(Action(type="inspect", artifact="preprocessing"))
        assert env._state["steps_used"] == 1
        env.step(Action(type="inspect", artifact="split_config"))
        assert env._state["steps_used"] == 2

    def test_observation_shows_steps_used(self):
        env = MLAuditEnv(task="easy")
        env.reset(task="easy", seed=42)

        obs, _, _, _ = env.step(Action(type="inspect", artifact="preprocessing"))
        assert obs.steps_used == 1

        obs, _, _, _ = env.step(Action(type="inspect", artifact="split_config"))
        assert obs.steps_used == 2

    def test_observation_shows_step_budget(self):
        env = MLAuditEnv(task="medium")
        env.reset(task="medium", seed=42)

        obs, _, _, _ = env.step(Action(type="inspect", artifact="preprocessing"))
        assert obs.step_budget == 12


class TestEfficiencyBonus:
    def test_fast_completion_higher_score(self):
        """Completing in fewer steps should give higher score."""
        # Fast completion
        env1 = MLAuditEnv(task="easy")
        env1.reset(task="easy", seed=42)
        env1.step(Action(type="inspect", artifact="preprocessing"))
        _, _, _, info1 = env1.step(Action(type="submit", verdict="pass", summary="quick"))

        # Slow completion (same experiment)
        env2 = MLAuditEnv(task="easy")
        env2.reset(task="easy", seed=42)
        for _ in range(6):
            env2.step(Action(type="inspect", artifact="preprocessing"))
        _, _, _, info2 = env2.step(Action(type="submit", verdict="pass", summary="slow"))

        # Fast should score higher due to efficiency bonus
        assert info1["score"] >= info2["score"], \
            f"Fast ({info1['score']}) should beat slow ({info2['score']})"
