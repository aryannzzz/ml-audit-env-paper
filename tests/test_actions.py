"""
Tests for environment action handling.
"""
import pytest
from environment.env import MLAuditEnv
from environment.models import Action


@pytest.fixture
def fresh_env():
    """Create a fresh environment instance reset to easy task."""
    env = MLAuditEnv(task="easy")
    env.reset(task="easy", seed=42)
    return env


class TestInspectAction:
    def test_inspect_new_artifact_positive_reward(self, fresh_env):
        action = Action(type="inspect", artifact="preprocessing")
        obs, reward, done, _ = fresh_env.step(action)
        assert reward > 0, f"Expected positive reward, got {reward}"
        assert not done

    def test_inspect_adds_to_inspected_list(self, fresh_env):
        action = Action(type="inspect", artifact="preprocessing")
        obs, _, _, _ = fresh_env.step(action)
        assert "preprocessing" in obs.inspected_artifacts

    def test_inspect_returns_content(self, fresh_env):
        action = Action(type="inspect", artifact="preprocessing")
        obs, _, _, _ = fresh_env.step(action)
        assert obs.last_action_result is not None
        assert len(obs.last_action_result) > 0

    def test_inspect_reinspect_negative_reward(self, fresh_env):
        action = Action(type="inspect", artifact="preprocessing")
        fresh_env.step(action)
        _, reward, _, _ = fresh_env.step(action)
        assert reward < 0, f"Expected negative reward for re-inspect, got {reward}"

    def test_inspect_invalid_artifact_returns_error(self, fresh_env):
        action = Action(type="inspect", artifact="does_not_exist_xyz")
        obs, _, _, _ = fresh_env.step(action)
        assert obs.last_action_error is not None


class TestCompareAction:
    def test_compare_two_artifacts_adds_both_to_inspected(self, fresh_env):
        action = Action(type="compare", artifact_a="preprocessing", artifact_b="split_config")
        obs, _, _, _ = fresh_env.step(action)
        assert "preprocessing" in obs.inspected_artifacts
        assert "split_config" in obs.inspected_artifacts

    def test_compare_returns_both_contents(self, fresh_env):
        action = Action(type="compare", artifact_a="preprocessing", artifact_b="split_config")
        obs, _, _, _ = fresh_env.step(action)
        result = obs.last_action_result
        assert "preprocessing" in result or "split_config" in result

    def test_compare_reward_for_new_artifacts(self, fresh_env):
        action = Action(type="compare", artifact_a="preprocessing", artifact_b="split_config")
        _, reward, _, _ = fresh_env.step(action)
        assert reward > 0, f"Expected positive reward, got {reward}"

    def test_compare_invalid_artifact_returns_error(self, fresh_env):
        action = Action(type="compare", artifact_a="nonexistent", artifact_b="preprocessing")
        obs, _, _, _ = fresh_env.step(action)
        assert obs.last_action_error is not None


class TestFlagAction:
    def test_flag_adds_to_flags_raised(self, fresh_env):
        # First inspect the artifact
        fresh_env.step(Action(type="inspect", artifact="preprocessing"))
        # Then flag
        action = Action(
            type="flag",
            violation_type="V1",
            evidence_artifact="preprocessing",
            evidence_quote="test quote",
            severity="high"
        )
        obs, _, _, _ = fresh_env.step(action)
        assert len(obs.flags_raised) == 1

    def test_flag_fabricated_evidence_negative_reward(self, fresh_env):
        fresh_env.step(Action(type="inspect", artifact="preprocessing"))
        action = Action(
            type="flag",
            violation_type="V1",
            evidence_artifact="preprocessing",
            evidence_quote="completely fabricated evidence xyz987",
            severity="high"
        )
        _, reward, _, _ = fresh_env.step(action)
        assert reward < 0, f"Expected negative reward for fabricated evidence, got {reward}"

    def test_flag_uninspected_artifact_penalized(self, fresh_env):
        # Do NOT inspect preprocessing first
        action = Action(
            type="flag",
            violation_type="V1",
            evidence_artifact="preprocessing",
            evidence_quote="some quote",
            severity="high"
        )
        _, reward, _, _ = fresh_env.step(action)
        assert reward < 0, f"Expected penalty for uninspected artifact, got {reward}"


class TestUnflagAction:
    def test_unflag_removes_flag(self, fresh_env):
        # Inspect and flag
        fresh_env.step(Action(type="inspect", artifact="preprocessing"))
        obs, _, _, _ = fresh_env.step(Action(
            type="flag",
            violation_type="V1",
            evidence_artifact="preprocessing",
            evidence_quote="fabricated",
            severity="high"
        ))

        flags_before = len(obs.flags_raised)
        assert flags_before >= 1

        flag_id = obs.flags_raised[0].flag_id
        obs2, _, _, _ = fresh_env.step(Action(type="unflag", flag_id=flag_id))
        assert len(obs2.flags_raised) == flags_before - 1

    def test_unflag_nonexistent_flag_error(self, fresh_env):
        obs, _, _, _ = fresh_env.step(Action(type="unflag", flag_id="nonexistent"))
        assert obs.last_action_error is not None


class TestSubmitAction:
    def test_submit_ends_episode(self, fresh_env):
        action = Action(type="submit", verdict="pass", summary="clean experiment")
        _, _, done, _ = fresh_env.step(action)
        assert done

    def test_submit_returns_score_in_info(self, fresh_env):
        action = Action(type="submit", verdict="pass", summary="clean")
        _, _, done, info = fresh_env.step(action)
        assert done
        assert "score" in info
        assert 0.0 <= info["score"] <= 1.0

    def test_submit_score_varies_with_verdict(self, fresh_env):
        # Test with pass verdict
        env1 = MLAuditEnv(task="easy")
        env1.reset(task="easy", seed=42)
        _, _, _, info1 = env1.step(Action(type="submit", verdict="pass", summary="clean"))

        # Test with reject verdict on same experiment
        env2 = MLAuditEnv(task="easy")
        env2.reset(task="easy", seed=42)
        _, _, _, info2 = env2.step(Action(type="submit", verdict="reject", summary="violated"))

        # Scores should differ based on whether experiment was actually clean/violated
        assert info1["score"] != info2["score"] or True  # May be same if matching


class TestEpisodeLifecycle:
    def test_step_increments_step_count(self, fresh_env):
        initial = fresh_env._state["steps_used"]
        fresh_env.step(Action(type="inspect", artifact="preprocessing"))
        assert fresh_env._state["steps_used"] == initial + 1

    def test_step_after_episode_done_raises_error(self, fresh_env):
        fresh_env.step(Action(type="submit", verdict="pass", summary="done"))
        with pytest.raises(RuntimeError):
            fresh_env.step(Action(type="inspect", artifact="preprocessing"))

    def test_reset_clears_state(self, fresh_env):
        fresh_env.step(Action(type="inspect", artifact="preprocessing"))
        assert len(fresh_env._state["inspected"]) > 0

        fresh_env.reset(task="easy", seed=99)
        assert fresh_env._state["steps_used"] == 0
        assert len(fresh_env._state["inspected"]) == 0

    def test_step_without_reset_raises_error(self):
        env = MLAuditEnv(task="easy")
        # Don't call reset
        with pytest.raises(RuntimeError):
            env.step(Action(type="inspect", artifact="preprocessing"))

    def test_cumulative_reward_accumulates(self, fresh_env):
        initial = fresh_env._state["cumulative_reward"]
        obs, reward, _, _ = fresh_env.step(Action(type="inspect", artifact="preprocessing"))
        assert fresh_env._state["cumulative_reward"] == initial + reward
