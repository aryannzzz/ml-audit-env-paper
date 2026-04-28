"""Unit tests for inference helper logic that requires no API calls."""

from inference import COMPARE_HINTS, maybe_add_compare_hint


def test_v5_compare_hint_returned_once():
    """V5 hint should fire after both artifacts are inspected, exactly once."""
    inspected_set = {"run_history", "experiment_notes"}
    already_hinted = set()

    first = maybe_add_compare_hint(inspected_set, COMPARE_HINTS, already_hinted)
    second = maybe_add_compare_hint(inspected_set, COMPARE_HINTS, already_hinted)

    assert first is not None
    assert "run_history" in first
    assert "experiment_notes" in first
    assert "V5" in first
    assert second is None


def test_hint_fires_before_llm_call():
    """Hint must be available before the next LLM call and not fire twice."""
    inspected = {"run_history", "experiment_notes"}
    already_hinted = set()
    hint = maybe_add_compare_hint(inspected, COMPARE_HINTS, already_hinted)
    assert hint is not None
    assert "V5" in already_hinted
    hint2 = maybe_add_compare_hint(inspected, COMPARE_HINTS, already_hinted)
    assert hint2 is None
