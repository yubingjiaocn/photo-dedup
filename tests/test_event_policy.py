import pytest

from src.event_policy import compile_event_policy, validate_event_policy


def policy(**updates):
    value = {"schema_version": 1, "event_kind": "stage", "confidence": 0.9,
             "valuable_states": ["action_phase", "singleton"], "mixed_theme": False,
             "identity_sensitive": False}
    value.update(updates)
    return value


def test_schema_rejects_arbitrary_threshold_or_delete_vocabulary():
    with pytest.raises(ValueError):
        validate_event_policy({**policy(), "delete": True})
    with pytest.raises(ValueError):
        validate_event_policy({**policy(), "dinov2_threshold": 0.1})


def test_full_stage_policy_keeps_multiple_action_phases():
    compiled = compile_event_policy(policy())
    assert compiled["max_visual_group"] == 5
    assert compiled["keepers_per_phase"] == 2
    assert compiled["protect_singletons"] is True
    assert compiled["protect_rare_states"] is True
    assert compiled["auto_remove_scope"] == "byte_identical_only"


def test_fallback_is_monotonically_safer():
    full = compile_event_policy(policy(event_kind="convention"), "full")
    for capability in ("no_vlm", "no_llm", "offline"):
        fallback = compile_event_policy(policy(event_kind="convention"), capability)
        assert fallback["max_visual_group"] <= full["max_visual_group"]
        assert fallback["keepers_per_phase"] >= full["keepers_per_phase"]
        assert fallback["require_identity"] is True


def test_invalid_or_low_confidence_policy_fails_closed():
    invalid = compile_event_policy({"delete": "everything"})
    low = compile_event_policy(policy(confidence=0.2, mixed_theme=True))
    for compiled in (invalid, low):
        assert compiled["max_visual_group"] <= 4
        assert compiled["keepers_per_phase"] >= 2
        assert compiled["review_required"] is True
