import json

import pytest

from src.dense_recovery_review import PRODUCER, context
from src import phase_selection as ps
from tests.test_region_set_quality import frame, phases


def packet():
    return {
        "schema_version": 1,
        "producer": PRODUCER,
        "group_member_ids": [1, 2],
        "review_only": True,
        "keeper_authority": False,
        "proposals": [
            {
                "source_file_id": 1,
                "target_file_id": 2,
                "source_detector_class": 0,
                "target_detector_class": 77,
                "source_confidence": 0.8,
                "target_confidence": 0.8,
                "source_detector_confirmed": True,
                "target_detector_confirmed": True,
                "identity_scope": "visible_region_only",
                "whole_body_completeness": "unknown",
                "semantic_identity_authority": False,
                "object_unique": True,
                "novelty_checked": True,
                "native_no_resize": True,
                "native_patch_size": 14,
                "matches": 20,
                "inliers": 16,
                "source_patches": 128,
                "target_patches": 128,
                "coverage": [0.4, 0.4],
                "median_similarity": 0.95,
                "scale": 1.0,
                "source_box_normalized": [0.1, 0.1, 0.5, 0.9],
                "target_box_normalized": [0.1, 0.1, 0.5, 0.9],
                "source_crop_edge": True,
                "target_crop_edge": True,
            }
        ],
    }


def members(p=None):
    ms = [frame(1), frame(2)]
    for m in ms:
        meta = json.loads(m["quality_meta"])
        meta["dense_instance_recovery"] = packet() if p is None else p
        m["quality_meta"] = json.dumps(meta)
    return ms


def test_partial_visible_region_is_review_information_not_whole_body_or_keeper():
    ms = members()
    before = ps.select_phase_keepers(ms, phases(0, 1))
    after = ps.select_phase_keepers(
        ms, phases(0, 1), instance_recovery_policy="review_only"
    )
    for key in ["keepers", "utility_scores", "group_keeper_budget", "phases"]:
        assert before[key] == after[key]
    assert after["review_required"] and after["mandatory_review"]
    dense = after["instance_recovery_context"]["dense_visible_region_context"]
    assert len(dense["proposals"]) == 1 and not dense["keeper_authority"]
    assert (
        not dense["stable_observed_set"]
        and dense["stable_observed_set_status"] == "not_established"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"whole_body_completeness": "complete"},
        {"semantic_identity_authority": True},
        {"identity_scope": "person_identity"},
        {"object_unique": False},
        {"novelty_checked": False},
        {"source_detector_confirmed": False},
        {"native_no_resize": False},
        {"target_detector_class": 16},
        {"target_file_id": 3},
        {"source_file_id": True},
        {"inliers": 10},
        {"inliers": 21},
        {"source_patches": 1024},
        {"coverage": [0.1, 0.6]},
        {"scale": 2.0},
        {"median_similarity": 0.89},
        {"source_confidence": 0.5},
        {"source_box_normalized": [0, 0, float("nan"), 1]},
    ],
)
def test_invalid_or_overclaiming_dense_evidence_abstains(change):
    p = packet()
    p["proposals"][0].update(change)
    assert not context(members(p))["proposals"]


def test_group_binding_cannot_leak_to_other_members():
    ms = members()
    ms[1]["id"] = 3
    assert not context(ms)["proposals"]
    ms = members()
    ms[1]["quality_meta"] = "{}"
    assert not context(ms)["proposals"]


def test_legacy_only_metadata_remains_unchanged():
    from src.instance_recovery import _legacy_review_context, review_context

    ms = [frame(1), frame(2)]
    assert context(ms) is None
    assert review_context(ms) == _legacy_review_context(ms)
