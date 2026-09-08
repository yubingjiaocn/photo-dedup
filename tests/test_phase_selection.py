import json

import numpy as np
import pytest

from src import decision, phase_selection, quality


def member(fid, ts, vector, *, q=60, center=(0.5, 0.5), complete=0.8, occlusion=0.1,
           include_optional=True, face_count=0):
    vec = np.asarray(vector, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    meta = {
        "face_quality": 0.8, "subject_center": list(center),
        "exposure": {"clip_hi": 0.01, "clip_lo": 0.01, "blob_hi": 0.0,
                     "blob_lo": 0.0, "anchor_mass": 0.8,
                     "entropy_nonclip": 5.0, "mass_usable": 0.9},
    }
    if include_optional:
        meta.update({"subject_completeness": complete, "occlusion": occlusion})
    return {
        "id": fid, "exif_timestamp": ts,
        "dinov2_embedding": quality.embedding_to_blob(vec),
        "quality_score": q, "quality_meta": json.dumps(meta),
        "faces_json": "[]", "face_count": face_count, "width": 100, "height": 100,
        "size_bytes": 100, "phash": b"12345678", "content_sha256": str(fid),
    }


def test_cross_phase_group_cannot_collapse_to_single_keeper():
    members = [
        member(1, 0, [1, 0, 0]), member(2, 1, [1, .01, 0]),
        member(3, 2, [0, 1, 0]), member(4, 3, [.01, 1, 0]),
    ]
    phases = phase_selection.segment_phases(members)
    result = phase_selection.select_phase_keepers(members, phases)
    assert len(phases) == 2
    assert len(result["keepers"]) >= 2
    assert result["physical_group_split"] is False
    assert result["semantics"].startswith("logical_phase_protection")


def test_same_phase_near_duplicates_can_use_one_keeper():
    members = [member(1, 0, [1, 0]), member(2, 1, [.999, .001]), member(3, 2, [.998, .002])]
    result = phase_selection.select_phase_keepers(members, phase_selection.segment_phases(members))
    assert len(result["keepers"]) == 1


def test_size_alone_does_not_add_keeper():
    members = [member(i, i, [1, i * .001], q=80 - i) for i in range(1, 9)]
    phases = phase_selection.segment_phases(members, embedding_boundary=0.8)
    result = phase_selection.select_phase_keepers(members, phases, max_group_keepers=3)
    assert len(result["keepers"]) == 1
    assert "OBSERVED_VARIATION_EXTRA_KEEPER" not in result["phases"][0]["reason_codes"]


def test_observed_variation_adds_mmr_quality_diversity_keeper():
    members = [
        member(1, 0, [1, 0, 0], q=95),
        member(2, 1, [.999, .001, 0], q=94),
        member(3, 2, [.8, .6, 0], q=80),
    ]
    phases = phase_selection.segment_phases(members, embedding_boundary=0.7)
    result = phase_selection.select_phase_keepers(
        members, phases, diversity_similarity=0.95, mmr_quality_weight=0.7
    )
    assert result["keepers"] == [0, 2]
    assert "OBSERVED_VARIATION_EXTRA_KEEPER" in result["phases"][0]["reason_codes"]
    assert "MMR_QUALITY_DIVERSITY" in result["phases"][0]["reason_codes"]


def test_mmr_weight_is_validated():
    members = [member(1, 0, [1, 0])]
    with pytest.raises(ValueError, match="mmr_quality_weight"):
        phase_selection.select_phase_keepers(
            members, phase_selection.segment_phases(members), mmr_quality_weight=1.1
        )


def test_production_optional_evidence_gap_is_bounded_not_keep_all():
    members = [member(i, i, [1, i * .001], q=70 + i, include_optional=False) for i in range(1, 7)]
    phases = phase_selection.segment_phases(members)
    result = phase_selection.select_phase_keepers(members, phases)
    assert result["review_required"] is False
    assert len(result["keepers"]) == 1
    evidence = result["phases"][0]["evidence"]["0"]
    assert evidence["subject_completeness"] is None
    assert evidence["occlusion_inverse"] is None
    assert "OPTIONAL_SCORING_EVIDENCE_MISSING" in result["phases"][0]["reason_codes"]


def test_optional_gap_routes_close_face_choice_to_review_without_extra_keeper():
    members = [
        member(1, 0, [1, 0], q=70, include_optional=False, face_count=1),
        member(2, 1, [1, .001], q=71, include_optional=False, face_count=1),
    ]
    result = phase_selection.select_phase_keepers(members, phase_selection.segment_phases(members))
    assert result["review_required"] is True
    assert len(result["keepers"]) == 1
    assert "OPTIONAL_EVIDENCE_LOW_MARGIN_REVIEW" in result["phases"][0]["reason_codes"]


def test_primary_feature_gap_adds_at_most_one_bounded_keeper():
    members = [member(i, i, [1, i * .001], include_optional=False) for i in range(1, 6)]
    members[2]["dinov2_embedding"] = None
    result = phase_selection.select_phase_keepers(members, phase_selection.segment_phases(members))
    assert result["review_required"] is True
    assert 1 < len(result["keepers"]) < len(members)


def test_group_budget_prevents_keep_all_across_noisy_logical_phases():
    members = [member(i, i, [1 if i % 2 else 0, 0 if i % 2 else 1], q=80 - i) for i in range(1, 7)]
    phases = phase_selection.segment_phases(members, embedding_boundary=0.95)
    result = phase_selection.select_phase_keepers(members, phases, max_group_keepers=3)
    assert result["budget_limited"] is True
    assert len(result["keepers"]) == 3
    assert len(result["keepers"]) < len(members)


def test_local_baseline_and_minimum_length_suppress_mild_singleton_cut():
    members = [
        member(1, 0, [1, 0, 0]),
        member(2, 1, [.96, .28, 0]),
        member(3, 2, [.94, .34, 0]),
        member(4, 3, [.97, .24, 0]),
    ]
    phases = phase_selection.segment_phases(members, group_type="burst")
    assert [phase.members for phase in phases] == [(0, 1, 2, 3)]


def test_phash_near_requires_stronger_boundary_consensus_than_burst():
    members = [
        member(1, 0, [1, 0, 0], center=(.5, .5)),
        member(2, 1, [.96, .28, 0], center=(.56, .5)),
        member(3, 2, [.97, .24, 0], center=(.57, .5)),
    ]
    assert len(phase_selection.segment_phases(members, group_type="phash_near")) == 1


def test_authoritative_score_flow_can_feed_decision_evidence():
    members = [member(1, 0, [1, 0], q=90), member(2, 1, [1, .001], q=60)]
    selection = phase_selection.select_phase_keepers(members, phase_selection.segment_phases(members))
    keeper = selection["keepers"][0]
    scores = selection["utility_scores"]
    result = decision.decide_group(members, keeper, scores, "burst", safe_duplicates={1: False})
    for idx, record in result["members"].items():
        assert record["evidence"]["utility_score"] == pytest.approx(scores[idx])
    other = next(idx for idx in range(2) if idx != keeper)
    assert result["members"][other]["evidence"]["pair_margin"] == pytest.approx(
        scores[keeper] - scores[other]
    )


def test_auto_remove_authority_remains_byte_identical_only():
    members = [member(1, 0, [1, 0]), member(2, 1, [1, .01])]
    scores = {0: .8, 1: .2}
    visual = decision.decide_group(members, 0, scores, "burst", safe_duplicates={1: False})
    assert visual["members"][1]["decision"] != "AUTO_REMOVE"
    exact = decision.decide_group(members, 0, scores, "sha_exact", safe_duplicates={1: True})
    assert exact["members"][1]["reason"] == "BYTE_IDENTICAL"
