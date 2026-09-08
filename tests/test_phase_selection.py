import json

import numpy as np

from src import decision, phase_selection, quality


def member(fid, ts, vector, *, q=60, center=(0.5, 0.5), complete=0.8, occlusion=0.1):
    vec = np.asarray(vector, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return {
        "id": fid, "exif_timestamp": ts,
        "dinov2_embedding": quality.embedding_to_blob(vec),
        "quality_score": q, "quality_meta": json.dumps({
            "face_quality": 0.8, "subject_center": list(center),
            "subject_completeness": complete, "occlusion": occlusion,
            "exposure": {"clip_hi": 0.01, "clip_lo": 0.01, "blob_hi": 0.0,
                         "blob_lo": 0.0, "anchor_mass": 0.8,
                         "entropy_nonclip": 5.0, "mass_usable": 0.9},
        }),
        "faces_json": "[]", "face_count": 0, "width": 100, "height": 100,
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
    assert {p["phase_id"] for p in result["phases"]} == {0, 1}


def test_same_phase_near_duplicates_can_use_one_keeper():
    members = [member(1, 0, [1, 0]), member(2, 1, [.999, .001]), member(3, 2, [.998, .002])]
    phases = phase_selection.segment_phases(members)
    result = phase_selection.select_phase_keepers(members, phases)
    assert len(phases) == 1
    assert len(result["keepers"]) == 1


def test_large_variable_phase_adds_diverse_keeper_with_reasons():
    members = [member(i, i, [1, i * .02, .01], q=80 - i) for i in range(1, 7)]
    phases = phase_selection.segment_phases(members, embedding_boundary=0.8)
    result = phase_selection.select_phase_keepers(members, phases, large_phase_size=6)
    assert len(result["keepers"]) >= 2
    assert "LARGE_PHASE_EXTRA_KEEPER" in result["phases"][0]["reason_codes"]
    assert result["phases"][0]["evidence"]


def test_missing_features_fail_safe_add_keeper_and_review():
    members = [member(1, 0, [1, 0]), member(2, 1, [1, .01])]
    members[1]["dinov2_embedding"] = None
    members[1]["quality_meta"] = "{}"
    phases = phase_selection.segment_phases(members)
    result = phase_selection.select_phase_keepers(members, phases)
    assert result["review_required"] is True
    assert len(result["keepers"]) == 2
    assert result["authority"] == "shadow_review_only"


def test_auto_remove_authority_remains_byte_identical_only():
    members = [member(1, 0, [1, 0]), member(2, 1, [1, .01])]
    scores = {0: .8, 1: .2}
    visual = decision.decide_group(members, 0, scores, "burst", safe_duplicates={1: False})
    assert visual["members"][1]["decision"] != "AUTO_REMOVE"
    exact = decision.decide_group(members, 0, scores, "sha_exact", safe_duplicates={1: True})
    assert exact["members"][1]["reason"] == "BYTE_IDENTICAL"
