from src.semantic_observer import (
    frame_observations,
    group_cv_context,
    persisted_group_cv_context,
)


def test_cv_observations_are_review_only_and_do_not_claim_crop():
    result = frame_observations(
        frame_alias="F01", image_size=(1000, 800),
        eye_result={
            "status": "ALL_OPEN", "detected_face_count": 1,
            "detector_count_disagreement": False, "reasons": [],
            "faces": [{"face_index": 0, "bbox_xywh": [1, 100, 200, 200],
                       "state": "OPEN", "reasons": ["BOTH_EYES_OPEN"]}],
        },
        yunet_faces=[{"bbox": [1, 100, 200, 200], "score": .9}],
        face_quality=.8, sharpness=120.0,
        exposure_metrics={"clip_hi": 0, "clip_lo": 0, "anchor_mass": .7},
        exposure_class=("ok", "", 0),
    )
    assert result["mode"] == "shadow_review_only"
    assert result["invariants"]["deletion_authority"] == "none"
    edge = next(item for item in result["observations"]
                if item["code"] == "FACE_FRAME_EDGE_PROXIMITY")
    assert edge["state"] == "NEAR_EDGE"
    assert "EDGE_PROXIMITY_IS_NOT_CROP_PROOF" in result["abstentions"]


def test_persisted_context_exports_all_cv_leaves_but_not_paths():
    context = persisted_group_cv_context([{
        "frame_alias": "F01", "quality_score": 80.0, "face_count": 1,
        "quality_meta": '{"clipiqa":0.7,"shadow":{"weak":0.01},"path":"/secret/a.jpg"}',
        "evidence_json": {"utility_score": .6, "nested": {"default_off": True}},
    }])
    refs = context["evidence_refs"]
    assert "F01:quality_meta.clipiqa" in refs
    assert "F01:quality_meta.shadow.weak" in refs
    assert "F01:evidence_json.nested.default_off" in refs
    assert not any("path" in ref.lower() for ref in refs)
    assert context["policy"]["drop_shadow_signals"] is False


def test_group_context_keeps_weak_and_shadow_observations_for_vlm():
    context = group_cv_context([{
        "frame_alias": "F01",
        "observations": [
            {"code": "FACE_LOCAL_QUALITY", "state": "MEASURED", "value": .51,
             "authority": "observation_only"},
            {"code": "WEAK_SHADOW_SIGNAL", "state": "MAYBE", "value": .02,
             "authority": "observation_only"},
        ],
        "abstentions": ["NOT_DECISIVE_ALONE"],
    }])
    assert context["evidence_refs"] == [
        "F01:FACE_LOCAL_QUALITY", "F01:WEAK_SHADOW_SIGNAL",
    ]
    assert context["frames"][0]["observations"][1]["value"] == .02
    assert context["mode"] == "shadow_review_only"


def test_missing_eye_evidence_stays_unknown():
    result = frame_observations(
        frame_alias="F02", image_size=(1000, 800),
        eye_result={"status": "UNKNOWN", "faces": [], "reasons": ["SMALL_FACE"]},
        yunet_faces=[], face_quality=None, sharpness=None,
        exposure_metrics={}, exposure_class=("maybe", "unknown", .5),
    )
    eye = next(item for item in result["observations"] if item["code"] == "EYE_UNKNOWN")
    assert eye["applicability"] == "UNKNOWN"
    assert "EYE_STATE_NOT_DECISIVE" in result["abstentions"]
