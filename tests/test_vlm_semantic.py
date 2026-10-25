from src.vlm_semantic import sanitize_group


def test_accepts_bounded_visible_semantic_claim():
    result = sanitize_group({
        "group_id": "g1",
        "semantic_claims": [{
            "code": "POSE_ACTION_DIFFERENCE",
            "subjects": ["F02"],
            "compared_to": ["F01"],
            "observation": "F02 has one arm raised while F01 has both arms lowered.",
            "confidence": "high",
            "visual_refs": [{"frame": "F02", "region": "subject"}],
        }],
    }, ["F01", "F02"])
    assert len(result["accepted"]) == 1
    assert result["invariants"]["deletion_authority"] == "none"


def test_deep_judgment_combines_images_and_shadow_cv_references():
    result = sanitize_group({
        "group_id": "g1",
        "judgment": {
            "recommended": ["F02"],
            "confidence": "high",
            "discriminators": [{
                "code": "COMPOSITE_ADVANTAGE",
                "subjects": ["F02"],
                "compared_to": ["F01"],
                "reason": "F02 has the stronger expression while several small CV advantages point the same way.",
                "confidence": "high",
                "visual_refs": [{"frame": "F02", "region": "face"}],
                "cv_refs": ["F02:FACE_LOCAL_QUALITY", "F02:GLOBAL_SHARPNESS"],
            }],
        },
    }, ["F01", "F02"], [
        "F02:FACE_LOCAL_QUALITY", "F02:GLOBAL_SHARPNESS",
    ])
    assert result["recommendation"] == {
        "recommended": ["F02"], "confidence": "high", "status": "grounded",
    }
    assert result["accepted"][0]["visual_refs"] == [{
        "frame": "F02", "region": "face", "region_kind": "face",
    }]
    assert len(result["accepted"]) == 1


def test_rejects_low_confidence_unknown_alias_unknown_cv_and_ungrounded_claims():
    result = sanitize_group({
        "group_id": "g1",
        "judgment": {
            "recommended": ["F02"],
            "confidence": "medium",
            "discriminators": [
                {
                    "code": "UNIQUE_MOMENT", "subjects": ["F03"],
                    "compared_to": ["F01"], "reason": "A distinct pose.",
                    "confidence": "low", "visual_refs": [],
                },
                {
                    "code": "FACE_QUALITY_DIFFERENCE", "subjects": ["F02"],
                    "compared_to": ["F01"], "reason": "F02 has the better face.",
                    "confidence": "medium", "cv_refs": ["F02:NOT_SUPPLIED"],
                },
                {
                    "code": "EXPRESSION_DIFFERENCE", "subjects": ["F02"],
                    "compared_to": ["F01"], "reason": "F02 looks better.",
                    "confidence": "medium",
                },
            ],
        },
    }, ["F01", "F02"], ["F02:FACE_LOCAL_QUALITY"])
    assert not result["accepted"]
    assert result["rejected"][0]["reasons"] == [
        "INVALID_SUBJECT_ALIAS", "UNGROUNDED_DISCRIMINATOR",
    ]
    assert result["rejected"][1]["reasons"] == [
        "UNKNOWN_CV_REFERENCE", "UNGROUNDED_DISCRIMINATOR",
    ]
    assert result["rejected"][2]["reasons"] == ["UNGROUNDED_DISCRIMINATOR"]


def test_pairwise_cv_ref_grounds_each_participating_frame():
    result = sanitize_group({
        "group_id": "g1",
        "judgment": {
            "recommended": ["F01", "F02"],
            "confidence": "high",
            "discriminators": [
                {
                    "code": "TECHNICAL_QUALITY_DIFFERENCE", "subjects": ["F01"],
                    "compared_to": ["F02"], "reason": "Pairwise DINO evidence.",
                    "confidence": "high", "visual_refs": [],
                    "cv_refs": ["PAIR:F01:F02:dinov2_cosine_similarity_distance"],
                },
                {
                    "code": "UNIQUE_MOMENT", "subjects": ["F02"],
                    "compared_to": ["F01"], "reason": "A separately grounded phase.",
                    "confidence": "high",
                    "visual_refs": [{"frame": "F02", "region": "full_frame"}],
                    "cv_refs": [],
                },
            ],
        },
    }, ["F01", "F02"], ["PAIR:F01:F02:dinov2_cosine_similarity_distance"])
    assert result["recommendation"]["recommended"] == ["F01", "F02"]


def test_accepts_bounded_descriptive_region_for_precise_human_attention():
    result = sanitize_group({
        "group_id": "g1",
        "semantic_claims": [{
            "code": "POSE_ACTION_DIFFERENCE", "subjects": ["F02"],
            "compared_to": ["F01"], "observation": "F02 has the cleaner pose.",
            "confidence": "medium",
            "visual_refs": [{"frame": "F02", "region": "raised hand near the rail"}],
        }],
    }, ["F01", "F02"])
    assert result["accepted"][0]["visual_refs"][0] == {
        "frame": "F02", "region": "raised hand near the rail",
        "region_kind": "described_region",
    }


def test_rejects_sensitive_inference_even_when_visually_referenced():
    result = sanitize_group({
        "group_id": "g1",
        "semantic_claims": [{
            "code": "EXPRESSION_DIFFERENCE", "subjects": ["F02"],
            "compared_to": ["F01"],
            "observation": "F02 reveals the subject's internal emotion.",
            "confidence": "high",
            "visual_refs": [{"frame": "F02", "region": "face"}],
        }],
    }, ["F01", "F02"])
    assert result["rejected"][0]["reasons"] == ["SENSITIVE_OR_UNSUPPORTED_INFERENCE"]


def test_recommendation_must_be_grounded_to_the_same_recommended_frame():
    result = sanitize_group({
        "group_id": "g1",
        "judgment": {
            "recommended": ["F02"],
            "confidence": "high",
            "discriminators": [{
                "code": "EXPRESSION_DIFFERENCE", "subjects": ["F01"],
                "compared_to": ["F02"], "reason": "F01 has a visible smile.",
                "confidence": "high",
                "visual_refs": [{"frame": "F01", "region": "face"}],
            }],
        },
    }, ["F01", "F02"])
    assert result["recommendation"]["status"] == "abstained"
    assert result["recommendation"]["recommended"] == []
    assert result["recommendation"]["rejections"] == ["UNGROUNDED_RECOMMENDATION"]
