"""Test AUTO_REMOVE only triggers when content_sha256 matches (requirement 3)."""
from src import decision


def test_auto_remove_requires_sha256_match():
    """AUTO_REMOVE only allowed when content_sha256 matches keeper."""
    keeper = {
        "phash": b"\x12\x34\x56\x78" * 2,
        "dinov2_embedding": b"\x00" * 1536,
        "quality_score": 75.0,
        "quality_meta": '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}',
        "content_sha256": "aabbccdd" + "00" * 28,
    }
    duplicate_sha = {
        "phash": b"\x12\x34\x56\x78" * 2,
        "dinov2_embedding": b"\x00" * 1536,
        "quality_score": 74.0,
        "quality_meta": '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}',
        "content_sha256": "aabbccdd" + "00" * 28,  # Same SHA
    }
    duplicate_no_sha = {
        "phash": b"\x12\x34\x56\x78" * 2,
        "dinov2_embedding": b"\x00" * 1536,
        "quality_score": 73.0,
        "quality_meta": '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}',
        "content_sha256": "11223344" + "00" * 28,  # Different SHA
    }

    members = [keeper, duplicate_sha, duplicate_no_sha]
    scores = {0: 75.0, 1: 74.0, 2: 73.0}
    safe_duplicates = {1: True, 2: False}

    result = decision.decide_group(members, keeper=0, scores=scores, group_type="sha_exact",
                                   profile="balanced", group_trusted=True, safe_duplicates=safe_duplicates)

    decisions = result["members"]
    assert decisions[0]["decision"] == "KEEP"
    assert decisions[1]["decision"] == "AUTO_REMOVE"
    assert decisions[1]["reason"] == "BYTE_IDENTICAL"
    assert decisions[2]["decision"] != "AUTO_REMOVE"


def test_auto_remove_never_without_safe_duplicates():
    """Without safe_duplicates dict, no AUTO_REMOVE decisions (safe default)."""
    keeper = {
        "phash": b"\x12\x34\x56\x78" * 2,
        "dinov2_embedding": b"\x00" * 1536,
        "quality_score": 75.0,
        "quality_meta": '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}',
        "content_sha256": "aabbccdd" + "00" * 28,
    }
    duplicate = {
        "phash": b"\x12\x34\x56\x78" * 2,
        "dinov2_embedding": b"\x00" * 1536,
        "quality_score": 74.0,
        "quality_meta": '{"exposure":{"clip_hi":0.0,"clip_lo":0.0,"blob_hi":0.0,"blob_lo":0.0,"anchor_mass":0.5,"entropy_nonclip":5.0,"mass_usable":0.6}}',
        "content_sha256": "aabbccdd" + "00" * 28,
    }

    members = [keeper, duplicate]
    scores = {0: 75.0, 1: 74.0}

    result = decision.decide_group(members, keeper=0, scores=scores, group_type="sha_exact",
                                   profile="balanced", group_trusted=True, safe_duplicates=None)

    decisions = result["members"]
    assert decisions[0]["decision"] == "KEEP"
    assert decisions[1]["decision"] != "AUTO_REMOVE"
