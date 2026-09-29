"""Shadow keeper reasons remain deterministic, faithful and decision-neutral."""
from __future__ import annotations

import copy
import hashlib
import json

from src import db, review_server
from src.selection_reason import explain_group, faithfulness


def member(file_id, *, keep=False, reason="RELATIVE_ONLY", quality=None,
           clipiqa=None, width=4000, height=3000, utility=None, margin=None):
    meta = {"backend": "torch"}
    if clipiqa is not None:
        meta["clipiqa"] = clipiqa
    evidence = {}
    if utility is not None:
        evidence["utility_score"] = utility
    if margin is not None:
        evidence["pair_margin"] = margin
    return {
        "file_id": file_id, "basename": f"IMG_{file_id}.jpg",
        "is_keep": int(keep), "decision": "KEEP" if keep else "MAYBE",
        "reason": reason, "quality_score": quality,
        "quality_meta": json.dumps(meta), "evidence_json": json.dumps(evidence),
        "width": width, "height": height, "face_count": 0,
    }


def test_deterministic_faithful_and_input_unchanged():
    group = {"group_id": 7, "keep_file_id": 1, "group_type": "burst"}
    members = [
        member(1, keep=True, reason="GROUP_KEEPER", quality=82,
               clipiqa=.72, utility=.81, margin=0),
        member(2, quality=70, clipiqa=.62, utility=.65, margin=.16),
    ]
    before = copy.deepcopy((group, members))
    first = explain_group(group, members)
    assert first == explain_group(group, members)
    assert (group, members) == before
    assert first["status"] == "explained"
    assert faithfulness(first)["faithfulness"] == 1.0
    margin = next(item for item in first["evidence"]
                  if item["code"] == "PERSISTED_PAIR_MARGIN")
    assert margin["subject_file_id"] == 2
    assert margin["compared_to_file_id"] == 1
    assert first["invariants"] == {
        "keeper_file_ids": [1], "changes_keeper": False,
        "changes_sorting": False, "changes_routing": False,
        "deletion_authority": "none", "model_calls": 0,
    }


def test_rival_advantage_is_tradeoff_not_reselection():
    result = explain_group({"keep_file_id": 1}, [
        member(1, keep=True, reason="GROUP_KEEPER", quality=80,
               width=3000, height=2000),
        member(2, quality=76, width=6000, height=4000,
               utility=.60, margin=.10),
    ])
    assert result["recommendation"]["code"] == "REVIEW_EXISTING_KEEPERS"
    assert result["invariants"]["keeper_file_ids"] == [1]
    assert any(item["code"] == "RIVAL_RESOLUTION_ADVANTAGE"
               for item in result["tradeoffs"])


def test_nonidentical_close_call_explains_without_backing_keeper():
    result = explain_group({"keep_file_id": 1}, [
        member(1, keep=True, reason="GROUP_KEEPER", quality=80,
               utility=.70, margin=0),
        member(2, quality=79, utility=.66, margin=.04),
    ])
    assert result["status"] == "explained"
    assert result["recommendation"]["code"] == "REVIEW_CLOSE_CALL"
    assert "不为该选择背书" in result["recommendation"]["text"]


def test_tradeoffs_consider_every_rival_not_only_closest_utility():
    result = explain_group({"keep_file_id": 1}, [
        member(1, keep=True, reason="GROUP_KEEPER", quality=80,
               width=2000, height=1500, utility=.8, margin=0),
        member(2, quality=70, width=2000, height=1500,
               utility=.7, margin=.1),
        member(3, quality=60, width=8000, height=6000,
               utility=.2, margin=.6),
    ])
    resolution = next(item for item in result["tradeoffs"]
                      if item["code"] == "RIVAL_RESOLUTION_ADVANTAGE")
    assert "16.00 倍" in resolution["text"]
    assert not any(item["code"] == "RIVAL_ADVANTAGE_ABSTAIN"
                   for item in result["tradeoffs"])


def test_byte_identical_tie_does_not_invent_utility_support_or_instability():
    result = explain_group({"keep_file_id": 1}, [
        member(1, keep=True, reason="GROUP_KEEPER", quality=70,
               utility=.7, margin=0),
        member(2, reason="BYTE_IDENTICAL", quality=70,
               utility=.7, margin=0),
    ])
    assert result["status"] == "explained"
    assert result["confidence"]["level"] == "high"
    assert {item["code"] for item in result["support"]} == {"BYTE_IDENTICAL_PEERS"}
    assert not any(item["code"] == "LOW_UTILITY_MARGIN"
                   for item in result["tradeoffs"])


def test_missing_scoped_primary_abstains_and_never_rebinds_pair_margin():
    result = explain_group({"keep_file_id": 1}, [
        member(3, keep=True, reason="LOGICAL_PHASE_KEEPER", quality=80,
               utility=.7, margin=0),
        member(4, quality=70, utility=.4, margin=.3),
    ])
    assert result["status"] == "abstained"
    assert not any(item["code"] == "PERSISTED_PAIR_MARGIN"
                   for item in result["evidence"])


def test_protection_only_does_not_explain_or_raise_confidence():
    result = explain_group({"keep_file_id": 1}, [
        member(1, keep=True, reason="GROUP_KEEPER"),
        member(2, keep=True, reason="POSE_VARIANT_KEEPER"),
        member(3),
    ])
    assert result["status"] == "abstained"
    assert any(item["code"] == "POSE_VARIANT_PROTECTION"
               for item in result["support"])


def test_rival_advantage_evidence_points_from_winner_to_compared_member():
    result = explain_group({"keep_file_id": 1}, [
        member(1, keep=True, reason="GROUP_KEEPER", quality=70,
               width=100, height=100, utility=.8, margin=0),
        member(2, quality=80, width=200, height=200,
               utility=.6, margin=.2),
    ])
    quality = next(item for item in result["evidence"]
                   if item["code"] == "RIVAL_QUALITY_SCORE_ADVANTAGE")
    assert quality["subject_file_id"] == 2
    assert quality["compared_to_file_id"] == 1
    assert quality["value"] == {"subject": 80.0, "compared_to": 70.0,
                                "delta": 10.0}
    resolution = next(item for item in result["evidence"]
                      if item["code"] == "RIVAL_RESOLUTION_ADVANTAGE")
    assert resolution["subject_file_id"] == 2
    assert resolution["compared_to_file_id"] == 1


def test_sqlite_projection_is_read_only_for_decision_tables(tmp_path):
    conn = db.open_db(tmp_path / "inventory.sqlite")
    file_ids = []
    for index in range(2):
        file_id = db.insert_file(conn, {
            "path": str(tmp_path / f"{index}.jpg"),
            "basename": f"{index}.jpg", "size_bytes": 10 + index,
            "mtime_ns": 1, "width": 100, "height": 100,
            "file_kind": "jpg", "scan_status": "done",
        })
        file_ids.append(file_id)
        db.batch_insert_features(conn, [{
            "file_id": file_id, "phash": None, "dinov2_embedding": None,
            "quality_score": 80.0 - index * 10, "quality_meta": "{}",
            "face_count": 0, "faces_json": "[]", "status": "done",
        }])
    group_id = db.insert_group(
        conn, "burst", file_ids[0],
        [(file_ids[0], True, "GROUP_KEEPER"),
         (file_ids[1], False, "RELATIVE_ONLY")], 1,
    )
    db.update_member_decisions(conn, group_id, [
        {"file_id": file_ids[0], "decision": "KEEP", "confidence": 0,
         "reason": "GROUP_KEEPER",
         "evidence_json": json.dumps({"utility_score": .8, "pair_margin": 0})},
        {"file_id": file_ids[1], "decision": "MAYBE", "confidence": 0,
         "reason": "RELATIVE_ONLY",
         "evidence_json": json.dumps({"utility_score": .6, "pair_margin": .2})},
    ])
    conn.commit()

    def digest():
        rows = []
        for table in ("groups", "group_members"):
            rows.extend((table, *tuple(row)) for row in conn.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            ))
        return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()

    before = digest()
    projected = review_server._group_items(db.group_page(conn, 0, 10))
    after = digest()
    assert projected[0]["selection_reason"]["invariants"]["changes_keeper"] is False
    assert before == after
    conn.close()


def test_missing_and_contradictory_evidence_abstain():
    missing = explain_group({"keep_file_id": 1}, [
        member(1, keep=True, reason="GROUP_KEEPER"), member(2),
    ])
    assert missing["status"] == "abstained"
    assert missing["recommendation"]["code"] == "ABSTAIN_INSUFFICIENT_EVIDENCE"
    contradictory = explain_group({"keep_file_id": 1}, [
        member(1, keep=True, reason="GROUP_KEEPER"),
        member(2, utility=.9, margin=-.2),
    ])
    assert contradictory["status"] == "abstained"
    assert any(item["code"] == "NEGATIVE_PERSISTED_MARGIN"
               for item in contradictory["tradeoffs"])


def test_protected_keeper_is_nonsemantic_and_review_only():
    result = explain_group({"keep_file_id": 1}, [
        member(1, keep=True, reason="GROUP_KEEPER", quality=80),
        member(2, keep=True, reason="POSE_VARIANT_KEEPER", quality=78),
        member(3, quality=70, utility=.6, margin=.1),
    ])
    pose = next(item for item in result["support"]
                if item["code"] == "POSE_VARIANT_PROTECTION")
    assert "没有语义阶段或删除权限" in pose["text"]
    assert result["invariants"]["keeper_file_ids"] == [1, 2]


def test_api_projection_adds_reason_without_changing_decisions_or_paths():
    base = {
        "group_id": 3, "group_type": "burst", "keep_file_id": 10,
        "decision_state": "REVIEW_REQUIRED", "group_confidence": 0.0,
        "policy_version": "p0-safety-v2", "member_count": 2,
        "member_confidence": 0.0, "quality_meta": "{}", "face_count": 0,
        "thumb_status": "ok", "thumb_error": None, "file_kind": "jpg",
        "size_bytes": 1, "exif_datetime": None,
    }
    rows = [
        {**base, "file_id": 10, "is_keep": 1, "decision": "KEEP",
         "reason": "GROUP_KEEPER", "evidence_json": '{"utility_score":0.8,"pair_margin":0}',
         "basename": "a.jpg", "width": 100, "height": 100, "quality_score": 80.0},
        {**base, "file_id": 11, "is_keep": 0, "decision": "MAYBE",
         "reason": "RELATIVE_ONLY", "evidence_json": '{"utility_score":0.6,"pair_margin":0.2}',
         "basename": "b.jpg", "width": 100, "height": 100, "quality_score": 60.0},
    ]
    projected = review_server._group_items(rows)[0]
    assert [item["decision"] for item in projected["members"]] == ["KEEP", "MAYBE"]
    assert projected["selection_reason"]["status"] == "explained"
    assert projected["selection_reason"]["invariants"]["changes_keeper"] is False
    assert "path" not in json.dumps(projected)
