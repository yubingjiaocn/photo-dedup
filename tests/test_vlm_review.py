"""Offline group VLM review artifacts are complete, grounded, stale-safe and inert."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from src import db, review_server, vlm_review
from src.semantic_observer import persisted_group_cv_context


def test_review_prompt_requests_chinese_user_facing_reasons():
    assert "Simplified Chinese" in vlm_review.PROMPT
    assert "schema defines them in English" in vlm_review.PROMPT


def _digest(conn) -> str:
    rows = []
    for table in ("groups", "group_members", "review_index"):
        rows.extend((table, *tuple(row)) for row in conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()


def _measurement(context, criterion: str):
    measurements = context["pairwise_measurements"][0]["measurements"]
    return next(item for item in measurements if item["criterion"] == criterion)


def _audit_value(context, ref: str):
    alias, suffix = ref.split(":", 1)
    if context.get("layer") == "L0":
        frame = context["frames"][int(alias[1:]) - 1]
        ref = frame["frame_identity"] + ":" + suffix
    for frame in context["frames"]:
        for measurement in frame["measurements"]:
            if measurement["ref"] == ref:
                return measurement["value"]
    raise AssertionError(f"missing audit ref: {ref}")


def _output(tmp_path: Path, *, exact: bool = False):
    output = tmp_path / "output"
    photos = tmp_path / "photos"
    output.mkdir()
    photos.mkdir()
    conn = db.open_db(output / "inventory.sqlite")
    ids = []
    for index in range(2):
        source = photos / f"photo-{index}.jpg"
        Image.new("RGB", (1200, 900), (20 + (0 if exact else index) * 30, 40, 60)).save(source, "JPEG")
        stat = source.stat()
        file_id = db.insert_file(conn, {
            "path": str(source), "basename": source.name, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "width": 1200, "height": 900,
            "file_kind": "jpg", "scan_status": "done",
        })
        ids.append(file_id)
        dino = np.array([1.0 - index, float(index)], dtype=np.float16)
        local_vector = np.zeros(768, dtype=np.float16)
        local_vector[index] = 1.0
        quality_meta = {
            "weak": {"score": 0.01 + index},
            "routing": {"scene_context": {"tags": ["portrait"]}},
            "default_off": {"enabled": False, "raw": index},
            "shadow_only": {"raw_scores": [0.1, 0.2]},
            "nested_path": "/private/library/photo.jpg",
            "local_quality": {
                "schema_version": 1,
                "method": "native_coco_crop_v1",
                "catalog_complete_single": True,
                "detected_subject_count": 1,
                "regions": [{
                    "kind": "subject", "class_id": 0,
                    "box_normalized": [0.1, 0.1, 0.9, 0.9],
                    "embedding_offset": 0, "confidence": 0.99,
                    "musiq": 80.0 - index, "clipiqa": 0.8 - index * 0.1,
                    "native_size": [1200, 900],
                }],
            },
        }
        db.batch_insert_features(conn, [{
            "file_id": file_id,
            "phash": (0 if exact else (0, 7)[index]).to_bytes(8, "big"),
            "content_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dinov2_embedding": dino.tobytes(),
            "quality_score": 80.0 - index,
            "quality_meta": json.dumps(quality_meta),
            "face_count": 1,
            "faces_json": json.dumps([{"bbox": [200, 150, 300, 300], "score": 0.9}]),
            "status": "done",
        }])
        conn.execute(
            "UPDATE features SET local_quality_embedding=? WHERE file_id=?",
            (local_vector.tobytes(), file_id),
        )
    group_type = "sha_exact" if exact else "phash_near"
    state = "AUTO_READY" if exact else "REVIEW_REQUIRED"
    rival_reason = "BYTE_IDENTICAL" if exact else "LOW_MARGIN"
    gid = db.insert_group(conn, group_type, ids[0],
        [(ids[0], True, "GROUP_KEEPER"), (ids[1], False, rival_reason)], 1,
        decision_state=state, confidence=1.0 if exact else 0.0,
        policy_version="test", decision_json=json.dumps({"shadow": {"weak": True}}))
    db.update_member_decisions(conn, gid, [
        {"file_id": ids[0], "decision": "KEEP", "confidence": 1.0,
         "reason": "GROUP_KEEPER", "evidence_json": json.dumps({"utility_score": .8})},
        {"file_id": ids[1], "decision": "AUTO_REMOVE" if exact else "MAYBE",
         "confidence": 1.0 if exact else 0.0, "reason": rival_reason,
         "evidence_json": json.dumps({"utility_score": .79, "pair_margin": .01,
                                      "shadow_signal": {"default_off": True}})},
    ])
    conn.commit()
    return output, conn, gid, ids


def test_complete_context_prunes_only_path_privacy_fields():
    context = persisted_group_cv_context([{
        "frame_alias": "F01", "quality_score": 80.0, "phash": b"12345678",
        "dinov2_embedding": b"abc", "local_quality_embedding": b"xyz",
        "quality_meta": json.dumps({
            "weak": .01, "default_off": False, "shadow": {"raw": [1, 2]},
            "source_path": "/secret/a.jpg",
        }),
        "faces_json": json.dumps([{"bbox": [1, 2, 3, 4]}]),
        "evidence_json": json.dumps({"nested": {"unused": None}}),
        "basename": "private.jpg",
    }], group={"group_id": 7, "decision_json": json.dumps({"weak": True})})
    refs = set(context["evidence_refs"])
    expected = {
        "F01:quality_score", "F01:phash", "F01:dinov2_embedding",
        "F01:local_quality_embedding",
        "F01:quality_meta.weak", "F01:quality_meta.default_off",
        "F01:quality_meta.shadow.raw[0]", "F01:quality_meta.shadow.raw[1]",
        "F01:faces_json[0].bbox[0]", "F01:faces_json[0].bbox[1]",
        "F01:faces_json[0].bbox[2]", "F01:faces_json[0].bbox[3]",
        "F01:evidence_json.nested.unused", "GROUP:group_id", "GROUP:decision_json.weak",
    }
    assert expected <= refs
    rendered = json.dumps(context)
    assert "/secret/a.jpg" not in rendered and "private.jpg" not in rendered
    assert {item["field"] for item in context["redactions"]} >= {
        "basename", "quality_meta.source_path",
    }
    assert context["policy"]["drop_weak_signals"] is False
    assert context["policy"]["drop_default_off_signals"] is False
    assert context["policy"]["drop_shadow_signals"] is False


def test_audit_and_model_contexts_split_binary_from_pairwise_cv(tmp_path):
    _output_dir, conn, gid, _ids = _output(tmp_path)
    rows = db.group_page_by_id(conn, gid)
    _members, audit_context, derived, model_context = vlm_review.build_layers(rows)
    rendered = json.dumps(model_context, sort_keys=True)
    assert '"encoding": "base64"' not in rendered
    assert '"data":' not in rendered
    assert "utility_score" not in rendered and "pair_margin" not in rendered
    assert "frame_identity" not in rendered
    assert derived["layer"] == "L1" and model_context["layer"] == "L2"
    phash = _measurement(derived, "phash")
    assert phash["value"] == {"hamming_distance": 3, "bit_length": 64}
    dino = _measurement(derived, "dinov2_embedding")
    assert dino["value"]["cosine_similarity"] == 0.0
    assert dino["value"]["cosine_distance"] == 1.0
    local = _measurement(derived, "local_quality_embedding")
    assert local["value"]["top_regions"] == [{
        "kind": "subject", "class_id": 0,
        "cosine_similarity": 0.0, "cosine_distance": 1.0,
    }]
    assert all(item["source_refs"] for item in (phash, dino, local))
    assert base64.b64decode(_audit_value(audit_context, "F01:dinov2_embedding")["data"]) == np.array(
        [1.0, 0.0], dtype=np.float16,
    ).tobytes()
    assert base64.b64decode(_audit_value(audit_context, "F01:local_quality_embedding")["data"]) == np.eye(
        1, 768, dtype=np.float16,
    ).tobytes()
    assert base64.b64decode(_audit_value(audit_context, "F01:phash")["data"]) == (0).to_bytes(8, "big")
    conn.close()


def test_model_context_guard_rejects_disguised_raw_embedding_field():
    with pytest.raises(ValueError, match="raw binary field"):
        vlm_review.assert_model_context_safe({"dinov2_embedding": "AAAA"})


def test_invoke_payload_contains_only_transformed_context(tmp_path, monkeypatch):
    _output_dir, conn, gid, _ids = _output(tmp_path)
    rows = db.group_page_by_id(conn, gid)
    _members, audit_context, model_context = vlm_review.build_contexts(rows)
    asset = tmp_path / "asset.jpg"
    Image.new("RGB", (16, 16), "black").save(asset, "JPEG")
    captured = {}

    class Client:
        def converse(self, **kwargs):
            captured.update(kwargs)
            raw = {"schema_version": 1, "group_id": gid,
                   "judgment": {"recommended": ["F01"], "confidence": "high",
                                "discriminators": []}, "abstentions": []}
            return {"output": {"message": {"content": [{"text": json.dumps(raw)}]}},
                    "usage": {}, "stopReason": "end_turn"}

    boto3 = ModuleType("boto3")
    boto3.client = lambda *_args, **_kwargs: Client()
    botocore = ModuleType("botocore")
    botocore_config = ModuleType("botocore.config")
    botocore_config.Config = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)

    vlm_review.invoke_bedrock(
        "global.anthropic.claude-sonnet-5-5", gid, [asset], model_context,
    )
    texts = [block["text"] for block in captured["messages"][0]["content"]
             if "text" in block]
    payload_text = "\n".join(texts)
    assert "MODEL_CONTEXT_JSON" in payload_text
    assert "CV_CONTEXT_JSON" not in payload_text
    assert '"encoding":"base64"' not in payload_text
    for ref in ("F01:phash", "F01:dinov2_embedding", "F01:local_quality_embedding"):
        raw_value = _audit_value(audit_context, ref)
        assert raw_value["data"] not in payload_text
    conn.close()


def test_easy_exact_group_bypasses_model_and_never_builds_visuals(tmp_path):
    output, conn, gid, _ids = _output(tmp_path, exact=True)
    before = _digest(conn)
    conn.close()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("easy exact group must not invoke a VLM")

    report = vlm_review.run_job(output, group_ids=[gid], model_id=None, invoke=forbidden)
    assert report["groups"] == [{"group_id": gid, "status": "ready", "route": "cv_only"}]
    artifact = json.loads(vlm_review.cache_path(output, gid).read_text())
    assert artifact["invariants"]["model_calls"] == 0
    assert not (output / vlm_review.CACHE_DIR / "assets" / str(gid)).exists()
    check = db.open_db(output / "inventory.sqlite")
    assert _digest(check) == before
    check.close()


def test_prepare_accepts_restored_file_when_mtime_changed_but_sha_matches(tmp_path):
    output, conn, gid, ids = _output(tmp_path)
    row = conn.execute("SELECT path FROM files WHERE id=?", (ids[0],)).fetchone()
    source = Path(row[0])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    conn.execute("UPDATE features SET content_sha256=? WHERE file_id=?", (digest, ids[0]))
    conn.commit()
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    conn.close()
    report = vlm_review.run_job(output, group_ids=[gid], prepare_only=True)
    assert report["groups"] == [{"group_id": gid, "status": "prepared", "route": "vlm"}]


def test_hard_group_gets_all_cv_fields_grounded_max_three_and_no_db_mutation(tmp_path):
    output, conn, gid, ids = _output(tmp_path)
    before = _digest(conn)
    conn.close()
    captured = {}

    def fake(model_id, group_id, assets, context):
        captured.update(model_id=model_id, group_id=group_id, assets=assets, context=context)
        refs = set(context["evidence_refs"])
        weak = next(ref for ref, item in context["evidence"].items()
                    if item["criterion"] == "quality_meta.weak.score")
        weak_ref = f"F02:{weak}"
        assert weak_ref in refs
        assert any(item["criterion"] == "phash" for item in context["pairs"].values())
        assert "utility_score" not in json.dumps(context)
        assert "pair_margin" not in json.dumps(context)
        claims = []
        for index in range(4):
            claims.append({
                "code": "COMPOSITE_ADVANTAGE", "subjects": ["F02"],
                "compared_to": ["F01"], "reason": f"visible distinction {index}",
                "confidence": "high", "visual_refs": [{"frame": "F02", "region": "face"}],
                "cv_refs": [weak_ref],
            })
        return ({"schema_version": 1, "group_id": group_id,
                 "judgment": {"recommended": ["F02"], "confidence": "high",
                              "discriminators": claims}, "abstentions": []},
                {"usage": {"inputTokens": 1, "outputTokens": 1}})

    report = vlm_review.run_job(
        output, group_ids=[gid], model_id="global.anthropic.claude-sonnet-5-5",
        invoke=fake,
    )
    assert report["groups"][0]["route"] == "vlm"
    assert captured["assets"]
    artifact = json.loads(vlm_review.cache_path(output, gid).read_text())
    assert len(artifact["presentation"]["distinctions"]) == 3
    assert artifact["presentation"]["recommended_file_ids"] == [ids[1]]
    assert artifact["presentation"]["distinctions"][0]["targets"] == [
        {"file_id": ids[1], "region": "face"}
    ]
    check = db.open_db(output / "inventory.sqlite")
    assert _digest(check) == before
    check.close()


def test_phase_distinct_mock_output_can_recommend_multiple_grounded_frames(tmp_path):
    output, conn, gid, ids = _output(tmp_path)
    conn.close()

    def fake(_model_id, group_id, _assets, _context):
        return ({
            "schema_version": 1, "group_id": group_id,
            "judgment": {
                "recommended": ["F01", "F02"], "confidence": "high",
                "discriminators": [
                    {
                        "code": "POSE_ACTION_DIFFERENCE", "subjects": ["F01"],
                        "compared_to": ["F02"],
                        "reason": "F01 preserves the raised-arm action phase.",
                        "confidence": "high",
                        "visual_refs": [{"frame": "F01", "region": "subject"}],
                        "cv_refs": [],
                    },
                    {
                        "code": "UNIQUE_MOMENT", "subjects": ["F02"],
                        "compared_to": ["F01"],
                        "reason": "F02 preserves the later interaction phase.",
                        "confidence": "high",
                        "visual_refs": [{"frame": "F02", "region": "full_frame"}],
                        "cv_refs": [],
                    },
                ],
            },
            "abstentions": [],
        }, {})

    vlm_review.run_job(
        output, group_ids=[gid], model_id="global.anthropic.claude-sonnet-5-5",
        invoke=fake,
    )
    artifact = json.loads(vlm_review.cache_path(output, gid).read_text())
    assert artifact["presentation"]["recommended_file_ids"] == ids
    assert len(artifact["presentation"]["distinctions"]) == 2


def test_low_confidence_observation_is_attention_only_not_recommendation_grounding(tmp_path):
    output, conn, gid, _ids = _output(tmp_path)
    conn.close()

    def fake(_model_id, group_id, _assets, _context):
        return ({
            "schema_version": 1, "group_id": group_id,
            "judgment": {
                "recommended": ["F02"], "confidence": "medium",
                "discriminators": [{
                    "code": "EXPRESSION_DIFFERENCE", "subjects": ["F02"],
                    "compared_to": ["F01"], "reason": "A subtle mouth-shape change.",
                    "confidence": "low",
                    "visual_refs": [{"frame": "F02", "region": "face"}],
                    "cv_refs": [],
                }],
            },
            "abstentions": [],
        }, {})

    vlm_review.run_job(
        output, group_ids=[gid], model_id="global.anthropic.claude-sonnet-5-5",
        invoke=fake,
    )
    presentation = json.loads(vlm_review.cache_path(output, gid).read_text())["presentation"]
    assert presentation["status"] == "abstained"
    assert presentation["recommended_file_ids"] == []
    assert presentation["distinctions"] == []
    assert presentation["attention_pointers"][0]["authority"] == \
        "optional_attention_pointer_only"


def test_transformation_version_change_invalidates_cache(tmp_path, monkeypatch):
    output, conn, gid, _ids = _output(tmp_path)
    conn.close()

    def fake(_model_id, group_id, _assets, _context):
        return ({
            "schema_version": 1, "group_id": group_id,
            "judgment": {
                "recommended": ["F01"], "confidence": "high",
                "discriminators": [{
                    "code": "UNIQUE_MOMENT", "subjects": ["F01"],
                    "compared_to": ["F02"], "reason": "F01 preserves a distinct phase.",
                    "confidence": "high",
                    "visual_refs": [{"frame": "F01", "region": "full_frame"}],
                    "cv_refs": [],
                }],
            },
            "abstentions": [],
        }, {})

    vlm_review.run_job(
        output, group_ids=[gid], model_id="global.anthropic.claude-sonnet-5-5",
        invoke=fake,
    )
    check = db.open_db(output / "inventory.sqlite")
    rows = db.group_page_by_id(check, gid)
    assert vlm_review.load_cached(output, rows) is not None
    monkeypatch.setattr(vlm_review, "CONTEXT_TRANSFORMATION_VERSION", "pairwise-binary-cv-v2")
    assert vlm_review.load_cached(output, rows) is None
    check.close()


def test_hash_mismatched_cache_is_ignored_by_api_projection(tmp_path):
    output, conn, gid, _ids = _output(tmp_path)
    rows = db.group_page_by_id(conn, gid)

    def fake(_model_id, group_id, _assets, _context):
        return ({
            "schema_version": 1, "group_id": group_id,
            "judgment": {"recommended": ["F02"], "confidence": "high",
                "discriminators": [{"code": "COMPOSITE_ADVANTAGE",
                    "subjects": ["F02"], "compared_to": ["F01"],
                    "reason": "F02 has a visible difference.", "confidence": "high",
                    "visual_refs": [{"frame": "F02", "region": "full_frame"}],
                    "cv_refs": []}]}, "abstentions": []}, {})

    vlm_review.run_job(
        output, group_ids=[gid], model_id="global.anthropic.claude-sonnet-5-5",
        invoke=fake,
    )
    path = vlm_review.cache_path(output, gid)
    artifact = json.loads(path.read_text())
    current_hash = artifact["input_hash"]
    artifact["input_hash"] = "not-current"
    artifact["presentation"] = {
        "source": "tampered", "status": "grounded",
        "distinctions": [{"text": "must not be trusted"}],
    }
    path.write_text(json.dumps(artifact))
    projected = review_server._group_items(rows, output)[0]
    assert "review_judgment" not in projected
    assert projected["selection_reason"]
    artifact["input_hash"] = current_hash
    path.write_text(json.dumps(artifact))
    projected = review_server._group_items(rows, output)[0]
    assert projected["review_judgment"]["source"] == "vlm"
    assert projected["review_judgment"]["distinctions"][0]["text"] == \
        "F02 has a visible difference."
    conn.close()
