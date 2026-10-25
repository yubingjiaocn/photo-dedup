"""No paid calls; local prefilter and single-authority projection regressions."""

import copy
import hashlib

import jsonschema
import numpy as np
import pytest
from PIL import Image

from scripts import run_large_group_development as runner
from src import cloud_judge_batch as b
from src import large_group_projection as p
from src import large_group_review as v4
from src import photo_exposure as cv
from src import two_pass_review as tp
from tests.test_large_group_review import fixture


def raw_fixture(n=3):
    rows, packet, validation, answer, request = fixture(n)
    answer.pop("recommended")
    answer["contract"] = p.CONTRACT
    answer["multi_select_reason"] = answer["multi_select_reason"]["text"]
    return rows, packet, validation, answer, request


@pytest.mark.parametrize("n", [3, 9, 11, 16])
def test_projection_has_one_authoritative_shortlist(n):
    rows, packet, validation, raw, _ = raw_fixture(n)
    checked = p.validate_response(raw, rows, packet, validation, {})
    assert [r["frame"] for r in checked["recommended"]] == rows
    assert len(checked["multi_select_reason"]["evidence"]) == n
    assert "recommended" not in raw
    assert checked["contract"] == p.CONTRACT


def test_duplicate_model_shortlist_is_rejected_not_repaired():
    rows, packet, validation, raw, _ = raw_fixture()
    raw["recommended"] = []
    with pytest.raises(jsonschema.ValidationError):
        p.validate_response(raw, rows, packet, validation, {})


def test_projection_preserves_pair_bytes():
    rows, _, _, _, request = fixture(2)
    assert tp.canonical(p.build_request(request, rows)) == tp.canonical(request)


def test_projection_rejects_bad_sha_and_weak_census():
    for bad_sha in (True, False):
        rows, packet, validation, raw, _ = raw_fixture()
        raw = copy.deepcopy(raw)
        if bad_sha:
            raw["clusters"][0]["representative"]["content_sha256"] = "0" * 64
        else:
            raw["phase_census"][0]["evidence"][0]["confidence"] = "low"
        with pytest.raises(tp.ReviewError):
            p.validate_response(raw, rows, packet, validation, {})


def test_wire_counts_source_photos_not_contact_sheet_parts(tmp_path):
    rows, _, _, _, request = raw_fixture()
    request = p.build_request(request, rows)
    path = tmp_path / "sheet.jpg"
    Image.new("RGB", (90, 30), "white").save(path)
    data = path.read_bytes()
    request["images"] = [
        {
            "kind": "complete_temporal_storyboard",
            "path": path.name,
            "bytes": len(data),
            "size": [90, 30],
            "sha256": hashlib.sha256(data).hexdigest(),
            "cells": [{"frame": tp.identity(r)} for r in rows],
        }
    ]
    wire, exposure = runner.wire_request(request, tmp_path)
    assert exposure["unique_source_photos"] == 3
    assert exposure["image_parts"] == 1
    assert wire["inferenceConfig"]["maxTokens"] == 8192
    assert (
        "English visual observations only"
        not in wire["messages"][0]["content"][0]["text"]
    )
    path.write_bytes(b"changed")
    with pytest.raises(tp.ReviewError, match="binding"):
        runner.wire_request(request, tmp_path)


def test_failed_reservation_is_never_resent(tmp_path):
    group = "synthetic-dev"
    b.save(tmp_path / group / "state.json", {"status": "failed"})

    class Client:
        def converse(self, **kwargs):
            pytest.fail("retry occurred")

    with pytest.raises(tp.ReviewError, match="cannot be retried"):
        runner.run_group(group, [group], Client(), run_root=tmp_path)


def frames(values, times):
    return [
        {
            "identity": str(i),
            "time": time,
            "descriptor": {
                "rgb": np.full((64, 64, 3), value, dtype=np.float32),
                "dhash": np.zeros((8, 8), dtype=bool),
                "aspect": 1,
                "quality": 10 - i,
            },
        }
        for i, (value, time) in enumerate(zip(values, times))
    ]


def test_local_clusters_direct_comparison_not_transitive_chain():
    groups = cv.micro_clusters(
        frames([0, 0.02, 0.04], [0, 0, 0]), cv.POLICIES["balanced"]
    )
    assert groups[0]["members"] == ["0", "1"]
    assert groups[1]["members"] == ["2"]


def test_unknown_or_distant_time_never_pruned():
    for times in ([None, None], [0, 2]):
        assert (
            len(cv.micro_clusters(frames([0, 0], times), cv.POLICIES["balanced"])) == 2
        )


def test_identical_local_descriptor_dedups_and_saves_ledger():
    result = cv.micro_clusters(frames([0, 0], [0, 0]), cv.POLICIES["strict"])
    assert len(result) == 1
    assert result[0]["omissions"][0]["comparison"]["dhash"] == 0


def test_router_emits_exact_exposure_allowlist_and_version():
    result = cv.route_group(frames([0, 0.2, 0.4], [0, 0, 0]), "balanced")
    assert result["router_version"] == "photo-exposure-router-v1"
    assert result["vlm_required"] is True
    assert result["local_selected"] == []
    assert set(result["vlm_exposed"]) == {"0", "1", "2"}
    assert result["member_to_representative_ratio"] == 1


def test_router_can_local_resolve_only_homogeneous_clear_quality_winner():
    sample = frames([0, 0, 0], [0, 0, 0])
    sample[0]["descriptor"]["quality"] = 10
    sample[1]["descriptor"]["quality"] = 1
    sample[2]["descriptor"]["quality"] = 2
    result = cv.route_group(sample, "recall_first")
    assert result["route"] == "local_resolved"
    assert result["vlm_exposed"] == []
    assert result["local_selected"] == ["0"]


def test_two_frame_selection_contract_remains_v3_even_when_homogeneous():
    sample = frames([0, 0], [0, 0])
    sample[0]["descriptor"]["quality"] = 10
    sample[1]["descriptor"]["quality"] = 1
    result = cv.route_group(sample, "efficient")
    assert result["pair_selection_contract"] == "v3_unchanged"
    assert result["route"] == "vlm_required"
    assert result["local_selected"] == []


def test_projection_schema_artifact_matches_function():
    assert p.schema() == b.read(
        b.ROOT / "schemas/large-group-judge-v4-candidate-2.schema.json"
    )


def test_unreadable_remains_attention_only():
    rows, packet, validation, raw, _ = raw_fixture()
    raw.update(decision="attention_only", multi_select_reason=None)
    raw["phase_census"][0]["readability"] = "unreadable"
    raw["clusters"][0].update(
        relation="unassessable",
        representative=None,
        omissions=[{"frame": rows[0], "reason": "细节不清，需人工检查。"}],
    )
    assert p.validate_response(raw, rows, packet, validation, {})["recommended"] == []


def test_old_v3_limits_still_unchanged():
    assert v4.contract_for(2) == "v3"
    assert tp.BUDGET["B"]["output_tokens"] == 4096


def test_b_transport_contains_only_representatives(tmp_path, monkeypatch):
    group = "synthetic-dev"
    case = tmp_path / "source" / group
    case.mkdir(parents=True)
    rows, _packet, validation, _, request = fixture(5)
    request["context"].update(
        temporal=[{"frame": r, "capture_time": 1} for r in rows],
        evidence={
            "frame_bindings": {f"F{i}": tp.identity(r) for i, r in enumerate(rows, 1)}
        },
    )
    for i, asset in enumerate(request["images"]):
        path = case / f"page-{i}.jpg"
        Image.new("RGB", (20, 20), (i * 20, 40, 60)).save(path)
        data = path.read_bytes()
        asset.update(
            path=path.name, sha256=tp.sha(data), bytes=len(data), size=[20, 20]
        )
    b.save(case / "request.json", request)
    b.save(case / "view-validation.json", validation)
    monkeypatch.setattr(runner, "SOURCE", case.parent)
    _, _, _, raw, _ = raw_fixture(4)

    class Client:
        calls = 0

        def converse(self, **kwargs):
            self.calls += 1
            content = kwargs["messages"][0]["content"]
            assert sum("image" in c for c in content) == 4
            assert rows[4]["content_sha256"] not in content[0]["text"]
            return {
                "stopReason": "end_turn",
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "output": {
                    "message": {"content": [{"text": tp.canonical(raw).decode()}]}
                },
            }

    client = Client()
    state = runner.run_group(
        group,
        [group],
        client,
        run_root=tmp_path / "result",
        exposure_plan={"group": group, "exposed": [tp.identity(r) for r in rows[:4]]},
    )
    assert state["member_count"] == 5
    assert state["exposure"]["unique_source_photos"] == 4
    assert client.calls == 1
    assert (
        state["decision_writes"]
        == state["keeper_writes"]
        == state["delete_writes"]
        == 0
    )


def test_converse_reasoning_envelope_preserved_but_not_used_as_judgment():
    rows, _packet, validation, raw, request = raw_fixture()
    request["context"]["evidence"] = {}
    request = p.build_request(request, rows)
    response = {
        "stopReason": "end_turn",
        "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
        "output": {
            "message": {
                "content": [
                    {"reasoningContent": {"reasoningText": {"signature": "opaque"}}},
                    {"text": tp.canonical(raw).decode()},
                ]
            }
        },
    }
    assert runner.validate_provider(response, request, rows, validation)["recommended"]
    response["output"]["message"]["content"][0] = {"toolUse": {"name": "unwanted"}}
    with pytest.raises(tp.ReviewError, match="unexpected response"):
        runner.validate_provider(response, request, rows, validation)
