"""Synthetic, offline v4 boundary tests; no development labels or model calls."""

import copy
import hashlib

import jsonschema
import pytest

from src import cloud_judge_batch as b
from src import large_group_review as v4
from src import two_pass_review as tp


def fixture(n=3):
    rows = [
        {"file_id": i, "content_sha256": hashlib.sha256(str(i).encode()).hexdigest()}
        for i in range(1, n + 1)
    ]
    views = [
        {
            "id": f"R{i}",
            "frame": r,
            "box": [0, 0, 1, 1],
            "purpose": "complete_frame",
            "role": "frame_view",
            "confidence": "high",
        }
        for i, r in enumerate(rows, 1)
    ]
    packet = {"complete_frames": [tp.identity(r) for r in rows], "views": views}
    validation = {
        "lookup": {v["id"]: v for v in views},
        "blockers": [],
        "decision_ceiling": "select",
    }
    ev = lambda i: [{"region": f"R{i}", "confidence": "medium"}]
    answer = {
        "contract": v4.CONTRACT,
        "decision": "select",
        "confidence": "medium",
        "recommended": [{"frame": r, "evidence": ev(i)} for i, r in enumerate(rows, 1)],
        "multi_select_reason": {
            "text": "分别保留不同动作和表情。",
            "evidence": [ev(i)[0] for i in range(1, n + 1)],
        },
        "irreplaceable": [],
        "discriminators": [{"text": "动作阶段清晰可辨。", "evidence": ev(1)}],
        "phase_census": [
            {
                "frame": r,
                "cluster": f"P{i}",
                "state": "可见独特动作阶段。",
                "readability": "usable",
                "evidence": ev(i),
            }
            for i, r in enumerate(rows, 1)
        ],
        "clusters": [
            {
                "id": f"P{i}",
                "members": [r],
                "relation": "distinct",
                "representative": r,
                "reason": "不同动作值得保留。",
                "omissions": [],
            }
            for i, r in enumerate(rows, 1)
        ],
    }
    request = {
        "version": b.VERSION,
        "pass": "single_judge",
        "prompt": b.JUDGE_PROMPT,
        "context": {"visual_packet": packet},
        "response_schema": b.read(b.ROOT / "schemas/cloud-scene-judge-v1.schema.json"),
        "max_output_tokens": 4096,
        "images": [
            {
                "kind": "full_group_page",
                "cells": [{"frame": tp.identity(r), "region": f"R{i}"}],
            }
            for i, r in enumerate(rows, 1)
        ],
    }
    return rows, packet, validation, answer, request


def validate(data):
    rows, packet, validation, answer, _ = data
    return v4.validate_response(answer, rows, packet, validation, {})


@pytest.mark.parametrize("n", [3, 8, 9, 11, 16])
def test_all_useful_phases_can_be_kept_without_eight_item_cap(n):
    data = fixture(n)
    assert validate(data) == data[3]
    assert v4.build_request(data[4], data[0])["judgment_contract"] == v4.CONTRACT


def test_pair_request_is_byte_identical_and_never_loads_new_schema(monkeypatch):
    rows, packet, validation, answer, request = fixture(2)

    def forbidden():
        pytest.fail("pair entered v4 schema")

    monkeypatch.setattr(v4, "response_schema", forbidden)
    before = tp.canonical(request)
    assert tp.canonical(v4.build_request(request, rows)) == before
    for k in ("contract", "phase_census", "clusters"):
        answer.pop(k)
    answer["discriminators"][0]["text"] = "Existing English v3 output is unchanged."
    legacy = tp.validate_judgment(
        answer, rows, packet, validation, {}, allow_visual_views=True
    )
    assert v4.validate_response(answer, rows, packet, validation, {}) == legacy
    assert v4.contract_for(2) == "v3"


def test_near_duplicates_one_representative():
    data = fixture()
    a = data[3]
    a["clusters"] = [
        {
            **a["clusters"][0],
            "members": data[0],
            "omissions": [
                {"frame": r, "reason": "表情动作相同，代表帧更清晰。"}
                for r in data[0][1:]
            ],
        }
    ]
    for c in a["phase_census"]:
        c["cluster"] = "P1"
    a["recommended"] = a["recommended"][:1]
    a["multi_select_reason"] = None
    assert validate(data) == a


def test_uncertain_substitution_not_low_visual_evidence():
    data = fixture()
    data[3]["clusters"][1]["relation"] = "uncertain_distinct"
    assert validate(data)
    data[3]["phase_census"][1]["evidence"][0]["confidence"] = "low"
    with pytest.raises(tp.ReviewError, match="medium/high"):
        validate(data)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["phase_census"].pop(),
        lambda a: a["phase_census"].append(a["phase_census"][0]),
        lambda a: a["phase_census"][0]["frame"].update(content_sha256="0" * 64),
        lambda a: a["phase_census"][0]["evidence"][0].update(region="R2"),
        lambda a: a["phase_census"][0].update(cluster="P9"),
        lambda a: a["phase_census"][0].update(readability="uncertain"),
        lambda a: a["clusters"][0].update(
            representative=a["clusters"][1]["representative"]
        ),
        lambda a: a["clusters"][0].update(relation="unassessable", representative=None),
        lambda a: a["clusters"].pop(),
        lambda a: a["clusters"][0]["omissions"].append(
            {"frame": a["clusters"][0]["representative"], "reason": "不应省略代表帧。"}
        ),
        lambda a: a["recommended"].pop(),
        lambda a: a["recommended"].append(a["recommended"][0]),
        lambda a: a["multi_select_reason"]["evidence"].pop(),
        lambda a: a["multi_select_reason"]["evidence"][0].update(confidence="low"),
        lambda a: a["discriminators"][0].update(text="English only"),
        lambda a: a.update(confidence="low"),
        lambda a: a.update(contract="v3"),
    ],
)
def test_invalid_coverage_or_binding_rejected(mutation):
    data = list(fixture())
    # Break shared references so adversarial edits affect only the response.
    data[3] = copy.deepcopy(data[3])
    mutation(data[3])
    with pytest.raises((tp.ReviewError, jsonschema.ValidationError)):
        validate(data)


def test_attention_can_record_unreadable_without_recommendations():
    data = fixture()
    a = data[3]
    a.update(decision="attention_only", recommended=[], multi_select_reason=None)
    a["phase_census"][0]["readability"] = "unreadable"
    a["clusters"][0].update(
        relation="unassessable",
        representative=None,
        omissions=[{"frame": data[0][0], "reason": "无法辨认，需人工检查。"}],
    )
    assert validate(data)


@pytest.mark.parametrize(
    "kind", ["missing_page", "missing_packet", "bad_binding", "over_budget", "too_many"]
)
def test_request_refuses_partial_or_oversize_input(kind):
    rows, _, _, _, request = fixture(17 if kind == "too_many" else 3)
    if kind == "missing_page":
        request["images"].pop()
    if kind == "missing_packet":
        request["context"]["visual_packet"]["complete_frames"].pop()
    if kind == "bad_binding":
        request["images"][0]["cells"][0]["region"] = "R2"
    if kind == "over_budget":
        request["context"]["extra"] = "x" * 48000
    with pytest.raises(tp.ReviewError):
        v4.build_request(request, rows)


def test_current_runner_has_no_v4_import_or_activation():
    assert "large_group_review" not in (b.ROOT / "src/cloud_judge_batch.py").read_text()
