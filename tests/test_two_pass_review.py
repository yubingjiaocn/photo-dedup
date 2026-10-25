"""Deterministic mechanics, not VLM quality or production admission tests."""

import copy
import json
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from scripts.probe_two_pass_review import fake_judgment, fake_map, variants
from src import two_pass_review as tp

ROOT = Path(__file__).resolve().parents[1]


def test_future_judge_reasons_are_requested_in_simplified_chinese():
    assert "Simplified Chinese" in tp.CHINESE_OUTPUT_INSTRUCTION
    assert "schema defines them in English" in tp.CHINESE_OUTPUT_INSTRUCTION
    assert "English visual observations only" in tp.PRIVACY_INSTRUCTION


@pytest.fixture
def fixture(tmp_path):
    rows, sources = [], {}
    for fid in (1, 2):
        path = tmp_path / f"{fid}.png"
        Image.new("RGB", (1600, 1200), (80 * fid, 40, 30)).save(path)
        rows.append(
            {
                "file_id": fid,
                "content_sha256": tp.sha(path.read_bytes()),
                "width": 1600,
                "height": 1200,
                "quality_meta": {},
                "faces_json": [],
            }
        )
        sources[fid] = path
    return rows, sources


def run(fixture, a=None, b=None, output=None):
    rows, sources = fixture
    a = a or fake_map(rows, "stage")
    b = b or fake_judgment(a)
    runner = tp.FakeModel({"A": a, "B": b})
    output = output or ROOT / "artifacts/two-pass-multi-subject/tests" / uuid4().hex
    result = tp.run_offline(
        rows, sources, output, model_a="fake:A", model_b="fake:B", runner=runner
    )
    return result, output, runner


def test_e2e_multi_select_irreplaceable_and_full_views(fixture):
    result, output, runner = run(fixture)
    assert result["judgment"]["decision"] == "select"
    assert len(result["judgment"]["recommended"]) == 2
    assert len(result["judgment"]["irreplaceable"]) == 2
    assert len(runner.calls) == 2
    assert result["external_model_calls"] == result["decision_table_writes"] == 0
    assert not result["production_enabled"]
    req = json.loads((output / "pass-A-request.json").read_text())
    assert all(a["kind"] == "full_group_page" for a in req["images"])
    assert set(req["context"]) == {"frames", "scene_hint"}
    l2 = json.loads((output / "L2.json").read_text())
    assert set(l2["frame_bindings"]) == set(l2["full_frame"]["frames"])
    assert set(l2["frame_bindings"].values()) == {tp.identity(r) for r in fixture[0]}
    for a in result["assets"]:
        for cell in a["cells"]:
            assert min(cell["actual_image_size"]) >= (
                1024 if a["kind"] == "full_group_page" else 256
            )
            assert not cell["upsampled"]
            assert all(
                x <= n for x, n in zip(cell["actual_image_size"], cell["native_size"])
            )
    assert result["subject_set_evidence"]["full_subject_coverage"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(name="real identity"),
        lambda p: p["regions"][0]["frame"].update(content_sha256="f" * 64),
        lambda p: p["regions"][0]["frame"].update(file_id=True),
        lambda p: p["regions"][0].update(box=[-0.1, 0.1, 0.5, 0.8]),
        lambda p: p["regions"][0].update(box=[0.1, 0.1, float("nan"), 0.8]),
        lambda p: p["regions"][0].update(box=[0, 0, 1, 1]),
        lambda p: p["regions"][0].update(box=[0, 0, 0.001, 0.001]),
        lambda p: p["regions"][0].update(box=[False, 0, 0.5, 0.5]),
        lambda p: p["regions"][0]["grounding"].update(page="F01"),
        lambda p: p["regions"][0]["grounding"].update(visible=False),
        lambda p: p["regions"].append(copy.deepcopy(p["regions"][0])),
        lambda p: p["coverage"].append(copy.deepcopy(p["coverage"][0])),
        lambda p: p["correspondence"][0].update(regions=["R1", "R1"]),
    ],
)
def test_schema_rejects_ungrounded_or_invalid_map(fixture, mutation):
    rows, _ = fixture
    a = fake_map(rows, "stage")
    mutation(a)
    with pytest.raises((tp.ReviewError, ValueError)):
        tp.validate_map(a, rows)


@pytest.mark.parametrize("case", ["missing", "bystander-pollution", "ambiguous"])
def test_fail_closed_before_pass_b(fixture, case):
    rows, _ = fixture
    a = next(a for name, a, _ in variants(fake_map(rows, "stage")) if name == case)
    result, _, runner = run(fixture, a=a, b=fake_judgment(fake_map(rows, "stage")))
    assert result["judgment"]["decision"] == "attention_only"
    assert len(runner.calls) == 1
    assert not result["judgment"]["recommended"]
    assert not any(a["kind"] != "full_group_page" for a in result["assets"])


def test_bystander_never_enters_strips(fixture):
    rows, _ = fixture
    a = next(
        a for name, a, _ in variants(fake_map(rows, "stage")) if name == "positive"
    )
    result, _, _ = run(fixture, a=a)
    assert result["judgment"]["decision"] == "select"
    refs = {
        c["region"]
        for asset in result["assets"]
        if "strip" in asset["kind"]
        for c in asset["cells"]
    }
    assert not refs.intersection({"R10", "R11"})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b.update(confidence="low"),
        lambda b: b["recommended"][0]["evidence"][0].update(confidence="low"),
        lambda b: b["recommended"][0]["evidence"][0].update(region="R2"),
        lambda b: b["recommended"][0]["frame"].update(content_sha256="f" * 64),
        lambda b: b.update(multi_select_reason=None),
        lambda b: b["multi_select_reason"].update(
            evidence=[{"region": "R1", "confidence": "medium"}]
        ),
        lambda b: b["irreplaceable"][0].update(subject="R2"),
        lambda b: b["irreplaceable"][0].update(evidence=[]),
        lambda b: b["discriminators"].extend(copy.deepcopy(b["discriminators"]) * 3),
    ],
)
def test_select_requires_frame_bound_medium_high_evidence(fixture, mutation):
    rows, _ = fixture
    a = fake_map(rows, "stage")
    b = fake_judgment(a)
    mutation(b)
    with pytest.raises((tp.ReviewError, TypeError)):
        tp.validate_judgment(b, rows, a, tp.validate_map(a, rows), {})


def test_low_attention_and_abstain_allowed(fixture):
    rows, _ = fixture
    a = fake_map(rows, "stage")
    for decision in ("attention_only", "abstain"):
        b = fake_judgment(a, decision)
        b["discriminators"][0]["evidence"][0]["confidence"] = "low"
        result, _, _ = run(fixture, a=a, b=b)
        assert result["judgment"]["decision"] == decision


def test_small_native_roi_and_text_budget_stop(fixture, monkeypatch):
    rows, _ = fixture
    a = fake_map(rows, "stage")
    for r in a["regions"]:
        r["box"] = [0.1, 0.1, 0.2, 0.2]
    result, _, runner = run(fixture, a=a)
    assert "ROI_NATIVE_SHORT_EDGE_BELOW_256" in result["blockers"][0]
    assert len(runner.calls) == 1
    monkeypatch.setitem(tp.BUDGET["A"], "text_tokens_upper_bound", 1)
    result, _, runner = run(fixture)
    assert "budget" in result["blockers"][0]
    assert not runner.calls


def test_cache_invalidates_models_cv_response_assets_and_source(fixture):
    result, output, _ = run(fixture)
    cached, _, runner = run(fixture, output=output)
    assert cached["cache_hit"] and not runner.calls
    (output / result["assets"][0]["path"]).write_bytes(b"damaged")
    regenerated, _, runner = run(fixture, output=output)
    assert not regenerated["cache_hit"] and len(runner.calls) == 2
    changed = copy.deepcopy(fixture)
    changed[0][0]["quality_meta"] = {"sharpness": 0.8}
    cv, _, _ = run(changed, output=output)
    assert cv["cache_key"] != result["cache_key"]
    a = fake_map(fixture[0], "stage")
    a["coverage"][0].update(state="uncertain", uncertainty="resolution")
    response, _, _ = run(fixture, a=a, output=output)
    assert response["cache_key"] != result["cache_key"]
    runner = tp.FakeModel({"A": a, "B": fake_judgment(a)})
    model = tp.run_offline(
        *fixture, output, model_a="fake:other", model_b="fake:B", runner=runner
    )
    assert model["cache_key"] != response["cache_key"]
    fixture[1][1].write_bytes(b"wrong content")
    with pytest.raises(tp.ReviewError, match="source SHA"):
        run(fixture, output=output)


def test_explicit_models_no_retry_or_transport(fixture):
    rows, sources = fixture
    a = fake_map(rows, "stage")
    runner = tp.FakeModel({"A": a, "B": fake_judgment(a)})
    with pytest.raises(tp.ReviewError, match="BOTH"):
        tp.run_offline(
            rows,
            sources,
            ROOT / "artifacts/two-pass-multi-subject/tests/unused",
            model_a="fake:A",
            model_b="",
            runner=runner,
        )
    with pytest.raises(tp.ReviewError, match="FakeModel"):
        tp.run_offline(
            rows,
            sources,
            ROOT / "artifacts/two-pass-multi-subject/tests/unused",
            model_a="fake:A",
            model_b="fake:B",
            runner=object(),
        )
    _, output, runner = run(fixture)
    with pytest.raises(tp.ReviewError, match="retry"):
        tp.run_offline(
            rows, sources, output, model_a="fake:A", model_b="fake:B", runner=runner
        )


def test_local_cv_joins_eyes_without_broadcast_or_detector_subject_authority(fixture):
    rows, sources = fixture
    rows[0]["quality_meta"] = {
        "eye_detection": {
            "status": "ALL_OPEN",
            "faces": [
                {
                    "bbox_xywh": [500, 400, 100, 100],
                    "state": "CLOSED",
                    "blink_left": 0.9,
                },
                {"bbox_xywh": [10, 10, 100, 100], "state": "OPEN"},
            ],
        },
        "utility": 123456,
        "pair_margin": 999,
        "keeper": True,
    }
    result, output, _ = run((rows, sources))
    l1 = json.loads((output / "L1.json").read_text())
    assert l1["ROI"]["R1"]["eyes"][0]["state"] == "CLOSED"
    assert len(l1["ROI"]["R1"]["eyes"]) == 1
    assert l1["ROI"]["R2"]["eyes"].startswith("unknown")
    assert l1["ROI"]["R1"]["completeness"] == "unknown"
    text = (output / "pass-B-request.json").read_text()
    assert (
        "123456" not in text and '"pair_margin"' not in text and '"keeper"' not in text
    )
    assert result["judgment"]["decision"] == "select"


def test_real_reports_preserved_and_fake_cases_inert():
    root = ROOT / "artifacts/two-pass-multi-subject"
    if not (root / "report.json").exists():
        pytest.skip("offline frozen probe not run")
    report = json.loads((root / "report.json").read_text())
    assert len(report["cases"]) == 10
    assert all(
        c["decision_tables_before"] == c["decision_tables_after"]
        for c in report["cases"]
    )
    assert report["external_model_calls"] == 0
    for name, digest in report["old_real_results_sha256"].items():
        assert (
            tp.sha((ROOT / "artifacts/multi-subject-narrow" / name).read_bytes())
            == digest
        )
    old = json.loads((ROOT / "artifacts/multi-subject-narrow/report.json").read_text())
    assert old["real_slice_gate_passes"] == 0
    assert all(g["status"] == "attention_only" for g in old["groups"])


def test_published_schemas_match_fake_contract(fixture):
    import jsonschema

    a = fake_map(fixture[0], "stage")
    b = fake_judgment(a)
    for name, value in (("subject-map", a), ("deep-judge", b)):
        schema = json.loads(
            (ROOT / f"schemas/two-pass-{name}-v1.schema.json").read_text()
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(value, schema)


def test_overlap_and_track_completeness(fixture):
    rows, _ = fixture
    a = fake_map(rows, "stage")
    r = copy.deepcopy(a["regions"][0])
    r["id"] = "R9"
    a["regions"].append(r)
    with pytest.raises(tp.ReviewError, match="overlapping"):
        tp.validate_map(a, rows)
    a = fake_map(rows, "stage")
    a["correspondence"] = []
    assert "KEY_REGION_NOT_COMPARED" in tp.validate_map(a, rows)["blockers"]


@pytest.mark.parametrize(
    "phase,field,value",
    [
        ("A", "images", 1),
        ("B", "images", 1),
        ("B", "text_tokens_upper_bound", 1),
        ("A", "output_bytes", 1),
    ],
)
def test_hard_call_and_request_budgets(fixture, monkeypatch, phase, field, value):
    monkeypatch.setitem(tp.BUDGET[phase], field, value)
    result, _, runner = run(fixture)
    assert result["judgment"]["decision"] == "attention_only"
    assert "budget" in result["blockers"][0]
    assert len(runner.calls) <= (1 if phase == "B" or field == "output_bytes" else 0)


def test_roi_count_and_frame_budget(fixture):
    rows, _ = fixture
    a = fake_map(rows, "stage")
    a["regions"] *= 9
    with pytest.raises(tp.ReviewError, match="budget"):
        tp.validate_map(a, rows)
    with pytest.raises(tp.ReviewError, match="frame budget"):
        run((rows * 3, fixture[1]))


@pytest.mark.parametrize("uncertainty", ["resolution", "occlusion"])
def test_uncertain_coverage_enriches_but_select_ceiling_is_enforced(
    fixture, uncertainty
):
    a = fake_map(fixture[0], "stage")
    a["coverage"][0].update(state="uncertain", uncertainty=uncertainty)
    result, output, runner = run(fixture, a=a, b=fake_judgment(a, "attention_only"))
    assert len(runner.calls) == 2
    assert result["decision_ceiling"] == "attention_only"
    assert json.loads((output / "L2.json").read_text())["ROI"]
    assert result["judgment"]["decision"] == "attention_only"
    with pytest.raises(tp.ReviewError, match="ceiling"):
        tp.validate_judgment(
            fake_judgment(a), fixture[0], a, tp.validate_map(a, fixture[0]), {}
        )
    rejected, _, calls = run(fixture, a=a, b=fake_judgment(a))
    assert len(calls.calls) == 2
    assert not rejected["judgment"]["recommended"]
    assert rejected["judgment"]["decision"] == "attention_only"


@pytest.mark.parametrize("role", ["bystander", "unknown"])
def test_small_nonmain_track_is_ledger_only_even_if_ambiguous(fixture, role):
    a = next(
        a
        for name, a, _ in variants(fake_map(fixture[0], "stage"))
        if name == "positive"
    )
    for region in a["regions"][-2:]:
        region.update(role=role, box=[0.01, 0.01, 0.02, 0.02])
    a["correspondence"].append(
        {
            "id": "T2",
            "regions": ["R10", "R11"],
            "confidence": "low",
            "state": "ambiguous",
        }
    )
    result, output, runner = run(fixture, a=a, b=fake_judgment(a, "attention_only"))
    assert len(runner.calls) == 2
    assert result["subject_set_evidence"]["eligible_tracks"] == ["T1"]
    ledger = json.loads((output / "L1.json").read_text())["coverage_ledger"]
    assert all(not r["strip_eligible"] for r in ledger if r["role"] == role)
    assert len([asset for asset in result["assets"] if "strip" in asset["kind"]]) == 1


def test_supporting_track_and_attention_evidence_allowed(fixture):
    a = fake_map(fixture[0], "group")
    for region in a["regions"]:
        if region["id"] in {"R2", "R4"}:
            region["role"] = "supporting"
            region["box"] = [0.7, 0.1, 0.99, 0.8]
    a["coverage"][0].update(state="uncertain", uncertainty="occlusion")
    b = fake_judgment(a, "attention_only")
    b["discriminators"][0]["evidence"] = [{"region": "R2", "confidence": "medium"}]
    result, _, runner = run(fixture, a=a, b=b)
    assert len(runner.calls) == 2
    assert result["subject_set_evidence"]["eligible_tracks"] == ["T1", "T2"]
    assert len(result["assets"]) == 4
    assert result["judgment"]["discriminators"][0]["evidence"][0]["region"] == "R2"
    a["regions"][3]["role"] = "key_subject"
    with pytest.raises(tp.ReviewError, match="incompatible"):
        tp.validate_map(a, fixture[0])


def test_tiny_nonmain_box_still_requires_finite_grounded_coordinates(fixture):
    a = fake_map(fixture[0], "stage")
    a["regions"][0].update(role="bystander", box=[0, 0, float("nan"), 0.001])
    with pytest.raises(tp.ReviewError, match="non-finite"):
        tp.validate_map(a, fixture[0])
