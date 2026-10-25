"""Offline contract/resume/safety tests. No SDK/network invocation."""

import copy
from pathlib import Path
from uuid import uuid4

import jsonschema
import pytest

from scripts import revalidate_saved_judge
from scripts.probe_two_pass_review import fake_judgment, fake_map
from src import cloud_judge_batch as b
from src import two_pass_review as tp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mapped():
    rows = [{"file_id": i, "content_sha256": str(i) * 64} for i in (1, 2)]
    packet = fake_map(rows, "stage")
    packet["regions"].append(
        {
            "id": "R9",
            "frame": rows[0],
            "box": [0.01, 0.01, 0.02, 0.02],
            "role": "bystander",
            "kind": "subject",
            "confidence": "low",
            "grounding": {"page": tp.identity(rows[0]), "visible": True},
        }
    )
    return rows, packet


@pytest.mark.parametrize("length", [240, 241, 249, 500])
def test_human_prose_unicode_limit_not_display_limit(mapped, length):
    rows, packet = mapped
    answer = fake_judgment(packet, "attention_only")
    answer["discriminators"][0]["text"] = "观察" * (length // 2) + "看" * (length % 2)
    original = copy.deepcopy(answer)
    jsonschema.validate(
        answer, b.read(ROOT / "schemas/two-pass-deep-judge-v1.schema.json")
    )
    assert (
        tp.validate_judgment(answer, rows, packet, tp.validate_map(packet, rows), {})
        == original
    )


def test_over_safety_limit_and_four_points_rejected(mapped):
    rows, packet = mapped
    for long in (True, False):
        answer = fake_judgment(packet, "attention_only")
        if long:
            answer["discriminators"][0]["text"] = "x" * 501
        else:
            answer["discriminators"] *= 4
        with pytest.raises(tp.ReviewError):
            tp.validate_judgment(
                answer, rows, packet, tp.validate_map(packet, rows), {}
            )


@pytest.mark.parametrize("role", ["bystander", "unknown"])
@pytest.mark.parametrize("decision", ["select", "attention_only", "abstain"])
def test_nonmain_only_attention_context(mapped, role, decision):
    rows, packet = mapped
    packet["regions"][-1]["role"] = role
    answer = fake_judgment(packet, "attention_only")
    answer["decision"] = decision
    answer["discriminators"][0] = {
        "text": "Background overlap creates occlusion and composition interference.",
        "evidence": [{"region": "R9", "confidence": "low"}],
    }
    if decision == "attention_only":
        assert (
            tp.validate_judgment(
                answer, rows, packet, tp.validate_map(packet, rows), {}
            )
            == answer
        )
    else:
        with pytest.raises(tp.ReviewError):
            tp.validate_judgment(
                answer, rows, packet, tp.validate_map(packet, rows), {}
            )


def test_nonmain_cannot_support_preference_or_unbound_frame(mapped):
    rows, packet = mapped
    answer = fake_judgment(packet, "attention_only")
    answer["discriminators"][0] = {
        "text": "This person has the best expression.",
        "evidence": [{"region": "R9", "confidence": "medium"}],
    }
    validation = tp.validate_map(packet, rows)
    with pytest.raises(tp.ReviewError, match="visual context"):
        tp.validate_judgment(answer, rows, packet, validation, {})
    answer["discriminators"][0]["text"] = "Background occlusion."
    validation["lookup"]["R9"]["frame"] = {"file_id": 999, "content_sha256": "9" * 64}
    with pytest.raises(tp.ReviewError, match="grounding mismatch"):
        tp.validate_judgment(answer, rows, packet, validation, {})


@pytest.mark.parametrize(
    "text",
    ["phone number: 1234567890", "named Alice", "email a@b.com", "phone: hidden"],
)
def test_privacy_contact_tripwires_remain(text):
    with pytest.raises(tp.ReviewError):
        tp.privacy_check({"text": text})


def test_phone_object_is_not_contact_information():
    tp.privacy_check({"text": "Supporting subject holds a phone and looks down."})


@pytest.fixture
def batch_env(tmp_path, monkeypatch, mapped):
    rows, packet = mapped
    frozen = {
        "version": b.VERSION,
        "snapshot_sha256": "a" * 64,
        "totals": b.TOTALS,
        "groups": [
            {
                "group": f"2026-04-04-G{i:03d}",
                "date": "2026-04-04",
                "group_id": i,
                "frames": rows,
            }
            for i in range(1, 195)
        ],
    }
    fp = tmp_path / "frozen.json"
    b.save(fp, frozen)
    mp = tmp_path / "map.json"
    b.save(mp, packet)
    hp = tmp_path / "handoff.json"
    b.save(
        hp,
        {
            "version": "local8b-scene-router-v1",
            "snapshot_sha256": "a" * 64,
            "groups": [
                {
                    "group": "2026-04-04-G001",
                    "status": "routed",
                    "scene": "stage",
                    "origin": "fixture",
                    "model": "fixture:local8b",
                    "attention_hints": {
                        "path": "map.json",
                        "sha256": tp.sha(mp.read_bytes()),
                    },
                },
                {
                    "group": "2026-04-04-G002",
                    "status": "failed",
                    "reason": "local router failed",
                },
            ],
        },
    )
    monkeypatch.setattr(b, "verify_snapshot", lambda _: None)

    def prepare(frozen, group, entry, base, output, confirmed=None):
        warnings = []
        try:
            loaded = b.load_ref(base, entry["attention_hints"])
        except tp.ReviewError as exc:
            loaded = packet
            warnings.append("untrusted_8b_hints_ignored: " + str(exc))
        return (
            {
                "response_schema": b.read(
                    ROOT / "schemas/two-pass-deep-judge-v1.schema.json"
                ),
                "context": {"visual_warnings": warnings},
            },
            rows,
            loaded,
            tp.validate_map(loaded, rows),
            {},
        )

    monkeypatch.setattr(b, "prepare", prepare)
    output = ROOT / "artifacts/cloud-judge-batch/tests" / uuid4().hex
    answer = fake_judgment(packet, "attention_only")
    judge = b.FixtureJudge({"2026-04-04-G001": answer})
    return fp, hp, output, judge


def test_batch_one_call_resume_routes_schema(batch_env):
    fp, hp, output, judge = batch_env
    result = b.run(fp, hp, output, fixture=judge, concurrency=4)
    assert judge.calls == ["2026-04-04-G001"]
    assert result["routes"] == {"attention_only": 194}
    assert result["external_calls_this_run"] == result["decision_table_writes"] == 0
    bindings = b.read(output / "2026-04-04-G001/evidence-bindings.json")
    assert bindings["discriminators"][0]["evidence"][0]["frame"]["file_id"] == 1
    jsonschema.validate(
        result, b.read(ROOT / "schemas/cloud-judge-batch-v1.schema.json")
    )
    resumed = b.run(fp, hp, output, fixture=judge)
    assert judge.calls == ["2026-04-04-G001"]
    assert (
        resumed["groups"][0]["cache_hit"]
        and resumed["groups"][0]["calls_this_run"] == 0
    )


def test_failure_and_uncertain_reservation_never_retry(batch_env):
    fp, hp, output, judge = batch_env
    judge.responses["2026-04-04-G001"] = TimeoutError("uncertain")
    result = b.run(fp, hp, output, fixture=judge)
    assert result["groups"][0]["state"] == "failed_no_retry"
    path = output / "2026-04-04-G001/state.json"
    state = b.read(path)
    state["state"] = "reserved"
    b.save(path, state)
    resumed = b.run(fp, hp, output, fixture=judge)
    assert resumed["groups"][0]["state"] == "reserved"
    assert len(judge.calls) == 1


def test_invalid_response_preserved_without_repair(batch_env):
    fp, hp, output, judge = batch_env
    judge.responses["2026-04-04-G001"]["discriminators"][0]["text"] = "x" * 501
    result = b.run(fp, hp, output, fixture=judge)
    assert result["groups"][0]["state"] == "failed_no_retry"
    assert (output / "2026-04-04-G001/raw.txt").exists()
    assert not (output / "2026-04-04-G001/validated.json").exists()
    b.run(fp, hp, output, fixture=judge)
    assert len(judge.calls) == 1


@pytest.mark.parametrize("what", ["raw.txt", "validated.json", "map"])
def test_cache_tamper_never_triggers_recall(batch_env, what):
    fp, hp, output, judge = batch_env
    b.run(fp, hp, output, fixture=judge)
    p = hp.parent / "map.json" if what == "map" else output / "2026-04-04-G001" / what
    p.write_text("{}")
    with pytest.raises(tp.ReviewError):
        b.run(fp, hp, output, fixture=judge)
    assert len(judge.calls) == 1


@pytest.mark.parametrize("concurrency", [0, 5, True])
def test_bad_concurrency(batch_env, concurrency):
    fp, hp, output, judge = batch_env
    with pytest.raises(tp.ReviewError, match="concurrency"):
        b.run(fp, hp, output, fixture=judge, concurrency=concurrency)


def test_missing_manifest_and_fixture_cloud_mix_fail_before_transport(batch_env):
    fp, hp, output, judge = batch_env
    with pytest.raises(FileNotFoundError):
        b.run(fp, hp.with_name("absent"), output)
    with pytest.raises(tp.ReviewError, match="fixture cannot"):
        b.run(fp, hp, output, fixture=judge, execute=True)
    with pytest.raises(tp.ReviewError, match="provenance mismatch"):
        b.run(fp, hp, output, execute=True)
    assert not judge.calls


def test_bad_optional_hint_hash_does_not_gate_complete_frame_judge(batch_env):
    fp, hp, output, judge = batch_env
    (hp.parent / "map.json").write_text("{}")
    result = b.run(fp, hp, output, fixture=judge)
    assert result["groups"][0]["state"] == "validated"
    assert "SHA mismatch" in result["groups"][0]["visual_warnings"][0]
    assert len(judge.calls) == 1


def test_saved_real_revalidation(tmp_path):
    pilot = ROOT / "artifacts/two-pass-real-pilot"
    if not pilot.exists():
        pytest.skip("private saved real receipts absent")
    output = ROOT / "artifacts/cloud-judge-batch/tests" / uuid4().hex
    report = revalidate_saved_judge.run(pilot, output)
    assert len(report["groups"]) == 2 and report["external_calls_this_run"] == 0
    assert all(
        g["decision"] == "attention_only" and g["raw_unchanged"]
        for g in report["groups"]
    )
    assert report["groups"][1]["evidence_frame_bindings"]["R3"]["file_id"] == 35


def test_no_database_or_deletion_mutator_in_batch():
    source = Path(b.__file__).read_text()
    assert "readonly_inventory" in source
    import inspect

    from src.readonly_features import readonly_inventory
    assert "mode=ro&immutable=1" in inspect.getsource(readonly_inventory)
    assert "INSERT INTO" not in source and "DELETE FROM" not in source
    assert ".unlink(" not in source
    assert "total_max_attempts" in source


def test_sonnet_transport_sdk_stub_one_attempt_and_no_fallback(tmp_path, monkeypatch):
    import sys
    import types

    from PIL import Image

    image = tmp_path / "page.jpg"
    Image.new("RGB", (32, 32)).save(image)
    asset = tp._asset(image, [], "full_group_page")
    captured = {}

    class Client:
        def converse(self, **kwargs):
            captured["wire"] = kwargs
            raise TimeoutError("uncertain I/O")

    def client(service, **kwargs):
        captured["service"] = service
        captured["options"] = kwargs
        return Client()

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(Session=lambda: types.SimpleNamespace(client=client)),
    )
    request = {"images": [asset], "model": b.MODEL, "context": {}}
    with pytest.raises(TimeoutError):
        b.SonnetJudge().invoke(request, tmp_path, "fixture")
    assert captured["service"] == "bedrock-runtime"
    assert captured["options"]["config"].retries["total_max_attempts"] == 1
    assert captured["wire"]["modelId"] == b.MODEL
    assert (tmp_path / "wire-request.json").exists()


def test_duplicate_handoff_is_rejected(batch_env):
    fp, hp, output, judge = batch_env
    handoff = b.read(hp)
    handoff["groups"].append(copy.deepcopy(handoff["groups"][0]))
    b.save(hp, handoff)
    with pytest.raises(tp.ReviewError, match="duplicate"):
        b.run(fp, hp, output, fixture=judge)
    assert not judge.calls


def test_unknown_usage_does_not_become_free_cloud_cost(batch_env, monkeypatch):
    fp, hp, output, judge = batch_env

    def invoke(*_):
        return {"usage": {"inputTokens": "unknown"}, "stopReason": "end_turn"}

    monkeypatch.setattr(judge, "invoke", invoke)
    result = b.run(fp, hp, output, fixture=judge)
    assert result["groups"][0]["state"] == "failed_no_retry"
    assert result["groups"][0]["estimated_cost_usd"] is None
    assert result["usage"]["inputTokens"] == 0


def test_saved_real_raw_tamper_rejected_before_validation(tmp_path):
    import shutil

    original = ROOT / "artifacts/two-pass-real-pilot"
    if not original.exists():
        pytest.skip("private saved real receipts absent")
    group = "2026-07-25-G004"
    pilot = tmp_path / "pilot"
    (pilot / group).mkdir(parents=True)
    shutil.copy2(original / "call-ledger.json", pilot / "call-ledger.json")
    for suffix in (
        "wire-request.json",
        "request.json",
        "provider-response.json",
        "raw.txt",
    ):
        name = f"pass-A-{suffix}"
        shutil.copy2(original / group / name, pilot / group / name)
    (pilot / group / "pass-A-raw.txt").write_text("{}")
    with pytest.raises(tp.ReviewError, match="hash mismatch"):
        revalidate_saved_judge.run(
            pilot, ROOT / "artifacts/cloud-judge-batch/tests" / uuid4().hex
        )


@pytest.mark.parametrize(
    "scene", ["action", "interaction", "stage", "group", "ordinary", "unknown"]
)
def test_scene_packet_always_preserves_every_complete_frame(tmp_path, mapped, scene):
    from PIL import Image

    rows, _ = mapped
    images = {
        tp.identity(r): Image.new("RGB", (1600, 1200), (10 * r["file_id"], 30, 40))
        for r in rows
    }
    assets, packet, validation = b.visual_packet(rows, images, scene, tmp_path)
    full = [a for a in assets if a["kind"] == "full_group_page"]
    assert len(full) == len(rows)
    assert {a["cells"][0]["frame"] for a in full} == set(images)
    assert all(min(a["cells"][0]["actual_image_size"]) >= 1024 for a in full)
    assert all(view["role"] == "frame_view" for view in packet["views"])
    if scene in {"action", "interaction"}:
        story = next(a for a in assets if a["kind"] == "complete_temporal_storyboard")
        assert [c["frame"] for c in story["cells"]] == [tp.identity(r) for r in rows]
        assert not any(a["kind"] == "operator_confirmed_strips" for a in assets)
    elif scene in {"stage", "group"}:
        grid = next(a for a in assets if a["kind"] == "fixed_multiscale_grid")
        assert len(grid["cells"]) == len(rows) * (6 if scene == "stage" else 4)
        assert {c["frame"] for c in grid["cells"]} == set(images)
    else:
        assert len(assets) == len(rows)
    answer = {
        "decision": "select",
        "confidence": "medium",
        "recommended": [
            {"frame": rows[0], "evidence": [{"region": "R1", "confidence": "medium"}]}
        ],
        "multi_select_reason": None,
        "irreplaceable": [],
        "discriminators": [
            {
                "text": "The complete frame has a distinct lighting phase.",
                "evidence": [{"region": "R1", "confidence": "medium"}],
            }
        ],
    }
    jsonschema.validate(
        answer, b.read(ROOT / "schemas/cloud-scene-judge-v1.schema.json")
    )
    assert (
        tp.validate_judgment(
            answer, rows, packet, validation, {}, allow_visual_views=True
        )
        == answer
    )
    with pytest.raises(tp.ReviewError):
        tp.validate_judgment(answer, rows, packet, validation, {})


def test_confirmed_action_strip_is_supplemental_never_replaces_pages(tmp_path, mapped):
    from PIL import Image

    rows, _ = mapped
    images = {tp.identity(r): Image.new("RGB", (1600, 1200)) for r in rows}
    confirmed = {
        "confirmed_by": "parent_visual_review",
        "tracks": [
            {"regions": [{"frame": r, "box": [0.2, 0.2, 0.8, 0.8]} for r in rows]}
        ],
    }
    assets, _, _ = b.visual_packet(rows, images, "action", tmp_path, confirmed)
    assert len([a for a in assets if a["kind"] == "full_group_page"]) == 2
    assert len([a for a in assets if a["kind"] == "operator_confirmed_strips"]) == 1
    confirmed["confirmed_by"] = "local8b"
    with pytest.raises(tp.ReviewError, match="operator confirmation"):
        b.visual_packet(rows, images, "action", tmp_path, confirmed)


def test_scene_judge_disallows_subject_irreplaceability_without_subject_grounding(
    mapped,
):
    _rows, packet = mapped
    answer = fake_judgment(packet)
    assert answer["irreplaceable"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            answer, b.read(ROOT / "schemas/cloud-scene-judge-v1.schema.json")
        )


def test_real_prepare_ignores_bad_8b_boxes_without_cropping_complete_frames(tmp_path):
    frozen_path = ROOT / "artifacts/cloud-judge-batch/frozen.json"
    if not frozen_path.exists():
        pytest.skip("private frozen data absent")
    frozen = b.read(frozen_path)
    group = next(g for g in frozen["groups"] if g["group"] == "2026-07-25-G004")
    hints = tmp_path / "hints.json"
    b.save(
        hints,
        [{"frame": group["frames"][0], "box": [2, 2, 3, 3], "confidence": "high"}],
    )
    entry = {
        "status": "routed",
        "scene": "stage",
        "attention_hints": {"path": hints.name, "sha256": tp.sha(hints.read_bytes())},
    }
    output = ROOT / "artifacts/cloud-judge-batch/tests" / uuid4().hex
    output.mkdir(parents=True)
    request, rows, packet, _, _ = b.prepare(frozen, group, entry, tmp_path, output)
    assert request["context"]["untrusted_local_attention_hints"] == []
    assert request["context"]["visual_warnings"][0].startswith(
        "untrusted_8b_hints_ignored:"
    )
    full = [a for a in request["images"] if a["kind"] == "full_group_page"]
    assert len(full) == len(rows)
    assert all(v["role"] == "frame_view" for v in packet["views"])
    assert request["context"]["hint_authority"] == "none_for_selection_or_coverage"


def test_missing_local_scene_still_reaches_complete_frame_preparation(
    batch_env, monkeypatch
):
    fp, hp, output, judge = batch_env
    data = b.read(hp)
    data["groups"] = []
    b.save(hp, data)
    original = b.prepare

    # Use a deterministic stub to prove missing routing is passed as unknown, not skipped.
    def prepare(frozen, group, entry, base, case, confirmed=None):
        assert entry is None
        raise tp.ReviewError("complete_frame_preparation_reached")

    monkeypatch.setattr(b, "prepare", prepare)
    result = b.run(fp, hp, output, fixture=judge)
    assert "complete_frame_preparation_reached" in result["groups"][0]["reason"]
    assert result["groups"][0]["scene_route"] == "unknown"
    assert original is not prepare


def test_selected_phase_never_dispatches_outside_allowlist(batch_env):
    fp, hp, output, judge = batch_env
    skipped = b.run(fp, hp, output, fixture=judge, selected_groups=["2026-04-04-G002"])
    assert not judge.calls and skipped["external_calls_total"] == 0
    assert skipped["groups"][0]["reason"] == "not_selected_this_phase"
    result = b.run(fp, hp, output, fixture=judge, selected_groups=["2026-04-04-G001"])
    assert len(judge.calls) == 1 and result["groups"][0]["state"] == "validated"
    b.run(fp, hp, output, fixture=judge, selected_groups=["2026-04-04-G002"])
    assert len(judge.calls) == 1


def test_authorized_canary_gate_thresholds_and_cost_stop():
    from scripts.run_authorized_cloud_judge import gate

    states = [
        {
            "group": str(i),
            "state": "validated",
            "reserved": True,
            "response_sha256": "a" * 64,
            "external_calls": 1,
            "estimated_cost_usd": 0.1,
            "decision_table_writes": 0,
        }
        for i in range(12)
    ]
    assert gate(states, finished=True)["passed"]
    states[0].update(
        state="failed_no_retry", reason="ValidationError: isolated response"
    )
    states[1].update(state="failed_no_retry", reason="ReviewError: isolated semantics")
    assert gate(states, finished=True)["passed"]
    states[2].update(estimated_cost_usd=0.201)
    assert gate(states, finished=True)["early_stop"]
    states[2].update(estimated_cost_usd=None)
    assert gate(states, finished=True)["early_stop"]
    states[2].update(estimated_cost_usd=0.1)
    states[1]["reason"] = "ValidationError: repeated schema"
    assert "repeated_failure_family" in gate(states, finished=False)["reasons"]
    states[1].update(state="validated", decision_table_writes=1)
    assert "decision_table_write_detected" in gate(states, finished=True)["reasons"]


def test_authorized_canary_has_two_per_date_and_covers_largest_groups():
    from scripts.run_authorized_cloud_judge import choose_canary

    frozen = {
        "groups": [
            {"group": f"{date}-G{i:03d}", "date": date, "frames": [None] * i}
            for date in b.DATES
            for i in (2, 3, 11)
        ]
    }
    selected = choose_canary(frozen, {"groups": []})
    assert len(selected) == len(set(selected)) == 12
    assert all(sum(g.startswith(date) for g in selected) == 2 for date in b.DATES)
    assert all(f"{date}-G011" in selected for date in b.DATES)


def test_batch_l2_projection_preserves_all_frames_and_discloses_omissions():
    full = {
        "layer": "L2",
        "schema_version": 1,
        "frames": [f"F{i:02d}" for i in range(11)],
        "evidence": {
            f"D{i:03d}": {
                "criterion": f"quality_meta.sharpness.metric{i}",
                "values": {f"F{j:02d}": j + i for j in range(11)},
            }
            for i in range(120)
        },
        "pairs": {},
        "evidence_refs": [str(i) for i in range(1320)],
    }
    original = copy.deepcopy(full)
    projected = b.project_batch_l2(full)
    assert full == original
    assert projected["frames"] == full["frames"]
    assert len(projected["evidence"]) == 24
    assert projected["projection"]["omitted_scalar_count"] == 96
    assert len(projected["evidence_refs"]) == 264
    assert all(
        projected["evidence"][key] == full["evidence"][key]
        for key in projected["evidence"]
    )
    tp.vlm_review.assert_model_context_safe(projected)


@pytest.mark.parametrize("weak_only", [False, True])
def test_v3_select_discriminators_allow_weak_observations(mapped, weak_only):
    rows, packet = mapped
    answer = fake_judgment(packet)
    answer["discriminators"] = copy.deepcopy(answer["discriminators"])
    for item in answer["discriminators"]:
        item["evidence"][0]["confidence"] = "low"
        if weak_only:
            for e in item["evidence"]:
                e["confidence"] = "low"
    assert (
        tp.validate_judgment(answer, rows, packet, tp.validate_map(packet, rows), {})
        == answer
    )


@pytest.mark.parametrize(
    "target", ["recommended", "multi_select_reason", "irreplaceable"]
)
def test_v3_weak_evidence_cannot_be_necessary_decision_support(mapped, target):
    rows, packet = mapped
    answer = fake_judgment(packet)
    answer[target] = copy.deepcopy(answer[target])
    item = answer[target] if target == "multi_select_reason" else answer[target][0]
    for e in item["evidence"]:
        e["confidence"] = "low"
    with pytest.raises(tp.ReviewError, match="low evidence"):
        tp.validate_judgment(answer, rows, packet, tp.validate_map(packet, rows), {})


def test_v3_recommended_can_include_weak_note_with_strong_binding(mapped):
    rows, packet = mapped
    answer = fake_judgment(packet)
    answer["recommended"][0]["evidence"].append(
        {
            "region": answer["recommended"][0]["evidence"][0]["region"],
            "confidence": "low",
        }
    )
    assert (
        tp.validate_judgment(answer, rows, packet, tp.validate_map(packet, rows), {})
        == answer
    )


@pytest.mark.parametrize("gid", ["2026-04-04-G014", "2026-04-05-G007"])
def test_v3_saved_mixed_confidence_receipts_source_region_hash_validation(
    tmp_path, gid
):
    from scripts.continue_cloud_judge_v3 import validate_saved

    source = ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v2"
    if not source.exists():
        pytest.skip("private saved receipts absent")
    frozen = b.read(ROOT / "artifacts/cloud-judge-batch/frozen.json")
    router = b.read(
        ROOT / "artifacts/cloud-judge-batch/authorized-input/scene-router.json"
    )
    group = next(g for g in frozen["groups"] if g["group"] == gid)
    _, answer, _, _ = validate_saved(source, group, frozen, router, tmp_path / gid)
    assert answer == b.read(source / gid / "raw.json")
    assert answer["decision"] == "select"
    validation = b.read(tmp_path / gid / "view-validation.json")
    for evidence in answer["recommended"][0]["evidence"]:
        evidence["confidence"] = "low"
    with pytest.raises(tp.ReviewError, match="low evidence"):
        tp.validate_judgment(
            answer, group["frames"], {}, validation, {}, allow_visual_views=True
        )


def test_v3_saved_raw_tampering_rejected(tmp_path):
    import shutil

    from scripts.continue_cloud_judge_v3 import validate_saved

    source = ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v2"
    if not source.exists():
        pytest.skip("private saved receipts absent")
    gid = "2026-04-04-G014"
    shutil.copytree(source / gid, tmp_path / "source" / gid)
    shutil.copy2(source / "contract.json", tmp_path / "source/contract.json")
    (tmp_path / "source" / gid / "raw.txt").write_text("{}")
    frozen = b.read(ROOT / "artifacts/cloud-judge-batch/frozen.json")
    router = b.read(
        ROOT / "artifacts/cloud-judge-batch/authorized-input/scene-router.json"
    )
    group = next(g for g in frozen["groups"] if g["group"] == gid)
    with pytest.raises(tp.ReviewError, match="hash mismatch"):
        validate_saved(tmp_path / "source", group, frozen, router, tmp_path / "output")


def test_continuation_quality_cost_and_single_io_failures_do_not_stop(tmp_path):
    from scripts.run_authorized_cloud_judge import continuation_gate

    states = [
        {
            "group": str(i),
            "state": "failed_no_retry",
            "external_calls": 1,
            "decision_table_writes": 0,
            "estimated_cost_usd": None,
            "reason": "ValidationError: rejected model output",
        }
        for i in range(9)
    ]
    assert not continuation_gate(states, tmp_path)["hard_stop"]
    states.append({**states[-1], "group": "9", "estimated_cost_usd": 0.3})
    assert continuation_gate(states, tmp_path)["hard_stop"]
    states[-1]["response_sha256"] = "a" * 64
    b.save(
        tmp_path / "9/provider-response.json",
        {
            "output": {
                "message": {
                    "content": [{"text": "Malformed but reviewable model answer"}]
                }
            }
        },
    )
    check = continuation_gate(states, tmp_path)
    assert not check["hard_stop"]
    assert check["consecutive_without_effective_response"] == 0


@pytest.mark.parametrize(
    "reason", ["source SHA mismatch", "asset path escape", "frozen membership mismatch"]
)
def test_continuation_source_safety_stops_immediately(tmp_path, reason):
    from scripts.run_authorized_cloud_judge import continuation_gate

    state = {
        "group": "source",
        "state": "attention",
        "external_calls": 0,
        "decision_table_writes": 0,
        "reason": reason,
    }
    assert continuation_gate([state], tmp_path)["hard_stop"]


def test_continuation_writes_and_duplicate_dispatch_are_hard_stops(tmp_path):
    from scripts.run_authorized_cloud_judge import continuation_gate

    state = {
        "group": "source",
        "state": "validated",
        "external_calls": 1,
        "decision_table_writes": 1,
        "reason": None,
    }
    assert continuation_gate([state], tmp_path)["hard_stop"]
    state.update(decision_table_writes=0, external_calls=2)
    assert continuation_gate([state], tmp_path)["hard_stop"]
