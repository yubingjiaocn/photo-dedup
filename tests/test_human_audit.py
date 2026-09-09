"""Synthetic human records only; never label real photos in tests."""
import copy
from pathlib import Path

import pytest
from PIL import Image

from src.human_audit import (
    canonical, development_scope, digest, freeze_design, load_json, make_plan,
    read_labels, report, seal, validate_labels,
)
from src.human_audit_bundle import export_bundle, main, verify_media
from src.risk_coverage import stratified_audit_sample


def inputs():
    scopes = {"demo": {"schema_version": 1, "dataset_id": "demo", "owner_scope": "synthetic",
                        "split": "tune", "events": ["synthetic-event"]}}
    rows = [{"dataset": "demo", "group_id": i, "member_ids": [i*10, i*10+1],
             "member_count": 2, "candidate_keeper_ids": [i*10], "risk_score": 0.1,
             "risk_decile": 1 if i <= 4 else 2, "changed": False,
             "group_type": "burst", "risk_reason": "SECRET_MODEL_REASON", "reason_codes": [],
             "selective_output": "SAFE_SILENT" if i < 7 else "REVIEW_PRIMARY"}
            for i in range(1, 8)]
    audit = stratified_audit_sample(rows, 4, "synthetic-seed")
    provenance = {"opened_original_media": False, "lin_held_out_touched": False,
                  "writeback": False, "authority": "shadow_review_only",
                  "auto_remove_authority": "BYTE_IDENTICAL_ONLY",
                  "source_sha256_before_after": {"demo": "a"*64}}
    fixture = {"schema_version": 1, "audit_id": "demo:G7", "member_ids": [70, 71],
               "source_sha256": "a"*64, "status": "pending", "purpose": "narrative_recheck"}
    return rows, audit, scopes, provenance, fixture


def bundle():
    rows, audit, scopes, provenance, fixture = inputs()
    plan = make_plan(rows, audit, scopes, provenance, "blind-seed", fixture)
    for task in plan["tasks"]:
        task["media_complete"] = True
    return seal(plan)


def label(b, task, *, error=False, status="reviewed"):
    return {"schema_version": 1, "bundle_id": b["bundle_id"], "task_id": task["task_id"],
            "status": status, "annotator": "synthetic-reviewer", "human_attested": True,
            "phases": [{"members": ["M1", "M2"], "acceptable_keepers": ["M2" if error else "M1"]}]
            if status == "reviewed" else [],
            "group_impure": False if status == "reviewed" else None, "note": "SYNTHETIC TEST ONLY"}


def files(tmp_path):
    rows, audit, scopes, provenance, fixture = inputs()
    paths = {}
    for name, value in [("groups", {"groups": rows}), ("audit", audit), ("scopes", scopes),
                        ("provenance", provenance), ("fixture", fixture)]:
        path = tmp_path / f"{name}.json"
        path.write_text(canonical(value))
        paths[f"{name}_path"] = path
    cache = tmp_path / "cache"
    cache.mkdir()
    for row in rows:
        for member in row["member_ids"]:
            Image.new("RGB", (32, 24), (member, 30, 10)).save(cache / f"{member}.jpg")
    return {**paths, "cache_roots": {"demo": cache}, "output": tmp_path / "bundle", "seed": "blind-seed"}


def test_deterministic_plan_and_sampling_replay():
    assert bundle() == bundle()
    rows, audit, scopes, provenance, fixture = inputs()
    plan = make_plan(rows, audit, scopes, provenance, "other-seed", fixture)
    assert [t["task_id"] for t in plan["tasks"]] != [t["task_id"] for t in bundle()["tasks"]]
    assert plan["silent_population"] == 6
    assert sum(t["kind"] == "probability" for t in plan["tasks"]) == 4
    assert sum(t["kind"] == "purposive" for t in plan["tasks"]) == 1


@pytest.mark.parametrize("field", ["inclusion_probability", "analysis_weight", "member_ids",
                                   "candidate_keeper_ids", "stratum_population", "label_status"])
def test_sampling_tamper_rejected(field):
    rows, audit, scopes, _, _ = inputs()
    audit["groups"][0][field] = None
    with pytest.raises(ValueError, match="sample row mismatch"):
        freeze_design(rows, audit, scopes)


def test_stale_fixture_fails():
    rows, audit, scopes, provenance, fixture = inputs()
    fixture["member_ids"] = [70, 72]
    with pytest.raises(ValueError, match="stale narrative"):
        make_plan(rows, audit, scopes, provenance, "seed", fixture)


@pytest.mark.parametrize("split,owner", [("held_out", "willy"), ("tune", "lin"), ("train", "unknown")])
def test_held_out_and_other_owners_rejected_before_data_read(tmp_path, monkeypatch, split, owner):
    args = files(tmp_path)
    scopes = load_json(args["scopes_path"])
    scopes["demo"].update(split=split, owner_scope=owner)
    args["scopes_path"].write_text(canonical(scopes))
    args["groups_path"] = tmp_path / "forbidden-input-must-not-be-opened"
    monkeypatch.setattr("src.human_audit_bundle.cached_thumbnail", lambda *a: pytest.fail("media read"))
    with pytest.raises(ValueError, match="development datasets"):
        export_bundle(**args)
    assert not args["output"].exists()


def test_export_blinding_metadata_only_and_immutable(tmp_path):
    args = files(tmp_path)
    b = export_bundle(**args)
    text = (args["output"] / "reviewer/index.html").read_text()
    for forbidden in ["candidate_keeper", "risk_score", "SECRET_MODEL_REASON", "demo:G",
                      "stratum", "narrative_recheck", str(tmp_path), "file://", "http://", "https://"]:
        assert forbidden not in text
    assert b["tasks"][0]["media_complete"]
    verify_media(b, args["output"] / "reviewer")
    args2 = {**args, "output": tmp_path / "second-bundle"}
    assert export_bundle(**args2) == b
    assert (args2["output"] / "reviewer/index.html").read_bytes() == (args["output"] / "reviewer/index.html").read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        export_bundle(**args)


def test_missing_thumbnail_blocks_review_but_allows_unassessable(tmp_path):
    args = files(tmp_path)
    (args["cache_roots"]["demo"] / "70.jpg").unlink()
    b = export_bundle(**args)
    task = next(t for t in b["tasks"] if t["kind"] == "purposive")
    assert not task["media_complete"]
    with pytest.raises(ValueError, match="missing thumbnails"):
        validate_labels(b, [label(b, task)])
    assert report(b, [label(b, task, status="unassessable")])["weighted_error_rate"] is None


def test_cache_and_output_symlinks_fail_closed(tmp_path):
    args = files(tmp_path)
    root = args["cache_roots"]["demo"]
    (root / "70.jpg").unlink()
    (root / "70.jpg").symlink_to(root / "71.jpg")
    with pytest.raises(ValueError, match="symlink"):
        export_bundle(**args)
    assert not args["output"].exists()
    (root / "70.jpg").unlink()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_bundle(**{**args, "cache_roots": {"demo": alias}})


def test_report_reviewer_mapping_tamper(tmp_path):
    args = files(tmp_path)
    b = export_bundle(**args)
    page = args["output"] / "reviewer/index.html"
    text = page.read_text()
    page.write_text(text.replace('-M1.jpg', '-M2.jpg', 1))
    with pytest.raises(ValueError, match="reviewer page fingerprint"):
        verify_media(b, args["output"] / "reviewer")


def test_report_media_tamper(tmp_path):
    args = files(tmp_path)
    b = export_bundle(**args)
    p = next((args["output"] / "reviewer/thumbs").iterdir())
    p.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_media(b, args["output"] / "reviewer")


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(bundle_id="stale"),
    lambda x: x.update(schema_version=True),
    lambda x: x.update(task_id="unknown"),
    lambda x: x.update(human_attested=False),
    lambda x: x.update(group_impure=0),
    lambda x: x.update(extra="model_score"),
    lambda x: x.update(phases=[]),
    lambda x: x["phases"][0].update(members=["M1"]),
    lambda x: x["phases"][0].update(acceptable_keepers=["M3"]),
    lambda x: x["phases"].append(copy.deepcopy(x["phases"][0])),
    lambda x: x.update(status="pending"),
])
def test_invalid_labels_fail_closed(mutate):
    b = bundle()
    value = label(b, b["tasks"][0])
    mutate(value)
    with pytest.raises(ValueError):
        report(b, [value])


def test_duplicate_and_tampered_bundle_rejected():
    b = bundle()
    value = label(b, b["tasks"][0])
    with pytest.raises(ValueError, match="duplicate"):
        report(b, [value, value])
    b["tasks"][0]["candidate_keeper_ids"] = [999]
    with pytest.raises(ValueError, match="integrity"):
        report(b, [])


def test_incomplete_uncertain_and_zero_selection_never_estimate():
    b = bundle()
    assert report(b, [])["upper_95"] is None
    values = [label(b, t) for t in b["tasks"]]
    probability = next(t for t in b["tasks"] if t["kind"] == "probability")
    values = [x for x in values if x["task_id"] != probability["task_id"]]
    assert report(b, values)["weighted_error_rate"] is None
    values.append(label(b, probability, status="uncertain"))
    assert report(b, values)["upper_95"] is None
    rows, audit, scopes, provenance, fixture = inputs()
    for row in rows:
        if row["selective_output"] == "DIAGNOSTIC_SAMPLE":
            row["selective_output"] = "SAFE_SILENT"
    small = stratified_audit_sample(rows, 1, "tiny")
    p = make_plan(rows, small, scopes, provenance, "seed", fixture)
    for t in p["tasks"]:
        t["media_complete"] = True
    b = seal(p)
    assert report(b, [label(b, t) for t in b["tasks"]])["upper_95"] is None


def test_weighted_risk_purposive_exclusion_and_census_exact():
    b = bundle()
    values = [label(b, t, error=t["stratum"] == "d2|unchanged" or t["kind"] == "purposive") for t in b["tasks"]]
    r = report(b, values)
    assert r["weighted_error_rate"] == pytest.approx(2/6)
    assert r["upper_95"] >= r["weighted_error_rate"]
    assert r["purposive_count"] == 1
    assert r["deletion_authority"] == "none"
    assert not r["safety_validated"]
    rows, _, scopes, provenance, _ = inputs()
    for row in rows:
        if row["selective_output"] == "DIAGNOSTIC_SAMPLE":
            row["selective_output"] = "SAFE_SILENT"
    audit = stratified_audit_sample(rows, 6, "census")
    plan = make_plan(rows, audit, scopes, provenance, "seed")
    for task in plan["tasks"]:
        task["media_complete"] = True
    b = seal(plan)
    result = report(b, [label(b, t, error=t["stratum"] == "d2|unchanged") for t in b["tasks"]])
    assert result["upper_95"] == result["weighted_error_rate"] == pytest.approx(2/6)


def test_human_phase_miss_derived_after_blind_labels():
    b = bundle()
    task = b["tasks"][0]
    value = label(b, task)
    value["phases"] = [{"members": ["M1"], "acceptable_keepers": ["M1"]},
                       {"members": ["M2"], "acceptable_keepers": ["M2"]}]
    detail = next(d for d in report(b, [value])["tasks"] if d["task_id"] == task["task_id"])
    assert detail["phase_miss"] and detail["error"]
    assert not detail["keeper_bad"]


def test_cli_end_to_end_replay_and_strict_json(tmp_path):
    args = files(tmp_path)
    argv = ["export"]
    for flag in ("groups", "audit", "scopes", "provenance", "fixture"):
        argv += [f"--{flag}", str(args[f"{flag}_path"])]
    argv += ["--cache", "demo", str(args["cache_roots"]["demo"]), "--output", str(args["output"]), "--seed", "blind-seed"]
    main(argv)
    b = load_json(args["output"] / "manifest.json")
    labels_path = tmp_path / "synthetic-labels.jsonl"
    values = [label(b, t) for t in b["tasks"]]
    labels_path.write_text("\n".join(canonical(x) for x in values))
    output = tmp_path / "report.json"
    main(["report", "--bundle", str(args["output"]), "--labels", str(labels_path), "--output", str(output)])
    assert load_json(output)["weighted_error_rate"] == 0
    assert read_labels(labels_path) == values
    labels_path.write_text('{"x":1,"x":2}\n')
    with pytest.raises(ValueError, match="duplicate"):
        read_labels(labels_path)
    labels_path.write_text('{"x":NaN}\n')
    with pytest.raises(ValueError, match="finite"):
        read_labels(labels_path)


def test_g10_fixture_is_pending_not_human_truth():
    fixture = load_json(Path(__file__).resolve().parents[1] / "fixtures/disney-g10-narrative-review-v1.json")
    assert fixture["status"] == "pending"
    assert fixture["member_ids"] == [430, 431, 432, 433]
    assert not {"expected_phases", "verdict", "human_reviewed"}.intersection(fixture)
    assert len(digest(fixture)) == 64


def test_scope_unknown_keys_rejected():
    scopes = inputs()[2]
    scopes["demo"]["allow_held_out"] = True
    with pytest.raises(ValueError):
        development_scope(scopes)


def test_model_packet_is_blind_disjoint_and_never_human_evidence(tmp_path):
    from scripts.prepare_model_audit import prepare
    args = files(tmp_path)
    b = export_bundle(**args)
    output = tmp_path / "model-packet"
    packet = prepare(args["output"], output, strata_count=2)
    second = prepare(args["output"], tmp_path / "model-packet-2", strata_count=2)
    assert packet == second
    assert len(packet["cases"]) == 3
    assert packet["annotation_kind"] == "model_provisional"
    assert not packet["human_annotation"] and not packet["selective_risk_eligible"]
    assert packet["status"] == "pending_local_inference"
    public = (output / "blind-input/tasks.json").read_text()
    assert all(word not in public for word in ("candidate", "stratum", "demo:G", "risk_score"))
    with pytest.raises(ValueError, match="invalid label fields"):
        report(b, [packet])
    assert report(b, [])["upper_95"] is None
    with pytest.raises(ValueError):
        prepare(args["output"], output)


def test_ui_javascript_syntax(tmp_path):
    import shutil
    import subprocess
    if not shutil.which("node"):
        pytest.skip("node unavailable")
    args = files(tmp_path)
    export_bundle(**args)
    page = (args["output"] / "reviewer/index.html").read_text()
    js = page.split('<script>')[1].split('</script>')[0]
    path = tmp_path / "ui.js"
    path.write_text(js)
    subprocess.run(["node", "--check", str(path)], check=True, capture_output=True)
