"""Synthetic human records only; never label real photos in tests."""
import copy
import json
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
    value = {"schema_version": 1, "bundle_id": b["bundle_id"], "task_id": task["task_id"],
            "status": status, "annotator": "synthetic-reviewer", "human_attested": True,
            "phases": [{"members": ["M1", "M2"], "acceptable_keepers": ["M2" if error else "M1"]}]
            if status == "reviewed" else [],
            "group_impure": False if status == "reviewed" else None, "note": "SYNTHETIC TEST ONLY"}

    if b.get("label_schema_version") == 2:
        value["schema_version"] = 2
        for phase in value["phases"]:
            phase["keeper_status"] = "assessed"
    return value


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


def legacy_bundle_files(args):
    """Synthetic v1 envelope/media fixture, independent of repository history.

    Source HTML is just a sealed blind data payload; upgrade replaces presentation.
    """
    import hashlib
    b = export_bundle(**args)
    plan = {k: v for k, v in b.items() if k not in {"bundle_id", "label_schema_version"}}
    public = {"schema_version": 1, "bundle_id": "0" * 64, "tasks": [
        {"task_id": t["task_id"], "media_complete": t["media_complete"],
         "members": [{"alias": m["alias"], "file": m["file"]} for m in t["media"]]}
        for t in b["tasks"]]}
    page = "<script>const DATA=" + canonical(public) + ";</script>"
    plan["reviewer_content_sha256"] = hashlib.sha256(page.encode()).hexdigest()
    old = seal(plan)
    page = page.replace('"bundle_id":"' + "0" * 64 + '"',
                        '"bundle_id":"' + old["bundle_id"] + '"')
    (args["output"] / "reviewer/index.html").write_text(page)
    (args["output"] / "manifest.json").write_text(canonical(old))
    verify_media(old, args["output"] / "reviewer")
    return old


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


def partial_label(b, task):
    value = label(b, task)
    value.update(schema_version=2, phases=[
        {"members": ["M1"], "keeper_status": "assessed", "acceptable_keepers": ["M1"]},
        {"members": ["M2"], "keeper_status": "quality_abstain"},
    ])
    return value


@pytest.mark.parametrize("status", ["uncertain", "unassessable"])
def test_v2_abstention_notes_roundtrip(tmp_path, status):
    from src.human_audit import labels_v2
    b = bundle()
    value = label(b, b["tasks"][0], status=status)
    value.update(schema_version=2, note="看不清表情，但备注必须保存\n第二行")
    path = tmp_path / "notes.jsonl"
    path.write_text(canonical(value) + "\n")
    assert read_labels(path) == labels_v2(b, [value]) == [value]
    detail = report(b, [value])["tasks"][0]
    assert detail["phase_coverage_error"] is None
    assert detail["keeper_quality_error"] is None
    assert detail["error"] is None


def test_partial_phase_labels_count_coverage_but_not_joint_or_group_quality():
    b = bundle()
    values = [partial_label(b, t) for t in b["tasks"]]
    r = report(b, values)
    assert r["weighted_error_rate"] is r["upper_95"] is None
    assert r["phase_only"]["weighted_phase_coverage_error_rate"] == 1
    assert r["phase_only"]["descriptive_only"]
    assert not r["phase_only"]["safety_validated"]
    counts = r["evidence_counts"]["probability"]
    assert counts["phase_coverage_eligible_count"] == counts["phase_coverage_error_count"] == 4
    assert counts["keeper_quality_eligible_count"] == counts["keeper_quality_error_count"] == 0
    assert counts["joint_eligible_count"] == 0
    assert counts["quality_abstain_group_count"] == counts["quality_assessed_phase_count"] == 4
    assert counts["quality_assessed_phase_errors"] == 0
    for d in r["tasks"]:
        assert d["phase_coverage_error"] is True
        assert d["error"] is d["keeper_bad"] is d["phase_miss"] is d["keeper_quality_error"] is None


def test_quality_abstain_all_phases_neither_pass_nor_error():
    b = bundle()
    value = partial_label(b, b["tasks"][0])
    value["phases"] = [{"members": ["M1", "M2"], "keeper_status": "quality_abstain"}]
    detail = report(b, [value])["tasks"][0]
    assert detail["phase_coverage_error"] is False
    assert detail["quality_assessed_phase_count"] == 0
    assert detail["keeper_quality_error"] is detail["error"] is None
    # Even a known false merge must not manufacture a joint label under abstention.
    value["group_impure"] = True
    assert report(b, [value])["tasks"][0]["error"] is None


def test_v1_fully_assessed_semantics_preserved_without_note_inference():
    from src.human_audit import labels_v2
    b = bundle()
    value = label(b, b["tasks"][0], error=True)
    value["note"] = "缩略图看不清，无法判断 keeper"
    old = report(b, [value])["tasks"][0]
    assert old["phase_miss"] and old["keeper_bad"] and old["error"]
    assert old["phase_coverage_error"] is False
    assert old["keeper_quality_error"] is True
    converted = labels_v2(b, [value])
    assert converted[0]["phases"][0]["keeper_status"] == "assessed"
    assert converted[0]["note"] == value["note"]
    assert report(b, converted)["tasks"] == report(b, [value])["tasks"]
    assert "keeper_status" not in value["phases"][0]  # No input mutation.


@pytest.mark.parametrize("mutation", [
    lambda x: x["phases"][1].update(acceptable_keepers=[]),
    lambda x: x["phases"][1].update(acceptable_keepers=["M2"]),
    lambda x: x["phases"][1].update(keeper_status="uncertain"),
    lambda x: x["phases"][0].update(acceptable_keepers=[]),
    lambda x: x["phases"][0].update(acceptable_keepers=["M1", "M1"]),
    lambda x: x["phases"][0].update(acceptable_keepers=["M2"]),
    lambda x: x["phases"][1].update(members=["M1"]),
    lambda x: x["phases"].pop(),
    lambda x: x["phases"][1].update(extra=True),
    lambda x: x.update(schema_version=2.0),
    lambda x: x.update(schema_version=True),
    lambda x: x.update(extra="unknown"),
])
def test_strict_v2_partition_and_quality_schema(mutation):
    b = bundle()
    value = partial_label(b, b["tasks"][0])
    mutation(value)
    with pytest.raises(ValueError):
        validate_labels(b, [value])


def test_one_probability_abstain_blocks_overall_but_purposive_does_not():
    b = bundle()
    values = [label(b, t) if t["kind"] == "probability" else partial_label(b, t) for t in b["tasks"]]
    assert report(b, values)["weighted_error_rate"] == 0
    t = next(t for t in b["tasks"] if t["kind"] == "probability")
    values = [partial_label(b, t) if x["task_id"] == t["task_id"] else x for x in values]
    r = report(b, values)
    assert r["weighted_error_rate"] is r["upper_95"] is None
    assert r["phase_only"]["weighted_phase_coverage_error_rate"] is not None
    assert r["evidence_counts"]["probability"]["keeper_quality_eligible_count"] == 3


def test_upgrade_preserves_design_media_labels_and_seals_initial_records(tmp_path):
    from src.human_audit import labels_v2
    from src.human_audit_bundle import upgrade_bundle
    args = files(tmp_path)
    old = legacy_bundle_files(args)
    values = [label(old, old["tasks"][0]), label(old, old["tasks"][1], status="uncertain")]
    values[0]["note"] = "看不清清晰度，但保持旧 reviewed 原义"
    path = tmp_path / "legacy.jsonl"
    original = "\n".join(canonical(x) for x in values) + "\n"
    path.write_text(original)
    output = tmp_path / "r3"
    new = upgrade_bundle(args["output"], path, output)
    assert old["bundle_id"] != new["bundle_id"]
    assert new["legacy_bundle_id"] == old["bundle_id"]
    for key in ("tasks", "source_identity", "source_sha256", "seed", "strata", "sampling_seed", "silent_population"):
        assert new[key] == old[key]
    for task in old["tasks"]:
        for m in task["media"]:
            assert (output / "reviewer" / m["file"]).read_bytes() == (args["output"] / "reviewer" / m["file"]).read_bytes()
    assert (output / "imported-v1-labels.jsonl").read_text() == path.read_text() == original
    page = (output / "reviewer/index.html").read_text()
    public = json.loads(page.split("const DATA=", 1)[1].split(";\n", 1)[0])
    assert public["initial_labels"] == values
    assert not any(k in page for k in ("candidate_keeper", "demo:G", "narrative_recheck"))
    assert report(new, values)["task_status_counts"] == {"reviewed": 1, "uncertain": 1, "pending": 3}
    assert read_labels(output / "labels-v2.jsonl") == labels_v2(new, values)
    verify_media(new, output / "reviewer")
    with pytest.raises(ValueError, match="immutable"):
        upgrade_bundle(args["output"], path, output)
    bad = copy.deepcopy(values[0])
    bad["bundle_id"] = "unrelated"
    with pytest.raises(ValueError, match="stale"):
        validate_labels(new, [bad])
    bad = labels_v2(new, values)[0]
    bad["bundle_id"] = old["bundle_id"]
    with pytest.raises(ValueError, match="stale"):
        validate_labels(new, [bad])
    (output / "reviewer/index.html").write_text(page.replace(values[0]["note"], "tampered"))
    with pytest.raises(ValueError, match="fingerprint"):
        verify_media(new, output / "reviewer")


def test_upgrade_tampered_source_fails_before_output(tmp_path):
    from src.human_audit_bundle import upgrade_bundle
    args = files(tmp_path)
    old = legacy_bundle_files(args)
    path = tmp_path / "legacy.jsonl"
    path.write_text(canonical(label(old, old["tasks"][0])))
    media = args["output"] / "reviewer" / old["tasks"][0]["media"][0]["file"]
    media.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="fingerprint"):
        upgrade_bundle(args["output"], path, tmp_path / "r3")
    assert not (tmp_path / "r3").exists()


def test_upgrade_snapshots_labels_once_and_rejects_symlink_inputs(tmp_path, monkeypatch):
    from src.human_audit_bundle import upgrade_bundle
    args = files(tmp_path)
    old = legacy_bundle_files(args)
    path = tmp_path / "legacy.jsonl"
    raw = (canonical(label(old, old["tasks"][0])) + "\n").encode()
    path.write_bytes(raw)
    real_read = Path.read_bytes
    reads = []
    def count_read(p):
        if p == path:
            reads.append(p)
            assert len(reads) == 1, "migration must use the same validated byte snapshot"
        return real_read(p)
    monkeypatch.setattr(Path, "read_bytes", count_read)
    upgrade_bundle(args["output"], path, tmp_path / "r3")
    assert len(reads) == 1
    assert (tmp_path / "r3/imported-v1-labels.jsonl").read_bytes() == raw
    link = tmp_path / "label-link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        upgrade_bundle(args["output"], link, tmp_path / "bad-label-link")
    manifest = args["output"] / "manifest.json"
    data = manifest.read_bytes()
    manifest.unlink()
    other = tmp_path / "manifest-copy"
    other.write_bytes(data)
    manifest.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        upgrade_bundle(args["output"], path, tmp_path / "bad-manifest-link")


def test_v1_requires_sealed_legacy_lineage_on_v2_bundles(tmp_path):
    from src.human_audit_bundle import upgrade_bundle
    args = files(tmp_path)
    b = export_bundle(**args)
    fabricated = label(b, b["tasks"][0])
    fabricated["schema_version"] = 1
    for phase in fabricated["phases"]:
        del phase["keeper_status"]
    with pytest.raises(ValueError, match="stale"):
        validate_labels(b, [fabricated])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="original v1"):
        upgrade_bundle(args["output"], empty, tmp_path / "not-v1")
    old = legacy_bundle_files({**args, "output": tmp_path / "old"})
    legacy = label(old, old["tasks"][0])
    path = tmp_path / "legacy.jsonl"
    path.write_text(canonical(legacy))
    new = upgrade_bundle(tmp_path / "old", path, tmp_path / "r3")
    assert validate_labels(new, [legacy])
    legacy["bundle_id"] = new["bundle_id"]
    with pytest.raises(ValueError, match="stale"):
        validate_labels(new, [legacy])


def test_mixed_quality_exposes_assessed_phase_errors_without_joint_verdict():
    b = bundle()
    task = b["tasks"][0]
    task["member_ids"].append(999)
    task["aliases"].append("M3")
    b = seal({k: v for k, v in b.items() if k != "bundle_id"})
    value = label(b, task)
    value.update(schema_version=2, phases=[
        {"members": ["M1", "M2"], "keeper_status": "assessed", "acceptable_keepers": ["M2"]},
        {"members": ["M3"], "keeper_status": "quality_abstain"},
    ])
    d = report(b, [value])["tasks"][0]
    assert d["quality_assessed_phase_count"] == d["quality_assessed_phase_errors"] == 1
    assert d["quality_abstain_phase_count"] == 1
    assert d["phase_coverage_error"] is True
    assert d["keeper_quality_error"] is d["keeper_bad"] is d["error"] is None


def test_upgrade_cli_never_rereads_source_for_completion_message(tmp_path, monkeypatch, capsys):
    args = files(tmp_path)
    old = legacy_bundle_files(args)
    path = tmp_path / "legacy.jsonl"
    path.write_text(canonical(label(old, old["tasks"][0])))
    monkeypatch.setattr("src.human_audit_bundle.read_labels",
                        lambda *a: pytest.fail("CLI must not reread source after publication"))
    output = tmp_path / "r3"
    main(["upgrade", "--bundle", str(args["output"]), "--labels", str(path), "--output", str(output)])
    assert "5 tasks with editable legacy records" in capsys.readouterr().out
    assert (output / "imported-v1-labels.jsonl").read_bytes() == path.read_bytes()
