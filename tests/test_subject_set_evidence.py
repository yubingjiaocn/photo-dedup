"""Bounded ensemble evidence gates, independent of stored photo decisions."""
import copy
import json
from pathlib import Path

import pytest

from src import subject_set_evidence as ss
from src import vlm_visual_budget


def fixture(count=3):
    rows = []
    for fid in (1, 2):
        regions = [{"region_id": f"S{i}", "kind": "subject", "class_id": 0,
                    "confidence": .9, "box_normalized": [.03+i*.3, .1, .27+i*.3, .9],
                    "musiq": 60 + i*fid, "clipiqa": .7} for i in range(count)]
        rows.append({"file_id": fid, "content_sha256": str(fid)*64,
                     "quality_meta": {"local_region_set": {"regions": regions}}, "faces_json": []})
    scope = {ss.identity(r): {"coverage": "verified_complete", "authority": "human_verified",
                             "roles": {f"S{i}": "key_candidate" for i in range(count)}} for r in rows}
    return rows, scope


def packet(count=3):
    rows, scope = fixture(count)
    l0, l1, lookup = ss.compile_evidence(rows, regime="group", scope=scope)
    sizes = {ss.identity(r): (4000, 3000) for r in rows}
    return l0, l1, lookup, sizes


def test_correspondence_is_reciprocal_candidate_not_region_id_join():
    rows, scope = fixture()
    regions = rows[1]["quality_meta"]["local_region_set"]["regions"]
    regions[0]["region_id"], regions[2]["region_id"] = "S2", "S0"
    _, l1, _ = ss.compile_evidence(rows, regime="group", scope=scope)
    pair = l1["correspondence"][0]["pairs"][0]
    assert pair["observations"][0].endswith('/S0')
    assert pair["observations"][1].endswith('/S2')
    assert pair["authority"] == "visual_correspondence_candidate_not_identity"
    assert pair["confidence"] == "medium"


def test_ambiguity_unmatched_and_count_mismatch_fail_closed():
    rows, scope = fixture()
    regions = rows[1]["quality_meta"]["local_region_set"]["regions"]
    regions[1]["box_normalized"] = regions[0]["box_normalized"]
    regions.pop()
    _, l1, lookup = ss.compile_evidence(rows, regime="group", scope=scope)
    assert "SUBJECT_COUNT_MISMATCH" in l1["blockers"]
    assert "CORRESPONDENCE_UNKNOWN_OR_CONFLICT" in l1["blockers"]
    assert any(c["reason"] == "AMBIGUOUS" for c in l1["correspondence"][0]["conflicts"])
    plan = ss.visual_plan(l1, lookup, {ss.identity(r): (4000, 3000) for r in rows})
    assert plan["status"] == "attention_only"
    assert not any(c["shown"] for c in plan["coverage_ledger"])


def test_equal_counts_never_establish_ensemble_completeness():
    rows, _ = fixture()
    _, l1, _ = ss.compile_evidence(rows, regime="stage")
    assert "ENSEMBLE_COVERAGE_UNVERIFIED" in l1["blockers"]
    assert all(f["true_subject_count"] == "unknown" for f in l1["frames"])
    assert all(f["ensemble_completeness"] == "unknown" for f in l1["frames"])


def test_visual_annotation_cannot_authorize_verified_coverage():
    rows, scope = fixture()
    for annotation in scope.values():
        annotation["authority"] = "Astra_visual_inspection_candidate_not_human_truth"
    _, l1, _ = ss.compile_evidence(rows, regime="group", scope=scope)
    assert "ENSEMBLE_COVERAGE_UNVERIFIED" in l1["blockers"]


def test_bystanders_do_not_spend_subject_budget_or_mask_missing_performers():
    rows, scope = fixture()
    for entry in scope.values():
        entry["roles"]["S1"] = "bystander_candidate"
        entry["detector_missing"] = True
    _, l1, lookup = ss.compile_evidence(rows, regime="stage", scope=scope)
    plan = ss.visual_plan(l1, lookup, {ss.identity(r): (4000, 3000) for r in rows})
    assert "DETECTOR_MISSING_SUBJECTS" in plan["blockers"]
    assert "BYSTANDER_CONTAMINATION" in plan["blockers"]
    assert not any(ref.endswith('/S1') for track in plan["windows"] for ref in track["refs"])
    assert all(e["reason"] == "bystander_downweighted" for e in plan["coverage_ledger"] if e["ref"].endswith('/S1'))


def test_budget_ranks_difference_not_largest_face_and_ledgers_every_subject():
    _, l1, lookup, sizes = packet()
    plan = ss.visual_plan(l1, lookup, sizes)
    assert plan["status"] == "candidate"
    assert len(plan["windows"]) == 2
    assert plan["windows"][0]["refs"][0].endswith('/S2')
    assert len(plan["coverage_ledger"]) == 6
    assert sum(c["shown"] for c in plan["coverage_ledger"]) == 4
    assert sum(c["reason"] == "budget_or_unresolved" for c in plan["coverage_ledger"]) == 2
    assert all(c["face_coverage"] == "not_shown" for c in plan["coverage_ledger"])
    assert len(ss.visual_plan(l1, lookup, sizes, max_regions=3)["windows"]) == 3
    with pytest.raises(ValueError):
        ss.visual_plan(l1, lookup, sizes, max_regions=4)


def test_size_gates_use_native_image_not_padding():
    _, l1, lookup, sizes = packet()
    small = {k: (399, 300) for k in sizes}
    plan = ss.visual_plan(l1, lookup, small)
    assert "SHEET_NATIVE_SHORT_EDGE_BELOW_400" in plan["blockers"]
    assert "ROI_NATIVE_SHORT_EDGE_BELOW_256" in plan["blockers"]
    assert plan["status"] == "attention_only"


def test_l2_only_l1_unknown_faces_eyes_phase_and_no_raw_geometry():
    l0, l1, _, _ = packet()
    with pytest.raises(ValueError):
        ss.prompt_evidence(l0)
    text = json.dumps(ss.prompt_evidence(l1))
    for key in ('"box"', '"keypoints"', '"embedding"', '"keeper"'):
        assert key not in text
    assert "allow_multiple_non_substitutable_members_and_phases" in text
    assert l1["irreplaceable_candidates"] == []
    assert "subject_ref or phase_ref" in l1["irreplaceable_candidate_contract"]["key"]
    assert all(s["eye_state"] == "unknown" for f in l1["frames"] for s in f["subjects"])


def test_stale_cache_rejects_content_scope_detector_or_version_change():
    l0, _, _, _ = packet()
    cached = {"version": ss.VERSION, "cache_key": ss.cache_key(l0)}
    assert ss.cache_valid(cached, l0)
    for mutate in (
        lambda p: p["frames"][0].update(frame="changed-sha"),
        lambda p: p["frames"][0]["scope"].update(coverage="unknown"),
        lambda p: p["frames"][0]["regions"][0].update(confidence=.1),
    ):
        changed = copy.deepcopy(l0)
        mutate(changed)
        assert not ss.cache_valid(cached, changed)
    assert not ss.cache_valid({**cached, "version": "legacy"}, l0)


def test_stale_scope_and_duplicate_identity_rejected():
    rows, scope = fixture()
    rows[0]["content_sha256"] = "f"*64
    with pytest.raises(ValueError, match="stale scope"):
        ss.compile_evidence(rows, regime="group", scope=scope)
    with pytest.raises(ValueError, match="duplicate"):
        ss.compile_evidence([rows[0], rows[0]], regime="group")


def test_candidates_do_not_authorize_production_group_planner():
    rows, _ = fixture()
    with pytest.raises(vlm_visual_budget.VisualBudgetError, match="REGISTRATION_UNAVAILABLE"):
        vlm_visual_budget.plan(rows, regime="group")


def test_real_slice_reports_are_blocked_inert_and_schema_is_deferred():
    root = Path(__file__).resolve().parents[1] / "artifacts/multi-subject-narrow"
    if not (root / "report.json").exists():
        pytest.skip("run bounded offline probe for frozen artifact verification")
    summary = json.loads((root / "report.json").read_text())
    assert summary["real_slice_gate_passes"] == 0
    assert summary["new_decision_schema"] == "deferred_until_real_subject_set_gate_passes"
    assert all(g["decision_tables_unchanged"] for g in summary["groups"])
    for path in root.glob('*-report.json'):
        report = json.loads(path.read_text())
        assert report["recommendations"] == []
        assert report["contact_sheet"]["actual_short_edge_gate"]
        assert all(min(c["actual_image_size"]) >= 400 for c in report["contact_sheet"]["cells"])
        assert report["decision_tables_before"] == report["decision_tables_after"]
        assert not report["production_enabled"]
