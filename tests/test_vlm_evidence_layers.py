"""Deterministic gates for the Fable evidence-chain narrow slice."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src import db, vlm_identity, vlm_prompt_context, vlm_review, vlm_visual_budget
from tests.test_vlm_review import _output


def test_interface_descriptor_is_not_raw_payload_and_exception_is_exact():
    vlm_review.assert_model_context_safe({"measurement_interfaces": {
        "phash": "perceptual_hash_xor_popcount",
        "dinov2_embedding": "cosine_on_persisted_little_endian_float16_vectors",
    }})
    for value in ("AAAA", b"1234", [1, 2, 3], {"encoding": "base64", "data": "AA=="}):
        with pytest.raises((ValueError, TypeError), match="raw binary"):
            vlm_review.assert_model_context_safe({"measurement_interfaces": {"phash": value}})
    with pytest.raises(ValueError, match="raw binary"):
        vlm_review.assert_model_context_safe({"nested": {"phash": "perceptual_hash_xor_popcount"}})


def test_g002_panel_alias_reversal_rejected_but_identity_join_bijective():
    a, b = {"file_id": 107, "content_sha256": "a" * 64}, {"file_id": 99, "content_sha256": "b" * 64}
    panel, sidecar = {"F01": a, "F02": b}, {"F01": b, "F02": a}
    assert vlm_identity.join_maps(panel, sidecar) == {"F01": "F02", "F02": "F01"}
    with pytest.raises(ValueError, match="alias join forbidden"):
        vlm_identity.join_maps(panel, sidecar, by="alias")
    changed = {"F01": b, "F02": {**a, "content_sha256": "c" * 64}}
    with pytest.raises(ValueError, match="SHA changed"):
        vlm_identity.join_maps(panel, changed)
    with pytest.raises(ValueError, match="content_sha256"):
        vlm_identity.identity({"file_id": 107})


def test_three_layers_coverage_rank_delta_and_no_selection_leakage(tmp_path):
    _, conn, gid, ids = _output(tmp_path)
    rows = db.group_page_by_id(conn, gid)
    members, audit, derived, prompt = vlm_review.build_layers(rows)
    assert [x["layer"] for x in (audit, derived, prompt)] == ["L0", "L1", "L2"]
    assert "base64" in json.dumps(audit)
    assert "utility_score" in json.dumps(audit)
    assert "/private/" not in json.dumps(audit)
    assert all(frame["frame_identity"].startswith("I") for frame in audit["frames"])
    assert "F01" not in json.dumps(derived)
    assert "utility_score" not in json.dumps(prompt)
    assert "pair_margin" not in json.dumps(prompt)
    assert "box_normalized" not in json.dumps(prompt)
    assert "landmarks" not in json.dumps(prompt)
    assert "content_sha256" not in json.dumps(prompt["evidence"])
    assert all(row["content_sha256"] not in json.dumps(prompt) for row in members)
    ledger = derived["coverage_ledger"]
    assert {item["disposition"] for item in ledger} <= {"exact", "derived", "summarized", "non-evidence"}
    refs = set(derived["evidence_refs"])
    assert all(set(item["replacement_refs"]) <= refs for item in ledger)
    assert all(item["reason"] for item in ledger)
    assert len(derived["family_coverage"]) == 13
    assert derived["family_coverage"]["pose_action_phase"]["reason"] == "not_persisted"
    assert derived["family_coverage"]["selection_diagnostics"]["disposition"] == "non-evidence"
    quality = next(r for r in derived["scalar_comparisons"] if r["metric"] == "quality_score")
    keys = [vlm_identity.identity(row) for row in members]
    assert quality["rank"] == {keys[0]: 1, keys[1]: 2}
    assert quality["delta_best"] == {keys[0]: 0, keys[1]: -1}
    assert quality["delta_median"] == {keys[0]: .5, keys[1]: -.5}
    assert vlm_prompt_context.token_estimate(prompt) <= 15000
    assert vlm_review.build_layers(list(reversed(rows)))[1:] == (audit, derived, prompt)
    changed = [dict(row) for row in rows]
    changed[0]["keep_file_id"] = ids[1]
    changed[0]["is_keep"] = False
    assert vlm_review.input_hash(changed) == vlm_review.input_hash(rows)
    with pytest.raises(ValueError, match="L1"):
        vlm_prompt_context.compile_prompt(audit, {})
    with pytest.raises(ValueError, match="only L2"):
        vlm_review.invoke_bedrock("global.anthropic.claude-sonnet-5-5", gid, [], audit)
    conn.close()


def test_fail_closed_numeric_arrays_base64_schema_and_text_budget(tmp_path):
    _, conn, gid, _ = _output(tmp_path)
    prompt = vlm_review.build_layers(db.group_page_by_id(conn, gid))[3]
    for key, value, match in (("raw", [1] * 21, "array"),
                              ("encoded", {"encoding": "base64", "data": "AA=="}, "base64"),
                              ("raw", b"binary", "binary"),
                              ("x", float("nan"), "non-finite"),
                              ("boxes", [[0, 0, 1, 1]], "non-prompt"),
                              ("utility_score", 0.9, "non-prompt")):
        bad = {**prompt, key: value}
        with pytest.raises((ValueError, TypeError), match=match):
            vlm_review.assert_model_context_safe(bad)
    with pytest.raises(ValueError, match="schema"):
        vlm_review.assert_model_context_safe({"layer": "L2", "schema_version": 100})
    huge = copy.deepcopy(prompt)
    huge["evidence"] = {f"x{i}": {"value": "x" * 450} for i in range(110)}
    with pytest.raises(ValueError, match="per-group limit"):
        vlm_review.assert_model_context_safe(huge)
    conn.close()


def test_alias_map_tamper_and_derived_hash_invalidate_ready_cache(tmp_path):
    output, conn, gid, _ = _output(tmp_path, exact=True)
    rows = db.group_page_by_id(conn, gid)
    vlm_review.run_job(output, group_ids=[gid])
    path = vlm_review.cache_path(output, gid)
    artifact = json.loads(path.read_text())
    assert artifact["alias_map"] == vlm_identity.alias_map(vlm_review._rows(rows))
    assert vlm_review.load_cached(output, rows) is not None
    artifact["alias_map"]["F01"], artifact["alias_map"]["F02"] = artifact["alias_map"]["F02"], artifact["alias_map"]["F01"]
    path.write_text(json.dumps(artifact))
    assert vlm_review.load_cached(output, rows) is None
    artifact["alias_map"] = vlm_identity.alias_map(vlm_review._rows(rows))
    artifact["derived_context_sha256"] = "not-current"
    path.write_text(json.dumps(artifact))
    assert vlm_review.load_cached(output, rows) is None
    conn.close()


def test_action_rejects_bystander_person_and_face_without_primary_authority(tmp_path):
    _, conn, gid, _ = _output(tmp_path)
    rows = vlm_review._rows(db.group_page_by_id(conn, gid))
    with pytest.raises(vlm_visual_budget.VisualBudgetError, match="ACTION_PRIMARY_SUBJECT_UNLOCALIZED"):
        vlm_visual_budget.plan(rows, regime="action")
    for row in rows:
        q = json.loads(row["quality_meta"])
        q["local_region_set"] = {"regions": [{"kind": "subject", "role": "primary_subject",
                                               "role_authority": "human_verified",
                                               "box_normalized": [.4, .2, .7, .9]}]}
        row["quality_meta"] = json.dumps(q)
    plan = vlm_visual_budget.plan(rows, regime="action")
    assert plan["bystander_faces_use_budget"] is False
    assert len(plan["windows"]) == 1
    assert plan["max_strips"] == 2
    assert plan["windows"][0][0] > .3
    conn.close()


def test_visual_cells_have_real_minimum_pixels_and_one_cross_file_strip(tmp_path):
    output, conn, gid, _ = _output(tmp_path)
    rows = vlm_review._rows(db.group_page_by_id(conn, gid))
    sources = {row["file_id"]: Path(conn.execute(
        "SELECT path FROM files WHERE id=?", (row["file_id"],)).fetchone()[0]) for row in rows}
    assets, manifest = vlm_review.prepare_visuals(output, rows, sources)
    assert [p.name for p in assets] == ["sheet-1.jpg", "comparison-1.jpg"]
    assert all(min(cell["image_pixels"]) >= 400 for cell in manifest[0]["cells"])
    assert all(min(cell["image_pixels"]) >= 256 for cell in manifest[1]["cells"])
    assert all(min(cell["native_pixels"]) >= 256 for cell in manifest[1]["cells"])
    for row in rows:
        row["faces_json"] = json.dumps([{"bbox": [200, 150, 10, 10]}])
    with pytest.raises(vlm_visual_budget.VisualBudgetError, match="CRITICAL_ROI_SHORT_EDGE"):
        vlm_review.prepare_visuals(output, rows, sources)
    conn.close()


def test_large_group_keeps_exact_sets_and_nearest_phash_without_quadratic_prompt(tmp_path):
    _, conn, gid, _ = _output(tmp_path)
    source = dict(db.group_page_by_id(conn, gid)[0])
    rows = []
    for i in range(11):
        row = {**source, "file_id": i + 1, "content_sha256": f"{max(i, 1):064x}",
               "phash": i.to_bytes(8, "big")}
        rows.append(row)
    _, _, derived, prompt = vlm_review.build_layers(rows)
    assert len(derived["pairwise_measurements"]) == 55
    assert prompt["identical_sets"] == [["F01", "F02"]]
    phash = [p for p in prompt["pairs"].values() if p["criterion"] == "phash"]
    assert len(phash) <= 11
    assert {a for p in phash for a in p["frames"]} == set(prompt["frames"])
    assert len([p for p in prompt["pairs"].values() if p["criterion"] == "dinov2_embedding"]) <= 2
    conn.close()


def test_source_hash_mismatch_is_rejected_even_with_unchanged_size_and_mtime(tmp_path):
    import os

    output, conn, gid, _ = _output(tmp_path)
    rows = vlm_review._rows(db.group_page_by_id(conn, gid))
    sources = {row["file_id"]: Path(conn.execute(
        "SELECT path FROM files WHERE id=?", (row["file_id"],)).fetchone()[0]) for row in rows}
    path = sources[rows[0]["file_id"]]
    stat = path.stat()
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="source changed"):
        vlm_review.prepare_visuals(output, rows, sources)
    conn.close()


def test_layer_report_never_mixes_derived_and_prompt_bytes(tmp_path):
    from src import vlm_criterion_compiler

    _, conn, gid, _ = _output(tmp_path)
    rows = db.group_page_by_id(conn, gid)
    _, audit, derived, prompt = vlm_review.build_layers(rows)
    report = vlm_review.context_report(rows)
    a, d, p = [len(vlm_review._canonical(v)) for v in (audit, derived, prompt)]
    assert report['audit_bytes'] == a
    assert report['derived_bytes'] == d
    assert report['prompt_context_bytes'] == p
    assert report['derived_to_audit_ratio'] == d / a
    assert report['prompt_context_to_audit_ratio'] == p / a
    assert report['prompt_context_to_derived_ratio'] == p / d
    assert report['prompt_context_tokens_estimate'] == vlm_prompt_context.token_estimate(prompt)
    assert report['prompt_context_within_bytes_limit'] == (p <= vlm_prompt_context.MAX_BYTES)
    assert report['total_text_within_tokens_estimate_limit'] == (report['total_text_tokens_estimate'] <= 15000)
    forbidden = {'model_bytes', 'model_to_audit_ratio', 'within_model_context_limit', 'reduction_fraction'}
    assert not forbidden & report.keys()
    assert not forbidden & vlm_criterion_compiler.coverage_report(derived).keys()
    assert 'max_model_context_bytes' not in json.dumps(derived)
    conn.close()


def test_large_l1_small_l2_report_gate_and_aggregate_ratios(monkeypatch):
    layers = ([{'group_id': 2}], {'data': 'x' * 1000},
              {'data': 'x' * 200000, 'coverage_ledger': []}, {'data': 'x' * 9700})
    monkeypatch.setattr(vlm_review, 'build_layers', lambda rows, enforce_gate=False: layers)
    one = vlm_review.context_report([])
    assert one['derived_to_audit_ratio'] > 100
    assert one['prompt_context_within_bytes_limit'] is True
    assert one['total_text_within_tokens_estimate_limit'] is True
    aggregate = vlm_review.aggregate_context_reports([one, one])
    for key in ('audit_bytes', 'derived_bytes', 'prompt_context_bytes'):
        assert aggregate[key] == 2 * one[key]
    for key in ('derived_to_audit_ratio', 'prompt_context_to_audit_ratio', 'prompt_context_to_derived_ratio'):
        assert aggregate[key] == one[key]
    budget = aggregate['prompt_budget']
    assert budget['total_text_tokens_estimate_sum'] == 2 * one['total_text_tokens_estimate']
    assert budget['total_text_tokens_estimate_max'] == one['total_text_tokens_estimate']
    # Aggregate sums may exceed a per-group limit without invalidating any group.
    many = vlm_review.aggregate_context_reports([one] * 20)
    assert many['prompt_context_bytes'] > vlm_prompt_context.MAX_BYTES
    assert many['prompt_budget']['prompt_context_all_groups_within_bytes_limit'] is True
    assert 'model_bytes' not in aggregate
    empty = vlm_review.aggregate_context_reports([])
    assert empty['prompt_context_to_audit_ratio'] is None
    assert empty['prompt_budget']['total_text_all_groups_within_tokens_estimate_limit'] is None


def test_reports_can_measure_over_budget_and_match_actual_l2_gate(monkeypatch, tmp_path):
    _, conn, gid, _ = _output(tmp_path)
    prompt = vlm_review.build_layers(db.group_page_by_id(conn, gid))[3]
    huge = copy.deepcopy(prompt)
    huge['evidence'] = {f'x{i}': {'value': 'x' * 450} for i in range(100)}
    budget = vlm_review.prompt_budget_report(len(vlm_review._canonical(huge)))
    assert budget['total_text_within_tokens_estimate_limit'] is False
    with pytest.raises(ValueError, match='per-group limit'):
        vlm_review.assert_model_context_safe(huge)
    monkeypatch.setattr(vlm_review, 'build_layers', lambda rows, enforce_gate=False:
                        ([{'group_id': 1}], {}, {}, huge))
    assert vlm_review.context_report([])['total_text_within_tokens_estimate_limit'] is False
    conn.close()


def test_aggregate_report_api_uses_same_layer_contract(tmp_path):
    output, conn, gid, _ = _output(tmp_path)
    one = vlm_review.context_report(db.group_page_by_id(conn, gid))
    report = vlm_review.report_contexts(output, group_ids=[gid], select='all')
    assert report == vlm_review.aggregate_context_reports([one])
    conn.close()


def test_aggregate_ratios_are_weighted_totals_and_budget_boundary_is_exact():
    reports = []
    for a, d, p in ((1000, 2000, 100), (3000, 9000, 900)):
        reports.append({**vlm_review.layer_size_report(a, d, p),
                        **vlm_review.prompt_budget_report(p), 'coverage': {}})
    result = vlm_review.aggregate_context_reports(reports)
    assert result['derived_to_audit_ratio'] == 11000 / 4000
    assert result['prompt_context_to_audit_ratio'] == 1000 / 4000
    assert result['prompt_context_to_derived_ratio'] == 1000 / 11000
    available = 3 * vlm_prompt_context.MAX_TOKENS_ESTIMATE - len(vlm_review.PROMPT.encode('utf-8'))
    assert vlm_review.prompt_budget_report(available)['total_text_within_tokens_estimate_limit'] is True
    assert vlm_review.prompt_budget_report(available + 1)['total_text_within_tokens_estimate_limit'] is False
