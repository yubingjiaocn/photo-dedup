"""Offline calibration reports must remain auditable, resumable, and path-safe."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scene_shadow_report import REPORT_SCHEMA_VERSION, main, run
from src.siglip_router import build_shadow_routing_record, load_prompt_bank

BANK = Path(__file__).parents[1] / "research" / "siglip_prompt_bank_v1.yaml"


def _record(**audit_overrides) -> dict:
    bank = load_prompt_bank(BANK)
    scores = {code: {prompt: (index + 1) / 10 for index, prompt in enumerate(
        spec["positive_prompts"] + spec["hard_negative_prompts"] + [bank["generic_null_prompt"]])}
        for code, spec in bank["tags"].items()}
    audit = {"model_sha256": "a" * 64, "runtime": {"device": "cpu", "precision": "float32"}}
    audit.update(audit_overrides)
    return build_shadow_routing_record(bank, scores, model_name="siglip-test", model_revision="r1", model_audit=audit)


def _jsonl(path: Path, records: list[object]) -> None:
    path.write_text("".join(json.dumps(item, allow_nan=True) + "\n" if not isinstance(item, str) else item + "\n" for item in records))


def test_resume_is_idempotent_and_reports_auditable_raw_stats(tmp_path):
    source, out = tmp_path / "routing.jsonl", tmp_path / "out"
    _jsonl(source, [_record(), _record(), _record()])
    first = run(jsonl_path=str(source), output_dir=str(out), checkpoint_every=1, stop_after=2)
    assert first["records"]["valid"] == 2
    final = run(jsonl_path=str(source), output_dir=str(out), checkpoint_every=1)
    again = run(jsonl_path=str(source), output_dir=str(out), checkpoint_every=1)
    assert final == again
    assert final["provenance"]["checkpoint_prefix_verified"] is True
    assert final["records"]["valid"] == 3
    assert "FIREWORKS.hard_negative_gap" in final["siglip"]["raw_statistics"]
    assert final["unknown"]["scene_context"]["reasons"]["OUT_OF_CALIBRATION_DOMAIN"] == 3
    assert (out / "scene-shadow-summary.json").exists()
    assert (out / "scene-shadow-tag-statistics.csv").read_text().startswith("tag,metric,")


def test_truncated_checkpoint_restarts_fail_closed_without_double_counting(tmp_path):
    source, out = tmp_path / "routing.jsonl", tmp_path / "out"
    _jsonl(source, [_record(), _record()])
    out.mkdir()
    (out / "scene-shadow.checkpoint.json").write_text('{"next_index":')
    report = run(jsonl_path=str(source), output_dir=str(out), checkpoint_every=1)
    assert report["records"] == {"seen": 2, "valid": 2, "invalid": 0, "invalid_rate": 0.0}


@pytest.mark.parametrize("mutation", [
    lambda state: state.update(next_index=999, seen=999),
    lambda state: state.update(valid=state["valid"] + 1),
    lambda state: state["stats"][next(iter(state["stats"]))].update(sum=1e99),
    lambda state: state["stats"][next(iter(state["stats"]))].update(count=-1),
    lambda state: state.update(prefix_digest="f" * 64),
])
def test_well_formed_forged_checkpoint_is_never_partially_trusted(tmp_path, mutation):
    source, out = tmp_path / "routing.jsonl", tmp_path / "out"
    _jsonl(source, [_record(), _record(), _record()])
    run(jsonl_path=str(source), output_dir=str(out), checkpoint_every=1, stop_after=2)
    checkpoint = out / "scene-shadow.checkpoint.json"
    state = json.loads(checkpoint.read_text())
    mutation(state)
    checkpoint.write_text(json.dumps(state))
    report = run(jsonl_path=str(source), output_dir=str(out))
    assert report["records"] == {"seen": 3, "valid": 3, "invalid": 0, "invalid_rate": 0.0}


def test_bad_json_nan_wrong_schema_and_missing_routing_fail_closed(tmp_path):
    source, out = tmp_path / "routing.jsonl", tmp_path / "out"
    bad_schema = _record()
    bad_schema["schema_version"] = 999
    nan = _record()
    nan["model"]["shadow_prompt_audit"]["tags"]["INDOOR"]["positive_mean"] = float("nan")
    _jsonl(source, [_record(), "{broken", bad_schema, nan, {"other": "missing"}])
    report = run(jsonl_path=str(source), output_dir=str(out))
    assert report["records"]["valid"] == 1
    assert report["records"]["invalid"] == 4
    assert report["bad_records"]


def test_mixed_prompt_hash_is_explicitly_rejected(tmp_path):
    source = tmp_path / "routing.jsonl"
    changed = _record()
    changed["model"]["prompt_bank_hash"] = "0" * 64
    _jsonl(source, [_record(), changed])
    report = run(jsonl_path=str(source), output_dir=str(tmp_path / "out"))
    assert report["status"] == "REJECTED_MIXED_IDENTITY"
    assert len(report["siglip"]["calibration_identities"]) == 2


@pytest.mark.parametrize("field,value", [
    ("model_sha256", "b" * 64),
    ("revision", "display-only-r2"),
    ("device", "cuda"),
    ("precision", "float16"),
])
def test_identity_mixing_and_revision_display_semantics(tmp_path, field, value):
    changed = _record()
    if field in {"device", "precision"}:
        changed["model"]["runtime"][field] = value
    else:
        changed["model"][field] = value
    source = tmp_path / "routing.jsonl"
    _jsonl(source, [_record(), changed])
    report = run(jsonl_path=str(source), output_dir=str(tmp_path / "out"))
    assert report["status"] == "REJECTED_MIXED_IDENTITY"
    assert len(report["siglip"]["model_revisions"]) == (2 if field == "revision" else 1)


@pytest.mark.parametrize("missing", ["model_sha256", "runtime"])
def test_missing_identity_rejected_and_cli_is_nonzero(tmp_path, missing):
    record = _record()
    del record["model"][missing]
    source = tmp_path / "routing.jsonl"
    _jsonl(source, [record])
    report = run(jsonl_path=str(source), output_dir=str(tmp_path / "out"))
    assert report["status"] == "REJECTED_MISSING_IDENTITY"
    assert main(["--jsonl", str(source), "--output-dir", str(tmp_path / "cli")]) == 2


def test_empty_input_atomic_outputs_and_no_source_path_leak(tmp_path):
    secret = tmp_path / "private-source-name.jsonl"
    secret.write_text("")
    out = tmp_path / "out"
    report = run(jsonl_path=str(secret), output_dir=str(out))
    rendered = json.dumps(report)
    assert report["records"]["seen"] == 0
    assert report["status"] == "REJECTED_NO_VALID_RECORDS"
    assert secret.name not in rendered and str(secret) not in rendered
    assert not list(out.glob(".*.tmp"))
    saved = json.loads((out / "scene-shadow-summary.json").read_text())
    assert saved["report_schema_version"] == REPORT_SCHEMA_VERSION


def test_db_input_reads_quality_meta_only_and_omits_photo_path(tmp_path):
    import sqlite3
    db = tmp_path / "inventory.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT); CREATE TABLE features (file_id INTEGER, quality_meta TEXT);")
    conn.execute("INSERT INTO files VALUES (1, '/very/private/photo.jpg')")
    conn.execute("INSERT INTO features VALUES (?, ?)", (1, json.dumps({"routing": _record()})))
    conn.commit()
    conn.close()
    report = run(db_path=str(db), output_dir=str(tmp_path / "out"))
    assert report["records"]["valid"] == 1
    assert report["provenance"]["sqlite_snapshot"] is True
    assert "/very/private/photo.jpg" not in json.dumps(report)


def test_sqlite_wal_is_in_snapshot_and_changed_database_cannot_resume_old_state(tmp_path):
    import sqlite3
    db, out = tmp_path / "inventory.sqlite", tmp_path / "out"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT); CREATE TABLE features (file_id INTEGER, quality_meta TEXT);")
    conn.execute("INSERT INTO files VALUES (1, '/not-read.jpg')")
    conn.execute("INSERT INTO features VALUES (1, ?)", (json.dumps({"routing": _record()}),))
    conn.commit()
    first = run(db_path=str(db), output_dir=str(out), stop_after=1)
    assert first["records"]["valid"] == 1
    conn.execute("INSERT INTO files VALUES (2, '/also-not-read.jpg')")
    conn.execute("INSERT INTO features VALUES (2, ?)", (json.dumps({"routing": _record()}),))
    conn.commit()
    final = run(db_path=str(db), output_dir=str(out))
    conn.close()
    assert final["records"]["valid"] == 2
    assert final["provenance"]["input"]["content_sha256"] != first["provenance"]["input"]["content_sha256"]


@pytest.mark.parametrize("kwargs", [{}, {"db_path": "x", "jsonl_path": "y"}])
def test_exactly_one_source_is_required(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        run(**kwargs)
