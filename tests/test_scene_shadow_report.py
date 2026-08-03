"""Offline calibration reports must remain auditable, resumable, and path-safe."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scene_shadow_report import REPORT_SCHEMA_VERSION, run
from src.siglip_router import build_shadow_routing_record, load_prompt_bank

BANK = Path(__file__).parents[1] / "research" / "siglip_prompt_bank_v1.yaml"


def _record() -> dict:
    bank = load_prompt_bank(BANK)
    scores = {code: {prompt: (index + 1) / 10 for index, prompt in enumerate(
        spec["positive_prompts"] + spec["hard_negative_prompts"] + [bank["generic_null_prompt"]])}
        for code, spec in bank["tags"].items()}
    return build_shadow_routing_record(bank, scores, model_name="siglip-test", model_revision="r1")


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
    assert report["status"] == "REJECTED_MIXED_PROMPT_HASH"
    assert len(report["siglip"]["prompt_bank_hashes"]) == 2


def test_empty_input_atomic_outputs_and_no_source_path_leak(tmp_path):
    secret = tmp_path / "private-source-name.jsonl"
    secret.write_text("")
    out = tmp_path / "out"
    report = run(jsonl_path=str(secret), output_dir=str(out))
    rendered = json.dumps(report)
    assert report["records"]["seen"] == 0
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
    assert "/very/private/photo.jpg" not in json.dumps(report)


@pytest.mark.parametrize("kwargs", [{}, {"db_path": "x", "jsonl_path": "y"}])
def test_exactly_one_source_is_required(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        run(**kwargs)
