"""Offline, fail-closed aggregation for SigLIP shadow-routing records.

This module only reads routing metadata already recorded in SQLite or JSONL.  It
never opens source photos, writes decisions/manifests, chooses thresholds, or
loads a model.  Its reports contain aggregate evidence and stable anonymous
record IDs only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from .routing_schema import SCENE_TAGS, SUBJECT_TAGS, validate_routing_record

REPORT_SCHEMA_VERSION = 1
AUDIT_FIELDS = ("positive_mean", "positive_median", "positive_range", "positive_std", "hard_negative_max", "hard_negative_gap")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(path: Path, kind: str) -> dict[str, Any]:
    stat = path.stat()
    identity = {"kind": kind, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "source_name_hash": hashlib.sha256(str(path.resolve()).encode()).hexdigest()}
    # JSONL is normally a portable, modest shadow export; a byte hash prevents
    # resuming an altered stream.  SQLite can be large, so its stat identity is
    # deliberate and is recorded as a limitation rather than silently hashing it.
    if kind == "jsonl":
        identity["content_sha256"] = _sha256(path)
    return identity


def _new_state(identity: Mapping[str, Any], config_hash: str | None) -> dict[str, Any]:
    return {"checkpoint_schema_version": REPORT_SCHEMA_VERSION, "input_identity": dict(identity),
            "config_sha256": config_hash, "next_index": 0, "seen": 0, "valid": 0, "invalid": 0,
            "unknown": {"scene_context": 0, "subject_protection": 0}, "reasons": Counter(),
            "unknown_reasons": {"scene_context": Counter(), "subject_protection": Counter()},
            "conflict_records": 0, "missing_records": 0, "tag_present": Counter(), "stats": {},
            "prompt_hashes": Counter(), "model_revisions": Counter(), "bad_records": []}


def _load_checkpoint(path: Path, identity: Mapping[str, Any], config_hash: str | None) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("checkpoint_schema_version") != REPORT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema mismatch")
        if raw.get("input_identity") != dict(identity) or raw.get("config_sha256") != config_hash:
            raise ValueError("checkpoint input/config mismatch")
        return _restore_state(raw)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return _new_state(identity, config_hash)


def _restore_state(raw: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(raw)
    for name in ("reasons", "tag_present", "prompt_hashes", "model_revisions"):
        state[name] = Counter(state.get(name, {}))
    state["unknown_reasons"] = {key: Counter(value) for key, value in state.get("unknown_reasons", {}).items()}
    state.setdefault("stats", {})
    state.setdefault("bad_records", [])
    return state


def _iter_jsonl(path: Path) -> Iterator[tuple[int, str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                yield index, f"jsonl:{index + 1}", ValueError("blank JSONL line")
                continue
            try:
                value = json.loads(line)
                routing = value.get("routing", value) if isinstance(value, dict) else value
                yield index, f"jsonl:{index + 1}", routing
            except json.JSONDecodeError as exc:
                yield index, f"jsonl:{index + 1}", exc


def _iter_db(path: Path) -> Iterator[tuple[int, str, Any]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        query = "SELECT f.id, fe.quality_meta FROM features fe JOIN files f ON f.id=fe.file_id ORDER BY f.id"
        for index, (file_id, raw_meta) in enumerate(connection.execute(query)):
            try:
                meta = json.loads(raw_meta)
                yield index, f"db:{file_id}", meta.get("routing") if isinstance(meta, dict) else None
            except (TypeError, json.JSONDecodeError) as exc:
                yield index, f"db:{file_id}", exc
    finally:
        connection.close()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _add_stat(state: dict[str, Any], key: str, value: Any) -> None:
    value = _finite(value, key)
    item = state["stats"].setdefault(key, {"count": 0, "sum": 0.0, "min": value, "max": value})
    item["count"] += 1
    item["sum"] += value
    item["min"] = min(item["min"], value)
    item["max"] = max(item["max"], value)


def _record_error(state: dict[str, Any], anon_id: str, error: Exception) -> None:
    state["invalid"] += 1
    if len(state["bad_records"]) < 20:
        state["bad_records"].append({"record_id": anon_id, "error": str(error)[:240]})


def _consume(state: dict[str, Any], anon_id: str, raw: Any) -> None:
    state["seen"] += 1
    if isinstance(raw, Exception):
        _record_error(state, anon_id, raw)
        return
    try:
        record = validate_routing_record(raw)
        model = record["model"]
        prompt_hash = model.get("prompt_bank_hash")
        audit = model.get("shadow_prompt_audit")
        if not isinstance(prompt_hash, str) or len(prompt_hash) != 64 or not isinstance(audit, Mapping):
            raise ValueError("missing SigLIP prompt-bank provenance")
        tags = audit.get("tags")
        expected_tags = SCENE_TAGS | SUBJECT_TAGS
        if not isinstance(tags, Mapping) or set(tags) != expected_tags:
            raise ValueError("shadow prompt audit does not cover routing schema tags")
        for tag, values in tags.items():
            if not isinstance(values, Mapping):
                raise ValueError(f"audit.{tag} must be an object")
            for field in AUDIT_FIELDS:
                _add_stat(state, f"{tag}.{field}", values.get(field))
        state["prompt_hashes"][prompt_hash] += 1
        state["model_revisions"][f"{model.get('name', '')}@{model.get('revision', '')}"] += 1
        for namespace in ("scene_context", "subject_protection"):
            value = record[namespace]
            state["reasons"].update(value["reasons"])
            if value["state"] == "UNKNOWN":
                state["unknown"][namespace] += 1
                state["unknown_reasons"][namespace].update(value["reasons"])
            for tag in value["tags"]:
                state["tag_present"][tag["code"]] += 1
        quality = record["quality_evidence"]
        state["reasons"].update(quality["reasons"])
        if quality["conflicts"]:
            state["conflict_records"] += 1
        if quality["missing"]:
            state["missing_records"] += 1
        state["valid"] += 1
    except (TypeError, ValueError, KeyError) as exc:
        _record_error(state, anon_id, exc)


def _checkpoint_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe(state)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _summary(state: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    valid = state["valid"]
    hashes = dict(state["prompt_hashes"])
    mixed = len(hashes) > 1
    aggregate_stats = {key: {"count": item["count"], "mean": item["sum"] / item["count"],
                              "min": item["min"], "max": item["max"]}
                       for key, item in state["stats"].items() if item["count"]}
    return {"report_schema_version": REPORT_SCHEMA_VERSION, "provenance": dict(provenance),
            "status": "REJECTED_MIXED_PROMPT_HASH" if mixed else "OK",
            "records": {"seen": state["seen"], "valid": valid, "invalid": state["invalid"],
                        "invalid_rate": state["invalid"] / state["seen"] if state["seen"] else 0.0},
            "unknown": {key: {"count": value, "coverage": value / valid if valid else 0.0,
                                "reasons": dict(state["unknown_reasons"][key])}
                        for key, value in state["unknown"].items()},
            "reasons": dict(state["reasons"]), "tag_present": dict(state["tag_present"]),
            "quality_evidence": {"conflict_records": state["conflict_records"], "missing_records": state["missing_records"],
                "conflict_rate": state["conflict_records"] / valid if valid else 0.0,
                "missing_rate": state["missing_records"] / valid if valid else 0.0},
            "siglip": {"prompt_bank_hashes": hashes, "model_revisions": dict(state["model_revisions"]),
                         "raw_statistics": aggregate_stats}, "bad_records": state["bad_records"]}


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tag", "metric", "count", "mean", "min", "max"])
        writer.writeheader()
        for key, values in sorted(report["siglip"]["raw_statistics"].items()):
            tag, metric = key.split(".", 1)
            writer.writerow({"tag": tag, "metric": metric, **values})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def run(*, db_path: str | None = None, jsonl_path: str | None = None, output_dir: str = "output",
        config_path: str | None = None, checkpoint_every: int = 250, stop_after: int | None = None) -> dict[str, Any]:
    if bool(db_path) == bool(jsonl_path):
        raise ValueError("provide exactly one of db_path or jsonl_path")
    source = Path(db_path or jsonl_path or "")
    kind = "db" if db_path else "jsonl"
    identity = _source_identity(source, kind)
    config_hash = _sha256(Path(config_path)) if config_path else None
    out = Path(output_dir)
    checkpoint = out / "scene-shadow.checkpoint.json"
    state = _load_checkpoint(checkpoint, identity, config_hash)
    iterator = _iter_db(source) if kind == "db" else _iter_jsonl(source)
    completed = True
    for index, anon_id, raw in iterator:
        if index < state["next_index"]:
            continue
        _consume(state, anon_id, raw)
        state["next_index"] = index + 1
        if checkpoint_every > 0 and state["seen"] % checkpoint_every == 0:
            _atomic_json(checkpoint, _checkpoint_payload(state))
        if stop_after is not None and state["seen"] >= stop_after:
            completed = False
            break
    _atomic_json(checkpoint, _checkpoint_payload(state))
    provenance = {"input": identity, "config_sha256": config_hash, "checkpoint_used": checkpoint.name,
                  "offline": True, "source_paths_omitted": True, "production_thresholds": "not_set"}
    report = _summary(state, provenance)
    if completed:
        _atomic_json(out / "scene-shadow-summary.json", report)
        _write_csv(out / "scene-shadow-tag-statistics.csv", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline SigLIP shadow calibration report; reads records only.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", help="SQLite inventory DB; only features.quality_meta is read")
    source.add_argument("--jsonl", help="JSONL routing records or {routing: record} objects")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--config", default=None, help="optional config file hashed for provenance")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    args = parser.parse_args(argv)
    report = run(db_path=args.db, jsonl_path=args.jsonl, output_dir=args.output_dir, config_path=args.config,
                 checkpoint_every=args.checkpoint_every)
    print(f"[scene-shadow] {report['status']} valid={report['records']['valid']} invalid={report['records']['invalid']}")
    return 2 if report["status"] != "OK" else 0


if __name__ == "__main__":
    sys.exit(main())
