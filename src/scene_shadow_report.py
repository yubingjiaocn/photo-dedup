"""Offline, fail-closed aggregation for SigLIP shadow-routing records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, Mapping

from .routing_schema import SCENE_TAGS, SUBJECT_TAGS, validate_routing_record
from .scene_checkpoint import CHECKPOINT_SCHEMA_VERSION, equivalent, json_safe, new_state, restore_checkpoint

REPORT_SCHEMA_VERSION = 2
AUDIT_FIELDS = ("positive_mean", "positive_median", "positive_range", "positive_std", "hard_negative_max", "hard_negative_gap")
IDENTITY_KEYS = ("name", "revision", "model_sha256", "prompt_bank_hash", "bank_version", "routing_schema_version", "device", "precision")


class CalibrationIdentityError(ValueError):
    """A record cannot participate in calibration without a complete identity."""


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(path: Path, kind: str) -> dict[str, Any]:
    stat = path.stat()
    identity = {"kind": kind, "size_bytes": stat.st_size, "content_sha256": _sha256(path)}
    if kind == "jsonl":
        identity["source_name_hash"] = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    return identity


@contextmanager
def _fixed_source(path: Path, kind: str) -> Iterator[tuple[Path, dict[str, Any]]]:
    if kind == "jsonl":
        yield path, _source_identity(path, kind)
        return
    temp_dir = Path(tempfile.mkdtemp(prefix="scene-shadow-snapshot-"))
    snapshot = temp_dir / "snapshot.sqlite"
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    destination = sqlite3.connect(snapshot)
    try:
        source.backup(destination)
        destination.close()
        source.close()
        yield snapshot, _source_identity(snapshot, kind)
    finally:
        destination.close()
        source.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def _load_raw_checkpoint(path: Path, identity: Mapping[str, Any], config_hash: str | None) -> dict[str, Any] | None:
    try:
        return restore_checkpoint(json.loads(path.read_text(encoding="utf-8")), identity, config_hash)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _iter_jsonl(path: Path) -> Iterator[tuple[int, str, Any, bytes]]:
    with path.open("rb") as handle:
        for index, line in enumerate(handle):
            anon_id = f"jsonl:{index + 1}"
            try:
                value = json.loads(line)
                routing = value.get("routing", value) if isinstance(value, dict) else value
                yield index, anon_id, routing, line
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                yield index, anon_id, exc, line


def _iter_db(path: Path) -> Iterator[tuple[int, str, Any, bytes]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        query = "SELECT f.id, fe.quality_meta FROM features fe JOIN files f ON f.id=fe.file_id ORDER BY f.id"
        for index, (file_id, raw_meta) in enumerate(connection.execute(query)):
            canonical = json.dumps([file_id, raw_meta], ensure_ascii=False, separators=(",", ":")).encode()
            try:
                meta = json.loads(raw_meta)
                yield index, f"db:{file_id}", meta.get("routing") if isinstance(meta, dict) else None, canonical
            except (TypeError, json.JSONDecodeError) as exc:
                yield index, f"db:{file_id}", exc, canonical
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


def _record_error(state: dict[str, Any], anon_id: str, error: Exception, *, identity: bool = False) -> None:
    state["invalid"] += 1
    state["identity_errors"] += int(identity)
    if len(state["bad_records"]) < 20:
        state["bad_records"].append({"record_id": anon_id, "error": str(error)[:240]})


def _calibration_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    model = record["model"]
    runtime = model.get("runtime")
    if runtime is None:
        device = precision = "UNRECORDED"
    elif isinstance(runtime, Mapping):
        device, precision = runtime.get("device"), runtime.get("precision")
    else:
        raise CalibrationIdentityError("runtime identity must be an object")
    values = {
        "name": model.get("name"), "revision": model.get("revision"), "model_sha256": model.get("model_sha256"),
        "prompt_bank_hash": model.get("prompt_bank_hash"), "bank_version": model.get("bank_version"),
        "routing_schema_version": record.get("schema_version"), "device": device, "precision": precision,
    }
    if any(isinstance(values[key], bool) or not isinstance(values[key], (str, int)) or values[key] == "" for key in IDENTITY_KEYS):
        raise CalibrationIdentityError("missing complete calibration identity")
    if not isinstance(values["bank_version"], int) or values["bank_version"] < 1:
        raise CalibrationIdentityError("invalid bank_version identity")
    for key in ("model_sha256", "prompt_bank_hash"):
        value = values[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise CalibrationIdentityError(f"invalid {key} identity")
    identity = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return identity, f"{values['name']}@{values['revision']}"


def _consume(state: dict[str, Any], anon_id: str, raw: Any) -> None:
    state["seen"] += 1
    if isinstance(raw, Exception):
        _record_error(state, anon_id, raw)
        return
    try:
        record = validate_routing_record(raw)
        identity, revision = _calibration_identity(record)
        audit = record["model"].get("shadow_prompt_audit")
        tags = audit.get("tags") if isinstance(audit, Mapping) else None
        if not isinstance(tags, Mapping) or set(tags) != SCENE_TAGS | SUBJECT_TAGS:
            raise ValueError("shadow prompt audit does not cover routing schema tags")
        stat_values = []
        for tag, values in tags.items():
            if not isinstance(values, Mapping):
                raise ValueError(f"audit.{tag} must be an object")
            for field in AUDIT_FIELDS:
                key = f"{tag}.{field}"
                stat_values.append((key, _finite(values.get(field), key)))
        for key, value in stat_values:
            _add_stat(state, key, value)
        state["identities"][identity] += 1
        state["model_revisions"][revision] += 1
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
        state["conflict_records"] += int(bool(quality["conflicts"]))
        state["missing_records"] += int(bool(quality["missing"]))
        state["valid"] += 1
    except CalibrationIdentityError as exc:
        _record_error(state, anon_id, exc, identity=True)
    except (TypeError, ValueError, KeyError) as exc:
        _record_error(state, anon_id, exc)


def _advance_digest(state: dict[str, Any], raw_bytes: bytes) -> None:
    previous = bytes.fromhex(state["prefix_digest"])
    state["prefix_digest"] = hashlib.sha256(previous + raw_bytes).hexdigest()


def _validated_resume(iterator: Iterator[tuple[int, str, Any, bytes]], saved: dict[str, Any] | None,
                      identity: Mapping[str, Any], config_hash: str | None) -> tuple[dict[str, Any], list[tuple[int, str, Any, bytes]]]:
    state = new_state(identity, config_hash)
    buffered = []
    if saved is None:
        return state, buffered
    target = saved["next_index"]
    for item in iterator:
        if item[0] >= target:
            buffered.append(item)
            break
        _consume(state, item[1], item[2])
        _advance_digest(state, item[3])
        state["next_index"] = item[0] + 1
    if state["next_index"] != target or not equivalent(state, saved):
        return new_state(identity, config_hash), []
    return saved, buffered


def _summary(state: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    valid = state["valid"]
    identities = dict(state["identities"])
    if state["identity_errors"]:
        status = "REJECTED_MISSING_IDENTITY"
    elif len(identities) > 1:
        status = "REJECTED_MIXED_IDENTITY"
    else:
        status = "OK"
    aggregate = {key: {"count": item["count"], "mean": item["sum"] / item["count"], "min": item["min"], "max": item["max"]}
                 for key, item in state["stats"].items() if item["count"]}
    return {"report_schema_version": REPORT_SCHEMA_VERSION, "provenance": dict(provenance), "status": status,
            "records": {"seen": state["seen"], "valid": valid, "invalid": state["invalid"],
                        "invalid_rate": state["invalid"] / state["seen"] if state["seen"] else 0.0},
            "unknown": {key: {"count": value, "coverage": value / valid if valid else 0.0,
                              "reasons": dict(state["unknown_reasons"][key])} for key, value in state["unknown"].items()},
            "reasons": dict(state["reasons"]), "tag_present": dict(state["tag_present"]),
            "quality_evidence": {"conflict_records": state["conflict_records"], "missing_records": state["missing_records"],
                                 "conflict_rate": state["conflict_records"] / valid if valid else 0.0,
                                 "missing_rate": state["missing_records"] / valid if valid else 0.0},
            "siglip": {"calibration_identities": identities, "model_revisions": dict(state["model_revisions"]),
                       "raw_statistics": aggregate}, "bad_records": state["bad_records"]}


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["tag", "metric", "count", "mean", "min", "max"])
            writer.writeheader()
            for key, values in sorted(report["siglip"]["raw_statistics"].items()):
                tag, metric = key.split(".", 1)
                writer.writerow({"tag": tag, "metric": metric, **values})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def run(*, db_path: str | None = None, jsonl_path: str | None = None, output_dir: str = "output",
        config_path: str | None = None, checkpoint_every: int = 250, stop_after: int | None = None) -> dict[str, Any]:
    if bool(db_path) == bool(jsonl_path):
        raise ValueError("provide exactly one of db_path or jsonl_path")
    source, kind = Path(db_path or jsonl_path or ""), "db" if db_path else "jsonl"
    config_hash = _sha256(Path(config_path)) if config_path else None
    out, checkpoint = Path(output_dir), Path(output_dir) / "scene-shadow.checkpoint.json"
    with _fixed_source(source, kind) as (fixed, identity):
        saved = _load_raw_checkpoint(checkpoint, identity, config_hash)
        iterator = _iter_db(fixed) if kind == "db" else _iter_jsonl(fixed)
        state, buffered = _validated_resume(iterator, saved, identity, config_hash)
        prefix_verified = saved is not None and state is saved
        if saved is not None and state is not saved:
            iterator = _iter_db(fixed) if kind == "db" else _iter_jsonl(fixed)
        completed = True
        for index, anon_id, raw, raw_bytes in chain(buffered, iterator):
            if index < state["next_index"]:
                continue
            _consume(state, anon_id, raw)
            _advance_digest(state, raw_bytes)
            state["next_index"] = index + 1
            if checkpoint_every > 0 and state["seen"] % checkpoint_every == 0:
                _atomic_json(checkpoint, json_safe(state))
            if stop_after is not None and state["seen"] >= stop_after:
                completed = False
                break
        _atomic_json(checkpoint, json_safe(state))
        provenance = {"input": identity, "config_sha256": config_hash, "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                      "checkpoint_prefix_verified": prefix_verified, "sqlite_snapshot": kind == "db",
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
