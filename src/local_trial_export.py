"""Export saved human review choices as data-only records for later learning.

Usage::

    python -m src.local_trial_export --output "D:\\photo-review\\2026-10-01"

Reads ``inventory.sqlite`` (read-only URI) and ``review_state.json`` from one
existing local review output and writes, inside that same output directory::

    local_trial_export/selections-<UTC stamp>.jsonl   one record per group member
    local_trial_export/selections-<UTC stamp>.manifest.json   counts + sha256

Each record says, for a human-decided group, whether that photo was selected or
not, plus its byte identity (``content_sha256``, size, mtime) -- never a path,
never pixels. Nothing is uploaded, copied, trained, or sent to a model; originals
are never opened. Decisions whose group membership or stored SHA no longer match
the inventory are reported as stale and left out instead of being exported.

This is a local trial format (``local-trial-selection-v1``). It is *not* the
frozen cloud194 learning registry, and nothing here registers new dates there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import db, review_state, root_scope

FORMAT = "local-trial-selection-v1"
EXPORT_DIRNAME = "local_trial_export"


def _selected_ids(decision: Dict[str, Any], rows: List[sqlite3.Row]) -> Optional[List[int]]:
    action = decision.get("action")
    if action == "accept":
        return sorted(int(r["file_id"]) for r in rows if r["is_keep"])
    if action == "pick":
        return [int(decision["file_id"])]
    if action == "keep":
        return [int(i) for i in decision["file_ids"]]
    return None  # mark: not decided yet


def build_records(output: Path) -> Dict[str, Any]:
    """Return ``{"records": [...], "counts": {...}, "stale": [...]}``; no writes."""
    output = Path(output).resolve()
    db_path = output / "inventory.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"inventory DB not found: {db_path}")
    state = review_state.ReviewState(output)  # load only; never saved here
    if state.load_error:
        raise ValueError(f"review state unreadable, refusing to export: {state.load_error}")
    conn = sqlite3.connect(root_scope.readonly_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        scope = root_scope.recorded(conn)
        # Opaque root identity: lets later learning keep dates apart without
        # carrying the private folder path.
        root_id = (hashlib.sha256(scope.key.encode("utf-8")).hexdigest()[:16]
                   if scope.key else None)
        records: List[Dict[str, Any]] = []
        stale: List[Dict[str, Any]] = []
        counts = {"groups_exported": 0, "groups_later": 0, "groups_stale": 0,
                  "selected": 0, "nonselected": 0}
        for gid_str, decision in sorted(state.snapshot()["groups"].items(), key=lambda kv: int(kv[0])):
            gid = int(gid_str)
            if decision.get("action") == "mark":
                counts["groups_later"] += 1
                continue
            rows = db.group_page_by_id(conn, gid, scope=scope)
            reason = _stale_reason(decision, rows)
            if reason:
                counts["groups_stale"] += 1
                stale.append({"group_id": gid, "reason": reason})
                continue
            chosen = set(_selected_ids(decision, rows) or [])
            counts["groups_exported"] += 1
            for row in rows:
                fid = int(row["file_id"])
                selected = fid in chosen
                counts["selected" if selected else "nonselected"] += 1
                records.append({
                    "format": FORMAT,
                    "root_id": root_id,
                    "group_id": gid,
                    "group_type": row["group_type"],
                    "member_fingerprint": decision.get("member_fingerprint"),
                    "action": decision["action"],
                    "keep_all": bool(decision.get("keep_all", False)),
                    "decided_at": decision.get("timestamp"),
                    "file_id": fid,
                    "selected": selected,
                    "ai_keeper": bool(row["is_keep"]),
                    "content_sha256": row["content_sha256"],
                    "size_bytes": row["size_bytes"],
                    "mtime_ns": row["mtime_ns"],
                    "exif_datetime": row["exif_datetime"],
                    "basename": row["basename"],
                })
        return {"records": records, "counts": counts, "stale": stale}
    finally:
        conn.close()


def _stale_reason(decision: Dict[str, Any], rows: List[sqlite3.Row]) -> Optional[str]:
    if not rows:
        return "group no longer in scope"
    current_fp = review_state.compute_member_fingerprint([int(r["file_id"]) for r in rows])
    if decision.get("member_fingerprint") != current_fp:
        return "group members changed (or legacy decision without fingerprint)"
    members = {int(r["file_id"]): r["content_sha256"] for r in rows}
    chosen = _selected_ids(decision, rows)
    if not chosen:
        return "no selected photo in this scope"
    if any(fid not in members for fid in chosen):
        return "selected photo no longer a group member"
    if decision.get("action") == "keep":
        for fid, sha in zip(decision["file_ids"], decision["file_sha256"]):
            if sha is not None and members.get(fid) != sha:
                return f"content_sha256 changed for file_id {fid}"
    return None


def write_export(output: Path) -> Dict[str, Any]:
    output = Path(output).resolve()
    result = build_records(output)
    target = output / EXPORT_DIRNAME
    target.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    data_path = target / f"selections-{stamp}.jsonl"
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                      for r in result["records"]).encode("utf-8")
    data_path.write_bytes(payload)
    manifest = {
        "format": FORMAT,
        "created_utc": stamp,
        "data_file": data_path.name,
        "data_sha256": hashlib.sha256(payload).hexdigest(),
        "records": len(result["records"]),
        "counts": result["counts"],
        "stale": result["stale"],
        "contains_photos": False,
        "uploaded": False,
        "model_calls": 0,
        "registered_in_cloud194_registry": False,
    }
    manifest_path = target / f"selections-{stamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["data_path"] = str(data_path)
    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export saved human selections (data only, no photos, no upload).")
    parser.add_argument("--output", required=True, help="existing local review output directory")
    args = parser.parse_args(argv)
    try:
        manifest = write_export(Path(args.output).expanduser())
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"[export][ERROR] {exc}", file=sys.stderr)
        return 1
    counts = manifest["counts"]
    print(f"[export] {manifest['records']} records -> {manifest['data_path']}")
    print(f"[export] groups exported={counts['groups_exported']} later={counts['groups_later']} "
          f"stale={counts['groups_stale']}; selected={counts['selected']} "
          f"nonselected={counts['nonselected']}")
    print("[export] no photos copied, nothing uploaded, no model called")
    return 0


if __name__ == "__main__":
    sys.exit(main())
