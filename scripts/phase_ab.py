#!/usr/bin/env python3
"""Read-only baseline/candidate A/B over existing cached inventory features.

No source-media path is opened. SQLite is opened with mode=ro&immutable=1 and
all artifacts are written to a separate output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import phase_selection as PS

_REQUIRED_TABLES = {"files", "features", "groups", "group_members"}
_REQUIRED_FEATURES = {
    "file_id", "quality_score", "quality_meta", "face_count", "faces_json",
    "dinov2_embedding", "phash", "content_sha256",
}


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def open_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
        raise RuntimeError("SQLite query_only is not active")
    if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise RuntimeError("source inventory failed quick_check")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not _REQUIRED_TABLES <= tables:
        raise RuntimeError(f"inventory schema missing tables: {sorted(_REQUIRED_TABLES-tables)}")
    feature_columns = {row[1] for row in conn.execute("PRAGMA table_info(features)")}
    if not _REQUIRED_FEATURES <= feature_columns:
        raise RuntimeError(f"inventory schema missing feature columns: {sorted(_REQUIRED_FEATURES-feature_columns)}")
    return conn


def _groups(conn: sqlite3.Connection) -> list[tuple[sqlite3.Row, list[dict[str, Any]]]]:
    output = []
    groups = conn.execute(
        "SELECT id, group_type, keep_file_id, member_count FROM groups ORDER BY id"
    ).fetchall()
    for group in groups:
        members = [dict(row) for row in conn.execute(
            """
            SELECT f.id, f.basename, f.size_bytes, f.exif_timestamp, f.width, f.height,
                   fe.phash, fe.content_sha256, fe.dinov2_embedding, fe.quality_score,
                   fe.quality_meta, fe.face_count, fe.faces_json, gm.decision
            FROM group_members gm JOIN files f ON f.id=gm.file_id
            JOIN features fe ON fe.file_id=f.id WHERE gm.group_id=?
            ORDER BY COALESCE(f.exif_timestamp, f.mtime_ns/1000000000), f.id
            """, (group["id"],)
        )]
        if len(members) != int(group["member_count"]):
            raise RuntimeError(f"group {group['id']} member_count mismatch")
        output.append((group, members))
    return output


def run(
    name: str, source: Path, output: Path, *, source_sha: str | None = None,
) -> dict[str, Any]:
    before = fingerprint(source)
    conn = open_readonly(source)
    rows: list[dict[str, Any]] = []
    summary = Counter()
    summary["inventory_files"] = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
    for group, members in _groups(conn):
        summary["groups"] += 1
        summary["members"] += len(members)
        group_type = str(group["group_type"])
        try:
            baseline_keeper = next(
                i for i, member in enumerate(members)
                if int(member["id"]) == int(group["keep_file_id"])
            )
        except StopIteration as exc:
            raise RuntimeError(f"group {group['id']} keep_file_id is outside its members") from exc
        baseline = {baseline_keeper}
        if group_type == "sha_exact":
            candidate = set(baseline)
            phase_records = [{"phase_id": 0, "members": list(range(len(members))),
                              "keepers": list(candidate), "review_required": False}]
            review_required = False
        else:
            phases = PS.segment_phases(members, group_type=group_type)
            selected = PS.select_phase_keepers(members, phases)
            candidate = set(selected["keepers"])
            phase_records = selected["phases"]
            review_required = bool(selected["review_required"])
        base_ids = [int(members[i]["id"]) for i in sorted(baseline)]
        cand_ids = [int(members[i]["id"]) for i in sorted(candidate)]
        changed = baseline != candidate
        if changed:
            summary["changed_groups"] += 1
        if len(candidate) > len(baseline):
            summary["extra_keeper_groups"] += 1
        summary["baseline_keepers"] += len(baseline)
        summary["candidate_keepers"] += len(candidate)
        summary["candidate_logical_phases"] += len(phase_records)
        if len(phase_records) > 1:
            summary["multi_phase_groups"] += 1
        # Review load is group/member work, not every absent optional field.
        if review_required:
            summary["evidence_review_groups"] += 1
            summary["evidence_review_members"] += len(members)
        # Every changed keeper set is primary user decision work, including a
        # two-frame replacement.  Unchanged optional-evidence gaps remain
        # available as secondary diagnostics without flooding the main queue.
        primary_review = changed
        secondary_review = review_required and not changed
        if primary_review:
            summary["ab_review_groups"] += 1
            summary["ab_review_members"] += len(members)
        if secondary_review:
            summary["secondary_review_groups"] += 1
            summary["secondary_review_members"] += len(members)
        thumb_root = source.parent / "thumbs"
        rows.append({
            "dataset": name, "group_id": int(group["id"]), "group_type": group_type,
            "member_count": len(members), "member_ids": [int(item["id"]) for item in members],
            "baseline_keeper_ids": base_ids, "candidate_keeper_ids": cand_ids,
            "logical_phase_count": len(phase_records),
            "logical_phases": [{
                "phase_id": item["phase_id"],
                "member_ids": [int(members[i]["id"]) for i in item["members"]],
                "keeper_ids": [int(members[i]["id"]) for i in item["keepers"]],
                "reason_codes": item.get("reason_codes", []),
            } for item in phase_records],
            "changed": changed, "review_required": primary_review,
            "review_tier": "primary" if primary_review else ("secondary" if secondary_review else "none"),
            "review_reason": "CANDIDATE_CHANGED_HIGH_IMPACT" if primary_review else ("EVIDENCE_GAP_DIAGNOSTIC" if secondary_review else ""),
            "thumbnail_paths": [str(thumb_root / f"{item['id']}.jpg") for item in members],
        })
    conn.close()
    after = fingerprint(source)
    if before != after:
        raise RuntimeError("source inventory changed during read-only A/B")
    member_count = summary["members"]
    result = {
        "schema_version": 3, "dataset": name, "source_inventory": str(source),
        "source_sha": source_sha,
        "source_fingerprint_before": before, "source_fingerprint_after": after,
        "sqlite_mode": "ro+immutable", "opened_original_media": False,
        "physical_group_split": False,
        "phase_semantics": "logical_phase_protection_within_existing_db_group",
        "authority": "shadow_review_only", "auto_remove_authority": "BYTE_IDENTICAL_ONLY",
        "metrics": {
            **dict(summary),
            "retention_scope": "existing_group_members_only",
            "baseline_retention": summary["baseline_keepers"] / member_count if member_count else 0.0,
            "candidate_retention": summary["candidate_keepers"] / member_count if member_count else 0.0,
            "baseline_inventory_retention": (
                summary["inventory_files"] - member_count + summary["baseline_keepers"]
            ) / summary["inventory_files"] if summary["inventory_files"] else 0.0,
            "candidate_inventory_retention": (
                summary["inventory_files"] - member_count + summary["candidate_keepers"]
            ) / summary["inventory_files"] if summary["inventory_files"] else 0.0,
            "evidence_review_group_rate": summary["evidence_review_groups"] / summary["groups"] if summary["groups"] else 0.0,
            "evidence_review_member_rate": summary["evidence_review_members"] / member_count if member_count else 0.0,
            "ab_review_group_rate": summary["ab_review_groups"] / summary["groups"] if summary["groups"] else 0.0,
            "ab_review_member_rate": summary["ab_review_members"] / member_count if member_count else 0.0,
            "primary_review_groups": summary["ab_review_groups"],
            "primary_review_members": summary["ab_review_members"],
            "secondary_review_groups": summary["secondary_review_groups"],
            "secondary_review_members": summary["secondary_review_members"],
        },
        "groups": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / f"{name}-groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "group_id", "group_type", "member_count", "baseline_keeper_ids",
            "candidate_keeper_ids", "logical_phase_count", "changed", "review_required", "review_reason",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    review = [row for row in rows if row["review_tier"] == "primary"]
    secondary = [row for row in rows if row["review_tier"] == "secondary"]
    (output / f"{name}-review-queue.json").write_text(
        json.dumps({"dataset": name, "count": len(review), "groups": review}, indent=2),
        encoding="utf-8",
    )
    (output / f"{name}-secondary-diagnostics.json").write_text(
        json.dumps({"dataset": name, "count": len(secondary), "groups": secondary}, indent=2),
        encoding="utf-8",
    )
    cards = []
    for row in review:
        keepers = set(row["candidate_keeper_ids"])
        baseline_ids = set(row["baseline_keeper_ids"])
        images = []
        for member_id, thumb in zip(row["member_ids"], row["thumbnail_paths"], strict=True):
            badges = (["candidate"] if member_id in keepers else []) + (["baseline"] if member_id in baseline_ids else [])
            images.append(
                f'<figure class="{" ".join(badges)}"><img src="{Path(thumb).as_uri()}" '
                f'alt="cached thumbnail {member_id}"><figcaption>{member_id} '
                f'{" / ".join(badges) or "review"}</figcaption></figure>'
            )
        cards.append(
            f'<section><h2>Group {row["group_id"]} · {row["group_type"]} · '
            f'{row["logical_phase_count"]} logical phases</h2>'
            f'<p>{row["review_reason"]}; baseline={row["baseline_keeper_ids"]}; '
            f'candidate={row["candidate_keeper_ids"]}</p><div class="strip">{"".join(images)}</div></section>'
        )
    html = f'''<!doctype html><meta charset="utf-8"><title>{name} phase A/B review</title>
<style>body{{font:14px system-ui;margin:20px;background:#111;color:#eee}}section{{border-top:1px solid #555;padding:14px 0}}.strip{{display:flex;flex-wrap:wrap;gap:8px}}figure{{margin:0;padding:4px;border:2px solid #555}}figure.baseline{{border-color:#49f}}figure.candidate{{box-shadow:0 0 0 3px #4c6 inset}}img{{width:180px;height:135px;object-fit:contain;background:#222}}figcaption{{max-width:180px}}</style>
<h1>{name}: cached-thumbnail logical-phase A/B review</h1>
<p>Green inset = candidate keeper; blue border = existing inventory baseline keeper. Cached thumbnails only; no original media is opened.</p>
{"".join(cards)}'''
    (output / f"{name}-review.html").write_text(html, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "INVENTORY"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha")
    args = parser.parse_args()
    reports = [run(name, Path(path).resolve(), args.output.resolve(), source_sha=args.source_sha)
               for name, path in args.dataset]
    aggregate = {
        "schema_version": 3, "source_sha": args.source_sha,
        "datasets": [item["dataset"] for item in reports],
        "reports": {item["dataset"]: item["metrics"] for item in reports},
        "constraints": {"source_read_only": True, "original_media_opened": False,
                        "lin_held_out_touched": False, "writeback": False},
    }
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
