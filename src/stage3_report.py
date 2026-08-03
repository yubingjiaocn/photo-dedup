"""Stage 3 -- review report + delete manifests.

Turns the ``groups`` table into artifacts in ``paths.output_dir``:

* ``review.html``       -- paged review UI (ALL timeline / MAYBE / UNKNOWN /
  GROUPS), rendered by :mod:`src.review_page`. Thumbnails come from
  ``output/thumbs`` on the SSD, which Stage 1 wrote from its single decode of
  each photo; the UI never reads an original.
* ``review_index`` rows -- compact ordered pagination index inside the DB, so
  a 100k-photo library pages with LIMIT/OFFSET instead of one giant JSON.
* ``review_summary.json`` -- small counters for the page header.
* ``delete_local.txt``  -- absolute paths to delete locally, one per line.
  Motion-photo sidecar videos are appended automatically (jpg dies -> mp4 dies).
* ``delete_cloud.json`` -- [{filename, exif_datetime, size_bytes}] consumed by
  scripts/gptk_delete.js to trash the same photos in Google Photos.
* ``summary.txt``       -- group count, delete count, GB reclaimed.

Nothing is deleted here; this stage only *proposes*. Deletion is stage 4
(``execute_local.py`` + the GPTK script).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Any, Dict, List, Optional

from .config import load_config
from . import db
from .review_page import (  # noqa: F401 (cached_thumb_uri re-exported for callers)
    DEFAULT_REVIEW_LIMIT,
    cached_thumb_uri,
    render_html,
    rewrite_performance_panel,
)


# --- delete-list expansion -------------------------------------------------

def _partner_path(conn, file_id: Optional[int]) -> Optional[str]:
    if not file_id:
        return None
    row = db.get_file(conn, int(file_id))
    return row["path"] if row else None


def collect_deletions(conn) -> Dict[str, Any]:
    """Collect only AUTO_REMOVE manifests; retain review queues separately."""
    groups_out: List[Dict[str, Any]] = []
    delete_paths: List[str] = []
    cloud_items: List[Dict[str, Any]] = []
    total_delete_bytes = 0
    seen_delete: set[str] = set()
    queues: Dict[str, List[Dict[str, Any]]] = {"MAYBE": [], "UNKNOWN": []}

    for grp in db.iter_groups(conn):
        members = db.group_members(conn, grp["id"])
        keep_rec = None
        del_recs: List[Dict[str, Any]] = []
        for m in members:
            rec = {
                "file_id": m["file_id"], "group_id": grp["id"],
                "path": m["path"], "basename": m["basename"],
                "size_bytes": m["size_bytes"] or 0, "width": m["width"], "height": m["height"],
                "exif_datetime": m["exif_datetime"], "quality_score": m["quality_score"],
                "face_count": m["face_count"], "reason": m["reason"],
                "motion_partner_id": m["motion_partner_id"], "is_keep": bool(m["is_keep"]),
                "decision": m["decision"] or ("KEEP" if m["is_keep"] else "UNKNOWN"),
                "confidence": m["confidence"], "evidence": _json_obj(m["evidence_json"]),
                "quality_meta": _json_obj(m["quality_meta"]),
            }
            if rec["decision"] == "KEEP":
                keep_rec = rec
            elif rec["decision"] == "AUTO_REMOVE":
                del_recs.append(rec)
            elif rec["decision"] in queues:
                rec["risk"] = _risk(rec)
                queues[rec["decision"]].append(rec)

        for rec in del_recs:
            for p, size in _expand_with_partner(conn, rec):
                if p in seen_delete:
                    continue
                seen_delete.add(p)
                delete_paths.append(p)
                total_delete_bytes += size
            # Cloud manifest: only the still image itself (Google merges motion).
            # Filename alone is not unique in Google Photos. Omit cloud items
            # unless both capture time and byte size are present.
            if rec["exif_datetime"] and rec["size_bytes"] > 0:
                cloud_items.append({
                    "filename": rec["basename"],
                    "exif_datetime": _iso_t(rec["exif_datetime"]),
                    "size_bytes": rec["size_bytes"],
                })

        groups_out.append({
            "id": grp["id"], "type": grp["group_type"],
            "keep": keep_rec, "deletes": del_recs, "member_count": grp["member_count"],
        })

    return {
        "groups": groups_out,
        "delete_paths": delete_paths,
        "cloud_items": cloud_items,
        "total_delete_bytes": total_delete_bytes,
        "queues": {k: sorted(v, key=lambda x: x["risk"], reverse=True) for k, v in queues.items()},
    }


def _json_obj(raw: Optional[str]) -> Dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _risk(rec: Dict[str, Any]) -> float:
    ev = rec.get("evidence") or {}
    margin = abs(float(ev.get("pair_margin", 0.0)))
    exposure = (rec.get("quality_meta") or {}).get("exposure", {})
    return max(0.0, 1.0 - margin, float(exposure.get("clip_hi", 0.0)),
               float(exposure.get("clip_lo", 0.0)))


def _expand_with_partner(conn, rec: Dict[str, Any]) -> List[tuple]:
    """Expand only a uniquely/bidirectionally owned, unprotected sidecar."""
    out = [(rec["path"], rec["size_bytes"])]
    partner = db.get_file(conn, int(rec["motion_partner_id"])) if rec["motion_partner_id"] else None
    if partner is not None and partner["motion_partner_id"] == rec["file_id"]:
        owner_count = conn.execute(
            "SELECT COUNT(*) FROM files WHERE motion_partner_id = ?", (partner["id"],)
        ).fetchone()[0]
        protected = conn.execute(
            "SELECT 1 FROM group_members WHERE file_id = ? AND (is_keep = 1 OR decision = 'KEEP') LIMIT 1",
            (partner["id"],),
        ).fetchone()
        protected_ref = conn.execute(
            "SELECT 1 FROM files f JOIN group_members gm ON gm.file_id=f.id "
            "WHERE f.motion_partner_id=? AND f.id != ? AND (gm.is_keep=1 OR gm.decision='KEEP') LIMIT 1",
            (partner["id"], rec["file_id"]),
        ).fetchone()
        if owner_count != 1 or protected or protected_ref:
            return out
        out.append((partner["path"], partner["size_bytes"] or 0))
    return out


def _iso_t(exif_dt: Optional[str]) -> Optional[str]:
    """'YYYY-MM-DD HH:MM:SS' -> 'YYYY-MM-DDTHH:MM:SS' (cloud matcher format)."""
    if not exif_dt:
        return None
    return exif_dt.replace(" ", "T")



def build_review_indexes(conn, data: Dict[str, Any]) -> Dict[str, int]:
    """Write the compact pagination index for every view (no image reads)."""
    counts = {"ALL": db.build_all_view_index(conn)}
    for view in ("MAYBE", "UNKNOWN"):
        counts[view] = db.replace_review_index(conn, view, [
            {"file_id": rec["file_id"], "group_id": rec.get("group_id"),
             "decision": view, "risk": rec.get("risk")}
            for rec in data["queues"][view]
        ])
    counts["GROUPS"] = db.count_groups(conn)
    return counts


# --- orchestration ---------------------------------------------------------

def run(
    config_path: Optional[str] = None,
    no_thumbs: bool = False,
    review_limit: int = DEFAULT_REVIEW_LIMIT,
    performance_panel: str = "",
) -> Dict[str, Any]:
    if review_limit < 1:
        raise ValueError("review_limit must be at least 1")
    cfg = load_config(config_path)
    conn = db.open_db(cfg.db_path)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect_deletions(conn)
    data["decision_stats"] = _json_obj(db.get_meta(conn, "stage2_stats", "{}"))
    view_counts = build_review_indexes(conn, data)
    conn.commit()
    thumb_stats = db.thumbnail_stats(conn)
    thumb_failures = db.thumbnail_failures(conn)

    # delete_local.txt
    local_txt = out_dir / "delete_local.txt"
    with open(local_txt, "w", encoding="utf-8") as fh:
        for p in data["delete_paths"]:
            fh.write(p + "\n")

    stage2_run = db.get_meta(conn, "stage2_run_id") or db.get_meta(conn, "stage2_done_at")
    groups = list(db.iter_groups(conn))
    policy = groups[0]["policy_version"] if groups else None
    manifest_meta = {
        "schema": 1, "stage2_run_id": stage2_run, "policy_version": policy,
        "count": len(data["delete_paths"]),
        "paths_sha256": hashlib.sha256(local_txt.read_bytes()).hexdigest(),
    }
    with open(out_dir / "delete_local.meta.json", "w", encoding="utf-8") as fh:
        json.dump(manifest_meta, fh, indent=2)

    # delete_cloud.json
    cloud_json = out_dir / "delete_cloud.json"
    with open(cloud_json, "w", encoding="utf-8") as fh:
        json.dump(data["cloud_items"], fh, ensure_ascii=False, indent=2)

    # review.html: paged UI; the static fallback uses only cached SSD thumbs.
    review = out_dir / "review.html"
    html_str = render_html(
        data, out_dir,
        review_limit=0 if no_thumbs else review_limit,
        performance_panel=performance_panel,
    )
    with open(review, "w", encoding="utf-8") as fh:
        fh.write(html_str)

    summary_payload = {
        "schema": 1,
        "views": view_counts,
        "groups": len(data["groups"]),
        "delete_files": len(data["delete_paths"]),
        "reclaim_bytes": data["total_delete_bytes"],
        "thumbnails": thumb_stats,
        "thumbnail_failures": thumb_failures,
        "page_sizes": [50, 100, 200],
        "skipped_oversize": int(db.get_meta(conn, "stage1_skipped_oversize", "0") or 0),
        "skipped_pixel_limit": int(
            db.get_meta(conn, "stage1_skipped_pixel_limit", "0") or 0
        ),
        "skipped_aspect_ratio": int(
            db.get_meta(conn, "stage1_skipped_aspect_ratio", "0") or 0
        ),
    }
    with open(out_dir / "review_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)

    # summary.txt
    gb = data["total_delete_bytes"] / (1024 ** 3)
    summary = out_dir / "summary.txt"
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("Photo Dedup - Summary\n")
        fh.write(f"groups: {len(data['groups'])}\n")
        fh.write(f"files to delete (incl. motion sidecars): {len(data['delete_paths'])}\n")
        fh.write(f"cloud items to trash: {len(data['cloud_items'])}\n")
        fh.write(f"maybe review: {len(data['queues']['MAYBE'])}\n")
        fh.write(f"unknown review: {len(data['queues']['UNKNOWN'])}\n")
        stats = data["decision_stats"]
        fh.write(f"auto coverage: {float(stats.get('auto_coverage', 0)):.1%}\n")
        fh.write("decision reasons: " + json.dumps(stats.get("reasons", {}), ensure_ascii=False) + "\n")
        fh.write(f"reclaimable: {gb:.2f} GB ({data['total_delete_bytes']} bytes)\n")
        fh.write(f"ALL timeline entries: {view_counts['ALL']}\n")
        fh.write(f"skipped oversize: {summary_payload['skipped_oversize']}\n")
        fh.write(f"skipped pixel limit: {summary_payload['skipped_pixel_limit']}\n")
        fh.write(f"skipped aspect ratio: {summary_payload['skipped_aspect_ratio']}\n")
        fh.write(
            f"thumbnails cached: {thumb_stats['recorded_ok']} "
            f"({thumb_stats['recorded_bytes'] / (1024 ** 3):.2f} GiB), "
            f"failed: {thumb_stats['recorded_failed']}\n"
        )

    db.set_meta(conn, "stage3_done_at", str(int(time.time())))
    conn.commit()
    conn.close()
    print(
        f"[stage3] wrote {review.name}, {local_txt.name}, {cloud_json.name}, "
        f"review_summary.json, {summary.name} -> {out_dir}"
    )
    if thumb_stats["recorded_failed"]:
        print(
            f"[stage3][WARN] {thumb_stats['recorded_failed']} photo(s) have no thumbnail; "
            "their review tiles say so explicitly (the UI never re-reads the HDD)"
        )
    return {
        "groups": len(data["groups"]),
        "delete_files": len(data["delete_paths"]),
        "cloud_items": len(data["cloud_items"]),
        "maybe": len(data["queues"]["MAYBE"]),
        "unknown": len(data["queues"]["UNKNOWN"]),
        "all_items": view_counts["ALL"],
        "views": view_counts,
        "thumbnails": thumb_stats,
        "reclaim_gb": gb,
        "output_dir": str(out_dir),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3: review report + delete lists")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--no-thumbs", action="store_true",
        help="omit the static fallback tiles (the paged UI is unaffected)",
    )
    ap.add_argument(
        "--review-limit", type=int, default=DEFAULT_REVIEW_LIMIT,
        help=f"static fallback tile count (default: {DEFAULT_REVIEW_LIMIT})",
    )
    args = ap.parse_args(argv)
    run(config_path=args.config, no_thumbs=args.no_thumbs, review_limit=args.review_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
