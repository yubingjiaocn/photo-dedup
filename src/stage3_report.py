"""Stage 3 -- review report + delete manifests.

Turns the ``groups`` table into four artifacts in ``paths.output_dir``:

* ``review.html``       -- one section per group: the keeper (larger thumb) vs
  the delete candidates, with scores. For eyeballing a sample before deleting.
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
import base64
import hashlib
import html
import io
import json
import sys
import time
from typing import Any, Dict, List, Optional

from .config import load_config
from . import db

DEFAULT_REVIEW_LIMIT = 1000


# --- thumbnails ------------------------------------------------------------

def thumb_data_uri(path: str, max_px: int = 200) -> Optional[str]:
    """Return a base64 JPEG data URI thumbnail, or None if unreadable."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            if img.format == "JPEG":
                img.draft("RGB", (max_px, max_px))
            img = img.convert("RGB")
            img.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


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
                "file_id": m["file_id"], "path": m["path"], "basename": m["basename"],
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


# --- HTML ------------------------------------------------------------------

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Photo Dedup - Review</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:20px;background:#111;color:#eee}
 h1{font-size:20px} .stats{color:#9cf;margin-bottom:16px}
 .group{border:1px solid #333;border-radius:8px;padding:12px;margin-bottom:18px;background:#1a1a1a}
 .gtitle{font-weight:bold;color:#fc9;margin-bottom:8px}
 .row{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}
 .keep{border:2px solid #4c9;padding:6px;border-radius:6px}
 .del{border:2px solid #c55;padding:6px;border-radius:6px;opacity:.9}
 .maybe{border-color:#fc3}.unknown{border-color:#999}
 .cap{font-size:11px;color:#bbb;max-width:210px;word-break:break-all}
 img{display:block;border-radius:4px}
 .tag{font-size:11px;padding:2px 6px;border-radius:4px;color:#000}
 .tk{background:#4c9}.td{background:#c55}
</style></head><body>
"""


def _img_block(rec: Dict[str, Any], max_px: int, is_keep: bool, thumbs: bool = True) -> str:
    uri = thumb_data_uri(rec["path"], max_px) if thumbs else None
    decision = rec.get("decision", "KEEP" if is_keep else "AUTO_REMOVE")
    tag = f'<span class="tag {"tk" if decision == "KEEP" else "td"}">{decision}</span>'
    cls = "keep" if decision == "KEEP" else f'del {decision.lower()}'
    if uri:
        img = f'<img src="{uri}" width="{max_px}">'
    else:
        note = "[unreadable]" if thumbs else "[thumb skipped]"
        img = '<div style="width:%dpx;color:#888">%s</div>' % (max_px, note)
    score = rec.get("quality_score")
    score_txt = f"{score:.1f}" if isinstance(score, (int, float)) else "?"
    exp = (rec.get("quality_meta") or {}).get("exposure", {})
    exp_txt = ""
    if exp:
        exp_txt = (
            f'<br>clip={float(exp.get("clip_hi", 0)):.1%}/{float(exp.get("clip_lo", 0)):.1%} '
            f'anchor={float(exp.get("anchor_mass", 0)):.1%} '
            f'entropy={float(exp.get("entropy_nonclip", 0)):.2f}'
        )
    cap = (
        f'{tag}<br>{html.escape(rec["basename"])}<br>'
        f'{rec["width"]}x{rec["height"]} | q={score_txt} | faces={rec["face_count"]}{exp_txt}<br>'
        f'{html.escape(str(rec["reason"] or ""))}'
    )
    return f'<div class="{cls}">{img}<div class="cap">{cap}</div></div>'


def render_html(
    data: Dict[str, Any], thumbs: bool = True, review_limit: int = DEFAULT_REVIEW_LIMIT
) -> str:
    if review_limit < 1:
        raise ValueError("review_limit must be at least 1")
    parts = [_HTML_HEAD]
    n_groups = len(data["groups"])
    gb = data["total_delete_bytes"] / (1024 ** 3)
    n_del = len(data["delete_paths"])
    parts.append("<h1>Photo Dedup - Review</h1>")
    parts.append(
        f'<div class="stats">{n_groups} groups &middot; {n_del} files to delete '
        f'&middot; ~{gb:.2f} GB reclaimable</div>'
    )
    for g in data["groups"]:
        parts.append('<div class="group">')
        parts.append(
            f'<div class="gtitle">Group #{g["id"]} [{g["type"]}] '
            f'({g["member_count"]} items)</div><div class="row">'
        )
        if g["keep"]:
            parts.append(_img_block(g["keep"], 220, True, thumbs))
        for d in g["deletes"]:
            parts.append(_img_block(d, 150, False, thumbs))
        parts.append("</div></div>")
    for state in ("MAYBE", "UNKNOWN"):
        queue = data["queues"][state]
        shown = queue[:review_limit]
        omitted = len(queue) - len(shown)
        parts.append(
            f'<h1>{state} review queue</h1>'
            f'<div class="stats">total: {len(queue)} &middot; shown: {len(shown)} '
            f'&middot; omitted: {omitted}</div>'
        )
        parts.append('<div class="row">')
        for rec in shown:
            parts.append(_img_block(rec, 150, False, thumbs))
        parts.append('</div>')
    parts.append("</body></html>")
    return "".join(parts)


# --- orchestration ---------------------------------------------------------

def run(
    config_path: Optional[str] = None,
    no_thumbs: bool = False,
    review_limit: int = DEFAULT_REVIEW_LIMIT,
) -> Dict[str, Any]:
    if review_limit < 1:
        raise ValueError("review_limit must be at least 1")
    cfg = load_config(config_path)
    conn = db.open_db(cfg.db_path)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect_deletions(conn)
    data["decision_stats"] = _json_obj(db.get_meta(conn, "stage2_stats", "{}"))

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

    # review.html (base64 thumbnails inline unless --no-thumbs)
    review = out_dir / "review.html"
    html_str = render_html(data, thumbs=not no_thumbs, review_limit=review_limit)
    with open(review, "w", encoding="utf-8") as fh:
        fh.write(html_str)

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

    db.set_meta(conn, "stage3_done_at", str(int(time.time())))
    conn.commit()
    conn.close()
    print(
        f"[stage3] wrote {review.name}, {local_txt.name}, {cloud_json.name}, "
        f"{summary.name} -> {out_dir}"
    )
    return {
        "groups": len(data["groups"]),
        "delete_files": len(data["delete_paths"]),
        "cloud_items": len(data["cloud_items"]),
        "maybe": len(data["queues"]["MAYBE"]),
        "unknown": len(data["queues"]["UNKNOWN"]),
        "reclaim_gb": gb,
        "output_dir": str(out_dir),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3: review report + delete lists")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-thumbs", action="store_true", help="skip base64 thumbnails")
    ap.add_argument(
        "--review-limit", type=int, default=DEFAULT_REVIEW_LIMIT,
        help="maximum thumbnails shown in each MAYBE/UNKNOWN queue (default: 1000)",
    )
    args = ap.parse_args(argv)
    run(config_path=args.config, no_thumbs=args.no_thumbs, review_limit=args.review_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
