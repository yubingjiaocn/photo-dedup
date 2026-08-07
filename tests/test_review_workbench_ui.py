"""Front-end boundary gates that still belong to Python, after the review UI
moved from an inlined script to a compiled Preact app under ``frontend/``.

What changed and why:

* The old node-DOM-stub tests here executed the *inlined* ``review.html`` script.
  That script no longer exists -- the UI is TypeScript compiled by Vite -- so its
  behaviour is now gated where it lives: reducer/keyboard/component logic by
  Vitest (``frontend/test/*``) and real-browser behaviour by the Chromium harness
  (``frontend/harness/browser_check.py``). See ``FRONTEND_DECISION.md``.
* Two checks in this file were never about the inline script; they are server/DB
  *contract* checks that the front end depends on. They stay, unchanged.
* Three new checks read the front-end **source** (now the source of truth) to keep
  the path-free / read-only-API boundary enforced in CI without a browser.
"""

from __future__ import annotations

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "src"


def _frontend_sources() -> str:
    """Concatenated front-end TypeScript/TSX source (the review UI's authority)."""
    parts = []
    for path in sorted(FRONTEND.rglob("*.ts")) + sorted(FRONTEND.rglob("*.tsx")):
        parts.append(path.read_text(encoding="utf-8"))
    assert parts, "no front-end sources found; is frontend/src present?"
    return "\n".join(parts)


# --- server/DB contract the front end consumes (preserved from the node-stub file) --

def test_public_api_still_returns_the_diagnostic_fields():
    """Only the UI decides where to show these; the API contract is unchanged."""
    from src import review_server

    row = {"file_id": 5, "group_id": 1, "decision": "MAYBE", "basename": "a.jpg",
           "width": 10, "height": 8, "size_bytes": 99, "exif_datetime": None,
           "file_kind": "jpg", "quality_score": 55.5, "face_count": 2,
           "reason": "LOW_MARGIN", "thumb_status": "ok", "thumb_error": None,
           "is_keep": 0}
    item = review_server._public_item(row)
    assert item["quality_score"] == 55.5
    assert item["face_count"] == 2
    assert item["reason"] == "LOW_MARGIN"
    assert "path" not in item


def test_flat_view_query_selects_is_keep_so_the_ai_keeper_is_labelled(tmp_path):
    """ALL/MAYBE/UNKNOWN rows must carry is_keep, or the AI badge silently vanishes."""
    from src import db

    conn = db.open_db(tmp_path / "inventory.sqlite")
    ids = []
    for i in range(3):
        ids.append(db.insert_file(conn, {
            "path": f"/photos/IMG_{i}.jpg", "basename": f"IMG_{i}.jpg",
            "size_bytes": 10 + i, "mtime_ns": 1, "exif_datetime": f"2026-01-01 12:00:0{i}",
            "exif_timestamp": 100 + i, "width": 10, "height": 10,
            "file_kind": "jpg", "scan_status": "done"}))
    keeper, dup, lone = ids
    db.insert_group(conn, "phash_near", keeper,
                    [(keeper, True, "keep"), (dup, False, "dup")], 1)
    db.build_all_view_index(conn)
    conn.commit()

    rows = {r["file_id"]: r for r in db.review_page(conn, "ALL", 0, 50)}
    conn.close()
    assert "is_keep" in rows[keeper].keys()
    assert rows[keeper]["is_keep"] == 1
    assert rows[dup]["is_keep"] == 0
    assert rows[lone]["is_keep"] is None


# --- front-end SOURCE boundary (new: the UI is TS now, gate it without a browser) --

def test_front_end_uses_only_the_read_only_local_api():
    """The UI talks to the read-only API plus the single /api/action mutation."""
    js = _frontend_sources()
    endpoints = sorted(set(re.findall(r"/api/[a-z]+", js)))
    assert endpoints == [
        "/api/action", "/api/group", "/api/locate", "/api/next",
        "/api/original", "/api/page", "/api/status", "/api/thumb",
    ], endpoints


def test_front_end_never_reads_a_source_path():
    """The browser only ever sees file_id/basename; no path field is referenced."""
    js = _frontend_sources()
    for forbidden in ("file://", ".path", "['path']", '"path"', "record.path"):
        assert forbidden not in js, f"front-end source references {forbidden!r}"


def test_front_end_declares_the_queues_browse_views_and_page_sizes():
    """The entry queues, browse views and page sizes the server serves are wired."""
    js = _frontend_sources()
    for token in ("PENDING", "LATER", "DONE", "ALL", "MAYBE", "UNKNOWN"):
        assert token in js, token
    for size in ("50", "100", "200"):
        assert size in js
    # The AI-compare/blink and undo vocabulary the reviewer relies on.
    assert "双栏对比" in js
    assert "撤销上一步" in js
