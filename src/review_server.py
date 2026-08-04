"""Local-only paged review server for a 100k+ photo library.

Hard boundaries (the reason this file exists):

* ``/api/thumb/<id>.jpg`` reads **only** ``output/thumbs/<id>.jpg`` on the SSD,
  so normal paging never wakes the HDD. ``/api/original/<id>`` is the sole,
  explicit exception: a user click streams that one inventoried still image
  for focus inspection; source paths are never returned to the browser.
* Pages come from SQLite with ``LIMIT/OFFSET`` over the pre-built
  ``review_index`` table, so no giant JSON is loaded by the server or browser.
* Views: ``ALL`` (full timeline), ``MAYBE``, ``UNKNOWN``, ``GROUPS``.
* Page size is 50, 100 (default), or 200 -- an explicit allow-list.
* Binds ``127.0.0.1`` only; read-only, with no delete/move/keep operation.
* Every query is restricted to the photo root the output directory is bound to
  (:mod:`src.root_scope`), so a legacy database holding several roots still
  serves -- and counts -- only the current one.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from . import db, review_state, root_scope, thumbnails

DEFAULT_PAGE_SIZE = 100
PAGE_SIZES = (50, 100, 200)
VIEWS = ("ALL", "MAYBE", "UNKNOWN", "GROUPS")
_THUMB_RE = re.compile(r"^/api/thumb/(\d{1,18})\.jpg$")
_ORIGINAL_RE = re.compile(r"^/api/original/(\d{1,18})$")
_GROUP_RE = re.compile(r"^/api/group/(\d{1,18})$")
_ACTION_RE = re.compile(r"^/api/action$")
_DENY_NAMES = {"inventory.sqlite", "inventory.sqlite-wal", "inventory.sqlite-shm"}


@dataclass(frozen=True)
class OriginalRecord:
    path: Path
    size_bytes: int
    mtime_ns: int


def _public_item(row: Any) -> Dict[str, Any]:
    """Project a DB row to path-free display fields."""
    keys = set(row.keys()) if hasattr(row, "keys") else set(row)

    def get(key: str) -> Any:
        return row[key] if key in keys else None

    score = get("quality_score")
    return {
        "file_id": int(get("file_id")),
        "group_id": get("group_id"),
        "decision": get("decision") or "UNGROUPED",
        "basename": get("basename"),
        "width": get("width"),
        "height": get("height"),
        "size_bytes": get("size_bytes"),
        "exif_datetime": get("exif_datetime"),
        "file_kind": get("file_kind"),
        "quality_score": round(float(score), 1) if isinstance(score, (int, float)) else None,
        "face_count": get("face_count"),
        "reason": get("reason"),
        "thumb": get("thumb_status") or "missing",
        "thumb_error": get("thumb_error"),
        "is_keep": bool(get("is_keep")) if get("is_keep") is not None else None,
    }


def _group_items(rows: List[Any]) -> List[Dict[str, Any]]:
    """Fold flat (group, member) rows into one entry per group, order preserved."""
    groups: List[Dict[str, Any]] = []
    index: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        group_id = int(row["group_id"])
        entry = index.get(group_id)
        if entry is None:
            entry = {"group_id": group_id, "group_type": row["group_type"],
                     "member_count": row["member_count"], "members": []}
            index[group_id] = entry
            groups.append(entry)
        entry["members"].append(_public_item(row))
    return groups


class ReviewData:
    """Read-only, thread-safe paged access to the review DB and thumb cache."""

    def __init__(self, output: Path, db_path: Optional[Path] = None) -> None:
        self.output = Path(output).resolve()
        self.db_path = Path(db_path) if db_path else self.output / "inventory.sqlite"
        self.thumbs = thumbnails.thumbs_dir(self.output).resolve()
        self._lock = threading.Lock()
        # Correctly escaped read-only URI: a '#' or '%' in the output path must
        # not truncate or mis-decode the filename (see root_scope.readonly_uri).
        self._conn = sqlite3.connect(root_scope.readonly_uri(self.db_path), uri=True,
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Read-only viewer: it cannot create a binding, so a pre-identity output
        # directory is served unscoped *and says so*, rather than printing one
        # root while querying everything.
        self.scope = root_scope.recorded(self._conn)
        if not self.scope.bound:
            print("[review][WARN] this output directory has no recorded photo root "
                  "(created before root identity); serving every row it contains")
        self.summary = self._read_summary()
        self.state = review_state.ReviewState(self.output)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _read_summary(self) -> Dict[str, Any]:
        path = self.output / "review_summary.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                "ALL": db.review_index_count(self._conn, "ALL", scope=self.scope),
                "MAYBE": db.review_index_count(self._conn, "MAYBE", scope=self.scope),
                "UNKNOWN": db.review_index_count(self._conn, "UNKNOWN", scope=self.scope),
                "GROUPS": db.count_groups(self._conn, scope=self.scope),
            }

    def page(self, view: str, page: int, page_size: int) -> Dict[str, Any]:
        if view not in VIEWS:
            raise ValueError(f"view must be one of {', '.join(VIEWS)}")
        if page_size not in PAGE_SIZES:
            raise ValueError(f"page_size must be one of {', '.join(map(str, PAGE_SIZES))}")
        if page < 1:
            raise ValueError("page must be at least 1")
        with self._lock:
            total = (db.count_groups(self._conn, scope=self.scope) if view == "GROUPS"
                     else db.review_index_count(self._conn, view, scope=self.scope))
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            offset = (page - 1) * page_size
            if view == "GROUPS":
                items: List[Dict[str, Any]] = _group_items(
                    db.group_page(self._conn, offset, page_size, scope=self.scope)
                )
            else:
                items = [_public_item(row) for row in
                         db.review_page(self._conn, view, offset, page_size,
                                        scope=self.scope)]
        # P0-1: Get state outside lock, and provide both page and summary
        if view == "GROUPS":
            group_ids = [g["group_id"] for g in items]
            page_states = self.state.current_page_states(group_ids)
            summary = self.state.summary()
        else:
            page_states = {}
            summary = {}
        result = {
            "view": view, "page": page, "pages": pages, "page_size": page_size,
            "total": total, "shown": len(items),
            "remaining_after_page": max(0, total - (offset + len(items))),
            "items": items,
        }
        if view == "GROUPS":
            result["review_state"] = page_states
            result["review_summary"] = summary
        return result

    def group(self, group_id: int) -> Optional[Dict[str, Any]]:
        """Path-free members for lightbox navigation from any review queue."""
        with self._lock:
            rows = db.group_page_by_id(self._conn, int(group_id), scope=self.scope)
        if not rows:
            return None
        return _group_items(rows)[0]

    def thumb_bytes(self, file_id: int) -> Optional[bytes]:
        """Read a cached JPEG from the SSD only. No source fallback, ever.

        The id is checked against the current scope first: a thumbnail left in the
        cache by another root (same output directory reused before this fix, or a
        stale file) must not be reachable through the API.
        """
        if not self.in_scope(file_id):
            return None
        candidate = thumbnails.thumb_path(self.thumbs, file_id).resolve()
        if candidate.parent != self.thumbs:
            return None
        try:
            return candidate.read_bytes()
        except OSError:
            return None

    def in_scope(self, file_id: int) -> bool:
        """True when ``file_id`` belongs to the root this directory is bound to."""
        if not self.scope.bound:
            return True
        predicate, params = db.scope_sql(self.scope, "f")
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM files f WHERE f.id = :file_id AND {predicate}",
                {**params, "file_id": int(file_id)},
            ).fetchone()
        return row is not None

    def original_record(self, file_id: int) -> Optional[OriginalRecord]:
        """Resolve immutable inventory identity for one explicit preview click."""
        predicate, params = db.scope_sql(self.scope, "f")
        with self._lock:
            row = self._conn.execute(
                "SELECT f.path, f.file_kind, f.size_bytes, f.mtime_ns FROM files f "
                f"WHERE f.id = :file_id AND {predicate}",
                {**params, "file_id": int(file_id)},
            ).fetchone()
        if (row is None or row["file_kind"] not in ("jpg", "jpg_motion")
                or row["size_bytes"] is None or row["mtime_ns"] is None):
            return None
        return OriginalRecord(Path(row["path"]), int(row["size_bytes"]), int(row["mtime_ns"]))

    def status(self) -> Dict[str, Any]:
        files, cache_bytes = thumbnails.directory_usage(self.thumbs)
        # Get scoped group IDs to filter stale state
        with self._lock:
            scoped_group_ids = {
                int(row["group_id"])
                for row in self._conn.execute(
                    f"""
                    SELECT DISTINCT g.id AS group_id FROM groups g
                    WHERE EXISTS (
                        SELECT 1 FROM group_members gm JOIN files f ON f.id = gm.file_id
                        WHERE gm.group_id = g.id
                          AND {db.scope_sql(self.scope, 'f')[0]}
                    )
                    """,
                    db.scope_sql(self.scope, 'f')[1],
                )
            }
        # Use snapshot() to get state without accessing private _data
        snapshot = self.state.snapshot()
        scoped_total = 0
        scoped_reviewed = 0
        scoped_marked = 0
        for gid_str, decision in snapshot["groups"].items():
            gid = int(gid_str)
            if gid in scoped_group_ids:
                scoped_total += 1
                action = decision.get("action")
                if action in ("accept", "pick"):
                    scoped_reviewed += 1
                elif action == "mark":
                    scoped_marked += 1
        return {
            "counts": self.counts(),
            "page_sizes": list(PAGE_SIZES),
            "thumb_cache_files": files,
            "thumb_cache_bytes": cache_bytes,
            "summary": self.summary,
            "scope": {"root": self.scope.path, "root_key": self.scope.key},
            "review_state": {
                "total": scoped_total,
                "reviewed": scoped_reviewed,
                "marked": scoped_marked,
            },
        }

    def apply_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply review action. Returns success/error response."""
        group_id = payload.get("group_id")
        file_id = payload.get("file_id")
        action = payload.get("action")

        if not isinstance(group_id, int) or not isinstance(action, str):
            return {"error": "group_id (int) and action (str) are required"}

        error = review_state.validate_action(
            self._conn, group_id, file_id, action, self.scope, self._lock
        )
        if error:
            return {"error": error}

        if action == "clear":
            self.state.clear_group(group_id)
        else:
            decision = {
                "action": action,
                "timestamp": int(time.time()),
            }
            if file_id is not None:
                decision["file_id"] = int(file_id)
            self.state.set_group(group_id, decision)

        return {"ok": True, "summary": self.state.summary()}


def create_handler(data: ReviewData) -> type[http.server.SimpleHTTPRequestHandler]:
    """Handler serving the review page, the paged API, and cached thumbnails."""

    class ReviewHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(data.output), **kwargs)

        # -- routing --------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            parsed = urlparse(self.path)
            if parsed.path == "/api/page":
                self._serve_page(parsed.query)
                return
            if parsed.path == "/api/status":
                self._send_json(data.status())
                return
            match = _GROUP_RE.fullmatch(parsed.path)
            if match:
                group = data.group(int(match.group(1)))
                if group is None:
                    self.send_error(404, "group unavailable")
                else:
                    self._send_json(group)
                return
            match = _THUMB_RE.fullmatch(parsed.path)
            if match:
                self._serve_thumb(int(match.group(1)))
                return
            match = _ORIGINAL_RE.fullmatch(parsed.path)
            if match:
                self._serve_original(int(match.group(1)))
                return
            if not self._static_allowed(parsed.path):
                self.send_error(404)
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path != "/api/action":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length < 1 or length > 8192:
                    self._send_json({"error": "invalid content length"}, status=400)
                    return
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    self._send_json({"error": "malformed JSON or invalid UTF-8"}, status=400)
                    return
                if not isinstance(payload, dict):
                    self._send_json({"error": "payload must be JSON object"}, status=400)
                    return
                response = data.apply_action(payload)
                status = 200 if "ok" in response else 400
                self._send_json(response, status=status)
            except (OSError,) as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _static_allowed(self, raw_path: str) -> bool:
            parts = [part for part in unquote(raw_path).replace("\\", "/").split("/")
                     if part not in ("", ".")]
            if any(part == ".." for part in parts):
                return False
            # The DB and the thumbnail directory are served only through the API.
            return not (parts and (parts[0] in _DENY_NAMES
                                   or parts[0] == thumbnails.DIR_NAME))

        # -- endpoints ------------------------------------------------------
        def _serve_page(self, query: str) -> None:
            params = parse_qs(query)
            try:
                view = params.get("view", ["ALL"])[0].upper()
                page = int(params.get("page", ["1"])[0])
                size = int(params.get("page_size", [str(DEFAULT_PAGE_SIZE)])[0])
                self._send_json(data.page(view, page, size))
            except (KeyError, TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _serve_thumb(self, file_id: int) -> None:
            payload = data.thumb_bytes(file_id)
            if payload is None:
                self.send_error(404, "thumbnail not generated")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(payload)

        def _serve_original(self, file_id: int) -> None:
            """Stream one source JPEG only after an explicit lightbox request."""
            record = data.original_record(file_id)
            if record is None:
                self.send_error(404, "original image unavailable")
                return
            try:
                with record.path.open("rb") as source:
                    live = os.fstat(source.fileno())
                    if (live.st_size != record.size_bytes
                            or live.st_mtime_ns != record.mtime_ns):
                        self.send_error(409, "original changed since inventory")
                        return
                    head = source.read(16)
                    if head.startswith(b"\xff\xd8\xff"):
                        content_type = "image/jpeg"
                    elif head.startswith(b"\x89PNG\r\n\x1a\n"):
                        content_type = "image/png"
                    else:
                        self.send_error(415, "inventoried file is not JPEG or PNG")
                        return
                    source.seek(0)
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(live.st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    while chunk := source.read(1024 * 1024):
                        self.wfile.write(chunk)
            except (OSError, BrokenPipeError, ConnectionError):
                return

        def _send_json(self, value: Any, status: int = 200) -> None:
            payload = json.dumps(value, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return ReviewHandler


def start_server(output: Path, port: int = 0,
                 db_path: Optional[Path] = None) -> Tuple[http.server.ThreadingHTTPServer, str]:
    """Bind a local-only server rooted at the output directory."""
    output = Path(output).resolve()
    review = output / "review.html"
    if not review.is_file():
        raise FileNotFoundError(f"review HTML does not exist: {review}")
    resolved_db = Path(db_path) if db_path else output / "inventory.sqlite"
    if not resolved_db.is_file():
        raise FileNotFoundError(f"review database does not exist: {resolved_db}")
    data = ReviewData(output, resolved_db)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), create_handler(data))
    server.review_data = data  # type: ignore[attr-defined]
    return server, f"http://127.0.0.1:{server.server_port}/review.html"
