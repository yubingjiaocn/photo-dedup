"""Local-only paged review server for a 100k+ photo library.

Hard boundaries (the reason this file exists):

* ``/api/thumb/<id>.jpg`` reads **only** ``output/thumbs/<id>.jpg`` on the SSD,
  so normal paging never wakes the HDD. ``/api/original/<id>`` is the sole,
  explicit exception: a user click streams that one inventoried still image
  for focus inspection; source paths are never returned to the browser.
* Pages come from SQLite with ``LIMIT/OFFSET`` over the pre-built
  ``review_index`` table, so no giant JSON is loaded by the server or browser.
* Views: ``ALL`` (full timeline), ``MAYBE``, ``UNKNOWN``, ``GROUPS``.
  ``GROUPS`` additionally takes a queue -- ``PENDING`` (no human decision yet),
  ``LATER`` (marked) or ``DONE`` (accepted/picked). The queue is applied
  **server-side**: the browser receives one page of the filtered queue with the
  queue's own ``total``, never the whole group list to filter itself.
* Page size is 50, 100 (default), or 200 -- an explicit allow-list.
* Binds ``127.0.0.1`` only; the only mutation is the human review state in
  ``output/review_state.json`` (accept/pick/mark/clear/undo) -- no delete, move
  or write ever reaches an original photo.
* Every query is restricted to the photo root the output directory is bound to
  (:mod:`src.root_scope`), so a legacy database holding several roots still
  serves -- and counts -- only the current one.
"""

from __future__ import annotations

import bisect
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
QUEUES = review_state.QUEUES
DEFAULT_QUEUE = "PENDING"
_THUMB_RE = re.compile(r"^/api/thumb/(\d{1,18})\.jpg$")
_ORIGINAL_RE = re.compile(r"^/api/original/(\d{1,18})$")
_GROUP_RE = re.compile(r"^/api/group/(\d{1,18})$")
_ACTION_RE = re.compile(r"^/api/action$")
# Static GETs are allow-listed, not deny-listed: the output directory also holds
# the deletion manifests, the human review state, the summary and the run log, and
# a new file added there must not become readable by default. The page is
# self-contained (CSS and script are inlined), so this is the whole list.
_STATIC_ALLOWED = frozenset({"review.html", "favicon.ico"})


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
        self._id_lock = threading.Lock()
        self._group_ids: Optional[List[int]] = None
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
        # Validate fingerprints and prune stale decisions after stage2 rerun
        stale_count = self.state.validate_and_prune_stale(self._conn, self.scope)
        if stale_count > 0:
            print(f"[review] Pruned {stale_count} stale group decision(s) from prior stage2 run")

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

    # --- group queues ------------------------------------------------------

    def group_ids(self) -> List[int]:
        """All in-scope group ids, ascending, read once per server process.

        The review database is opened read-only, so the set of groups cannot
        change underneath us; only the human state layered on top does.
        """
        with self._id_lock:
            if self._group_ids is None:
                with self._lock:
                    self._group_ids = db.group_ids_in_scope(self._conn, scope=self.scope)
            return self._group_ids

    def queue_ids(self, queue: str) -> Tuple[List[int], Dict[str, int]]:
        """Ordered ids of one queue plus the counts of all three.

        Filtering happens here, against one integer list and one pass over the
        stored actions, so a page request never materialises groups the queue
        excludes and never copies the whole decision map to read one key.
        """
        if queue not in QUEUES:
            raise ValueError(f"queue must be one of {', '.join(QUEUES)}")
        ids = self.group_ids()
        tally = self.state.queue_tally(ids)
        placement = tally["queues"]
        return [gid for gid in ids if placement[gid] == queue], tally["counts"]

    def queue_counts(self) -> Dict[str, int]:
        """How many groups sit in each queue right now."""
        return self.state.queue_tally(self.group_ids())["counts"]

    def next_in_queue(self, after_group_id: int, queue: str,
                      page_size: int = DEFAULT_PAGE_SIZE) -> Dict[str, Any]:
        """Where to go after deciding ``after_group_id``, from the server's view.

        The decided group has just left ``queue``, which can also shrink the queue
        by a whole page. The client must not guess: it asks for the first
        surviving group *after* the one it decided, and gets that group's real
        page in the queue as it stands now. Ordering is therefore always forward,
        across a page boundary too, and the end of the queue answers with the
        last survivor rather than silently wrapping to page 1.
        """
        if queue not in QUEUES:
            raise ValueError(f"queue must be one of {', '.join(QUEUES)}")
        if page_size not in PAGE_SIZES:
            raise ValueError(f"page_size must be one of {', '.join(map(str, PAGE_SIZES))}")
        ids, counts = self.queue_ids(queue)
        total = len(ids)
        result: Dict[str, Any] = {"queue": queue, "total": total,
                                  "queue_counts": counts, "page_size": page_size,
                                  "after": int(after_group_id)}
        if total == 0:
            result.update({"found": False, "group_id": None, "index": None,
                           "page": 1, "wrapped": False})
            return result
        index = bisect.bisect_right(ids, int(after_group_id))
        wrapped = index >= total
        if wrapped:
            index = total - 1          # nothing left ahead: stay on the last one
        result.update({"found": True, "group_id": ids[index], "index": index,
                       "page": index // page_size + 1, "wrapped": wrapped})
        return result

    def locate(self, group_id: int, queue: str,
               page_size: int = DEFAULT_PAGE_SIZE) -> Dict[str, Any]:
        """Where ``group_id`` sits in ``queue``, or where to land instead.

        Used to restore a reload and to follow an undo back to its group. When
        the group is no longer in that queue (it was decided, or it left the
        scope) the answer points at the next surviving entry -- clamped to the
        last one -- so the caller never has to guess a page.
        """
        if page_size not in PAGE_SIZES:
            raise ValueError(f"page_size must be one of {', '.join(map(str, PAGE_SIZES))}")
        ids, counts = self.queue_ids(queue)
        total = len(ids)
        result: Dict[str, Any] = {"queue": queue, "total": total,
                                  "queue_counts": counts, "page_size": page_size}
        if total == 0:
            result.update({"found": False, "page": 1, "index": None, "group_id": None})
            return result
        target = int(group_id)
        index = bisect.bisect_left(ids, target)
        found = index < total and ids[index] == target
        index = min(index, total - 1)
        result.update({"found": found, "index": index, "group_id": ids[index],
                       "page": index // page_size + 1})
        return result

    def page(self, view: str, page: int, page_size: int,
             queue: str = DEFAULT_QUEUE) -> Dict[str, Any]:
        if view not in VIEWS:
            raise ValueError(f"view must be one of {', '.join(VIEWS)}")
        if page_size not in PAGE_SIZES:
            raise ValueError(f"page_size must be one of {', '.join(map(str, PAGE_SIZES))}")
        if page < 1:
            raise ValueError("page must be at least 1")
        if view == "GROUPS":
            return self._group_page(page, page_size, queue)
        with self._lock:
            total = db.review_index_count(self._conn, view, scope=self.scope)
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            offset = (page - 1) * page_size
            items = [_public_item(row) for row in
                     db.review_page(self._conn, view, offset, page_size,
                                    scope=self.scope)]
        return {
            "view": view, "page": page, "pages": pages, "page_size": page_size,
            "total": total, "shown": len(items),
            "remaining_after_page": max(0, total - (offset + len(items))),
            "items": items,
        }

    def _group_page(self, page: int, page_size: int, queue: str) -> Dict[str, Any]:
        """One page of the requested group queue, sliced before any join."""
        if queue not in QUEUES:
            raise ValueError(f"queue must be one of {', '.join(QUEUES)}")
        ids, counts = self.queue_ids(queue)
        total = len(ids)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, pages)
        offset = (page - 1) * page_size
        window = ids[offset:offset + page_size]
        with self._lock:
            items = _group_items(
                db.group_page_by_ids(self._conn, window, scope=self.scope))
        group_ids = [group["group_id"] for group in items]
        return {
            "view": "GROUPS", "queue": queue, "page": page, "pages": pages,
            "page_size": page_size, "total": total, "shown": len(items),
            "remaining_after_page": max(0, total - (offset + len(items))),
            "items": items,
            "review_state": self.state.current_page_states(group_ids),
            "review_summary": self.state.summary(),
            "queue_counts": counts,
            "group_total": sum(counts.values()),
            "undo_depth": self.state.history_depth(),
            # Carried on the page the workbench already loads, so the reviewer is
            # told their saved decisions were unreadable without a second request.
            "state_warning": self.state.warning(),
        }

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
        # The queue tally over the cached scoped id list already answers "how many
        # of *this root's* groups are decided/marked", so there is no second
        # DISTINCT-groups query and no full copy of the decision map here.
        counts = self.state.queue_tally(self.group_ids())["counts"]
        payload = {
            "counts": self.counts(),
            "page_sizes": list(PAGE_SIZES),
            "thumb_cache_files": files,
            "thumb_cache_bytes": cache_bytes,
            "summary": self.summary,
            "scope": {"root": self.scope.path, "root_key": self.scope.key},
            "queues": counts,
            "undo_depth": self.state.history_depth(),
            "review_state": {
                "total": counts["DONE"] + counts["LATER"],
                "reviewed": counts["DONE"],
                "marked": counts["LATER"],
            },
        }
        warning = self.state.warning()
        if warning:
            payload["state_warning"] = warning
        return payload

    def apply_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply one review action. Returns success/error response.

        ``accept``/``pick``/``mark`` are recorded together with the step that
        reverses them; ``undo`` replays the newest such step, so U always
        restores the group's previous state (including "undecided") and reports
        which group and photo to return to. ``clear`` stays a plain, targeted
        reset of one group.

        ``context_file_id`` (optional) is the photo the reviewer had on screen.
        It is validated as a member of the group but never written into the
        decision -- only beside the undo step, so undoing an ``accept`` or a
        ``mark`` comes back to that photo instead of the group's first member.

        ``accept`` additionally requires the group to *have* an AI keeper in this
        scope: "keep what the machine chose" is not a decision that can be
        recorded when the machine chose nothing here, which a group straddling two
        roots can produce.
        """
        action = payload.get("action")
        if action == "undo":
            if (payload.get("group_id") is not None or payload.get("file_id") is not None
                    or payload.get("context_file_id") is not None):
                return {"error": "undo action must not include group_id or file_id"}
            undone = self.state.undo_last(allowed=self.group_in_scope)
            if undone is None:
                return {"error": "nothing to undo"}
            return {"ok": True, "undo": undone, **self._state_envelope()}

        group_id = payload.get("group_id")
        file_id = payload.get("file_id")
        context_file_id = payload.get("context_file_id")

        if not isinstance(group_id, int) or not isinstance(action, str):
            return {"error": "group_id (int) and action (str) are required"}

        error = review_state.validate_action(
            self._conn, group_id, file_id, action, self.scope, self._lock,
            context_file_id=context_file_id,
        )
        if error:
            return {"error": error}
        focus = int(context_file_id) if context_file_id is not None else None

        if action == "clear":
            self.state.revert_decision(group_id, focus_file_id=focus)
        else:
            # One member read serves both checks below: the AI keeper's existence
            # and the fingerprint that pins this decision to these photos.
            with self._lock:
                member_rows = db.group_page_by_id(self._conn, group_id, scope=self.scope)
            member_fingerprint = None
            if member_rows:
                member_fingerprint = review_state.compute_member_fingerprint(
                    [int(row["file_id"]) for row in member_rows])
            if action == "accept" and not any(row["is_keep"] for row in member_rows):
                return {"error": "group has no AI keeper in this scope to accept"}

            decision = {
                "action": action,
                "timestamp": int(time.time()),
            }
            if file_id is not None:
                decision["file_id"] = int(file_id)
            self.state.apply_decision(group_id, decision, member_fingerprint,
                                      focus_file_id=focus)

        return {"ok": True, "group_id": int(group_id), **self._state_envelope()}

    def group_in_scope(self, group_id: int) -> bool:
        """True when ``group_id`` is one of the groups this root can review."""
        ids = self.group_ids()
        index = bisect.bisect_left(ids, int(group_id))
        return index < len(ids) and ids[index] == int(group_id)

    def _state_envelope(self) -> Dict[str, Any]:
        """Progress fields every mutating response carries back to the UI.

        One pass over the stored actions, not one per field: this runs on every
        keystroke, so it must not walk the decision map several times.
        """
        return {
            "summary": self.state.summary(),
            "queues": self.queue_counts(),
            "undo_depth": self.state.history_depth(),
        }


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
            if parsed.path == "/api/locate":
                self._serve_locate(parsed.query)
                return
            if parsed.path == "/api/next":
                self._serve_next(parsed.query)
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
            if parsed.path in ("", "/"):
                # Never generate a directory index: it would advertise the
                # manifests, the review state and the run log by name.
                self.send_response(302)
                self.send_header("Location", "/review.html")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if not self._static_allowed(parsed.path):
                self.send_error(404)
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path != "/api/action":
                self.send_error(404)
                return
            forbidden = self._reject_cross_site()
            if forbidden:
                self._send_json({"error": forbidden}, status=403)
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

        # -- request guards -------------------------------------------------
        def _reject_cross_site(self) -> Optional[str]:
            """Refuse a write that another site could have caused.

            The server listens on loopback, but any page in the same browser can
            still POST to it. Two cheap, standard checks close that:

            * ``Content-Type`` must be ``application/json``. A cross-origin form
              can only send the three CORS-safelisted types, so requiring JSON
              means such a form is rejected before it is parsed, and a real
              preflight would be needed instead.
            * ``Origin``/``Referer``, *when the browser sends one*, must be this
              loopback server. A missing header is allowed on purpose: curl and
              local scripts send none, and they are not the threat here.
            """
            media_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if media_type.lower() != "application/json":
                return "Content-Type must be application/json"
            for header in ("Origin", "Referer"):
                value = self.headers.get(header)
                if not value or value == "null":
                    continue
                if not self._is_own_origin(value):
                    return f"cross-site {header} is not allowed"
            return None

        def _is_own_origin(self, value: str) -> bool:
            """True when ``value`` addresses this very server on loopback."""
            try:
                parsed = urlparse(value)
            except ValueError:
                return False
            if parsed.scheme not in ("http", "https"):
                return False
            host = (parsed.hostname or "").lower()
            if host not in ("127.0.0.1", "::1", "localhost"):
                return False
            port = parsed.port if parsed.port is not None else (
                443 if parsed.scheme == "https" else 80)
            return port == int(self.server.server_port)

        def _static_allowed(self, raw_path: str) -> bool:
            """Fail-closed allow-list for plain static GETs.

            The output directory is a working directory, not a web root: it holds
            the deletion manifests, the human review state, the summary and the
            run log. Denying a handful of known names let anything new added
            there leak by default, so this inverts the rule -- only the review
            page itself is served statically, and everything else goes through
            the API or not at all.

            Decoded once and compared exactly, so ``/%2e%2e/x``, ``/./review.html``
            or a backslash separator cannot smuggle a different target through.
            """
            try:
                decoded = unquote(raw_path, errors="strict")
            except (UnicodeDecodeError, ValueError):
                return False
            if "\x00" in decoded:
                return False
            parts = [part for part in decoded.replace("\\", "/").split("/")
                     if part not in ("", ".")]
            return len(parts) == 1 and parts[0] in _STATIC_ALLOWED

        # -- endpoints ------------------------------------------------------
        def _serve_page(self, query: str) -> None:
            params = parse_qs(query)
            try:
                view = params.get("view", ["GROUPS"])[0].upper()
                queue = params.get("queue", [DEFAULT_QUEUE])[0].upper()
                page = int(params.get("page", ["1"])[0])
                size = int(params.get("page_size", [str(DEFAULT_PAGE_SIZE)])[0])
                self._send_json(data.page(view, page, size, queue))
            except (KeyError, TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _serve_next(self, query: str) -> None:
            """Answer "which group comes after the one I just decided"."""
            params = parse_qs(query)
            try:
                queue = params.get("queue", [DEFAULT_QUEUE])[0].upper()
                after = int(params.get("after", ["0"])[0])
                size = int(params.get("page_size", [str(DEFAULT_PAGE_SIZE)])[0])
                self._send_json(data.next_in_queue(after, queue, size))
            except (KeyError, TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _serve_locate(self, query: str) -> None:
            """Answer "which page holds this group" for reload/undo restoration."""
            params = parse_qs(query)
            try:
                queue = params.get("queue", [DEFAULT_QUEUE])[0].upper()
                group_id = int(params.get("group_id", ["0"])[0])
                size = int(params.get("page_size", [str(DEFAULT_PAGE_SIZE)])[0])
                self._send_json(data.locate(group_id, queue, size))
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
