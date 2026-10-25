"""Private cloud-batch review. Inventories and batch artifacts are strictly read-only.

Numeric API group ids are transport handles only. All durable membership and
selection identities use (file_id, content_sha256), never date/Fxx aliases.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from pathlib import Path

from . import db, review_assets
from .learning_io import atomic_json, file_lock, read_json, safe_path
from .readonly_features import readonly_inventory
from .review_events import EventJournal, HoldoutPolicy, make_event

FILTERS = ("ALL", "PENDING", "ACCEPTED", "MODIFIED", "AI_ISSUES")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def identity(frame):
    fid, sha = frame["file_id"], frame["content_sha256"]
    if type(fid) is not int or fid < 1 or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("invalid file_id+SHA identity")
    return fid, sha


class CloudReviewData:
    """Adapter for the existing /api/page, /group, /status and /action endpoints."""

    def __init__(self, output: Path, batch: Path, manifest: Path, *, allow_holdout=False):
        self.output = Path(output).expanduser().resolve()
        self.batch = Path(batch).resolve(strict=True)
        manifest = Path(manifest).resolve(strict=True)
        if self.output == self.batch or self.output.is_relative_to(self.batch):
            raise ValueError("review output must be separate from cloud batch")
        assets = review_assets.manifest_for()
        self.static_allowed = frozenset({"review.html", *assets["assets"]})
        self.static_root = Path(__file__).resolve().parents[1] / "frontend" / "dist"
        self._lock = threading.RLock()
        self._receipts = {manifest: digest(manifest.read_bytes())}
        self._inventories = {}
        self._db_hashes = {}
        self._groups = {}
        self._media = {}
        frozen = json.loads(manifest.read_bytes())
        # Policy always uses the full trusted registry, even for a development-only UI.
        registry = (
            Path(__file__).resolve().parents[1]
            / "artifacts/cloud-judge-batch/frozen.json"
        )
        self._holdouts = HoldoutPolicy(read_json(registry))
        if not allow_holdout and any(
            self._holdouts.blocked(g["group"], g["date"], g["frames"])
            for g in frozen["groups"]
        ):
            raise ValueError("holdout manifest refused; use development-only manifest or explicit manual review")
        for inv in frozen["inventories"]:
            path = Path(inv["path"]).resolve(strict=True)
            if digest(path.read_bytes()) != inv["sha256"]:
                raise ValueError("inventory fingerprint mismatch")
            if self.output == path.parent or self.output.is_relative_to(path.parent):
                raise ValueError(
                    "use an isolated review output, not source inventory directory"
                )
            self._inventories[inv["date"]] = path
            self._db_hashes[path] = inv["sha256"]
        for handle, spec in enumerate(frozen["groups"], 1):
            alias = spec["group"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", alias):
                raise ValueError("invalid display group alias")
            case = self.batch / alias
            state = self._read(case / "state.json")
            packet = self._read(case / "visual-packet.json")
            request = self._read(case / "request.json")
            rows = self._rows(spec)
            with readonly_inventory(self._inventories[spec["date"]]) as conn:
                for row in rows:
                    source = Path(
                        conn.execute(
                            "SELECT path FROM files WHERE id=?", (row["file_id"],)
                        ).fetchone()[0]
                    ).resolve()
                    if self.output == source.parent or self.output.is_relative_to(
                        source.parent
                    ):
                        raise ValueError(
                            "review output must not be inside an original photo directory"
                        )
            expected = sorted(identity(f) for f in spec["frames"])
            if sorted(identity(r) for r in rows) != expected:
                raise ValueError("inventory membership identity mismatch: " + alias)
            identities = {f"I{fid}@{sha}": (fid, sha) for fid, sha in expected}
            if set(packet["complete_frames"]) != set(identities):
                raise ValueError("incomplete cloud group: " + alias)
            images = {}
            for image in request["images"]:
                if image["kind"] != "full_group_page":
                    continue
                if len(image["cells"]) != 1:
                    raise ValueError("review requires individual complete-frame pages")
                key = identities[image["cells"][0]["frame"]]
                path = self._safe_artifact(case, image["path"])
                if digest(path.read_bytes()) != image["sha256"]:
                    raise ValueError("cloud image hash mismatch")
                images[key] = (path, image["sha256"])
            if set(images) != set(expected):
                raise ValueError("missing complete group photos")
            # Bind all consumed raw/validated artifacts. Failed raw never becomes a recommendation.
            for name, sha in state.get("artifact_hashes", {}).items():
                path = self._safe_artifact(case, name)
                if digest(path.read_bytes()) != sha:
                    raise ValueError("cloud artifact hash mismatch: " + name)
                self._receipts[path] = sha
            judgment = (
                self._read(case / "validated.json")
                if state["state"] == "validated"
                else None
            )
            recommended = (
                [r["frame"] for r in judgment["recommended"]] if judgment else []
            )
            if any(identity(f) not in expected for f in recommended):
                raise ValueError("recommended identity mismatch")
            views = {v["id"]: v for v in packet["views"]}
            observations = []
            for index, d in enumerate(
                judgment.get("discriminators", [])[:3] if judgment else []
            ):
                evidence = []
                for e in d["evidence"]:
                    frame = views[e["region"]]["frame"]
                    if identity(frame) not in expected:
                        raise ValueError("evidence identity mismatch")
                    evidence.append({"frame": frame, "confidence": e["confidence"]})
                observations.append(
                    {
                        "index": index,
                        "text": d["text"],
                        "evidence": evidence,
                        "confidence": "low"
                        if all(e["confidence"] == "low" for e in evidence)
                        else "medium",
                    }
                )
            members = []
            for row in sorted(
                rows, key=lambda r: (r.get("exif_datetime") or "", r["file_id"])
            ):
                frame = {
                    "file_id": row["file_id"],
                    "content_sha256": row["content_sha256"],
                }
                fid, sha = identity(frame)
                members.append(
                    {
                        **frame,
                        "basename": row["basename"],
                        "width": row["width"],
                        "height": row["height"],
                        "exif_datetime": row.get("exif_datetime"),
                        "image_url": f"/api/cloud-image/{handle}/{fid}/{sha}.jpg",
                    }
                )
                self._media[(handle, fid, sha)] = images[(fid, sha)]
            self._groups[handle] = {
                "group_id": handle,
                "alias": alias,
                "members": members,
                "member_count": len(members),
                "ai": {
                    "status": judgment["decision"] if judgment else "manual",
                    "recommended": recommended,
                    "discriminators": observations,
                    "multi_select_reason": judgment.get("multi_select_reason")
                    if judgment
                    else None,
                    "failure": state.get("reason") if not judgment else None,
                },
                "_spec": spec,
            }
        if not self._groups:
            raise ValueError("cloud batch has no groups")
        self._file_stats = {p: self._stamp(p) for p in self._receipts | self._db_hashes}
        self.batch_id = digest(
            canonical(
                {
                    "receipts": sorted((str(p), h) for p, h in self._receipts.items()),
                    "inventories": frozen["inventories"],
                }
            )
        )
        self.summary = {}
        if (self.batch / "summary.json").is_file():
            summary = json.loads((self.batch / "summary.json").read_bytes())
            self.summary = {
                "estimated_cost_usd": summary.get("estimated_cost_usd_known"),
                "cost_note": "估算，非实际账单",
            }
        self.state_dir = self.output / "review-state"
        self.state_path = self.state_dir / "cloud-review.json"
        self.journal = EventJournal(self.output / "review-events-v1")
        self._check_output()
        self.journal.recover_pending()
        self.states = {}
        self.warning = None
        self._stale = False
        if self.state_path.exists():
            try:
                saved = json.loads(self.state_path.read_bytes())
                if saved["batch_id"] != self.batch_id:
                    raise ValueError("batch fingerprint changed")
                for key, value in saved["groups"].items():
                    group = next(
                        g for g in self._groups.values() if self._key(g) == key
                    )
                    if value["members"] != self._frames(group):
                        raise ValueError("saved identity mismatch")
                    if any(
                        identity(f) not in {identity(m) for m in group["members"]}
                        for f in value["selected"]
                    ):
                        raise ValueError("saved selection mismatch")
                    if value["status"] not in (
                        "pending",
                        "later",
                        "accepted",
                        "modified",
                    ):
                        raise ValueError("invalid saved review status")
                    if (
                        value["status"] in ("accepted", "modified")
                        and not value["selected"]
                    ):
                        raise ValueError("empty saved selection")
                    if value["status"] == "accepted" and sorted(
                        identity(f) for f in value["selected"]
                    ) != sorted(identity(f) for f in group["ai"]["recommended"]):
                        raise ValueError(
                            "saved AI acceptance differs from recommendation"
                        )
                self.states = saved["groups"]
            except (ValueError, KeyError, TypeError, StopIteration, AttributeError):
                self.warning = "旧审阅状态与当前批次或照片身份不一致，已拒绝加载；请使用新的 output 目录。"
                self._stale = True

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _safe_artifact(self, case, name):
        path = (case / name).resolve(strict=True)
        if not path.is_relative_to(self.batch) or path.parent != case.resolve():
            raise ValueError("cloud artifact path escape")
        return path

    def _read(self, path):
        path = self._safe_artifact(path.parent, path.name)
        payload = path.read_bytes()
        self._receipts[path] = digest(payload)
        return json.loads(payload)

    def _rows(self, spec):
        path = self._inventories[spec["date"]]
        with readonly_inventory(path) as conn:
            return [dict(r) for r in db.group_page_by_id(conn, spec["group_id"])]

    @staticmethod
    def _frames(group):
        return sorted(
            (
                {"file_id": m["file_id"], "content_sha256": m["content_sha256"]}
                for m in group["members"]
            ),
            key=identity,
        )

    def _key(self, group):
        # Existing UI keys remain stable; events/features always carry date scope.
        return digest(canonical(self._frames(group)))

    def _check_output(self):
        safe_path(self.output, "review-events-v1")
        if self.state_dir.is_symlink() or self.state_path.is_symlink():
            raise ValueError("review-state symlinks are not allowed")
        if (
            self.output.resolve() != self.output
            or self.state_dir.resolve().parent != self.output
        ):
            raise ValueError("review-state path escape")

    def group(self, group_id):
        with self._lock:
            group = self._groups.get(group_id)
            if group is None:
                return None
            result = copy.deepcopy({k: v for k, v in group.items() if k != "_spec"})
            result["review"] = copy.deepcopy(self.states.get(self._key(group)))
            return result

    def status(self):
        with self._lock:
            counts = {key: 0 for key in FILTERS}
            counts["ALL"] = len(self._groups)
            for group in self._groups.values():
                state = self.states.get(self._key(group), {}).get("status", "pending")
                counts["PENDING"] += state in ("pending", "later")
                counts["ACCEPTED"] += state == "accepted"
                counts["MODIFIED"] += state == "modified"
                counts["AI_ISSUES"] += group["ai"]["status"] != "select"
            return {
                "mode": "cloud_review",
                "batch_id": self.batch_id,
                "counts": counts,
                "total": counts["ALL"],
                "processed": counts["ACCEPTED"] + counts["MODIFIED"],
                "remaining": counts["PENDING"],
                "state_warning": self.warning,
                **self.summary,
            }

    def page(self, view="GROUPS", page=1, page_size=100, queue="ALL"):
        if queue not in FILTERS or page < 1 or page_size not in (50, 100, 200):
            raise ValueError("invalid cloud review page/filter")
        with self._lock:
            groups = [self.group(gid) for gid in self._groups]

            def included(g):
                state = (g["review"] or {}).get("status", "pending")
                return (
                    queue == "ALL"
                    or (queue == "PENDING" and state in ("pending", "later"))
                    or (queue == "ACCEPTED" and state == "accepted")
                    or (queue == "MODIFIED" and state == "modified")
                    or (queue == "AI_ISSUES" and g["ai"]["status"] != "select")
                )

            groups = [g for g in groups if included(g)]
            total = len(groups)
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            return {
                "items": groups[(page - 1) * page_size : page * page_size],
                "page": page,
                "pages": pages,
                "total": total,
                "status": self.status(),
            }

    def image_bytes(self, group_id, file_id, sha):
        entry = self._media.get((group_id, file_id, sha))
        if not entry:
            return None
        path, expected = entry
        payload = path.read_bytes()
        if digest(payload) != expected:
            raise ValueError("cloud image changed; reload batch")
        return payload

    def apply_action(self, payload):
        with self._lock:
            self._check_output()
            with file_lock(self.state_dir / ".save.lock"):
                # Refresh under the process lock to avoid stale-instance lost updates.
                if self.state_path.exists() and not self._stale:
                    saved = read_json(self.state_path)
                    if saved["batch_id"] != self.batch_id:
                        return {"error": "batch/state mismatch"}
                    self.states = saved["groups"]
                    pending = saved.get("pending_event")
                    if pending:
                        try:
                            self.journal.append(pending)
                            atomic_json(self.state_path, {**saved, "pending_event": None})
                        except (ValueError, OSError) as exc:
                            return {"error": "state saved; event pending: " + str(exc)}
                return self._apply_locked(payload)

    def _apply_locked(self, payload):
        try:
            if self._stale or payload.get("batch_id") != self.batch_id:
                raise ValueError("batch/state mismatch; use a fresh output directory")
            group = self._groups[int(payload["group_id"])]
            frames = self._frames(group)
            if sorted(identity(r) for r in self._rows(group["_spec"])) != sorted(
                identity(f) for f in frames
            ):
                raise ValueError("inventory identity changed; action refused")
            # Every byte is hashed at load. Stamp changes invalidate the whole batch;
            # unchanged files need no 1GB re-read on every checkbox save.
            for path, stamp in self._file_stats.items():
                if self._stamp(path) != stamp:
                    raise ValueError("cloud batch or inventory changed; action refused")
            action = payload["action"]
            previous = copy.deepcopy(
                self.states.get(
                    self._key(group),
                    {
                        "status": "pending",
                        "selected": [],
                        "feedback": False,
                        "ignored_reasons": [],
                    },
                )
            )
            if action == "accept":
                if group["ai"]["status"] != "select":
                    raise ValueError("no AI recommendation to accept")
                previous.update(status="accepted", selected=group["ai"]["recommended"])
            elif action in ("modify", "keep_all"):
                chosen = frames if action == "keep_all" else payload["selected"]
                keys = [identity(f) for f in chosen]
                if (
                    not keys
                    or len(keys) != len(set(keys))
                    or not set(keys) <= {identity(f) for f in frames}
                ):
                    raise ValueError("choose at least one valid file_id+SHA")
                previous.update(status="modified", selected=chosen)
            elif action == "later":
                previous["status"] = "later"
            elif action == "feedback":
                if type(payload.get("inaccurate")) is not bool:
                    raise ValueError("feedback must be boolean")
                ignored = payload.get("ignored_reasons", previous["ignored_reasons"])
                if not isinstance(ignored, list) or any(
                    type(i) is not int
                    or not 0 <= i < len(group["ai"]["discriminators"])
                    for i in ignored
                ):
                    raise ValueError("invalid reason index")
                previous.update(
                    feedback=payload["inaccurate"], ignored_reasons=sorted(set(ignored))
                )
            else:
                raise ValueError("unsupported review action")
            previous["members"] = frames
            revision = (
                max((v.get("revision", 0) for v in self.states.values()), default=0) + 1
            )
            token = payload.get("request_id")
            if token is not None and (
                not isinstance(token, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", token)
            ):
                raise ValueError("invalid request_id")
            key = f"live:{self.batch_id}:{token or revision}"
            for old in self.journal.events():
                if old["idempotency_key"] == key:
                    if old["group"] != group["alias"] or old["human"]["action"] != action or old["human"]["final"] != sorted(previous["selected"], key=identity) or old["human"]["feedback"] != previous["feedback"] or old["human"]["ignored_reasons"] != previous["ignored_reasons"]:
                        raise ValueError("request_id reused with different action")
                    return {
                        "ok": True,
                        "event_id": old["event_id"],
                        "group": self.group(group["group_id"]),
                        "status": self.status(),
                    }
            previous["revision"] = revision
            sealed = self._holdouts.blocked(group["alias"], group["_spec"]["date"], frames)
            event = make_event(
                group["_spec"],
                {**previous, "action": action},
                self.batch,
                self.batch_id,
                key,
                holdout=sealed,
                sealed=sealed,
            )
            next_states = {**self.states, self._key(group): previous}
            self._check_output()
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._check_output()
            atomic_json(self.state_path, {
                "version": 1, "batch_id": self.batch_id,
                "groups": next_states, "pending_event": event,
            })
            self.states = next_states
            # The durable outbox in state survives an append failure or a crash.
            committed = self.journal.append(event)
            atomic_json(
                self.state_path,
                {
                    "version": 1,
                    "batch_id": self.batch_id,
                    "groups": next_states,
                    "pending_event": None,
                },
            )
            return {
                "ok": True,
                "event_id": committed["event_id"],
                "group": self.group(group["group_id"]),
                "status": self.status(),
            }
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return {"error": str(exc)}

    def close(self):
        pass
