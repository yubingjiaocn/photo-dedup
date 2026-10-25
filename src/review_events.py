"""Versioned review events; atomic immutable files form an append-only journal."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import jsonschema

from .learning_io import (
    atomic_json,
    canonical,
    file_lock,
    file_sha,
    read_json,
    safe_path,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "review-event-log-v1"


def identity(frame):
    return frame["file_id"], frame["content_sha256"]


def frame_set(frames):
    return {identity(f) for f in frames}


class HoldoutPolicy:
    """Trusted local splits + frozen membership, never event-supplied flags."""

    def __init__(self, frozen, split_paths=None, trust_path=None):
        split_paths = split_paths or [
            ROOT / "artifacts/pair-router-tuning/split.json",
            ROOT / "artifacts/large-group-tuning/split.json",
        ]
        self.frozen_sha256 = sha256(canonical(frozen))
        self.groups = set()
        self.hashes = []
        for path in split_paths:
            self.groups.update(read_json(path)["holdout_groups"])
            self.hashes.append(file_sha(path))
        trust = read_json(trust_path or ROOT / "schemas/learning-registry-trust.json")
        if self.frozen_sha256 != trust["frozen_sha256"] or self.hashes != trust["split_sha256"]:
            raise ValueError("trusted registry fingerprint mismatch; new datasets require explicit new trust configuration")
        self.specs = {g["group"]: g for g in frozen["groups"]}
        self.inventories = {i["date"]: i for i in frozen.get("inventories", [])}
        if not self.groups <= self.specs.keys():
            raise ValueError("holdout registry missing frozen identities")
        self.identities = {
            (g["date"], *identity(f))
            for g in frozen["groups"]
            if g["group"] in self.groups
            for f in g["frames"]
        }
        self.shas = {i[2] for i in self.identities}

    def blocked(self, group, date, frames):
        return group in self.groups or any(
            (date, *identity(f)) in self.identities or f["content_sha256"] in self.shas
            for f in frames
        )

    def check_event(self, event):
        spec = self.specs.get(event["group"])
        if (
            not spec
            or spec["date"] != event["date"]
            or frame_set(spec["frames"]) != frame_set(event["members"])
        ):
            raise ValueError("event frozen identity mismatch")
        if event["holdout"] or self.blocked(
            event["group"], event["date"], event["members"]
        ):
            raise ValueError("holdout content prohibited")


@lru_cache(maxsize=1)
def event_validator():
    return jsonschema.Draft202012Validator(
        read_json(ROOT / "schemas/review-event-log-v1.schema.json"),
        format_checker=jsonschema.FormatChecker(),
    )


def validate_event(event):
    event_validator().validate(event)
    if event["event_id"] != sha256(event["idempotency_key"].encode()):
        raise ValueError("event ID mismatch")
    if event["holdout"]:
        ai = event["ai"]
        if (ai["decision"] != "unavailable_sealed"
                or any(ai[k] is not None for k in ("prompt_sha256", "model", "contract", "receipt_ref"))
                or any(ai[k] for k in ("recommended", "reasons", "evidence_refs", "artifacts"))
                or event["assets"] or any(event["snapshots"].values())):
            raise ValueError("holdout event must be sealed")
    members = frame_set(event["members"])
    if len(members) != len(event["members"]):
        raise ValueError("duplicate event member")
    for frames in (
        event["ai"]["recommended"],
        event["human"]["final"],
        event["human"]["added"],
        event["human"]["removed"],
    ):
        if not frame_set(frames) <= members or len(frame_set(frames)) != len(frames):
            raise ValueError("event selection identity mismatch")
    ai, final = (
        frame_set(event["ai"]["recommended"]),
        frame_set(event["human"]["final"]),
    )
    if (
        frame_set(event["human"]["added"]) != final - ai
        or frame_set(event["human"]["removed"]) != ai - final
    ):
        raise ValueError("event difference mismatch")
    if event["human"]["replaced"] != bool(final - ai and ai - final):
        raise ValueError("event replaced mismatch")
    if event["human"]["status"] in ("accepted", "modified") and not final:
        raise ValueError("reviewed event requires final selection")
    if event["assets"] and frame_set(a["original"] for a in event["assets"]) != members:
        raise ValueError("asset mapping incomplete")
    for ref in [
        *event["snapshots"].values(),
        *event["ai"]["artifacts"],
        *[a["review"] for a in event["assets"]],
    ]:
        if ref:
            safe_path(ROOT, ref["path"])
    return event


class EventJournal:
    def __init__(self, root):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("journal symlink")

    def recover_pending(self):
        """Recover every visible outbox belonging to this output, before export."""
        state_dir = self.root.parent / "review-state"
        if not state_dir.exists():
            return
        safe_path(self.root.parent, "review-state")
        with file_lock(state_dir / ".save.lock"):
            for path in sorted(state_dir.glob("*.json")):
                safe_path(state_dir, path.name)
                saved = read_json(path)
                if saved.get("pending_event"):
                    self.append(saved["pending_event"])
                    atomic_json(path, {**saved, "pending_event": None})

    def export_events(self):
        # Failure is deliberately fatal: exporting an incomplete journal is unsafe.
        self.recover_pending()
        return self.events()

    def events(self):
        safe_path(self.root.parent, self.root.name)
        values = []
        for path in sorted(self.root.glob("[0-9]*.json")):
            if path.is_symlink():
                raise ValueError("event symlink")
            value = validate_event(read_json(path))
            if path.name != f"{value['sequence']:012d}-{value['event_id']}.json":
                raise ValueError("event sequence filename mismatch")
            values.append(value)
        if [e["sequence"] for e in values] != list(range(1, len(values) + 1)) or len(
            {e["event_id"] for e in values}
        ) != len(values):
            raise ValueError("event sequence gap or duplicate")
        return values

    def append(self, event):
        safe_path(self.root.parent, self.root.name)
        with file_lock(self.root / ".lock"):
            existing = self.events()
            for old in existing:
                if old["event_id"] == event["event_id"]:
                    if {
                        k: v
                        for k, v in old.items()
                        if k not in ("sequence", "timestamp")
                    } != {
                        k: v
                        for k, v in event.items()
                        if k not in ("sequence", "timestamp")
                    }:
                        raise ValueError("idempotency key reused with different event")
                    return old
            event = {**event, "sequence": len(existing) + 1}
            validate_event(event)
            path = self.root / f"{event['sequence']:012d}-{event['event_id']}.json"
            if path.exists():
                raise ValueError("event overwrite refused")
            atomic_json(path, event)
            return event


def artifact_ref(case, name):
    path = safe_path(case, name)
    return (
        {"path": case.name + "/" + name, "sha256": file_sha(path)}
        if path.is_file()
        else None
    )


def make_event(
    spec, saved, batch, session, key, holdout=False, provenance=None, sealed=False
):
    """Sealed backfill copies labels only; never opens holdout AI/images/features."""
    case = safe_path(batch, spec["group"])
    members = sorted(spec["frames"], key=identity)
    artifacts, assets, snapshots = [], [], {"L0": None, "L1": None}
    ai = {
        "decision": "unavailable_sealed",
        "recommended": [],
        "reasons": [],
        "evidence_refs": [],
        "prompt_sha256": None,
        "model": None,
        "contract": None,
        "receipt_ref": None,
        "artifacts": artifacts,
    }
    sealed = sealed or holdout
    if not sealed:
        request = read_json(case / "request.json")
        state = read_json(case / "state.json")
        judgment = (
            read_json(case / "validated.json") if state["state"] == "validated" else {}
        )
        ai.update(
            decision=judgment.get("decision", "manual"),
            recommended=[r["frame"] for r in judgment.get("recommended", [])],
            reasons=judgment.get("discriminators", []),
            evidence_refs=[
                e
                for r in judgment.get("recommended", [])
                for e in r.get("evidence", [])
            ],
            prompt_sha256=sha256(canonical(request["prompt"])),
            model=request["model"],
            contract=judgment.get(
                "contract", request.get("response_schema", {}).get("$id", "unknown")
            ),
            receipt_ref=artifact_ref(case, "provider-response.json"),
        )
        artifacts.extend(
            ref
            for name in (
                "raw.json",
                "raw.txt",
                "validated.json",
                "state.json",
                "provider-response.json",
            )
            if (ref := artifact_ref(case, name))
        )
        snapshots = {name: artifact_ref(case, name + ".json") for name in ("L0", "L1")}
        for image in request["images"]:
            if image["kind"] != "full_group_page":
                continue
            if len(image["cells"]) != 1:
                raise ValueError("one source per review asset required")
            match = re.fullmatch(
                r"I([0-9]+)@([0-9a-f]{64})", image["cells"][0]["frame"]
            )
            if not match:
                raise ValueError("bad asset identity")
            ref = artifact_ref(case, image["path"])
            if ref["sha256"] != image["sha256"]:
                raise ValueError("review asset hash mismatch")
            assets.append(
                {
                    "original": {"file_id": int(match[1]), "content_sha256": match[2]},
                    "review": ref,
                    "long_edge": 2048,
                }
            )
    final = sorted(saved.get("selected", []), key=identity)
    recommended = frame_set(ai["recommended"])
    added = [f for f in final if identity(f) not in recommended]
    removed = [f for f in ai["recommended"] if identity(f) not in frame_set(final)]
    return {
        "schema": SCHEMA,
        "event_id": sha256(key.encode()),
        "idempotency_key": key,
        "sequence": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "group": spec["group"],
        "date": spec["date"],
        "members": members,
        "holdout": holdout,
        "ai": ai,
        "human": {
            "status": saved["status"],
            "final": final,
            "added": added,
            "removed": removed,
            "replaced": bool(added and removed),
            "ignored_reasons": saved.get("ignored_reasons", []),
            "feedback": saved.get("feedback", False),
            "action": saved.get("action", "backfill"),
        },
        "snapshots": snapshots,
        "assets": assets,
        "provenance": provenance or {"kind": "live_review"},
    }


def backfill(snapshot, frozen_path, batch, journal, policy):
    payload = Path(snapshot).read_bytes()
    saved = json.loads(payload)
    fingerprint = sha256(payload)
    by_members = {
        frozenset(frame_set(v["members"])): v for v in saved["groups"].values()
    }
    count = 0
    for spec in read_json(frozen_path)["groups"]:
        state = by_members[frozenset(frame_set(spec["frames"]))]
        sealed = policy.blocked(spec["group"], spec["date"], spec["frames"])
        event = make_event(
            spec,
            state,
            batch,
            "snapshot-backfill",
            f"backfill:{fingerprint}:{spec['group']}",
            holdout=sealed,
            sealed=sealed,
            provenance={
                "kind": "frozen_label_snapshot",
                "snapshot_sha256": fingerprint,
            },
        )
        journal.append(event)
        count += 1
    return {
        "events": count,
        "holdout": sum(e["holdout"] for e in journal.events()),
        "snapshot_sha256": fingerprint,
    }
