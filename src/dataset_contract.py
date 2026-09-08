"""Versioned dataset/split contract for leakage-safe offline evaluation.

Dataset manifests contain identifiers and event membership, not source paths or
labels.  Splits are event-level: an event may occur in exactly one split.
Held-out datasets are report-only and are rejected by every tuning-purpose
entry point.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class Split(StrEnum):
    TRAIN = "train"
    TUNE = "tune"
    HELD_OUT = "held_out"


class Purpose(StrEnum):
    TRAIN = "train"
    THRESHOLD_SEARCH = "threshold_search"
    PROMPT_SELECTION = "prompt_selection"
    PRESET_SELECTION = "preset_selection"
    FINAL_EVALUATION = "final_evaluation"


_TUNING_PURPOSES = {
    Purpose.TRAIN, Purpose.THRESHOLD_SEARCH, Purpose.PROMPT_SELECTION,
    Purpose.PRESET_SELECTION,
}


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    split: Split
    owner_scope: str
    events: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "split": self.split.value,
            "owner_scope": self.owner_scope,
            "events": list(self.events),
        }


def validate_manifest(value: Mapping[str, Any]) -> DatasetManifest:
    required = {"schema_version", "dataset_id", "split", "owner_scope", "events"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("dataset manifest has invalid keys")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"dataset manifest schema_version must be {SCHEMA_VERSION}")
    dataset_id = value["dataset_id"]
    owner_scope = value["owner_scope"]
    events = value["events"]
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    if not isinstance(owner_scope, str) or not owner_scope.strip():
        raise ValueError("owner_scope must be a non-empty string")
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence) or not events:
        raise ValueError("events must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in events):
        raise ValueError("every event id must be a non-empty string")
    if len(events) != len(set(events)):
        raise ValueError("event ids must be unique within a dataset")
    try:
        split = Split(value["split"])
    except (ValueError, TypeError) as exc:
        raise ValueError("split must be train, tune, or held_out") from exc
    return DatasetManifest(dataset_id.strip(), split, owner_scope.strip(), tuple(events))


def load_manifest(path: str | Path) -> DatasetManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_manifest(data)


def validate_split_contract(manifests: Sequence[DatasetManifest]) -> None:
    """Reject duplicate dataset IDs and cross-split event leakage."""
    dataset_ids: set[str] = set()
    event_splits: dict[str, Split] = {}
    for manifest in manifests:
        if manifest.dataset_id in dataset_ids:
            raise ValueError(f"duplicate dataset_id: {manifest.dataset_id}")
        dataset_ids.add(manifest.dataset_id)
        for event_id in manifest.events:
            previous = event_splits.get(event_id)
            if previous is not None and previous != manifest.split:
                raise ValueError(
                    f"event leakage: {event_id} appears in {previous.value} and {manifest.split.value}"
                )
            event_splits[event_id] = manifest.split


def authorize(manifest: DatasetManifest, purpose: Purpose | str, *, labels_requested: bool = False) -> dict[str, Any]:
    """Authorize one offline operation; held-out tuning fails closed.

    Final held-out evaluation may read labels only to compute the final report.
    The returned contract always forbids policy/config/prompt write-back.
    """
    try:
        purpose = Purpose(purpose)
    except (ValueError, TypeError) as exc:
        raise ValueError("unknown dataset purpose") from exc
    if manifest.split == Split.HELD_OUT and purpose in _TUNING_PURPOSES:
        raise PermissionError(
            f"held_out dataset {manifest.dataset_id!r} is report-only; {purpose.value} is forbidden"
        )
    if labels_requested and purpose != Purpose.FINAL_EVALUATION:
        raise PermissionError("labels may only be read by final_evaluation")
    if purpose == Purpose.FINAL_EVALUATION and manifest.split != Split.HELD_OUT:
        raise PermissionError("final_evaluation requires a held_out dataset")
    return {
        "dataset_id": manifest.dataset_id,
        "split": manifest.split.value,
        "purpose": purpose.value,
        "manifest_fingerprint": manifest.fingerprint,
        "labels_authorized": bool(labels_requested),
        "report_only": manifest.split == Split.HELD_OUT,
        "policy_writeback_allowed": False,
        "delete_trash_authority": "none",
    }
