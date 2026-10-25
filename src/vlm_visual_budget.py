"""Fail-closed scene-driven visual plans; no detector face implies main subject.

Plans contain normalized shared windows, not per-file independently selected face
crops. A scene or primary subject that was not localized is a blocker, not a reason
to spend the action budget on bystanders. Coordinates remain in audit manifests.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


class VisualBudgetError(ValueError):
    """Visual evidence cannot support a scene-specific comparison."""


def _quality(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("quality_meta")
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value) if isinstance(value, Mapping) else {}


def plan(members: Sequence[Mapping[str, Any]], *, regime: str | None = None) -> dict[str, Any]:
    qualities = [_quality(row) for row in members]
    if regime is None:
        tags = {str(tag).lower() for q in qualities
                for tag in q.get("routing", {}).get("scene_context", {}).get("tags", [])}
        known = tags & {"action", "portrait", "group", "landscape"}
        if len(known) != 1:
            raise VisualBudgetError("SCENE_REGIME_UNRESOLVED: no unique persisted regime")
        regime = known.pop()
    if regime not in {"action", "portrait", "group", "landscape"}:
        raise VisualBudgetError("SCENE_REGIME_UNSUPPORTED")
    if regime == "action":
        # Native COCO 'person' detections cannot stand in for a mascot/action
        # subject. Require a source-backed primary-subject annotation in each file.
        candidates = []
        for row, q in zip(members, qualities):
            regions = q.get("local_region_set", {}).get("regions", [])
            primary = [r for r in regions if r.get("role") == "primary_subject"
                       and r.get("role_authority") in {"human_verified", "validated_subject_tracker"}]
            if len(primary) != 1:
                raise VisualBudgetError(
                    f"ACTION_PRIMARY_SUBJECT_UNLOCALIZED:file_id={row['file_id']}; "
                    "person/face detections have no main-subject authority"
                )
            candidates.append(primary[0].get("box_normalized"))
        windows = [_union(candidates)]
        label = "primary whole-body action; hands/feet/props"
    elif regime == "portrait":
        candidates = []
        for row in members:
            raw = row.get("faces_json") or []
            faces = json.loads(raw) if isinstance(raw, str) else raw
            boxes = [face["bbox"] for face in faces if isinstance(face.get("bbox"), list)
                     and len(face["bbox"]) == 4]
            if not boxes:
                raise VisualBudgetError("PORTRAIT_FACE_UNLOCALIZED")
            box = max(boxes, key=lambda b: b[2] * b[3])
            x, y, w, h = box
            candidates.append([x / row["width"], y / row["height"],
                               (x + w) / row["width"], (y + h) / row["height"]])
        windows = [_union(candidates)]
        label = "portrait face/eyes"
    else:
        # Subject correspondence and registered high-information maps must exist
        # before group/landscape strips are safe to label as comparable regions.
        raise VisualBudgetError(f"{regime.upper()}_SHARED_REGION_REGISTRATION_UNAVAILABLE")
    return {"regime": regime, "sheet_min_short_edge": 400,
            "strip_min_short_edge": 256, "max_strips": 2,
            "windows": windows, "label": label,
            "bystander_faces_use_budget": False}


def _union(boxes: Sequence[Any]) -> list[float]:
    if any(not isinstance(b, (list, tuple)) or len(b) != 4
           or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1) for b in boxes):
        raise VisualBudgetError("INVALID_PRIMARY_ROI")
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad = 0.08 * max(x2 - x1, y2 - y1)
    return [max(0, x1 - pad), max(0, y1 - pad), min(1, x2 + pad), min(1, y2 + pad)]


def subject_map_plan(subject_map, validation, sizes):
    """Offline VLM-map budget; independent from detector-first production plan.

    Every key region must fit, rather than dropping a low-ranked subject. A
    stage-area strip represents an area, not resolved individual faces/eyes.
    """
    import math

    blockers = list(validation["blockers"])
    windows = []
    for track in subject_map["correspondence"]:
        if track["id"] not in validation["eligible_tracks"]:
            continue
        cells = []
        for ref in track["regions"]:
            region = validation["lookup"][ref]
            from .vlm_identity import identity
            width, height = sizes[identity(region["frame"])]
            x, y, xx, yy = region["box"]
            native = [math.ceil(xx*width)-math.floor(x*width),
                      math.ceil(yy*height)-math.floor(y*height)]
            if region["role"] not in {"key_subject", "supporting"}:
                blockers.append("BYSTANDER_OR_NONKEY_MAIN_ROI")
            if min(native) < 256:
                blockers.append("ROI_NATIVE_SHORT_EDGE_BELOW_256")
            cells.append({"region": ref, "native_size": native})
        windows.append({"track": track["id"], "cells": cells})
    if len(windows) > 3:
        blockers.append("STRIP_BUDGET_EXCEEDED")
    return {"status": "attention_only" if blockers else "offline_candidate",
            "blockers": sorted(set(blockers)), "windows": windows,
            "strip_min_short_edge": 256, "max_strips": 3,
            "bystander_faces_use_budget": False, "full_subject_coverage": False,
            "coverage_authority": "VLM_candidate_not_semantic_verification",
            "production_authority": False}
