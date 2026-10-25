"""Decision-neutral subject-map -> local CV -> deep review.

Fake and explicitly authorized bounded Bedrock transports are isolated.
No database writes, retries or fallback. Geometry is not semantic truth.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import subject_set_evidence as subjects
from . import vlm_criterion_compiler, vlm_prompt_context, vlm_review, vlm_visual_budget
from .vlm_identity import identity

VERSION = "two-pass-review-v2"
BUDGET = {
    "frames": 4,
    "regions": 16,
    "regions_per_frame": 4,
    "tracks": 3,
    "A": {
        "images": 4,
        "text_tokens_upper_bound": 6000,
        "output_tokens": 4096,
        "output_bytes": 12000,
        "calls": 1,
    },
    "B": {
        "images": 7,
        "text_tokens_upper_bound": 36000,
        "output_tokens": 4096,
        "output_bytes": 12000,
        "calls": 1,
    },
    "image_bytes": 16_000_000,
    "image_pixels": 20_000_000,
}
PROMPTS = {
    "A": "Inspect ALL complete high-resolution group pages. Map key subjects or interaction/stage areas, not just detectable faces. Use anonymous region IDs; no real names, identity inference or facial recognition. Report missing subjects, reflection/occlusion ambiguity and coverage uncertainty. Correspondence is a visual candidate, never identity. Return only the strict Pass A schema. Pixel boxes are normalized in the oriented full frame. Never claim independently verified completeness.",
    "B": "Review full group pages AND all comparison strips with scene map and CV evidence. CV measures, not subject authority or preference. Unknown stays unknown. Do not infer identity or names. Consider ensemble coverage, faces/eyes, lighting intent, interaction and distinct phases. Select only with medium/high decision support and sufficient coverage. Discriminators may include mixed or entirely low-confidence weak attention notes, which never provide necessary decision support. Each recommended frame needs medium/high evidence; multi-select and irreplaceability use only medium/high evidence. Multiple useful frames are allowed with evidence-bound reasons and frame+subject/phase irreplaceability. At most three discriminators. Obey context decision_ceiling: attention_only forbids recommendations or irreplaceability claims. Stage-area evidence permits only stage structure, lighting and frame changes, never individual-performer coverage. Supporting-region evidence is allowed. Bystander/unknown regions may support attention_only occlusion, distraction or composition observations only, never selection or irreplaceability. Every region reference resolves to its exact frame in scene_map. Each discriminator text is at most 500 Unicode characters. Return only strict Pass B schema. Review only; never delete or change stored decisions.",
}
ROLES = {"key_subject", "supporting", "bystander", "unknown"}
CONFIDENCE = {"low", "medium", "high"}
_ID = re.compile(r"^[RT][0-9]{1,2}$")
_FORBIDDEN = re.compile(
    r"keeper|keep_file_id|pair_margin|utility|decision|user_override", re.IGNORECASE
)


class ReviewError(ValueError):
    """Fail closed; caller may expose the reason for manual attention."""


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def _keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required.split()):
        raise ReviewError("schema fields mismatch")


def _enum(value, choices):
    if not isinstance(value, str) or value not in choices:
        raise ReviewError("invalid enum")


def _text(value, limit=500):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ReviewError("invalid bounded text")


def _list(value, limit, minimum=0):
    if not isinstance(value, list) or not minimum <= len(value) <= limit:
        raise ReviewError("list budget/schema exceeded")


def _frame(value, members):
    _keys(value, "file_id content_sha256")
    try:
        key = identity(value)
    except ValueError as exc:
        raise ReviewError(str(exc)) from exc
    if key not in members:
        raise ReviewError("file_id + SHA grounding mismatch")
    return key


def _box(box, *, main_roi=True):
    _list(box, 4, 4)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in box
    ):
        raise ReviewError("non-finite normalized box")
    x, y, xx, yy = box
    if not 0 <= x < xx <= 1 or not 0 <= y < yy <= 1:
        raise ReviewError("normalized box out of bounds")
    if main_roi and not 0.005 <= (xx - x) * (yy - y) <= 0.85:
        raise ReviewError("region area budget")
    return box


def validate_map(packet, rows):
    """Structural grounding only; coverage is always a model claim."""
    _keys(packet, "scene regime confidence coverage regions correspondence")
    _enum(packet["scene"], {"stage", "multi_person", "interaction", "unknown"})
    _enum(packet["regime"], {"stage", "group", "unknown"})
    _enum(packet["confidence"], CONFIDENCE)
    members = {identity(r): r for r in rows}
    _list(packet["coverage"], len(rows), len(rows))
    coverage, blockers, ceiling_reasons = {}, [], []
    for c in packet["coverage"]:
        _keys(c, "frame state uncertainty confidence")
        key = _frame(c["frame"], members)
        if key in coverage:
            raise ReviewError("duplicate coverage frame")
        coverage[key] = c
        _enum(c["state"], {"candidate_complete", "missing", "uncertain"})
        _enum(
            c["uncertainty"],
            {
                "none_reported",
                "missing_key",
                "occlusion",
                "reflection",
                "unresolved_roles",
                "resolution",
            },
        )
        _enum(c["confidence"], CONFIDENCE)
        if (
            c["state"] != "candidate_complete"
            or c["uncertainty"] != "none_reported"
            or c["confidence"] == "low"
        ):
            ceiling_reasons.append("COVERAGE_UNCERTAIN")
        if c["state"] == "missing" or c["uncertainty"] in {
            "missing_key",
            "reflection",
            "unresolved_roles",
        }:
            blockers.append("COVERAGE_UNRESOLVED")
    if packet["confidence"] == "low" or packet["regime"] == "unknown":
        ceiling_reasons.append("SCENE_UNCERTAIN")
    _list(packet["regions"], BUDGET["regions"])
    lookup = {}
    for r in packet["regions"]:
        _keys(r, "id frame box role kind confidence grounding")
        if (
            not isinstance(r["id"], str)
            or not _ID.fullmatch(r["id"])
            or not r["id"].startswith("R")
            or r["id"] in lookup
        ):
            raise ReviewError("duplicate/invalid region ID")
        frame = _frame(r["frame"], members)
        _enum(r["role"], ROLES)
        _box(r["box"], main_roi=r["role"] in {"key_subject", "supporting"})
        _enum(r["kind"], {"subject", "interaction", "stage_area"})
        _enum(r["confidence"], CONFIDENCE)
        _keys(r["grounding"], "page visible")
        if r["grounding"]["page"] != frame or r["grounding"]["visible"] is not True:
            raise ReviewError("region lacks full-page grounding")
        lookup[r["id"]] = r
        if r["role"] == "unknown" or (
            r["role"] == "key_subject" and r["confidence"] == "low"
        ):
            ceiling_reasons.append("ROLE_OR_REGION_UNCERTAIN")
    for frame in members:
        regions = [r for r in lookup.values() if identity(r["frame"]) == frame]
        if len(regions) > BUDGET["regions_per_frame"]:
            raise ReviewError("regions per frame budget")
        if not any(r["role"] == "key_subject" for r in regions):
            blockers.append("NO_KEY_SUBJECT_REGION")
        for i, a in enumerate(regions):
            for b in regions[i + 1 :]:
                if a["kind"] == b["kind"] and subjects.iou(a["box"], b["box"]) > 0.8:
                    raise ReviewError("duplicate/overlapping regions")
                if {a["role"], b["role"]} == {
                    "key_subject",
                    "bystander",
                } and subjects.iou(a["box"], b["box"]) > 0.3:
                    raise ReviewError("BYSTANDER_ROLE_CONFLICT")
    _list(packet["correspondence"], BUDGET["tracks"])
    used, track_ids = set(), set()
    eligible_tracks, excluded_tracks = [], []
    compared = set()
    for t in packet["correspondence"]:
        _keys(t, "id regions confidence state")
        if (
            not isinstance(t["id"], str)
            or not _ID.fullmatch(t["id"])
            or not t["id"].startswith("T")
            or t["id"] in track_ids
        ):
            raise ReviewError("invalid track ID")
        track_ids.add(t["id"])
        _list(t["regions"], len(rows), 2)
        _enum(t["confidence"], CONFIDENCE)
        _enum(t["state"], {"candidate", "ambiguous"})
        if any(not isinstance(r, str) or r not in lookup for r in t["regions"]):
            raise ReviewError("track grounding")
        regs = [lookup[r] for r in t["regions"]]
        if len({identity(r["frame"]) for r in regs}) != len(regs) or used.intersection(
            t["regions"]
        ):
            raise ReviewError("ambiguous duplicate track membership")
        used.update(t["regions"])
        if len({r["role"] for r in regs}) != 1 or len({r["kind"] for r in regs}) != 1:
            raise ReviewError("incompatible track roles/kinds")
        if regs[0]["role"] in {"bystander", "unknown"}:
            excluded_tracks.append(
                {"track": t["id"], "regions": t["regions"], "reason": "non_main_role"}
            )
            continue
        if (
            len(regs) != len(rows)
            or t["confidence"] == "low"
            or t["state"] == "ambiguous"
        ):
            blockers.append("CORRESPONDENCE_UNCERTAIN")
        else:
            eligible_tracks.append(t["id"])
            compared.update(t["regions"])
    if {r["id"] for r in lookup.values() if r["role"] == "key_subject"} - compared:
        blockers.append("KEY_REGION_NOT_COMPARED")
    ceiling_reasons.extend(blockers)
    return {
        "lookup": lookup,
        "eligible_tracks": eligible_tracks,
        "excluded_tracks": excluded_tracks,
        "decision_ceiling": "attention_only" if ceiling_reasons else "select",
        "ceiling_reasons": sorted(set(ceiling_reasons)),
        "blockers": sorted(set(blockers)),
        "coverage_authority": "model_claim_not_verified_completeness",
        "full_subject_coverage": False,
        "criteria_priority": list(subjects.CRITERIA),
    }


def _safe(value):
    """Drop selection authority and raw vectors before the existing CV compiler."""
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items() if not _FORBIDDEN.search(k)}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return value


def _assert_no_selection_authority(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if _FORBIDDEN.search(key):
                raise ReviewError("selection authority leaked into model evidence")
            _assert_no_selection_authority(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_selection_authority(child)


def _object(value):
    return json.loads(value) if isinstance(value, str) else (value or {})


def _pixels(im, box):
    x, y, xx, yy = box
    return (
        math.floor(x * im.width),
        math.floor(y * im.height),
        math.ceil(xx * im.width),
        math.ceil(yy * im.height),
    )


def _measurement(im):
    gray = np.asarray(im.convert("L"), dtype=np.float32) / 255
    return {
        "luminance_mean": round(float(gray.mean()), 5),
        "dark_fraction": round(float((gray < 0.03).mean()), 5),
        "bright_fraction": round(float((gray > 0.97).mean()), 5),
        "gradient_energy": round(
            float(
                np.abs(np.diff(gray, axis=0)).mean()
                + np.abs(np.diff(gray, axis=1)).mean()
            ),
            5,
        ),
    }


def _contained(inner, outer):
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    intersection = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0])) * max(
        0, min(inner[3], outer[3]) - max(inner[1], outer[1])
    )
    return area > 0 and intersection / area >= 0.9


def _box_observations(records, field, im, box):
    """Keep all spatially joined scalar observations, not global aggregate state."""
    observations = []
    for index, record in enumerate(records):
        b = record.get(field)
        if not isinstance(b, list) or len(b) != 4:
            continue
        x, y, w, h = b
        normalized = [
            x / im.width,
            y / im.height,
            (x + w) / im.width,
            (y + h) / im.height,
        ]
        if _contained(normalized, box):
            observations.append(
                {
                    "source_index": index,
                    **{
                        k: v
                        for k, v in record.items()
                        if k != field
                        and (v is None or isinstance(v, (str, bool, int, float)))
                    },
                }
            )
    return observations


def enrich(rows, images, packet, validation):
    """Map observations spatially; no detector defines region membership/role.

    Face/eye observations are not inherited from group-level aggregates. Unknown
    metrics stay unknown; stored region IQA is labeled as overlapping measurement,
    never rebranded as newly measured ROI IQA.
    """
    clean_rows = []
    for row in rows:
        clean = {k: row[k] for k in vlm_review._CV_MEMBER_FIELDS if k in row}
        for k in ("quality_meta", "faces_json", "evidence_json"):
            if k in clean:
                clean[k] = _safe(_object(clean[k]))
        clean_rows.append(_safe(clean))
    members, audit, full_l1, full_l2 = vlm_review.build_layers(
        clean_rows, enforce_gate=False
    )
    metrics, ledger = {}, []
    by_key = {identity(r): r for r in clean_rows}
    for rid, region in validation["lookup"].items():
        key, box = identity(region["frame"]), region["box"]
        row, im = by_key[key], images[key]
        q = _object(row.get("quality_meta"))
        local = q.get("local_region_set", {}).get("regions", [])
        matched = [
            r
            for r in local
            if r.get("box_normalized") and subjects.iou(box, r["box_normalized"]) >= 0.5
        ]
        face_observations = _box_observations(
            _object(row.get("faces_json")) or [], "bbox", im, box
        )
        eye_observations = _box_observations(
            q.get("eye_detection", {}).get("faces", []), "bbox_xywh", im, box
        )
        poses = [
            p
            for p in q.get("pose_evidence", {}).get("people", [])
            if subjects.iou(box, p["box"]) >= 0.5
        ]
        pose = {"state": "unknown", "reason": "no_unique_spatial_join"}
        if len(poses) == 1:
            pose = {
                "state": "measured",
                "visible_keypoints": sum(
                    p[2] >= 0.5 for p in poses[0].get("keypoints", [])
                ),
                "phase": "unknown",
            }
        m = {
            "frame": key,
            "pixels": _measurement(im.crop(_pixels(im, box))),
            "stored_local_quality": [
                {k: r.get(k) for k in ("region_id", "kind", "musiq", "clipiqa")}
                for r in matched
            ],
            "stored_quality_scope": "overlapping_detector_region_not_new_ROI_measurement",
            "faces": face_observations or "unknown",
            "eyes": eye_observations or "unknown_no_eye_box_join",
            "pose": pose,
            "completeness": "unknown",
            "occlusion": "unknown",
            "edge_touch": min(box[0], box[1], 1 - box[2], 1 - box[3]) <= 0.01,
            "semantic_phase": "unknown",
            "new_ROI_musiq_clipiqa": "unknown_not_run",
        }
        metrics[rid] = m
        ledger.append(
            {
                "region": rid,
                "role": region["role"],
                "full_page": True,
                "strip_eligible": any(
                    t["id"] in validation["eligible_tracks"] and rid in t["regions"]
                    for t in packet["correspondence"]
                ),
                "exclusion_reason": (
                    "non_main_role"
                    if region["role"] in {"bystander", "unknown"}
                    else "not_excluded_by_role"
                ),
                "faces_eyes_complete": False,
            }
        )
    similarities = []
    for t in packet["correspondence"]:
        if t["id"] not in validation["eligible_tracks"]:
            continue
        signatures = []
        for rid in t["regions"]:
            r = validation["lookup"][rid]
            crop = images[identity(r["frame"])].crop(
                _pixels(images[identity(r["frame"])], r["box"])
            )
            signatures.append(
                np.asarray(crop.resize((32, 32)).convert("RGB"), dtype=np.float32) / 255
            )
        similarities.append(
            {
                "track": t["id"],
                "regions": t["regions"],
                "mean_absolute_rgb_delta": [
                    round(float(np.abs(a - signatures[0]).mean()), 5)
                    for a in signatures[1:]
                ],
                "authority": "unregistered_appearance_not_identity_or_semantic_similarity",
            }
        )
    l1 = {
        "layer": "L1",
        "ROI": metrics,
        "region_similarity": similarities,
        "full_frame": full_l1,
        "coverage_ledger": ledger,
        "criteria_priority": list(subjects.CRITERIA),
        "subject_map_authority": validation["coverage_authority"],
    }
    l2 = {
        "layer": "L2",
        "frame_bindings": {r["frame_alias"]: identity(r) for r in members},
        "ROI": metrics,
        "region_similarity": similarities,
        "full_frame": full_l2,
        "coverage_ledger": ledger,
        "criteria_priority": list(subjects.CRITERIA),
    }
    _assert_no_selection_authority(l2["ROI"])
    vlm_review.assert_model_context_safe(full_l2)
    return audit, l1, l2


def _asset(path, cells, kind):
    with Image.open(path) as im:
        size = list(im.size)
    return {
        "path": path.name,
        "sha256": sha(path.read_bytes()),
        "bytes": path.stat().st_size,
        "size": size,
        "kind": kind,
        "cells": cells,
    }


def _render_cell(im, minimum, target):
    if min(im.size) < minimum:
        raise ReviewError("native image short edge too small")
    scale = min(1, target / min(im.size))
    return im.resize(
        (round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS
    )


def render_pages(images, output):
    assets = []
    for i, (key, im) in enumerate(images.items()):
        shown = _render_cell(im, 1024, 1536)
        path = output / f"page-{i + 1}.jpg"
        shown.save(path, quality=92)
        assets.append(
            _asset(
                path,
                [
                    {
                        "frame": key,
                        "native_size": list(im.size),
                        "actual_image_size": list(shown.size),
                        "upsampled": False,
                    }
                ],
                "full_group_page",
            )
        )
    return assets


def render_strips(packet, validation, images, output):
    plan = vlm_visual_budget.subject_map_plan(
        packet, validation, {k: im.size for k, im in images.items()}
    )
    if plan["blockers"]:
        raise ReviewError(";".join(plan["blockers"]))
    assets = []
    for t in packet["correspondence"]:
        if t["id"] not in validation["eligible_tracks"]:
            continue
        cells, pictures = [], []
        for rid in t["regions"]:
            r = validation["lookup"][rid]
            im = images[identity(r["frame"])]
            box = _pixels(im, r["box"])
            crop = im.crop(box)
            shown = _render_cell(crop, 256, 384)
            pictures.append(shown)
            cells.append(
                {
                    "region": rid,
                    "frame": identity(r["frame"]),
                    "pixel_box": list(box),
                    "native_size": list(crop.size),
                    "actual_image_size": list(shown.size),
                    "upsampled": False,
                }
            )
        strip = Image.new(
            "RGB",
            (sum(p.width for p in pictures), max(p.height for p in pictures) + 32),
            "white",
        )
        x = 0
        for cell, pic in zip(cells, pictures):
            strip.paste(pic, (x, 32))
            ImageDraw.Draw(strip).text(
                (x + 4, 8),
                f"{t['id']} {cell['region']} {cell['frame'].split('@')[0]}",
                fill="black",
            )
            x += pic.width
        path = output / f"strip-{t['id']}.jpg"
        strip.save(path, quality=95)
        assets.append(_asset(path, cells, "subject_or_interaction_strip"))
    return assets, plan


def request(pass_name, model, text, assets, output):
    if not isinstance(model, str) or not model.strip():
        raise ReviewError("explicit model required for each pass")
    payload = {
        "version": VERSION,
        "pass": pass_name,
        "model": model,
        "prompt": PROMPTS[pass_name],
        "context": text,
        "images": assets,
        "max_output_tokens": BUDGET[pass_name]["output_tokens"],
    }
    schema_name = "subject-map" if pass_name == "A" else "deep-judge"
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / f"two-pass-{schema_name}-v1.schema.json"
    )
    payload["response_schema"] = json.loads(schema_path.read_text())
    # UTF-8 byte count is a conservative token ceiling for byte-level tokenizers;
    # image tokens are separately bounded by image pixels, not this text number.
    text_bound = len(
        canonical({k: v for k, v in payload.items() if k != "images"})
    ) + len(canonical(assets))
    if (
        len(assets) > BUDGET[pass_name]["images"]
        or text_bound > BUDGET[pass_name]["text_tokens_upper_bound"]
    ):
        raise ReviewError(f"{pass_name} text/image budget exceeded")
    if (
        sum(a["bytes"] for a in assets) > BUDGET["image_bytes"]
        or sum(math.prod(a["size"]) for a in assets) > BUDGET["image_pixels"]
    ):
        raise ReviewError("visual byte/pixel budget exceeded")
    for a in assets:
        if sha((output / a["path"]).read_bytes()) != a["sha256"]:
            raise ReviewError("asset SHA mismatch")
    payload["text_token_upper_bound"] = text_bound
    return payload


def validate_judgment(p, rows, packet, validation, l2, *, allow_visual_views=False):
    _keys(
        p,
        "decision confidence recommended multi_select_reason irreplaceable discriminators",
    )
    _enum(p["decision"], {"select", "attention_only", "abstain"})
    _enum(p["confidence"], CONFIDENCE)
    members = {identity(r): r for r in rows}
    _list(p["recommended"], len(rows))
    _list(p["irreplaceable"], 8)
    _list(p["discriminators"], 3)
    lookup = validation["lookup"]

    def evidence(
        items, frame=None, selecting=False, attention_context=False, require_one=False
    ):
        _list(items, 8, 1)
        strong = 0
        for e in items:
            _keys(e, "region confidence")
            if e["region"] not in lookup:
                raise ReviewError("ungrounded evidence region")
            r = lookup[e["region"]]
            bound_frame = _frame(r["frame"], members)
            if (
                r["role"] not in {"key_subject", "supporting"}
                and not (allow_visual_views and r["role"] == "frame_view")
                and not (attention_context and r["role"] in {"bystander", "unknown"})
            ) or (frame and bound_frame != frame):
                raise ReviewError("evidence not bound to recommended frame/key subject")
            _enum(e["confidence"], CONFIDENCE)
            is_strong = e["confidence"] != "low" and r["confidence"] != "low"
            strong += int(is_strong)
            if selecting and not require_one and not is_strong:
                raise ReviewError("low evidence cannot support select")
        if selecting and require_one and not strong:
            raise ReviewError(
                "low evidence cannot support select without medium/high binding"
            )

    selecting = p["decision"] == "select"
    seen = set()
    for r in p["recommended"]:
        _keys(r, "frame evidence")
        key = _frame(r["frame"], members)
        if key in seen:
            raise ReviewError("duplicate recommendation")
        seen.add(key)
        evidence(r["evidence"], key, selecting, require_one=True)
    if selecting and (
        not seen
        or not p["discriminators"]
        or p["confidence"] == "low"
        or validation["blockers"]
        or validation["decision_ceiling"] != "select"
    ):
        raise ReviewError(
            "select requires medium/high grounded evidence and coverage; decision ceiling enforced"
        )
    if p["decision"] != "select" and (seen or p["irreplaceable"]):
        raise ReviewError("attention/abstain cannot emit recommendations")
    if len(seen) > 1:
        _keys(p["multi_select_reason"], "text evidence")
        _text(p["multi_select_reason"]["text"])
        evidence(p["multi_select_reason"]["evidence"], selecting=True)
        if {
            identity(lookup[e["region"]]["frame"])
            for e in p["multi_select_reason"]["evidence"]
        } != seen:
            raise ReviewError("multi-select reason must bind every recommended frame")
    elif p["multi_select_reason"] is not None:
        raise ReviewError("unexpected multi-select reason")
    for item in p["irreplaceable"]:
        _keys(item, "frame subject phase evidence")
        key = _frame(item["frame"], members)
        if (
            key not in seen
            or item["subject"] not in lookup
            or identity(lookup[item["subject"]]["frame"]) != key
        ):
            raise ReviewError("irreplaceable subject/frame mismatch")
        if item["phase"] is not None:
            _enum(
                item["phase"],
                {"interaction", "gesture", "formation", "lighting", "appearance"},
            )
        evidence(item["evidence"], key, selecting)
        if item["subject"] not in {e["region"] for e in item["evidence"]}:
            raise ReviewError("irreplaceable lacks subject evidence")
    for item in p["discriminators"]:
        _keys(item, "text evidence")
        _text(item["text"])
        nonmain = any(
            lookup.get(e.get("region"), {}).get("role") in {"bystander", "unknown"}
            for e in item["evidence"]
        )
        attention_context = p["decision"] == "attention_only" and bool(
            re.search(
                r"occlu|obstruct|overlap|distract|interfer|composition|framing|遮挡|干扰|构图",
                item["text"],
                re.IGNORECASE,
            )
        )
        if nonmain and not attention_context:
            raise ReviewError(
                "non-main evidence requires attention-only visual context"
            )
        evidence(item["evidence"], attention_context=attention_context)
    return copy.deepcopy(p)


class FakeModel:
    """Deterministic transport substitute, explicitly incapable of network calls."""

    def __init__(self, responses):
        self.responses = copy.deepcopy(responses)
        self.calls = []

    def invoke(self, payload):
        phase = payload["pass"]
        if any(c["pass"] == phase for c in self.calls):
            raise ReviewError("implicit retry forbidden")
        if not payload["model"].startswith("fake:"):
            raise ReviewError("offline fake runner requires explicit fake: model IDs")
        self.calls.append(
            {
                "pass": phase,
                "model": payload["model"],
                "request_sha": sha(canonical(payload)),
                "text_token_upper_bound": payload["text_token_upper_bound"],
                "images": len(payload["images"]),
                "external_calls": 0,
                "billed_input_tokens": None,
                "billed_output_tokens": None,
            }
        )
        self.calls[-1]["response_bytes"] = len(canonical(self.responses[phase]))
        if self.calls[-1]["response_bytes"] > BUDGET[phase]["output_bytes"]:
            raise ReviewError("response byte budget exceeded; no retry")
        return copy.deepcopy(self.responses[phase])


def _input_key(rows, model_a, model_b, runner):
    def convert(value):
        if isinstance(value, bytes):
            return {"binary_sha": sha(value)}
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return sha(
        canonical(
            {
                "version": VERSION,
                "budget": BUDGET,
                "prompts": PROMPTS,
                "schema_hashes": {
                    p.name: sha(p.read_bytes())
                    for p in (Path(__file__).resolve().parents[1] / "schemas").glob(
                        "two-pass-*.schema.json"
                    )
                },
                "implementation_hashes": {
                    p.name: sha(p.read_bytes())
                    for p in (
                        Path(__file__),
                        Path(vlm_visual_budget.__file__),
                        Path(subjects.__file__),
                        Path(vlm_review.__file__),
                        Path(vlm_criterion_compiler.__file__),
                        Path(vlm_prompt_context.__file__),
                    )
                },
                "models": [model_a, model_b],
                "rows": convert(rows),
                "fake_responses": getattr(runner, "responses", None),
            }
        )
    )


def run_offline(rows, sources, output, *, model_a, model_b, runner):
    """One group, one attempt per pass, inert sidecar only; no production adapter."""
    if type(runner) is not FakeModel:
        raise ReviewError("only deterministic FakeModel is allowed")
    if runner.calls:
        raise ReviewError("runner already consumed; implicit retry forbidden")
    if any(
        not isinstance(m, str) or not m.startswith("fake:") or len(m) <= 5
        for m in (model_a, model_b)
    ):
        raise ReviewError("explicit fake model required for BOTH passes")
    if not 2 <= len(rows) <= BUDGET["frames"] or len(
        {identity(r) for r in rows}
    ) != len(rows):
        raise ReviewError("frame budget or duplicate identity")
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(root):
        raise ReviewError("output must stay in repository")
    output.mkdir(parents=True, exist_ok=True)
    images = {}
    for row in rows:
        path = Path(sources[row["file_id"]])
        if sha(path.read_bytes()) != row["content_sha256"]:
            raise ReviewError("source SHA mismatch")
        with Image.open(path) as opened:
            im = ImageOps.exif_transpose(opened).convert("RGB")
        if im.size != (row["width"], row["height"]):
            raise ReviewError("oriented image dimensions mismatch")
        images[identity(row)] = im
    key = _input_key(rows, model_a, model_b, runner)
    cache = output / "sidecar.json"
    if cache.exists():
        prior = json.loads(cache.read_text())
        if prior.get("cache_key") == key and prior.get("version") == VERSION:
            body = {k: v for k, v in prior.items() if k != "sidecar_sha"}
            if prior.get("sidecar_sha") == sha(canonical(body)) and all(
                (output / a["path"]).exists()
                and sha((output / a["path"]).read_bytes()) == a["sha256"]
                for a in prior["assets"]
            ):
                return {**prior, "cache_hit": True, "calls_this_run": 0}
    result = {
        "version": VERSION,
        "cache_key": key,
        "mode": "offline_fake_review_only",
        "production_enabled": False,
        "real_model_effectiveness": "not_tested",
        "assets": [],
        "external_model_calls": 0,
        "decision_table_writes": 0,
        "blockers": [],
    }
    try:
        pages = render_pages(images, output)
        result["assets"] = pages
        # Pass A receives no CV/detector/selection authority, only full pages and minimal header.
        a_request = request(
            "A",
            model_a,
            {"frames": list(images), "scene_hint": "multi_subject_or_stage"},
            pages,
            output,
        )
        (output / "pass-A-request.json").write_bytes(canonical(a_request))
        a = runner.invoke(a_request)
        result["pass_A"] = a
        validation = validate_map(a, rows)
        result["subject_set_evidence"] = validation
        result["decision_ceiling"] = validation["decision_ceiling"]
        if validation["blockers"]:
            raise ReviewError(";".join(validation["blockers"]))
        strips, plan = render_strips(a, validation, images, output)
        result["assets"] += strips
        result["visual_plan"] = plan
        audit, l1, l2 = enrich(rows, images, a, validation)
        for name, layer in (("L0", audit), ("L1", l1), ("L2", l2)):
            (output / f"{name}.json").write_bytes(canonical(layer))
        b_request = request(
            "B",
            model_b,
            {
                "scene_map": a,
                "evidence": l2,
                "decision_ceiling": validation["decision_ceiling"],
                "coverage_limits": validation["ceiling_reasons"],
            },
            pages + strips,
            output,
        )
        (output / "pass-B-request.json").write_bytes(canonical(b_request))
        b = runner.invoke(b_request)
        result["pass_B_raw"] = b
        result["judgment"] = validate_judgment(b, rows, a, validation, l2)
    except (ReviewError, ValueError, KeyError, TypeError) as exc:
        result["blockers"].append(str(exc))
        result["judgment"] = {
            "decision": "attention_only",
            "confidence": "low",
            "recommended": [],
            "multi_select_reason": None,
            "irreplaceable": [],
            "discriminators": [],
        }
    result["accounting"] = runner.calls
    result["calls_this_run"] = len(runner.calls)
    result["cache_hit"] = False
    result["sidecar_sha"] = sha(canonical(result))
    cache.write_bytes(canonical(result))
    return result


ALLOWED_BEDROCK_MODELS = frozenset(
    {
        "global.anthropic.claude-sonnet-5-5",
        "global.openai.gpt-5.6-terra",
    }
)
PRIVACY_INSTRUCTION = (
    " Return JSON only, without fences. No names, identity guesses, OCR, personal "
    "details or demographic/relationship inference. Use anonymous region IDs. "
    "For stage scenes prefer a stage_area when individual performers are too small; "
    "foreground audience are bystanders. For multi-person scenes cover all main "
    "subjects/interaction areas, not merely the largest face. Reflection ambiguity "
    "must remain uncertain. Text reasons must be English visual observations only."
)
# Frozen v2/v3 receipts above must remain byte-verifiable. New batches and the
# future LiteLLM adapter append this instruction under a new contract version.
CHINESE_OUTPUT_INSTRUCTION = (
    " Write every human-readable reason in concise Simplified Chinese. Keep JSON "
    "keys, enum values, frame IDs and region IDs exactly as the schema defines "
    "them in English."
)
# Conservative lexical tripwires, not a complete semantic privacy classifier.
_PRIVACY = re.compile(
    r"https?://|www\.|[\w.+-]+@[\w.-]+\.[a-z]{2,}|"
    r"\b(?:named|surname|celebrity|identity|recogniz(?:e|ed)|phone(?=\s*(?:number\b|:))|address|"
    r"wife|husband|daughter|son|ethnicity|religion)\b|\+?\d[\d ()-]{8,}\d",
    re.IGNORECASE,
)


def privacy_check(packet):
    """Inspect free prose only; schema removes undeclared personal-data fields."""
    if isinstance(packet, dict):
        for key, value in packet.items():
            if (
                key == "text"
                and isinstance(value, str)
                and (_PRIVACY.search(value) or any(ord(c) > 127 for c in value))
            ):
                raise ReviewError("privacy prose tripwire; manual review required")
            privacy_check(value)
    elif isinstance(packet, list):
        for value in packet:
            privacy_check(value)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReviewError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ReviewError(f"non-finite JSON: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


class BedrockAdapter:
    """Explicit opt-in transport; durable reservation makes uncertain failures final.

    One shared adapter owns the entire two-group/four-call budget. A reservation is
    written BEFORE network I/O. Restarting never repeats a consumed group/pass.
    Cost rates are explicit estimates, never represented as provider invoice rates.
    """

    def __init__(self, output, *, model_id, rates, client=None):
        if model_id not in ALLOWED_BEDROCK_MODELS:
            raise ReviewError("model outside strict Bedrock allowlist")
        self.output = Path(output).resolve()
        root = Path(__file__).resolve().parents[1]
        if not self.output.is_relative_to(root):
            raise ReviewError("output must stay in repository")
        self.output.mkdir(parents=True, exist_ok=True)
        _keys(rates, "input_per_million output_per_million basis")
        for key in ("input_per_million", "output_per_million"):
            if (
                type(rates[key]) not in (int, float)
                or not math.isfinite(rates[key])
                or rates[key] <= 0
            ):
                raise ReviewError("explicit positive estimated cost rates required")
        _text(rates["basis"], 500)
        self.rates = rates
        self.model_id = model_id
        self.ledger_path = self.output / "call-ledger.json"
        self.calls = (
            json.loads(self.ledger_path.read_text())
            if self.ledger_path.exists()
            else []
        )
        self.client = client

    def _save(self):
        self.ledger_path.write_bytes(canonical(self.calls))

    def invoke(self, payload, *, group, assets_dir):
        assets_dir = Path(assets_dir).resolve()
        if not assets_dir.is_relative_to(self.output):
            raise ReviewError("assets must stay inside pilot output")
        if (
            payload["model"] != self.model_id
            or payload["model"] not in ALLOWED_BEDROCK_MODELS
        ):
            raise ReviewError("model mismatch; no upgrade/fallback")
        if group not in {"2026-07-25-G004", "2026-04-05-G007"}:
            raise ReviewError("group outside pilot allowlist")
        phase = payload["pass"]
        if phase not in {"A", "B"}:
            raise ReviewError("invalid pass")
        if len(self.calls) >= 4 or any(
            c["group"] == group and c["pass"] == phase for c in self.calls
        ):
            raise ReviewError("call budget/implicit retry forbidden")
        # Reconstruct budgeted request so callers cannot smuggle larger metadata.
        expected = request(
            phase,
            self.model_id,
            payload["context"],
            payload["images"],
            Path(assets_dir),
        )
        if expected != payload:
            raise ReviewError("request differs from validated budget contract")
        _assert_no_selection_authority(
            payload["context"]
            if phase == "A"
            else payload["context"]["evidence"].get("ROI", {})
        )
        content = [
            {
                "text": canonical(
                    {k: v for k, v in payload.items() if k != "images"}
                ).decode()
                + PRIVACY_INSTRUCTION
            }
        ]
        total_bytes = total_pixels = 0
        for asset in payload["images"]:
            path = (Path(assets_dir) / asset["path"]).resolve()
            if not path.is_relative_to(Path(assets_dir).resolve()):
                raise ReviewError("asset path escape")
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG" or list(image.size) != asset["size"]:
                    raise ReviewError("image manifest mismatch")
                total_pixels += image.width * image.height
            if len(data) != asset["bytes"] or sha(data) != asset["sha256"]:
                raise ReviewError("image byte/SHA mismatch")
            total_bytes += len(data)
            content.extend(
                [
                    {"text": canonical(asset).decode()},
                    {"image": {"format": "jpeg", "source": {"bytes": data}}},
                ]
            )
        text_bytes = sum(len(c["text"].encode()) for c in content if "text" in c)
        if (
            text_bytes > BUDGET[phase]["text_tokens_upper_bound"]
            or total_bytes > BUDGET["image_bytes"]
            or total_pixels > BUDGET["image_pixels"]
        ):
            raise ReviewError("wire text/image budget exceeded")
        wire = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {"maxTokens": BUDGET[phase]["output_tokens"]},
        }
        # Base64 is a lossless audit serialization of SDK byte fields.
        serial = json.dumps(
            wire,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda b: {"bytes_base64": base64.b64encode(b).decode()},
        ).encode()
        case = Path(assets_dir)
        entry = {
            "group": group,
            "pass": phase,
            "model_id": self.model_id,
            "request_sha256": sha(serial),
            "payload_sha256": sha(canonical(payload)),
            "text_bytes": text_bytes,
            "image_bytes": total_bytes,
            "image_pixels": total_pixels,
            "images": len(payload["images"]),
            "external_calls": 1,
            "state": "reserved",
            "usage": None,
            "estimated_cost_usd": None,
            "rates": self.rates,
        }
        self.calls.append(entry)
        self._save()
        (case / f"pass-{phase}-wire-request.json").write_bytes(serial)
        started = time.perf_counter()
        try:
            if self.client is None:
                import boto3
                from botocore.config import Config

                self.client = boto3.client(
                    "bedrock-runtime",
                    region_name="us-east-1",
                    config=Config(
                        retries={"total_max_attempts": 1, "mode": "standard"},
                        connect_timeout=20,
                        read_timeout=300,
                    ),
                )
            response = self.client.converse(**wire)
            response_bytes = canonical(response)
            (case / f"pass-{phase}-provider-response.json").write_bytes(response_bytes)
            entry.update(
                response_sha256=sha(response_bytes),
                usage=response.get("usage"),
                stop_reason=response.get("stopReason"),
                provider_request_id=response.get("ResponseMetadata", {}).get(
                    "RequestId"
                ),
            )
            usage = response.get("usage", {})
            if any(
                type(usage.get(k)) is not int or usage[k] < 0
                for k in ("inputTokens", "outputTokens", "totalTokens")
            ):
                raise ReviewError("missing/invalid provider usage")
            entry["estimated_cost_usd"] = round(
                (
                    usage["inputTokens"] * self.rates["input_per_million"]
                    + usage["outputTokens"] * self.rates["output_per_million"]
                )
                / 1_000_000,
                8,
            )
            if (
                usage["totalTokens"] != usage["inputTokens"] + usage["outputTokens"]
                or usage["outputTokens"] > BUDGET[phase]["output_tokens"]
            ):
                raise ReviewError("usage/output token budget mismatch")
            if any(
                usage.get(k, 0)
                for k in ("cacheReadInputTokens", "cacheWriteInputTokens")
            ):
                raise ReviewError("unexpected cache usage; cost estimate invalid")
            if response.get("stopReason") != "end_turn":
                raise ReviewError("nonterminal/truncated response; no retry")
            blocks = response["output"]["message"]["content"]
            if any(set(b) != {"text"} for b in blocks):
                raise ReviewError("unexpected response content block")
            text = "\n".join(b["text"] for b in blocks)
            (case / f"pass-{phase}-raw.txt").write_text(text)
            entry["response_text_sha256"] = sha(text.encode())
            entry["response_text_bytes"] = len(text.encode())
            if len(text.encode()) > BUDGET[phase]["output_bytes"]:
                raise ReviewError("response byte budget exceeded")
            parsed = strict_json(text)
            import jsonschema

            try:
                jsonschema.validate(parsed, payload["response_schema"])
            except jsonschema.ValidationError as exc:
                raise ReviewError(
                    f"response schema invalid: {exc.message[:160]}"
                ) from exc
            privacy_check(parsed)
            entry["state"] = "response_validated"
            return parsed
        except Exception as exc:
            entry.update(
                state="failed_no_retry", error=f"{type(exc).__name__}: {str(exc)[:400]}"
            )
            raise ReviewError(entry["error"]) from exc
        finally:
            entry["latency_seconds"] = round(time.perf_counter() - started, 3)
            self._save()


def run_real(rows, sources, output, *, adapter, group):
    """Private review artifacts only. Invalid A never reaches enrichment or B."""
    output = Path(output).resolve()
    if (
        not output.is_relative_to(adapter.output)
        or len(rows) < 2
        or len(rows) > BUDGET["frames"]
        or len({identity(r) for r in rows}) != len(rows)
    ):
        raise ReviewError("pilot path/frame budget")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "sidecar.json").exists():
        raise ReviewError("real pilot output already exists; no automatic replay")
    result = {
        "version": VERSION,
        "mode": "real_bedrock_review_only",
        "production_enabled": False,
        "group": group,
        "assets": [],
        "blockers": [],
        "decision_table_writes": 0,
        "pass_A_status": "not_run",
        "pass_B_status": "not_run",
    }
    images = {}
    try:
        for row in rows:
            path = Path(sources[row["file_id"]])
            if sha(path.read_bytes()) != row["content_sha256"]:
                raise ReviewError("source SHA mismatch")
            with Image.open(path) as opened:
                im = ImageOps.exif_transpose(opened).convert("RGB")
            if im.size != (row["width"], row["height"]):
                raise ReviewError("oriented image dimensions mismatch")
            images[identity(row)] = im
        pages = render_pages(images, output)
        result["assets"] = pages
        ar = request(
            "A",
            adapter.model_id,
            {"frames": list(images), "scene_hint": "multi_subject_or_stage"},
            pages,
            output,
        )
        (output / "pass-A-request.json").write_bytes(canonical(ar))
        result["pass_A_status"] = "failed"
        a = adapter.invoke(ar, group=group, assets_dir=output)
        (output / "pass-A-raw.json").write_bytes(canonical(a))
        validation = validate_map(a, rows)
        (output / "pass-A-validated.json").write_bytes(
            canonical({"map": a, "validation": validation})
        )
        result["subject_set_evidence"] = validation
        result["decision_ceiling"] = validation["decision_ceiling"]
        if validation["blockers"]:
            raise ReviewError(";".join(validation["blockers"]))
        result["pass_A_status"] = "passed_structural_only"
        strips, plan = render_strips(a, validation, images, output)
        result["assets"] += strips
        result["visual_plan"] = plan
        audit, l1, l2 = enrich(rows, images, a, validation)
        for name, layer in (("L0", audit), ("L1", l1), ("L2", l2)):
            (output / f"{name}.json").write_bytes(canonical(layer))
        br = request(
            "B",
            adapter.model_id,
            {
                "scene_map": a,
                "evidence": l2,
                "decision_ceiling": validation["decision_ceiling"],
                "coverage_limits": validation["ceiling_reasons"],
            },
            pages + strips,
            output,
        )
        (output / "pass-B-request.json").write_bytes(canonical(br))
        result["pass_B_status"] = "failed"
        b = adapter.invoke(br, group=group, assets_dir=output)
        (output / "pass-B-raw.json").write_bytes(canonical(b))
        result["judgment"] = validate_judgment(b, rows, a, validation, l2)
        (output / "pass-B-validated.json").write_bytes(canonical(result["judgment"]))
        result["pass_B_status"] = "passed"
    except (ReviewError, ValueError, KeyError, TypeError) as exc:
        result["blockers"].append(str(exc))
        result["judgment"] = {
            "decision": "attention_only",
            "confidence": "low",
            "recommended": [],
            "multi_select_reason": None,
            "irreplaceable": [],
            "discriminators": [],
        }
    # Explicit absent-layer records prevent a stopped A being mistaken for fake CV/B.
    for name in (
        "pass-A-raw",
        "pass-A-validated",
        "pass-B-raw",
        "pass-B-validated",
        "L0",
        "L1",
        "L2",
    ):
        path = output / f"{name}.json"
        if not path.exists():
            path.write_bytes(
                canonical({"status": "not_produced", "reason": result["blockers"]})
            )
    result["accounting"] = [c for c in adapter.calls if c["group"] == group]
    result["external_model_calls"] = len(result["accounting"])
    result["sidecar_sha"] = sha(canonical(result))
    (output / "sidecar.json").write_bytes(canonical(result))
    return result
