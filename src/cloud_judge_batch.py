"""Read-only frozen CV -> local scene router -> complete visuals -> ONE Sonnet judge.

Artifacts only. Durable per-group reservations survive crashes; no retry/fallback.
The runner has no decision-table or source-file write API.
"""

from __future__ import annotations

import base64
import copy
import fcntl
import io
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jsonschema
from PIL import Image, ImageOps

from . import db
from . import two_pass_review as tp
from .readonly_features import readonly_inventory

ROOT = Path(__file__).resolve().parents[1]
VERSION = "cloud-judge-batch-v1"
MODEL = "global.anthropic.claude-sonnet-5-5"
DATES = (
    "2026-04-04",
    "2026-04-05",
    "2026-06-23",
    "2026-07-25",
    "2026-07-26",
    "2026-07-31",
)
TOTALS = {"dates": 6, "files": 2065, "groups": 194}


def save(path, value):
    """Atomic/fsynced checkpoints, including before dispatching external I/O."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        f.write(tp.canonical(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path):
    return tp.strict_json(Path(path).read_text())


def verify(condition, reason):
    if not condition:
        raise tp.ReviewError(reason)


def artifact_root(path):
    path = Path(path).resolve()
    verify(
        path.is_relative_to(ROOT / "artifacts"), "output must stay in repo artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot(eval_root):
    """Exactly one inventory per date; variants/duplicate June dates are excluded."""
    root = Path(eval_root).resolve()
    inventories, groups = [], []
    files = 0
    for date in DATES:
        p = (
            root
            / ("local-holdout" if date == "2026-06-23" else "development")
            / date
            / "inventory.sqlite"
        )
        with readonly_inventory(p) as conn:
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            files += count
            ids = [r[0] for r in conn.execute("SELECT id FROM groups ORDER BY id")]
            for gid in ids:
                rows = [dict(r) for r in db.group_page_by_id(conn, gid)]
                groups.append(
                    {
                        "group": f"{date}-G{gid:03d}",
                        "date": date,
                        "group_id": gid,
                        "frames": [
                            {k: r[k] for k in ("file_id", "content_sha256")}
                            for r in rows
                        ],
                    }
                )
        inventories.append(
            {
                "date": date,
                "path": str(p),
                "sha256": tp.sha(p.read_bytes()),
                "files": count,
                "groups": len(ids),
            }
        )
    totals = {"dates": len(inventories), "files": files, "groups": len(groups)}
    verify(totals == TOTALS, "frozen dataset count mismatch")
    body = {
        "version": VERSION,
        "totals": totals,
        "inventories": inventories,
        "groups": groups,
    }
    return {**body, "snapshot_sha256": tp.sha(tp.canonical(body))}


def verify_snapshot(frozen):
    verify(
        frozen["version"] == VERSION and frozen["totals"] == TOTALS,
        "frozen contract mismatch",
    )
    verify(
        frozen["snapshot_sha256"]
        == tp.sha(
            tp.canonical({k: v for k, v in frozen.items() if k != "snapshot_sha256"})
        ),
        "snapshot hash mismatch",
    )
    inventories = frozen["inventories"]
    verify(
        sorted(i["date"] for i in inventories) == list(DATES),
        "six unique dates required",
    )
    verify(
        len(frozen["groups"]) == 194
        and len({g["group"] for g in frozen["groups"]}) == 194,
        "194 unique groups required",
    )
    for inv in inventories:
        verify(
            tp.sha(Path(inv["path"]).read_bytes()) == inv["sha256"],
            "frozen inventory changed",
        )
    # Re-query memberships/counts rather than trusting a self-hashed manifest.
    root = Path(inventories[0]["path"]).parents[2]
    verify(snapshot(root) == frozen, "snapshot differs from read-only inventory")


def load_ref(base, ref):
    tp._keys(ref, "path sha256")
    path = (base / ref["path"]).resolve()
    verify(path.is_relative_to(base.resolve()), "handoff path escape")
    data = path.read_bytes()
    verify(tp.sha(data) == ref["sha256"], "handoff artifact SHA mismatch")
    return tp.strict_json(data.decode())


SCENES = {"action", "stage", "group", "interaction", "ordinary", "unknown"}
VISUAL_BUDGET = {
    "images": 20,
    "image_bytes": 32_000_000,
    "image_pixels": 48_000_000,
    "text_bytes": 48000,
    "frames": 16,
}
JUDGE_PROMPT = (
    "Review every complete frame page before any supplemental storyboard, grid or strip. "
    "The local 8B scene route and boxes are UNTRUSTED attention hints, never subject coverage or selection authority. "
    "Infer the scene and useful differences yourself from complete images and compact CV L2. "
    "Fixed grids and centers are geometric views, NOT detected subjects. Do not infer identity or names. "
    "Return the strict JSON judgment schema, at most three evidence-bound observations, each <=500 Unicode characters. "
    "Region IDs reference deterministic visual views with exact frame bindings. Recommendations require medium/high visual evidence; "
    "unknown or bystander hints never support selection. Distinguish decision support from weak attention notes: "
    "select requires medium/high global confidence and at least one medium/high evidence binding per recommended frame. "
    "Discriminators may mix medium/high/low evidence or contain only low evidence as weak attention notes; "
    "low observations must not be necessary decision support. Multi-select reasons must cover every recommended frame using only medium/high evidence. "
    "Do not claim individual-subject coverage from a grid/stage view. Irreplaceability is not available without semantic subject grounding. "
    "Consider useful temporal phases and allow multiple frames when justified. Never delete or alter stored decisions."
)


def _mosaic(cells, output, name, kind):
    """Cells are (image, view metadata). No crop or frame is silently omitted."""
    columns = min(6, len(cells))
    width = max(im.width for im, _ in cells)
    height = max(im.height for im, _ in cells) + 26
    canvas = Image.new(
        "RGB", (columns * width, math.ceil(len(cells) / columns) * height), "white"
    )
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for i, (image, meta) in enumerate(cells):
        x, y = (i % columns) * width, (i // columns) * height
        canvas.paste(image, (x, y + 26))
        draw.text(
            (x + 3, y + 5),
            meta["region"] + " " + meta["frame"].split("@")[0],
            fill="black",
        )
    verify(
        max(canvas.size) <= 8000,
        "supplemental mosaic exceeds native service dimensions",
    )
    path = output / name
    canvas.save(path, quality=92)
    return tp._asset(path, [m for _, m in cells], kind)


def visual_packet(rows, images, scene, output, confirmed=None):
    """Scene chooses deterministic supplements. Complete high-res pages are mandatory."""
    verify(scene in SCENES, "invalid scene route")
    assets, lookup = [], {}

    def view(frame, box, purpose):
        rid = f"R{len(lookup) + 1}"
        verify(len(lookup) < 99, "visual reference budget")
        row = next(r for r in rows if tp.identity(r) == frame)
        record = {
            "id": rid,
            "frame": {k: row[k] for k in ("file_id", "content_sha256")},
            "box": box,
            "role": "frame_view",
            "confidence": "high",
            "purpose": purpose,
        }
        lookup[rid] = record
        return rid

    for index, row in enumerate(rows):
        frame = tp.identity(row)
        image = images[frame]
        shown = tp._render_cell(image, 1024, 1536)
        path = output / f"page-{index + 1}.jpg"
        shown.save(path, quality=92)
        rid = view(frame, [0, 0, 1, 1], "complete_frame")
        assets.append(
            tp._asset(
                path,
                [
                    {
                        "region": rid,
                        "frame": frame,
                        "native_size": list(image.size),
                        "actual_image_size": list(shown.size),
                        "upsampled": False,
                    }
                ],
                "full_group_page",
            )
        )
    if scene in {"action", "interaction"}:
        cells = []
        for row, asset in zip(rows, assets):
            frame = tp.identity(row)
            shown = tp._render_cell(images[frame], 384, 512)
            cells.append(
                (shown, {**asset["cells"][0], "actual_image_size": list(shown.size)})
            )
        assets.append(
            _mosaic(cells, output, "storyboard.jpg", "complete_temporal_storyboard")
        )
    if scene in {"stage", "group"}:
        boxes = [[0, 0, 0.5, 0.5], [0.5, 0, 1, 0.5], [0, 0.5, 0.5, 1], [0.5, 0.5, 1, 1]]
        if scene == "stage":
            boxes += [[0.1, 0.1, 0.9, 0.9], [0.25, 0.25, 0.75, 0.75]]
        cells = []
        for row in rows:
            frame = tp.identity(row)
            image = images[frame]
            for box in boxes:
                crop = image.crop(tp._pixels(image, box))
                shown = tp._render_cell(crop, 256, 256)
                rid = view(
                    frame,
                    box,
                    "fixed_multiscale_center_or_grid"
                    if scene == "stage"
                    else "fixed_grid",
                )
                cells.append(
                    (
                        shown,
                        {
                            "region": rid,
                            "frame": frame,
                            "pixel_box": list(tp._pixels(image, box)),
                            "native_size": list(crop.size),
                            "actual_image_size": list(shown.size),
                            "upsampled": False,
                        },
                    )
                )
        assets.append(_mosaic(cells, output, "fixed-grid.jpg", "fixed_multiscale_grid"))
    # Confirmation is supplied separately by an operator, never by the scene-router model.
    if confirmed and scene in {"action", "interaction", "group"}:
        verify(
            confirmed.get("confirmed_by") in {"human", "parent_visual_review"},
            "operator confirmation required",
        )
        cells = []
        for track in confirmed["tracks"]:
            for region in track["regions"]:
                frame = tp.identity(region["frame"])
                verify(frame in images, "confirmed strip frame mismatch")
                box = tp._box(region["box"], main_roi=False)
                image = images[frame]
                crop = image.crop(tp._pixels(image, box))
                shown = tp._render_cell(crop, 256, 384)
                rid = view(frame, box, "operator_confirmed_strip")
                cells.append(
                    (
                        shown,
                        {
                            "region": rid,
                            "frame": frame,
                            "pixel_box": list(tp._pixels(image, box)),
                            "native_size": list(crop.size),
                            "actual_image_size": list(shown.size),
                            "upsampled": False,
                        },
                    )
                )
        verify(bool(cells), "empty confirmed strips")
        assets.append(
            _mosaic(cells, output, "confirmed-strips.jpg", "operator_confirmed_strips")
        )
    verify(len(assets) <= VISUAL_BUDGET["images"], "complete packet image budget")
    verify(
        sum(a["bytes"] for a in assets) <= VISUAL_BUDGET["image_bytes"],
        "complete packet byte budget",
    )
    verify(
        sum(math.prod(a["size"]) for a in assets) <= VISUAL_BUDGET["image_pixels"],
        "complete packet pixel budget",
    )
    validation = {
        "lookup": lookup,
        "blockers": [],
        "decision_ceiling": "select",
        "ceiling_reasons": [],
        "coverage_authority": "complete_frame_views_not_verified_subject_completeness",
    }
    packet = {
        "scene_route": scene,
        "route_authority": "untrusted_local_hint",
        "views": list(lookup.values()),
        "complete_frames": [tp.identity(r) for r in rows],
        "individual_subject_coverage": "unknown",
    }
    return assets, packet, validation


def project_batch_l2(full):
    """Bound discriminating scalars by family; full measurements remain in L1.

    This is an explicit view projection, not a change to observations or frame coverage.
    """
    groups = {}

    def category(item):
        name = item["criterion"].lower()
        for family in (
            "eye",
            "face",
            "pose",
            "exposure",
            "sharpness",
            "musiq",
            "clipiqa",
            "local",
            "routing",
            "quality_score",
        ):
            if family in name:
                return family
        return "other"

    def contrast(item):
        values = list(item["values"].values())
        numbers = [v for v in values if type(v) in (int, float)]
        if len(numbers) == len(values) and numbers:
            return (max(numbers) - min(numbers)) / max(
                max(abs(v) for v in numbers), 1e-9
            )
        return len({tp.canonical(v) for v in values}) / max(1, len(values))

    for ref, item in full["evidence"].items():
        groups.setdefault(category(item), []).append((ref, item))
    for values in groups.values():
        values.sort(
            key=lambda pair: (-contrast(pair[1]), pair[1]["criterion"], pair[0])
        )
    selected = []
    while len(selected) < 24 and any(groups.values()):
        for family in sorted(groups):
            if groups[family] and len(selected) < 24:
                selected.append(groups[family].pop(0))
    result = copy.deepcopy(full)
    result["evidence"] = dict(selected)
    result["evidence_refs"] = [
        f"{alias}:{ref}" for ref, item in selected for alias in item["values"]
    ]
    result["evidence_refs"].extend(
        f"PAIR:{':'.join(item['frames'])}:{ref}"
        for ref, item in result["pairs"].items()
    )
    result["projection"] = {
        "method": "family_round_robin_normalized_measurement_contrast",
        "scalar_limit": 24,
        "original_scalar_count": len(full["evidence"]),
        "selected_scalar_count": len(selected),
        "omitted_scalar_count": len(full["evidence"]) - len(selected),
        "complete_measurements_location": "L1 artifact; not all scalars are model-visible",
        "frame_coverage": "all_frames_retained",
        "authority": "measurements_not_preferences",
    }
    tp.vlm_review.assert_model_context_safe(result)
    return result


def prepare(frozen, group, entry, base, output, confirmed=None):
    scene = (
        entry.get("scene", "unknown")
        if entry and entry.get("status") == "routed"
        else "unknown"
    )
    inv = next(i for i in frozen["inventories"] if i["date"] == group["date"])
    with readonly_inventory(inv['path']) as conn:
        rows = [dict(r) for r in db.group_page_by_id(conn, group["group_id"])]
        sources = {
            r["file_id"]: Path(
                conn.execute(
                    "SELECT path FROM files WHERE id=?", (r["file_id"],)
                ).fetchone()[0]
            )
            for r in rows
        }
    verify(
        [{k: r[k] for k in ("file_id", "content_sha256")} for r in rows]
        == group["frames"],
        "frozen membership mismatch",
    )
    verify(
        2 <= len(rows) <= VISUAL_BUDGET["frames"],
        "complete group exceeds visual budget; no sampling",
    )
    # Stored keeper ordering has no authority. Sequence uses capture time, then stable file ID.
    rows.sort(
        key=lambda r: (
            r.get("exif_timestamp") is None,
            r.get("exif_timestamp") or 0,
            r["file_id"],
        )
    )
    images = {}
    for r in rows:
        path = sources[r["file_id"]]
        verify(tp.sha(path.read_bytes()) == r["content_sha256"], "source SHA mismatch")
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        verify(
            image.size == (r["width"], r["height"]),
            "oriented source dimensions mismatch",
        )
        images[tp.identity(r)] = image
    warnings = []
    try:
        assets, packet, validation = visual_packet(
            rows, images, scene, output, confirmed
        )
    except (tp.ReviewError, KeyError, TypeError, ValueError) as exc:
        if confirmed is None:
            raise
        warnings.append(
            "confirmed_strip_unusable_full_view_fallback: " + str(exc)[:200]
        )
        assets, packet, validation = visual_packet(rows, images, scene, output, None)
    # 8B hints never feed rendering, CV spatial joins, coverage or recommendations.
    hints = []
    if entry and entry.get("attention_hints"):
        try:
            candidate = load_ref(base, entry["attention_hints"])
            verify(isinstance(candidate, list) and len(candidate) <= 32, "hint budget")
            for hint in candidate:
                tp._keys(hint, "frame box confidence")
                tp._frame(hint["frame"], images)
                tp._box(hint["box"], main_roi=False)
                tp._enum(hint["confidence"], tp.CONFIDENCE)
            verify(len(tp.canonical(candidate)) <= 6000, "untrusted hint byte budget")
            hints = candidate
        except (tp.ReviewError, OSError, KeyError, TypeError, ValueError) as exc:
            warnings.append("untrusted_8b_hints_ignored: " + str(exc)[:200])
    clean = []
    for row in rows:
        item = {k: row[k] for k in tp.vlm_review._CV_MEMBER_FIELDS if k in row}
        for key in ("quality_meta", "faces_json", "evidence_json"):
            if key in item:
                item[key] = tp._safe(tp._object(item[key]))
        clean.append(tp._safe(item))
    members, audit, l1, diagnostic_l2 = tp.vlm_review.build_layers(
        clean, enforce_gate=False
    )
    full_l2 = project_batch_l2(diagnostic_l2)
    l2 = {
        "layer": "L2",
        "frame_bindings": {r["frame_alias"]: tp.identity(r) for r in members},
        "full_frame": full_l2,
    }
    tp.vlm_review.assert_model_context_safe(full_l2)
    verify(len(tp.canonical(l2)) <= 30000, "compact CV L2 exceeds 30KB; no truncation")
    temporal = [
        {
            "frame": {k: r[k] for k in ("file_id", "content_sha256")},
            "capture_time": r.get("exif_timestamp"),
            "authority": "stored_timestamp_not_inferred_phase",
        }
        for r in rows
    ]
    for name, value in (
        ("L0", audit),
        ("L1", l1),
        ("L2", l2),
        ("visual-packet", packet),
        ("view-validation", validation),
    ):
        save(output / f"{name}.json", value)
    request = {
        "version": VERSION,
        "pass": "single_judge",
        "model": MODEL,
        "prompt": JUDGE_PROMPT,
        "context": {
            "visual_packet": packet,
            "evidence": l2,
            "untrusted_local_attention_hints": hints,
            "hint_authority": "none_for_selection_or_coverage",
            "visual_warnings": warnings,
            "decision_ceiling": "select",
            "temporal": temporal,
        },
        "images": assets,
        "response_schema": read(ROOT / "schemas/cloud-scene-judge-v1.schema.json"),
        "max_output_tokens": tp.BUDGET["B"]["output_tokens"],
    }
    verify(
        len(tp.canonical(request)) + len(tp.PRIVACY_INSTRUCTION.encode())
        <= VISUAL_BUDGET["text_bytes"],
        "request text budget",
    )
    save(output / "request.json", request)
    return request, rows, packet, validation, l2


class FixtureJudge:
    """Explicit test transport; never imports an SDK or connects to a service."""

    external = False

    def __init__(self, responses):
        self.responses = copy.deepcopy(responses)
        self.calls = []

    def invoke(self, request, case, group):
        self.calls.append(group)
        answer = self.responses[group]
        if isinstance(answer, Exception):
            raise answer
        text = tp.canonical(answer).decode()
        return {
            "output": {"message": {"content": [{"text": text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "fixture": True,
        }


class SonnetJudge:
    external = True

    def invoke(self, request, case, group):
        # SDK retries disabled; failure consumes this group's durable reservation.
        import boto3
        from botocore.config import Config

        content = [
            {
                "text": tp.canonical(
                    {k: v for k, v in request.items() if k != "images"}
                ).decode()
                + tp.PRIVACY_INSTRUCTION
            }
        ]
        for a in request["images"]:
            path = (case / a["path"]).resolve()
            verify(path.is_relative_to(case.resolve()), "asset path escape")
            data = path.read_bytes()
            verify(
                tp.sha(data) == a["sha256"] and len(data) == a["bytes"], "asset changed"
            )
            with Image.open(io.BytesIO(data)) as image:
                verify(
                    image.format == "JPEG" and list(image.size) == a["size"],
                    "asset dimensions mismatch",
                )
            content.extend(
                [
                    {"text": tp.canonical(a).decode()},
                    {"image": {"format": "jpeg", "source": {"bytes": data}}},
                ]
            )
        verify(
            sum(len(c["text"].encode()) for c in content if "text" in c)
            <= VISUAL_BUDGET["text_bytes"],
            "wire text budget",
        )
        wire = {
            "modelId": MODEL,
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {"maxTokens": tp.BUDGET["B"]["output_tokens"]},
        }
        serial = json.loads(
            json.dumps(
                wire, default=lambda b: {"bytes_base64": base64.b64encode(b).decode()}
            )
        )
        save(case / "wire-request.json", serial)
        client = boto3.Session().client(
            "bedrock-runtime",
            region_name="us-east-1",
            config=Config(
                retries={"total_max_attempts": 1, "mode": "standard"},
                connect_timeout=20,
                read_timeout=300,
            ),
        )
        return client.converse(**wire)


def response_judgment(response, request, case, rows, packet, validation, l2, rates):
    save(case / "provider-response.json", response)
    usage = response.get("usage", {})
    for k in ("inputTokens", "outputTokens", "totalTokens"):
        verify(type(usage.get(k)) is int and usage[k] >= 0, "missing/invalid usage")
    verify(
        usage["totalTokens"] == usage["inputTokens"] + usage["outputTokens"],
        "usage sum mismatch",
    )
    verify(
        usage["outputTokens"] <= tp.BUDGET["B"]["output_tokens"], "output token budget"
    )
    verify(
        not any(
            usage.get(k, 0) for k in ("cacheReadInputTokens", "cacheWriteInputTokens")
        ),
        "unpriced provider cache usage",
    )
    verify(response.get("stopReason") == "end_turn", "truncated response; no retry")
    blocks = response["output"]["message"]["content"]
    verify(all(set(b) == {"text"} for b in blocks), "unexpected content")
    text = "\n".join(b["text"] for b in blocks)
    (case / "raw.txt").write_text(text)
    verify(len(text.encode()) <= tp.BUDGET["B"]["output_bytes"], "response byte budget")
    parsed = tp.strict_json(text)
    save(case / "raw.json", parsed)
    jsonschema.validate(parsed, request["response_schema"])
    tp.privacy_check(parsed)
    verify(
        not parsed["irreplaceable"],
        "visual views do not prove subject irreplaceability",
    )
    judgment = tp.validate_judgment(
        parsed, rows, packet, validation, l2, allow_visual_views=True
    )
    save(case / "validated.json", judgment)
    save(
        case / "evidence-bindings.json",
        {
            "authority": "deterministic_frame_views_not_subject_or_identity_truth",
            "discriminators": [
                {
                    "index": index,
                    "evidence": [
                        {
                            "region": e["region"],
                            "frame": validation["lookup"][e["region"]]["frame"],
                            "box": validation["lookup"][e["region"]]["box"],
                            "role": validation["lookup"][e["region"]]["role"],
                        }
                        for e in item["evidence"]
                    ],
                }
                for index, item in enumerate(judgment["discriminators"])
            ],
        },
    )
    cost = (
        usage["inputTokens"] * rates["input_per_million"]
        + usage["outputTokens"] * rates["output_per_million"]
    ) / 1_000_000
    return judgment, usage, round(cost, 8)


def run(
    frozen_path,
    handoff_path,
    output,
    *,
    execute=False,
    concurrency=1,
    rates=None,
    fixture=None,
    confirmed_visuals_path=None,
    selected_groups=None,
):
    verify(
        type(concurrency) is int and 1 <= concurrency <= 4, "concurrency must be 1..4"
    )
    verify(
        fixture is None or type(fixture) is FixtureJudge,
        "only explicit FixtureJudge is allowed offline",
    )
    verify(not (execute and fixture is not None), "fixture cannot execute cloud")
    # Required first: no manifest means no dataset/model work.
    handoff_path = Path(handoff_path).resolve()
    handoff = read(handoff_path)
    frozen = read(frozen_path)
    verify_snapshot(frozen)
    verify(
        handoff["version"] == "local8b-scene-router-v1"
        and handoff["snapshot_sha256"] == frozen["snapshot_sha256"],
        "handoff/frozen binding mismatch",
    )
    jsonschema.validate(
        handoff, read(ROOT / "schemas/local8b-scene-router-v1.schema.json")
    )
    entries = handoff["groups"]
    expected = {g["group"] for g in frozen["groups"]}
    verify(
        len({e["group"] for e in entries}) == len(entries)
        and {e["group"] for e in entries} <= expected,
        "duplicate/unknown handoff group",
    )
    if selected_groups is not None:
        verify(
            isinstance(selected_groups, (list, tuple))
            and len(selected_groups) == len(set(selected_groups))
            and set(selected_groups) <= expected,
            "invalid selected group set",
        )
    selected = set(selected_groups) if selected_groups is not None else expected
    indexed = {e["group"]: e for e in entries}
    confirmed_manifest = (
        read(confirmed_visuals_path) if confirmed_visuals_path else None
    )
    if confirmed_manifest:
        verify(
            confirmed_manifest["snapshot_sha256"] == frozen["snapshot_sha256"],
            "confirmed visuals/frozen mismatch",
        )
        verify(
            set(confirmed_manifest["groups"]) <= expected,
            "unknown confirmed visual group",
        )
    confirmed_groups = confirmed_manifest["groups"] if confirmed_manifest else {}
    output = artifact_root(output)
    rates = rates or {
        "input_per_million": 3.0,
        "output_per_million": 15.0,
        "basis": "Illustrative planning assumption, not verified tariff or invoice.",
    }
    for entry in entries:
        if entry["status"] == "routed":
            verify(
                entry["origin"] == ("fixture" if fixture is not None else "local8b"),
                "fixture/local provenance mismatch",
            )
    for key in ("input_per_million", "output_per_million"):
        verify(
            type(rates[key]) in (int, float)
            and math.isfinite(rates[key])
            and rates[key] > 0,
            "positive finite rates required",
        )
    tp._text(rates["basis"], 500)
    contract = {
        "version": VERSION,
        "judgment_contract": "v3",
        "snapshot_sha256": frozen["snapshot_sha256"],
        "model": MODEL,
        "rates": rates,
        "mode": "fixture" if fixture else "cloud",
        "confirmed_visuals_sha256": tp.sha(tp.canonical(confirmed_manifest)),
        "implementation": {
            p.name: tp.sha(p.read_bytes())
            for p in (
                Path(__file__),
                Path(tp.__file__),
                ROOT / "schemas/cloud-scene-judge-v1.schema.json",
                ROOT / "schemas/local8b-scene-router-v1.schema.json",
                ROOT / "schemas/two-pass-subject-map-v1.schema.json",
                ROOT / "src/vlm_visual_budget.py",
                ROOT / "src/vlm_criterion_compiler.py",
                ROOT / "src/vlm_review.py",
                ROOT / "src/vlm_prompt_context.py",
                ROOT / "src/subject_set_evidence.py",
                ROOT / "src/vlm_identity.py",
            )
        },
    }
    transport = fixture if fixture is not None else (SonnetJudge() if execute else None)
    lock = (output / "batch.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "contract.json").exists():
            verify(
                read(output / "contract.json") == contract,
                "resume contract changed; no replay",
            )
        else:
            save(output / "contract.json", contract)

        def one(group):
            gid = group["group"]
            entry = indexed.get(gid)
            case = output / gid
            case.mkdir(exist_ok=True)
            key = tp.sha(
                tp.canonical({"contract": contract, "group": group, "local": entry})
            )
            state_path = case / "state.json"
            old = read(state_path) if state_path.exists() else None
            if old and old.get("reserved"):
                verify(
                    old["input_sha256"] == key, "reserved group input changed; no retry"
                )
                if old["state"] == "validated":
                    verify(
                        tp.sha((case / "validated.json").read_bytes())
                        == old["validated_sha256"],
                        "validated cache changed",
                    )
                if (
                    entry
                    and entry.get("attention_hints")
                    and not any(
                        w.startswith("untrusted_8b_hints_ignored:")
                        for w in old.get("visual_warnings", [])
                    )
                ):
                    load_ref(handoff_path.parent, entry["attention_hints"])
                for name, expected_sha in old.get("artifact_hashes", {}).items():
                    verify(
                        tp.sha((case / name).read_bytes()) == expected_sha,
                        "cached artifact changed",
                    )
                return {**old, "cache_hit": True, "calls_this_run": 0}
            state = {
                "group": gid,
                "input_sha256": key,
                "state": "attention",
                "route": "attention_only",
                "reason": None,
                "reserved": False,
                "external_calls": 0,
                "calls_this_run": 0,
                "usage": None,
                "estimated_cost_usd": None,
                "decision_table_writes": 0,
                "cache_hit": False,
            }
            state["scene_route"] = (
                entry.get("scene", "unknown")
                if entry and entry.get("status") == "routed"
                else "unknown"
            )
            state["local_hint_authority"] = "none_for_selection_or_coverage"
            if gid not in selected:
                state["reason"] = "not_selected_this_phase"
                save(state_path, state)
                return state
            if fixture is not None and gid not in fixture.responses:
                state["reason"] = "fixture_case_not_selected"
                save(state_path, state)
                return state
            start = None
            try:
                req, rows, packet, validation, l2 = prepare(
                    frozen,
                    group,
                    entry,
                    handoff_path.parent,
                    case,
                    confirmed_groups.get(gid),
                )
                state["request_sha256"] = tp.sha(tp.canonical(req))
                state["visual_warnings"] = req.get("context", {}).get(
                    "visual_warnings", []
                )
                if transport is None:
                    state.update(state="prepared", reason="explicit_execute_required")
                    save(state_path, state)
                    return state
                state.update(
                    state="reserved",
                    reserved=True,
                    external_calls=int(transport.external),
                    calls_this_run=1,
                )
                save(state_path, state)
                start = time.perf_counter()
                response = transport.invoke(req, case, gid)
                # Receipt and raw usage survive even output-schema/semantic rejection.
                save(case / "provider-response.json", response)
                state["response_sha256"] = tp.sha(tp.canonical(response))
                state["usage"] = response.get("usage")
                usage = state["usage"] or {}
                if all(
                    type(usage.get(k)) is int and usage[k] >= 0
                    for k in ("inputTokens", "outputTokens")
                ) and not any(
                    usage.get(k, 0)
                    for k in ("cacheReadInputTokens", "cacheWriteInputTokens")
                ):
                    state["estimated_cost_usd"] = round(
                        (
                            usage["inputTokens"] * rates["input_per_million"]
                            + usage["outputTokens"] * rates["output_per_million"]
                        )
                        / 1_000_000,
                        8,
                    )
                judgment, usage, cost = response_judgment(
                    response, req, case, rows, packet, validation, l2, rates
                )
                state.update(
                    state="validated",
                    route=judgment["decision"],
                    usage=usage,
                    estimated_cost_usd=cost,
                    validated_sha256=tp.sha(tp.canonical(judgment)),
                    latency_seconds=round(time.perf_counter() - start, 3),
                )
            except Exception as exc:  # noqa: BLE001 -- isolate a consumed group, never retry
                state.update(
                    state="failed_no_retry" if state["reserved"] else "attention",
                    route="attention_only",
                    reason=f"{type(exc).__name__}: {str(exc)[:400]}",
                )
            if start is not None:
                state["latency_seconds"] = round(time.perf_counter() - start, 3)
            state["artifact_hashes"] = {
                p.name: tp.sha(p.read_bytes())
                for p in case.iterdir()
                if p.is_file()
                and p.name != "state.json"
                and not p.name.endswith(".tmp")
            }
            save(state_path, state)
            return state

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(one, frozen["groups"]))
        verify_snapshot(frozen)
        summary = {
            "version": VERSION,
            "mode": "fixture" if fixture else ("execute" if execute else "prepare"),
            "totals": frozen["totals"],
            "concurrency": concurrency,
            "selected_groups": sorted(selected),
            "routes": dict(Counter(r["route"] for r in results)),
            "states": dict(Counter(r["state"] for r in results)),
            "scene_routes": dict(
                Counter(r.get("scene_route", "unknown") for r in results)
            ),
            "reason_counts": dict(Counter(r["reason"] for r in results if r["reason"])),
            "local_model_counts": dict(
                Counter(e.get("model", "not_provided") for e in entries)
            ),
            "provider_receipts": sum(bool(r.get("response_sha256")) for r in results),
            "external_calls_total": sum(r["external_calls"] for r in results),
            "external_calls_this_run": sum(
                r["calls_this_run"] * r["external_calls"] for r in results
            ),
            "usage": {
                k: sum(
                    (r["usage"] or {}).get(k, 0)
                    for r in results
                    if type((r["usage"] or {}).get(k, 0)) is int
                    and (r["usage"] or {}).get(k, 0) >= 0
                )
                for k in ("inputTokens", "outputTokens", "totalTokens")
            },
            "estimated_cost_usd_known": round(
                sum(r["estimated_cost_usd"] or 0 for r in results), 8
            ),
            "cost_unknown_groups": [
                r["group"]
                for r in results
                if r["external_calls"] and r["estimated_cost_usd"] is None
            ],
            "cost_basis": rates,
            "decision_table_writes": 0,
            "groups": results,
        }
        save(output / "report.json", summary)
        return summary
    finally:
        lock.close()
