"""Read-only G002 narrow slice. No VLM, detector weights, decision writes or batch ROI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_vlm_rebuild import digest
from src import action_roi_candidates, db, vlm_identity, vlm_review


def run(eval_root: Path, output: Path):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("writes must stay in repository")
    output.mkdir(parents=True, exist_ok=True)
    base = eval_root / "astra-regime-validation-20260910"
    inventory = base / "local-holdout/2026-06-23/inventory.sqlite"
    conn = sqlite3.connect(f"file:{inventory}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    before = digest(conn)
    members, *_ = vlm_review.build_layers(db.group_page_by_id(conn, 2))
    if {r["file_id"] for r in members} != {99, 107}:
        raise ValueError("not the frozen G002 narrow slice")
    images, sources = [], []
    for row in members:
        path = Path(conn.execute("SELECT path FROM files WHERE id=?", (row["file_id"],)).fetchone()[0])
        if not path.resolve().is_relative_to(eval_root.resolve()):
            raise ValueError("source escaped read-only evaluation root")
        sha = vlm_review._sha_file(path)
        if sha != row["content_sha256"]:
            raise ValueError("source SHA mismatch")
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        images.append(image)
        sources.append({"alias": row["frame_alias"], **vlm_identity.alias_map([row])[row["frame_alias"]],
                        "oriented_size": list(image.size), "source_sha_verified": True})
    result = action_roi_candidates.propose_pair([np.array(im) for im in images])
    masks = result.pop("masks", [])
    result.update(schema_version=1, group_id=2, dataset="2026-06-23", sources=sources,
                  model_calls=0, decision_tables_before=before, decision_tables_after=digest(conn),
                  independent_acceptance="pending parent visual review", production_roi_enabled=False)
    result["context_report"] = vlm_review.context_report(db.group_page_by_id(conn, 2))
    conn.close()
    if result["status"] == "candidate":
        window = result["shared_window_normalized"]
        cells, cell_records = [], []
        overlay = Image.new("RGB", (2048, 828), "white")
        for i, (image, source, mask) in enumerate(zip(images, sources, masks)):
            mask_path = output / f'G002-{source["alias"]}-mask.png'
            Image.fromarray(mask).save(mask_path)
            bounds = [math.floor(window[0] * image.width), math.floor(window[1] * image.height),
                      math.ceil(window[2] * image.width), math.ceil(window[3] * image.height)]
            crop = image.crop(bounds)
            if min(crop.size) < 256:
                raise ValueError("native crop short edge below 256")
            scale = min(1, 512 / min(crop.size))
            displayed = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
            if min(displayed.size) < 256:
                raise ValueError("displayed actual-image short edge below 256")
            cells.append(displayed)
            cell_records.append({"alias": source["alias"], "file_id": source["file_id"],
                                 "native_box_pixels": bounds, "native_crop_size": list(crop.size),
                                 "displayed_actual_image_size": list(displayed.size), "upsampled": False})
            thumb = image.resize((1024, 768), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(thumb)
            draw.rectangle([window[0] * 1024, window[1] * 768, window[2] * 1024, window[3] * 768], outline="lime", width=3)
            overlay.paste(thumb, (i * 1024, 60))
            ImageDraw.Draw(overlay).text((i * 1024 + 20, 15), f'{source["alias"]} file_id={source["file_id"]}: review-only moving-region candidate', fill="black")
        overlay.save(output / "G002-candidate-overlay.jpg", quality=92)
        strip = Image.new("RGB", (sum(im.width for im in cells), max(im.height for im in cells) + 60), "white")
        x = 0
        for im, source in zip(cells, sources):
            strip.paste(im, (x, 60))
            ImageDraw.Draw(strip).text((x + 12, 15), f'{source["alias"]} / file_id={source["file_id"]} / review candidate', fill="black")
            x += im.width
        strip_path = output / "G002-comparison-strip.jpg"
        strip.save(strip_path, quality=95)
        result["comparison_strip"] = {"path": strip_path.name, "sha256": hashlib.sha256(strip_path.read_bytes()).hexdigest(),
                                       "cells": cell_records, "actual_short_edge_gate": True,
                                       "label": "same candidate moving foreground region; semantic identity unestablished"}
        result["mask_artifacts"] = [f'G002-{s["alias"]}-mask.png' for s in sources]
    else:
        result["next_minimal_route"] = "human supplies one source-backed region per frame; compare extents before any batch expansion"
    (output / "G002-candidate-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "consistency", "comparison_strip") if k in result}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("/home/ubuntu/photo-dedup-eval"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/fable-roi-narrow")
    args = parser.parse_args()
    run(args.eval_root, args.output)
