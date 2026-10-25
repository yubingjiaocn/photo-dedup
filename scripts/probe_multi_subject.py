"""Three frozen development groups only; read-only inventory and source-backed sheets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_vlm_rebuild import digest
from src import subject_set_evidence as evidence

SLICES = (
    ("2026-04-04", 14, {248, 284}, "group", "human/animal interaction; unequal detections and clipped bodies"),
    ("2026-04-05", 7, {35, 131}, "group", "three visible people; reflection/occlusion and body extent ambiguity"),
    ("2026-07-25", 4, {470, 481}, "stage", "distant stage; detector captures audience, not performers"),
)


def run(eval_root, output):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("writes must stay in repository")
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for date, gid, expected, regime, scene in SLICES:
        inventory = eval_root / "astra-regime-validation-20260910/development" / date / "inventory.sqlite"
        with sqlite3.connect(f"file:{inventory}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            before = digest(conn)
            rows = [dict(r) for r in conn.execute(
                "SELECT f.id AS file_id, f.path, f.width, f.height, fe.content_sha256, "
                "fe.quality_meta, fe.faces_json FROM files f "
                "JOIN features fe ON fe.file_id=f.id JOIN group_members gm ON gm.file_id=f.id "
                "WHERE gm.group_id=? ORDER BY f.id", (gid,))]
            if {r["file_id"] for r in rows} != expected:
                raise ValueError("frozen slice changed")
            images, sources, scope, sizes = [], [], {}, {}
            for row in rows:
                path = Path(row.pop("path")).resolve()
                if not path.is_relative_to(eval_root.resolve()):
                    raise ValueError("source escaped frozen tree")
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if sha != row["content_sha256"]:
                    raise ValueError("source SHA mismatch")
                with Image.open(path) as opened:
                    im = ImageOps.exif_transpose(opened).convert("RGB")
                if im.size != (row["width"], row["height"]):
                    raise ValueError("oriented coordinate mismatch")
                key = evidence.identity(row)
                images.append(im)
                sizes[key] = im.size
                sources.append({"file_id": row["file_id"], "content_sha256": sha,
                                "source_sha_verified": True, "oriented_size": list(im.size)})
                regs = json.loads(row["quality_meta"])["local_region_set"]["regions"]
                role = "bystander_candidate" if regime == "stage" else "key_candidate"
                scope[key] = {"authority": "Astra_visual_inspection_candidate_not_human_truth",
                              "coverage": "unknown", "scene_observation": scene,
                              "detector_missing": regime == "stage",
                              "clipping_conflict": regime != "stage",
                              "roles": {r["region_id"]: role for r in regs if r["kind"] == "subject"}}
            l0, l1, lookup = evidence.compile_evidence(rows, regime=regime, scope=scope)
            plan = evidence.visual_plan(l1, lookup, sizes)
            # Full-frame context always retained; no blocked subject strip is mislabeled as coverage.
            cells, dimensions = [], []
            for im, source in zip(images, sources):
                if min(im.size) < 400:
                    raise ValueError("native sheet source too small")
                scale = 512 / min(im.size)
                shown = im.resize((round(im.width*scale), round(im.height*scale)), Image.Resampling.LANCZOS)
                cells.append(shown)
                dimensions.append({"file_id": source["file_id"], "actual_image_size": list(shown.size),
                                   "upsampled": False})
            sheet = Image.new("RGB", (sum(i.width for i in cells), max(i.height for i in cells)+64), "white")
            x = 0
            for im, source in zip(cells, sources):
                sheet.paste(im, (x, 64))
                ImageDraw.Draw(sheet).text((x+10, 10), f'{date} G{gid:03d} file_id={source["file_id"]}\nFull ensemble context; subject coverage UNKNOWN', fill="black")
                x += im.width
            prefix = f"{date}-G{gid:03d}"
            sheet_path = output / f"{prefix}-contact-sheet.jpg"
            sheet.save(sheet_path, quality=95)
            result = {"version": evidence.VERSION, "dataset": date, "group_id": gid, "scene": scene,
                      "sources": sources, "L0": l0, "L1": l1, "L2": evidence.prompt_evidence(l1),
                      "cache_key": evidence.cache_key(l0), "visual_plan": plan,
                      "contact_sheet": {"path": sheet_path.name, "cells": dimensions,
                                        "actual_short_edge_gate": all(min(c["actual_image_size"]) >= 400 for c in dimensions)},
                      "decision_tables_before": before, "decision_tables_after": digest(conn),
                      "recommendations": [], "external_model_calls": 0, "production_enabled": False}
            (output / f"{prefix}-report.json").write_text(json.dumps(result, indent=2)+"\n")
            reports.append({"dataset": date, "group_id": gid, "status": plan["status"],
                            "blockers": plan["blockers"], "correspondence_pairs": [len(m["pairs"]) for m in l1["correspondence"]],
                            "detected_counts": [f["detected_subject_count"] for f in l1["frames"]],
                            "decision_tables_unchanged": before == result["decision_tables_after"],
                            "sheet": sheet_path.name})
    summary = {"groups": reports, "real_slice_gate_passes": sum(r["status"] == "candidate" for r in reports),
               "new_decision_schema": "deferred_until_real_subject_set_gate_passes",
               "production_enabled": False, "external_model_calls": 0, "old_preference_labels_read": False,
               "next_minimal_action": "SHA-bound human subject-set/role/extent annotations or independent detections; no 22-group expansion"}
    (output / "report.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("/home/ubuntu/photo-dedup-eval"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/multi-subject-narrow")
    args = parser.parse_args()
    run(args.eval_root, args.output)
