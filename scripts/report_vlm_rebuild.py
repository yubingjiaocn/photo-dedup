"""Read-only frozen-set evidence/identity/ROI diagnostics. Never invokes a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (
    db,
    vlm_identity,
    vlm_prompt_context,
    vlm_review,
    vlm_visual_budget,
)


def digest(conn):
    result = hashlib.sha256()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    selected = {"groups", "group_members", "review_index", "review_actions", "review_action_log"}
    selected.update(table for table in tables if "action" in table)
    for table in sorted(selected & tables):
        result.update(table.encode())
        for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
            result.update(json.dumps(tuple(row), default=str).encode())
    return result.hexdigest()


def old_alias_map(artifact):
    context = artifact.get("audit_context") or artifact.get("cv_context")
    result = {}
    for frame in context["frames"]:
        leaves = {item["ref"].split(":", 1)[1]: item["value"] for item in frame["measurements"]}
        result[frame["frame_alias"]] = {k: leaves[k] for k in ("file_id", "content_sha256")}
    return result


def diagnostic_sheet(rows, sources, output):
    from PIL import Image, ImageDraw, ImageOps

    sheet = Image.new("RGB", (2400, 970), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 10), "G002 ACTION ROI BLOCKED: detectors locate bystanders, not mascot", fill="black")
    for i, row in enumerate(rows):
        with Image.open(sources[row["file_id"]]) as src:
            image = ImageOps.exif_transpose(src).convert("RGB").resize((1200, 900))
        annotate = ImageDraw.Draw(image)
        q = json.loads(row["quality_meta"])
        for region in q.get("local_region_set", {}).get("regions", []):
            box = region.get("box_normalized")
            if box:
                rect = [box[0] * 1200, box[1] * 900, box[2] * 1200, box[3] * 900]
                annotate.rectangle(rect, outline="red", width=4)
                annotate.text((rect[0] + 2, rect[1] + 2), region["region_id"], fill="red")
        sheet.paste(image, (1200 * i, 70))
        draw.text((1200 * i + 20, 35), f"{row['frame_alias']} file_id={row['file_id']}", fill="black")
    path = output / "G002-roi-blocked.jpg"
    sheet.save(path, "JPEG", quality=90)
    return path.name


def run(eval_root: Path, output: Path):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("report writes must stay inside repository")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    hashes = {}
    frozen = sorted(eval_root.glob("*/**/vlm-review/groups/*.json"))
    if len(frozen) != 22:
        raise ValueError(f"expected frozen 22-group set, got {len(frozen)}")
    for sidecar in frozen:
        inventory = sidecar.parents[2] / "inventory.sqlite"
        dataset = sidecar.parents[2].name
        project = sidecar.relative_to(eval_root).parts[0]
        conn = sqlite3.connect(f"file:{inventory}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        before = digest(conn)
        rows = db.group_page_by_id(conn, int(sidecar.stem))
        members, audit, derived, prompt = vlm_review.build_layers(rows, enforce_gate=False)
        current = vlm_identity.alias_map(members)
        old = old_alias_map(json.loads(sidecar.read_text()))
        old_join = vlm_identity.join_maps(old, current)
        sources = {}
        source_hashes_match = True
        for row in members:
            source = Path(conn.execute("SELECT path FROM files WHERE id=?", (row["file_id"],)).fetchone()[0])
            if not source.resolve().is_relative_to(eval_root.resolve()):
                raise ValueError("source escaped read-only evaluation tree")
            sources[row["file_id"]] = source
            if source not in hashes:
                hashes[source] = vlm_review._sha_file(source)
            source_hashes_match &= hashes[source] == row["content_sha256"]
        mapping_project = "astra-real-events-20260910" if dataset == "2026-04-04" else project
        panel_path = eval_root / mapping_project / "panels" / dataset / "mapping.json"
        panel_group = next(g for g in json.loads(panel_path.read_text()) if int(g["group_id"]) == int(sidecar.stem))
        by_id = {r["file_id"]: r for r in members}
        panel = {}
        for frame in panel_group["frames"]:
            row = by_id[frame["id"]]
            panel_source = Path(frame["path"])
            if not panel_source.resolve().is_relative_to(eval_root.resolve()):
                raise ValueError("panel source escaped read-only evaluation tree")
            if panel_source not in hashes:
                hashes[panel_source] = vlm_review._sha_file(panel_source)
            panel[frame["alias"]] = {"file_id": row["file_id"], "content_sha256": hashes[panel_source]}
        panel_join = vlm_identity.join_maps(panel, old)
        try:
            vlm_identity.join_maps(panel, old, by="alias")
            alias_rejected = False
        except ValueError:
            alias_rejected = True
        try:
            vlm_review.assert_model_context_safe(prompt)
            guard = {"pass": True}
        except (TypeError, ValueError) as exc:
            guard = {"pass": False, "error": str(exc)}
        is_action_slice = dataset == "2026-06-23" and sidecar.stem == "2"
        try:
            plan = vlm_visual_budget.plan(members, regime="action" if is_action_slice else None)
            roi = {"pass": True, "plan": plan, "render_validated": False}
        except vlm_visual_budget.VisualBudgetError as exc:
            roi = {"pass": False, "error": str(exc), "bystander_crops_emitted": 0}
        if is_action_slice:
            roi["artifact"] = diagnostic_sheet(members, sources, output)
            roi["regime_source"] = "assigned G002 action narrow slice; not old keeper/labels"
            roi["diagnostic_cell_image_short_edge"] = 900
            roi["production_comparison_strips"] = 0
            for name, value in (("L0", audit), ("L1", derived), ("L2", prompt)):
                (output / f"G002-{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        counts = Counter(item["disposition"] for item in derived["coverage_ledger"])
        refs = set(derived["evidence_refs"])
        missing_refs = sorted({ref for item in derived["coverage_ledger"]
                               for ref in item["replacement_refs"] if ref not in refs})
        report = {
            "dataset": dataset, "group_id": int(sidecar.stem), "members": len(members),
            "layers": {name: {"bytes": len(vlm_review._canonical(value)),
                               "tokens_estimate_byte_div_3": vlm_prompt_context.token_estimate(value)}
                       for name, value in (("L0", audit), ("L1", derived), ("L2", prompt))},
            "guard": guard, "coverage": dict(counts), "missing_replacement_refs": missing_refs,
            "families": {key: {k: v for k, v in item.items() if k != "replacement_refs"}
                         for key, item in derived["family_coverage"].items()},
            "join": {"old_to_current": old_join, "panel_to_old": panel_join,
                     "panel_alias_map": panel, "old_alias_map": old,
                     "alias_join_rejected": alias_rejected, "source_sha_verified": source_hashes_match},
            "roi": roi, "decision_tables_before": before,
            "decision_tables_after": digest(conn),
        }
        conn.close()
        results.append(report)
    tokens = [item["layers"]["L2"]["tokens_estimate_byte_div_3"] for item in results]
    template_tokens = math.ceil(len(vlm_review.PROMPT.encode("utf-8")) / 3)
    report = {"schema_version": 1, "model_calls": 0, "groups": results,
              "summary": {"groups": len(results), "L2_tokens_median_estimate": statistics.median(tokens),
                          "L2_tokens_max_estimate": max(tokens),
                          "prompt_template_tokens_estimate": template_tokens,
                          "total_text_tokens_median_estimate": statistics.median(tokens) + template_tokens,
                          "total_text_tokens_max_estimate": max(tokens) + template_tokens,
                          "token_estimator": "ceil(canonical UTF-8 bytes / 3), not model tokenizer",
                          "guard_pass": sum(r["guard"]["pass"] for r in results),
                          "stable_join_pass": sum(r["join"]["source_sha_verified"] for r in results),
                          "roi_plan_pass": sum(r["roi"]["pass"] for r in results),
                          "decision_table_digests_unchanged": all(r["decision_tables_before"] == r["decision_tables_after"] for r in results),
                          "text_hard_gates_estimate_pass": statistics.median(tokens) + template_tokens <= 8000 and max(tokens) + template_tokens <= 15000},
              "uncovered": ["G002 mascot/action primary subject absent from persisted detector boxes",
                            "registered top-3 shared local regions not available; only aggregates in L2",
                            "explicit scene applicability unavailable when router disabled",
                            "group decision_json phase shadows mix selection authority and index-based evidence; retained in L0 but not compiled",
                            "semantic action phase/completeness localization unavailable; no labels used as truth",
                            "select/attention_only/abstain and strict irreplaceable schema deferred at ROI stop",
                            "no external VLM or human preference/attention validation performed"]}
    (output / "report-22.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("/home/ubuntu/photo-dedup-eval"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/fable-rebuild")
    args = parser.parse_args()
    run(args.eval_root, args.output)
