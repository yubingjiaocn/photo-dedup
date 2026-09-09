#!/usr/bin/env python3
"""Read-only v4 selective-prediction A/B over cached Disney/JX3 evidence."""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_ab import _groups, fingerprint, open_readonly, run as phase_run
from src import phase_selection as PS
from src.risk_coverage import (
    DIAGNOSTIC_SAMPLE, REVIEW_PRIMARY, SAFE_SILENT, add_deciles,
    assign_primary_budget, frontier, risk_score, stratified_audit_sample,
)

BUDGETS = (0.03, 0.05, 0.10, 0.20)


def _load_reports(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for dataset in ("disney", "jx3"):
        value = json.loads((root / f"{dataset}.json").read_text(encoding="utf-8"))
        for row in value["groups"]:
            output[(dataset, int(row["group_id"]))] = row
    return output


def _evidence(group: Any, members: list[dict[str, Any]], row: dict[str, Any],
              v2: MappingLike, v3: MappingLike) -> dict[str, Any]:
    embeddings = [PS._embedding(member) for member in members]
    valid_pairs = []
    adjacent = []
    for left in range(len(members)):
        for right in range(left + 1, len(members)):
            a, b = embeddings[left], embeddings[right]
            if a is not None and b is not None and a.shape == b.shape:
                similarity = float(np.dot(a, b))
                valid_pairs.append(similarity)
                if right == left + 1:
                    adjacent.append(similarity)
    times = [int(item["exif_timestamp"]) for item in members if item.get("exif_timestamp") is not None]
    face_counts = {int(item.get("face_count") or 0) for item in members}
    centers = [PS._subject_center(item) for item in members]
    valid_centers = [item for item in centers if item is not None]
    position_conflict = any(
        max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 0.22
        for index, a in enumerate(valid_centers) for b in valid_centers[index + 1:]
    )
    utilities = sorted(PS.utility_scores(members).values(), reverse=True)
    quality_margin = utilities[0] - utilities[1] if len(utilities) > 1 else 1.0
    critical_missing = sum(
        item.get("exif_timestamp") is None or PS._embedding(item) is None for item in members
    )
    key = (row["dataset"], int(row["group_id"]))
    old2, old3 = v2.get(key), v3.get(key)
    v2_v3 = bool(old2 and old3 and (
        old2.get("candidate_keeper_ids") != old3.get("candidate_keeper_ids")
        or old2.get("logical_phase_count") != old3.get("logical_phase_count")
    ))
    return {
        "phase_coverage_diagnostics": row.get("phase_coverage_diagnostics", []),
        "group_type": str(group["group_type"]), "member_count": len(members),
        "time_span_seconds": max(times) - min(times) if len(times) > 1 else 0,
        "min_pair_similarity": min(valid_pairs) if valid_pairs else None,
        "min_adjacent_similarity": min(adjacent) if adjacent else None,
        "face_count_states": len(face_counts), "position_conflict": position_conflict,
        "quality_margin": quality_margin, "critical_missing_count": critical_missing,
        "v2_v3_disagreement": v2_v3, "keeper_count": len(row["candidate_keeper_ids"]),
        "logical_phase_count": row["logical_phase_count"], "changed": row["changed"],
        "baseline_keeper_ids": row["baseline_keeper_ids"],
        "candidate_keeper_ids": row["candidate_keeper_ids"],
    }


# Kept structural to avoid importing typing_extensions on the evaluation host.
MappingLike = dict[tuple[str, int], dict[str, Any]]


def _human_errors(annotation_path: Path | None, rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    if annotation_path is None:
        return {"status": "unavailable", "groups": 0, "errors": set(), "annotation_status": None}
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    errors: set[tuple[str, int]] = set()
    assessed = 0
    details = []
    for item in document.get("annotations", []):
        dataset = str(item["event_id"])
        number = int(str(item["group_id"]).split("G")[-1])
        row = rows.get((dataset, number))
        if not row:
            continue
        assessed += 1
        candidate = set(row["candidate_keeper_ids"])
        phase_miss = False
        keeper_bad = False
        for phase, member_ids in item.get("phase_ids", {}).items():
            acceptable = set(item.get("acceptable_keepers", {}).get(phase, []))
            selected_in_phase = candidate.intersection(member_ids)
            if not selected_in_phase.intersection(acceptable):
                phase_miss = True
            if selected_in_phase - acceptable:
                keeper_bad = True
        error = bool(item.get("group_impure") or item.get("phase_undersegmented")
                     or phase_miss or keeper_bad)
        if error:
            errors.add((dataset, number))
        details.append({"audit_id": f"{dataset}:G{number}", "error": error,
                        "phase_miss": phase_miss, "keeper_bad": keeper_bad})
    return {"status": "provisional_human_tune_not_random_audit", "groups": assessed,
            "errors": errors, "details": details,
            "annotation_status": document.get("annotation_status")}


def _contact_sheets(audit: dict[str, Any], row_map: dict[tuple[str, int], dict[str, Any]],
                    output: Path) -> list[str]:
    sheets = output / "audit-contact-sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for entry in audit["groups"]:
        row = row_map[(entry["dataset"], int(entry["group_id"]))]
        cells = []
        for member_id, thumb in zip(row["member_ids"], row["thumbnail_paths"], strict=True):
            path = Path(thumb)
            try:
                image = Image.open(path).convert("RGB")
                image.thumbnail((220, 160))
            except (OSError, ValueError):
                image = Image.new("RGB", (220, 160), "#333333")
            canvas = Image.new("RGB", (230, 195), "#151515")
            canvas.paste(image, ((230 - image.width) // 2, 5))
            draw = ImageDraw.Draw(canvas)
            badge = "KEEP" if member_id in row["candidate_keeper_ids"] else "shadow"
            draw.text((7, 172), f"M{member_id}  {badge}", fill="#ffffff")
            cells.append(canvas)
        width = min(4, max(1, len(cells))) * 230
        height = math.ceil(len(cells) / 4) * 195 + 55
        sheet = Image.new("RGB", (width, height), "#101010")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{entry['audit_id']}  risk={entry['risk_score']:.3f}  {entry['risk_reason']}", fill="#ffffff")
        for index, cell in enumerate(cells):
            sheet.paste(cell, ((index % 4) * 230, 55 + (index // 4) * 195))
        filename = f"{entry['dataset']}-G{entry['group_id']}.jpg"
        sheet.save(sheets / filename, quality=88)
        entry["contact_sheet"] = f"audit-contact-sheets/{filename}"
        artifacts.append(entry["contact_sheet"])
    return artifacts


def _review_ui(rows: list[dict[str, Any]], output: Path) -> None:
    queue = sorted((row for row in rows if row["selective_output"] in {REVIEW_PRIMARY, DIAGNOSTIC_SAMPLE}),
                   key=lambda row: (-row["expected_review_value"], row["dataset"], row["group_id"]))
    cards = []
    for row in queue:
        figures = []
        for member, thumb in zip(row["member_ids"], row["thumbnail_paths"], strict=True):
            badge = "keeper" if member in row["candidate_keeper_ids"] else ""
            figures.append(f'<figure class="{badge}"><img src="{html.escape(Path(thumb).as_uri())}"><figcaption>M{member} {badge}</figcaption></figure>')
        key = f"{row['dataset']}:G{row['group_id']}"
        cards.append(f'''<section data-id="{key}"><h2>{key} · {row['selective_output']} · risk {row['risk_score']:.3f}</h2>
<p>{html.escape(row['risk_reason'])}</p><div class="strip">{"".join(figures)}</div><div class="buttons">
<button data-v="keeper_correct">当前 keeper 正确</button><button data-v="keep_more">应多留</button>
<button data-v="keeper_wrong">keeper 选错</button><button data-v="group_false_merge">分组误并</button></div></section>''')
    page = f'''<!doctype html><meta charset="utf-8"><title>v4 risk review</title><style>
body{{font:14px system-ui;background:#111;color:#eee;margin:20px}}section{{padding:16px 0;border-top:1px solid #555}}.strip{{display:flex;flex-wrap:wrap;gap:8px}}figure{{margin:0;border:2px solid #555;padding:4px}}figure.keeper{{border-color:#5d7}}img{{width:180px;height:135px;object-fit:contain;background:#222}}button{{margin:10px 6px 0 0;padding:10px}}section.done{{opacity:.42}}pre{{white-space:pre-wrap}}</style>
<h1>Selective review / diagnostic audit</h1><p>Expected-value order. Anonymous IDs and cached thumbnails only. Export downloads JSON; no database write.</p>
<button id="export">导出匿名 label JSON</button><pre id="status"></pre>{''.join(cards)}<script>
const labels={{}}; document.querySelectorAll('section button').forEach(b=>b.onclick=()=>{{const s=b.closest('section'); labels[s.dataset.id]={{audit_id:s.dataset.id,verdict:b.dataset.v,label_status:'human_reviewed'}};s.classList.add('done');document.querySelector('#status').textContent=JSON.stringify(labels,null,2)}});
document.querySelector('#export').onclick=()=>{{const payload={{schema_version:1,annotation_kind:'risk_coverage_v4_audit',labels:Object.values(labels)}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}));a.download='risk-coverage-v4-labels.json';a.click()}};</script>'''
    (output / "review-prototype.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", nargs=2, required=True, metavar=("NAME", "DB"))
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--primary-budget", type=float, default=0.10)
    parser.add_argument("--audit-size", type=int, default=30)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v2, v3 = _load_reports(args.v2), _load_reports(args.v3)
    phase_root = output / "shadow-decisions"
    source_before = {}
    source_after = {}
    rows = []
    for name, source_text in args.dataset:
        source = Path(source_text).resolve()
        source_before[name] = fingerprint(source)
        report = phase_run(name, source, phase_root, source_sha=args.source_sha)
        report_rows = {int(row["group_id"]): row for row in report["groups"]}
        conn = open_readonly(source)
        for group, members in _groups(conn):
            row = report_rows[int(group["id"])]
            evidence = _evidence(group, members, row, v2, v3)
            row["risk_evidence"] = evidence
            row.update(risk_score(evidence))
            rows.append(row)
        conn.close()
        source_after[name] = fingerprint(source)
        if source_before[name] != source_after[name]:
            raise RuntimeError(f"source changed: {name}")
    add_deciles(rows)
    ranked = assign_primary_budget(rows, args.primary_budget)
    row_map = {(row["dataset"], int(row["group_id"])): row for row in rows}
    human = _human_errors(args.annotations, row_map)
    curve = frontier(rows, BUDGETS, human["errors"])
    audit = stratified_audit_sample(rows, args.audit_size)
    contacts = _contact_sheets(audit, row_map, output)
    _review_ui(rows, output)
    primary = [row for row in rows if row["selective_output"] == REVIEW_PRIMARY]
    diagnostics = [row for row in rows if row["selective_output"] == DIAGNOSTIC_SAMPLE]
    safe = [row for row in rows if row["selective_output"] == SAFE_SILENT]
    total_members = sum(row["member_count"] for row in rows)
    primary_members = sum(row["member_count"] for row in primary)
    buckets = {str(decile): sum(row["risk_decile"] == decile for row in rows) for decile in range(1, 11)}
    summary = {
        "schema_version": 4, "source_sha": args.source_sha,
        "authority": "shadow_review_only", "auto_remove_authority": "BYTE_IDENTICAL_ONLY",
        "opened_original_media": False, "lin_held_out_touched": False,
        "source_fingerprint_before": source_before, "source_fingerprint_after": source_after,
        "group_count": len(rows), "member_count": total_members,
        "v3_primary_baseline": {"groups": 46, "group_rate": 46 / 113,
                                "source": "phase-ab-v3 final: disney 28 + jx3 18"},
        "selective_outputs": {REVIEW_PRIMARY: len(primary), DIAGNOSTIC_SAMPLE: len(diagnostics), SAFE_SILENT: len(safe)},
        "mandatory_review_groups": sum(bool(row.get("mandatory_review")) for row in rows),
        "review_budget_overflow_groups": sum(bool(row.get("review_budget_overflow")) for row in rows),
        "primary_review": {"groups": len(primary), "group_rate": len(primary) / len(rows),
                           "members": primary_members, "member_rate": primary_members / total_members},
        "silent_coverage": (len(rows) - len(primary)) / len(rows), "risk_decile_counts": buckets,
        "frontier": curve,
        "human_label_metrics": {
            "status": human["status"], "annotation_status": human["annotation_status"],
            "assessed_groups": human["groups"], "observed_error_groups": len(human["errors"]),
            "warning": "bounded tune set is not a probability sample; no whole-silent-region CI",
        },
        "silent_error_upper_bound_95": None,
        "silent_error_upper_bound_status": "unavailable_until_human_labels_for_random_diagnostic_sample",
        "audit": {"sample_count": audit["sample_count"], "silent_population": audit["silent_population"],
                  "label_status": "unlabelled", "contact_sheets": len(contacts)},
        "keeper_phase": {"candidate_keepers": sum(len(row["candidate_keeper_ids"]) for row in rows),
                         "logical_phases": sum(row["logical_phase_count"] for row in rows),
                         "selected_per_phase": (sum(len(row["candidate_keeper_ids"]) for row in rows)
                                                / sum(row["logical_phase_count"] for row in rows))},
        "claims": {"risk_buckets_and_coverage": "full real inventories",
                   "error_capture": "provisional human tune subset only",
                   "precision": "not claimed", "audit_labels": "unlabelled"},
    }
    export_rows = [{key: value for key, value in row.items() if key != "thumbnail_paths"} for row in ranked]
    (output / "risk-groups.json").write_text(json.dumps({"groups": export_rows}, indent=2), encoding="utf-8")
    (output / "primary-review-queue.json").write_text(json.dumps({"groups": [{key: value for key, value in row.items() if key != "thumbnail_paths"} for row in primary]}, indent=2), encoding="utf-8")
    (output / "audit-manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "human-tune-evaluation.json").write_text(json.dumps({**human, "errors": sorted(f"{d}:G{g}" for d, g in human["errors"])}, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
