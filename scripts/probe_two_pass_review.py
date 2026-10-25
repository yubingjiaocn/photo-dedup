"""G004/G007 deterministic fake two-pass E2E; original tree is read-only."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_vlm_rebuild import digest
from src import db
from src import two_pass_review as tp


def fake_map(rows, regime):
    """Hand-authored fixture windows, NOT a real model result or subject truth.

    Stage uses the performance area because individual performers are too small
    for face/eye claims. G007 uses visible foreground/reflected interaction areas,
    not a claim of three unique people or confirmed mirror correspondence.
    """
    boxes = (
        [[0.24, 0.27, 0.72, 0.59]]
        if regime == "stage"
        else [[0.30, 0.03, 0.79, 0.995], [0.76, 0.0, 1.0, 0.65]]
    )
    regions, tracks, coverage = [], [], []
    for i, row in enumerate(rows):
        frame = {k: row[k] for k in ("file_id", "content_sha256")}
        coverage.append(
            {
                "frame": frame,
                "state": "candidate_complete",
                "uncertainty": "none_reported",
                "confidence": "medium",
            }
        )
        for j, box in enumerate(boxes):
            regions.append(
                {
                    "id": f"R{i * len(boxes) + j + 1}",
                    "frame": frame,
                    "box": box,
                    "role": "key_subject",
                    "kind": "stage_area" if regime == "stage" else "interaction",
                    "confidence": "medium",
                    "grounding": {"page": tp.identity(row), "visible": True},
                }
            )
    for j in range(len(boxes)):
        tracks.append(
            {
                "id": f"T{j + 1}",
                "regions": [f"R{i * len(boxes) + j + 1}" for i in range(len(rows))],
                "confidence": "medium",
                "state": "candidate",
            }
        )
    return {
        "scene": "stage" if regime == "stage" else "multi_person",
        "regime": regime,
        "confidence": "medium",
        "coverage": coverage,
        "regions": regions,
        "correspondence": tracks,
    }


def fake_judgment(subject_map, decision="select"):
    refs = subject_map["correspondence"][0]["regions"]
    lookup = {r["id"]: r for r in subject_map["regions"]}
    evidence = [{"region": r, "confidence": "medium"} for r in refs]
    return {
        "decision": decision,
        "confidence": "medium" if decision == "select" else "low",
        "recommended": [
            {"frame": lookup[r]["frame"], "evidence": [e]}
            for r, e in zip(refs, evidence)
        ]
        if decision == "select"
        else [],
        "multi_select_reason": {
            "text": "FAKE fixture: complementary interaction/lighting phases.",
            "evidence": evidence,
        }
        if decision == "select"
        else None,
        "irreplaceable": [
            {
                "frame": lookup[r]["frame"],
                "subject": r,
                "phase": "lighting",
                "evidence": [e],
            }
            for r, e in zip(refs, evidence)
        ]
        if decision == "select"
        else [],
        "discriminators": [
            {
                "text": "FAKE fixture distinction; not a real preference judgment.",
                "evidence": evidence,
            }
        ],
    }


def variants(base):
    positive = copy.deepcopy(base)
    # Explicit bystander is retained in the map/ledger, never a main track.
    for i, c in enumerate(positive["coverage"]):
        positive["regions"].append(
            {
                "id": f"R{10 + i}",
                "frame": c["frame"],
                "box": [0.03, 0.65, 0.25, 0.98],
                "role": "bystander",
                "kind": "subject",
                "confidence": "medium",
                "grounding": {"page": tp.identity(c["frame"]), "visible": True},
            }
        )
    missing = copy.deepcopy(base)
    missing["coverage"][0].update(state="missing", uncertainty="missing_key")
    pollution = copy.deepcopy(positive)
    pollution["correspondence"][0]["regions"][0] = "R10"
    ambiguous = copy.deepcopy(base)
    ambiguous["correspondence"][0]["state"] = "ambiguous"
    yield "positive", positive, "select"
    yield "missing", missing, "attention_only"
    yield "bystander-pollution", pollution, "attention_only"
    yield "ambiguous", ambiguous, "attention_only"
    yield "low-confidence", positive, "attention_only"


def run(eval_root, output):
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError("output must remain inside repo")
    output.mkdir(parents=True, exist_ok=True)
    old_root = ROOT / "artifacts/multi-subject-narrow"
    old_hashes = {p.name: tp.sha(p.read_bytes()) for p in old_root.glob("*.json")}
    reports = []
    for date, gid, expected, regime in (
        ("2026-07-25", 4, {470, 481}, "stage"),
        ("2026-04-05", 7, {35, 131}, "group"),
    ):
        inventory = (
            eval_root
            / "astra-regime-validation-20260910/development"
            / date
            / "inventory.sqlite"
        )
        with sqlite3.connect(f"file:{inventory}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            before = digest(conn)
            rows = [dict(r) for r in db.group_page_by_id(conn, gid)]
            assert {r["file_id"] for r in rows} == expected
            sources = {
                r["file_id"]: Path(
                    conn.execute(
                        "SELECT path FROM files WHERE id=?", (r["file_id"],)
                    ).fetchone()[0]
                )
                for r in rows
            }
            if any(
                not p.resolve().is_relative_to(eval_root.resolve())
                for p in sources.values()
            ):
                raise ValueError("source outside frozen tree")
            base = fake_map(rows, regime)
            for name, a, expected_decision in variants(base):
                b = fake_judgment(
                    base, "attention_only" if name == "low-confidence" else "select"
                )
                runner = tp.FakeModel({"A": a, "B": b})
                case = output / f"{date}-G{gid:03d}-{name}"
                result = tp.run_offline(
                    rows,
                    sources,
                    case,
                    model_a="fake:subject-map-v1",
                    model_b="fake:deep-judge-v1",
                    runner=runner,
                )
                assert result["judgment"]["decision"] == expected_decision, result[
                    "blockers"
                ]
                reports.append(
                    {
                        "case": case.name,
                        "decision": result["judgment"]["decision"],
                        "blockers": result["blockers"],
                        "calls": result["accounting"],
                        "strips": len(
                            [
                                a
                                for a in result["assets"]
                                if a["kind"] != "full_group_page"
                            ]
                        ),
                        "decision_tables_before": before,
                        "decision_tables_after": digest(conn),
                    }
                )
                assert before == reports[-1]["decision_tables_after"]
    assert old_hashes == {
        p.name: tp.sha(p.read_bytes()) for p in old_root.glob("*.json")
    }
    summary = {
        "scope": "fake mechanics only; no real model effectiveness or production admission",
        "cases": reports,
        "old_real_results_unchanged": True,
        "old_real_results_sha256": old_hashes,
        "external_model_calls": 0,
        "production_enabled": False,
        "budgets": tp.BUDGET,
    }
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root", type=Path, default=Path("/home/ubuntu/photo-dedup-eval")
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/two-pass-multi-subject"
    )
    args = parser.parse_args()
    run(args.eval_root, args.output)
