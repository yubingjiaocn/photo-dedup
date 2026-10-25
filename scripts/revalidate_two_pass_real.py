"""Verify immutable v1 provenance; reuse A locally and spend only unconsumed B slots."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jsonschema
from PIL import Image, ImageOps

from scripts.report_vlm_rebuild import digest
from src import db
from src import two_pass_review as tp

GROUPS = (("2026-07-25", 4, {470, 481}), ("2026-04-05", 7, {35, 131}))
IMPLEMENTATION = (
    "src/two_pass_review.py",
    "src/vlm_visual_budget.py",
    "src/subject_set_evidence.py",
    "src/vlm_review.py",
    "src/vlm_criterion_compiler.py",
    "src/vlm_prompt_context.py",
    "src/vlm_identity.py",
    "scripts/revalidate_two_pass_real.py",
    "schemas/two-pass-subject-map-v1.schema.json",
    "schemas/two-pass-deep-judge-v1.schema.json",
)


def check(condition, message):
    if not condition:
        raise tp.ReviewError(message)


def implementation_hashes():
    return {name: tp.sha((ROOT / name).read_bytes()) for name in IMPLEMENTATION}


def verify_saved_a(output, group, rows, sources):
    """No transport. Original implementation, wire, schema and source identities bind A."""
    old = output / "archive/v1"
    manifest = json.loads((old / "implementation-manifest.json").read_text())
    check(
        manifest == json.loads((output / "implementation-manifest.json").read_text()),
        "original implementation manifest changed",
    )
    for name, expected in manifest.items():
        check(
            tp.sha((old / "implementation" / name).read_bytes()) == expected,
            f"original implementation hash: {name}",
        )
    ledger = json.loads((old / "call-ledger.json").read_text())
    entries = [c for c in ledger if c["group"] == group and c["pass"] == "A"]
    check(len(entries) == 1, "one original A required")
    entry = entries[0]
    current = json.loads((output / "call-ledger.json").read_text())
    check(
        [c for c in current if c["group"] == group and c["pass"] == "A"] == entries,
        "original A ledger changed",
    )
    case = output / group
    for name in (
        "pass-A-wire-request.json",
        "pass-A-request.json",
        "pass-A-provider-response.json",
        "pass-A-raw.txt",
        "pass-A-raw.json",
    ):
        check(
            (case / name).read_bytes() == (old / group / name).read_bytes(),
            f"original A artifact changed: {name}",
        )
    for name, field in (
        ("pass-A-wire-request.json", "request_sha256"),
        ("pass-A-provider-response.json", "response_sha256"),
        ("pass-A-raw.txt", "response_text_sha256"),
        ("pass-A-request.json", "payload_sha256"),
    ):
        check(
            tp.sha((case / name).read_bytes()) == entry[field], f"hash mismatch: {name}"
        )
    ar = json.loads((case / "pass-A-request.json").read_text())
    original_schema = json.loads(
        (old / "implementation/schemas/two-pass-subject-map-v1.schema.json").read_text()
    )
    check(ar["response_schema"] == original_schema, "original schema mismatch")
    wire = json.loads((case / "pass-A-wire-request.json").read_text())
    check(
        wire["modelId"]
        == entry["model_id"]
        == ar["model"]
        == "global.anthropic.claude-sonnet-5-5",
        "model identity mismatch",
    )
    content = wire["messages"][0]["content"]
    wire_payload = json.JSONDecoder().raw_decode(content[0]["text"])[0]
    check(
        wire_payload == {k: v for k, v in ar.items() if k != "images"},
        "wire/payload mismatch",
    )
    check(
        ar["context"]["frames"] == [tp.identity(r) for r in rows],
        "source identity mismatch",
    )
    check(len(content) == 1 + 2 * len(ar["images"]), "wire image count")
    images = {}
    for row, asset, i in zip(rows, ar["images"], range(len(rows))):
        path = sources[row["file_id"]]
        check(tp.sha(path.read_bytes()) == row["content_sha256"], "source SHA mismatch")
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        check(image.size == (row["width"], row["height"]), "source dimensions mismatch")
        images[tp.identity(row)] = image
        check(json.loads(content[1 + 2 * i]["text"]) == asset, "wire image metadata")
        image_path = (case / asset["path"]).resolve()
        check(image_path.is_relative_to(case.resolve()), "image path escape")
        data = image_path.read_bytes()
        check(
            data == (old / group / asset["path"]).read_bytes(), "original page changed"
        )
        check(
            tp.sha(data) == asset["sha256"] and len(data) == asset["bytes"],
            "page hash/size",
        )
        check(
            base64.b64decode(
                content[2 + 2 * i]["image"]["source"]["bytes"]["bytes_base64"]
            )
            == data,
            "wire image bytes mismatch",
        )
        with Image.open(image_path) as page:
            check(list(page.size) == asset["size"], "page dimensions mismatch")
    response = json.loads((case / "pass-A-provider-response.json").read_text())
    text = "\n".join(b["text"] for b in response["output"]["message"]["content"])
    check(text == (case / "pass-A-raw.txt").read_text(), "response text mismatch")
    check(
        response["stopReason"] == "end_turn" and response["usage"] == entry["usage"],
        "provider receipt mismatch",
    )
    packet = tp.strict_json(text)
    check(
        packet == json.loads((case / "pass-A-raw.json").read_text()),
        "parsed raw mismatch",
    )
    jsonschema.validate(packet, original_schema)
    jsonschema.validate(
        packet,
        json.loads((ROOT / "schemas/two-pass-subject-map-v1.schema.json").read_text()),
    )
    tp.privacy_check(packet)
    validation = tp.validate_map(packet, rows)
    check(not validation["blockers"], f"structural blockers: {validation['blockers']}")
    check(
        validation["decision_ceiling"] == "attention_only",
        "pilot requires attention-only",
    )
    return packet, validation, images, ar["images"], entry


def run(output, eval_root, *, execute=False):
    output = output.resolve()
    check(output.is_relative_to(ROOT), "repo output only")
    version = output / "revalidate-v2"
    version.mkdir(exist_ok=True)
    lock = version / "implementation-manifest.json"
    hashes = implementation_hashes()
    if lock.exists():
        check(
            json.loads(lock.read_text()) == hashes,
            "v2 implementation/schema changed since preflight",
        )
    else:
        check(not execute, "preflight required before execution")
        lock.write_bytes(tp.canonical(hashes))
    original = json.loads((output / "archive/v1/report.json").read_text())
    adapter = tp.BedrockAdapter(
        output,
        model_id="global.anthropic.claude-sonnet-5-5",
        rates=original["cost_basis"],
    )
    check(
        len(adapter.calls) == 2 and all(c["pass"] == "A" for c in adapter.calls),
        "requires exactly two original A slots, no prior B attempt",
    )
    old_hashes = original["old_real_json_hashes_unchanged"]
    for name, expected in old_hashes.items():
        check(
            tp.sha((ROOT / "artifacts/multi-subject-narrow" / name).read_bytes())
            == expected,
            "old real evaluation changed",
        )
    prepared = []
    # Both groups must pass all local verification and budgets before ANY B call.
    for date, gid, expected in GROUPS:
        group = f"{date}-G{gid:03d}"
        case = version / group
        if execute:
            check(not (case / "sidecar.json").exists(), "replay forbidden")
        case.mkdir(exist_ok=True)
        inventory = (
            eval_root
            / "astra-regime-validation-20260910/development"
            / date
            / "inventory.sqlite"
        )
        prior = next(g for g in original["groups"] if g["group"] == group)
        database_sha = tp.sha(inventory.read_bytes())
        check(
            database_sha == prior["frozen_database_sha256_unchanged"],
            "frozen DB changed",
        )
        with sqlite3.connect(f"file:{inventory}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            before = digest(conn)
            check(before == prior["decision_tables_before"], "decision tables changed")
            rows = [dict(r) for r in db.group_page_by_id(conn, gid)]
            check({r["file_id"] for r in rows} == expected, "group membership changed")
            sources = {
                r["file_id"]: Path(
                    conn.execute(
                        "SELECT path FROM files WHERE id=?", (r["file_id"],)
                    ).fetchone()[0]
                )
                for r in rows
            }
        check(
            all(
                p.resolve().is_relative_to(eval_root.resolve())
                for p in sources.values()
            ),
            "source path",
        )
        a, validation, images, pages, receipt = verify_saved_a(
            output, group, rows, sources
        )
        for asset in pages:
            shutil.copy2(output / group / asset["path"], case / asset["path"])
        strips, plan = tp.render_strips(a, validation, images, case)
        audit, l1, l2 = tp.enrich(rows, images, a, validation)
        for name, data in (
            ("L0", audit),
            ("L1", l1),
            ("L2", l2),
            ("pass-A-validated", {"map": a, "validation": validation}),
        ):
            (case / f"{name}.json").write_bytes(tp.canonical(data))
        context = {
            "scene_map": a,
            "evidence": l2,
            "decision_ceiling": validation["decision_ceiling"],
            "coverage_limits": validation["ceiling_reasons"],
            "review_scope": (
                "Only stage structure, lighting and frame changes. No individual-performer coverage or face/eye claims."
                if gid == 4
                else "Key T1 and supporting T2; occlusion remains unresolved. At most three attention points; no selection."
            ),
        }
        br = tp.request("B", adapter.model_id, context, pages + strips, case)
        # Adapter adds privacy instruction and asset metadata on the wire.
        check(
            br["text_token_upper_bound"] + len(tp.PRIVACY_INSTRUCTION.encode())
            <= tp.BUDGET["B"]["text_tokens_upper_bound"],
            "wire text budget",
        )
        (case / "pass-B-request.json").write_bytes(tp.canonical(br))
        provenance = {
            "version": tp.VERSION,
            "source_A": f"../../{group}",
            "A_receipt": receipt,
            "implementation_hashes": hashes,
            "source_identities": [tp.identity(r) for r in rows],
            "all_hashes_schema_source_verified": True,
        }
        (case / "revalidation.json").write_bytes(tp.canonical(provenance))
        prepared.append(
            (case, rows, a, validation, l2, br, plan, inventory, before, database_sha)
        )
    summary = {
        "mode": "revalidate_v2_execute" if execute else "revalidate_v2_preflight",
        "external_calls": len(adapter.calls),
        "maximum_calls": 4,
        "groups": [],
        "production_enabled": False,
        "cost_basis": adapter.rates,
        "old_real_json_hashes_unchanged": old_hashes,
    }
    for (
        case,
        rows,
        a,
        validation,
        l2,
        br,
        plan,
        inventory,
        before,
        database_sha,
    ) in prepared:
        result = {
            "version": tp.VERSION,
            "group": case.name,
            "pass_A_status": "revalidated_original_raw_no_call",
            "pass_B_status": "not_run",
            "decision_ceiling": validation["decision_ceiling"],
            "subject_set_evidence": validation,
            "visual_plan": plan,
            "assets": br["images"],
            "production_enabled": False,
            "decision_table_writes": 0,
            "blockers": [],
        }
        if execute:
            try:
                b = adapter.invoke(br, group=case.name, assets_dir=case)
                (case / "pass-B-raw.json").write_bytes(tp.canonical(b))
                result["judgment"] = tp.validate_judgment(b, rows, a, validation, l2)
                (case / "pass-B-validated.json").write_bytes(
                    tp.canonical(result["judgment"])
                )
                result["pass_B_status"] = "passed"
            except tp.ReviewError as exc:
                result["pass_B_status"] = "failed_no_retry"
                result["blockers"].append(str(exc))
                result["judgment"] = {
                    "decision": "attention_only",
                    "confidence": "low",
                    "recommended": [],
                    "multi_select_reason": None,
                    "irreplaceable": [],
                    "discriminators": [],
                }
            result["accounting"] = [c for c in adapter.calls if c["group"] == case.name]
            result["external_model_calls"] = len(result["accounting"])
        with sqlite3.connect(f"file:{inventory}?mode=ro", uri=True) as conn:
            check(digest(conn) == before, "decision tables changed")
        check(tp.sha(inventory.read_bytes()) == database_sha, "frozen DB changed")
        result.update(
            decision_tables_before=before,
            decision_tables_after=before,
            frozen_database_sha256_unchanged=database_sha,
        )
        if execute:
            result["sidecar_sha"] = tp.sha(tp.canonical(result))
            (case / "sidecar.json").write_bytes(tp.canonical(result))
        summary["groups"].append(result)
    summary["external_calls"] = len(adapter.calls)
    summary["estimated_cost_usd"] = round(
        sum(c.get("estimated_cost_usd") or 0 for c in adapter.calls), 8
    )
    check(implementation_hashes() == hashes, "implementation changed during run")
    for name, expected in old_hashes.items():
        check(
            tp.sha((ROOT / "artifacts/multi-subject-narrow" / name).read_bytes())
            == expected,
            "old real tree changed",
        )
    (version / ("report.json" if execute else "preflight.json")).write_bytes(
        tp.canonical(summary)
    )
    if execute:
        (output / "report.json").write_bytes(tp.canonical(summary))
    print(
        json.dumps(
            {
                "groups": [
                    {
                        "group": g["group"],
                        "B": g["pass_B_status"],
                        "ceiling": g["decision_ceiling"],
                        "blockers": g["blockers"],
                    }
                    for g in summary["groups"]
                ],
                "external_calls": len(adapter.calls),
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/two-pass-real-pilot"
    )
    parser.add_argument(
        "--eval-root", type=Path, default=Path("/home/ubuntu/photo-dedup-eval")
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Use only the two remaining B slots; never invoke A",
    )
    args = parser.parse_args()
    run(args.output, args.eval_root, execute=args.execute)
