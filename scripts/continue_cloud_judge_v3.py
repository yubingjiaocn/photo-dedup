"""Import the three consumed v2 slots after local provenance/contract-v3 validation.

No transport. Original request/raw/receipt and assets remain byte-identical.
"""

from __future__ import annotations

import argparse
import base64
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cloud_judge_batch as b
from src import two_pass_review as tp

CONSUMED = ("2026-04-04-G014", "2026-04-04-G057", "2026-04-05-G007")


def validate_saved(source, group, frozen, router, destination):
    case = source / group["group"]
    state = b.read(case / "state.json")
    contract = b.read(source / "contract.json")
    entry = next((e for e in router["groups"] if e["group"] == group["group"]), None)
    b.verify(
        state["reserved"] and state["external_calls"] == 1, "missing consumed slot"
    )
    b.verify(
        state["input_sha256"]
        == tp.sha(tp.canonical({"contract": contract, "group": group, "local": entry})),
        "source input hash mismatch",
    )
    for name, sha in state["artifact_hashes"].items():
        b.verify(
            tp.sha((case / name).read_bytes()) == sha,
            "source artifact hash mismatch: " + name,
        )
    request = b.read(case / "request.json")
    response = b.read(case / "provider-response.json")
    b.verify(
        tp.sha(tp.canonical(request)) == state["request_sha256"],
        "request hash mismatch",
    )
    b.verify(
        tp.sha(tp.canonical(response)) == state["response_sha256"],
        "response hash mismatch",
    )
    wire = b.read(case / "wire-request.json")
    b.verify(wire["modelId"] == b.MODEL, "wire model mismatch")
    content = wire["messages"][0]["content"]
    b.verify(
        content[0]["text"]
        == tp.canonical({k: v for k, v in request.items() if k != "images"}).decode()
        + tp.PRIVACY_INSTRUCTION,
        "wire text mismatch",
    )
    b.verify(
        len(content) == 2 * len(request["images"]) + 1, "wire image count mismatch"
    )
    for i, asset in enumerate(request["images"]):
        b.verify(
            content[1 + 2 * i]["text"] == tp.canonical(asset).decode(),
            "wire asset label mismatch",
        )
        block = content[2 + 2 * i]
        data = base64.b64decode(block["image"]["source"]["bytes"]["bytes_base64"])
        b.verify(
            tp.sha(data)
            == asset["sha256"]
            == tp.sha((case / asset["path"]).read_bytes()),
            "wire asset mismatch",
        )
    text = "\n".join(x["text"] for x in response["output"]["message"]["content"])
    b.verify(text == (case / "raw.txt").read_text(), "receipt/raw mismatch")
    parsed = tp.strict_json(text)
    b.verify(parsed == b.read(case / "raw.json"), "parsed raw mismatch")
    # Re-render from SHA-verified frozen original sources, proving region/temporal/L2 binding.
    check = destination / "source-check"
    check.mkdir(parents=True, exist_ok=True)
    rebuilt, rows, packet, validation, l2 = b.prepare(
        frozen,
        group,
        entry,
        ROOT / "artifacts/cloud-judge-batch/authorized-input",
        check,
    )
    b.verify(rebuilt["context"] == request["context"], "source/region/context mismatch")
    b.verify(rebuilt["images"] == request["images"], "source images mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    for p in case.iterdir():
        if p.is_file() and p.name != "state.json":
            shutil.copy2(p, destination / p.name)
    judgment, usage, cost = b.response_judgment(
        response, request, destination, rows, packet, validation, l2, contract["rates"]
    )
    b.verify(judgment == parsed, "raw must not be rewritten")
    return state, judgment, usage, cost


def run(frozen_path, router_path, source, output):
    source, output = Path(source).resolve(), b.artifact_root(output)
    b.verify(
        source != output and not output.is_relative_to(source),
        "new continuation directory required",
    )
    frozen, router = b.read(frozen_path), b.read(router_path)
    b.verify_snapshot(frozen)
    original_hashes = {
        str(p.relative_to(source)): tp.sha(p.read_bytes())
        for p in source.rglob("*")
        if p.is_file()
    }
    b.verify(
        not (output / "continuation.json").exists(),
        "continuation already imported; resume authorized runner",
    )
    b.run(frozen_path, router_path, output, selected_groups=[])
    new_contract = b.read(output / "contract.json")
    for gid in CONSUMED:
        group = next(g for g in frozen["groups"] if g["group"] == gid)
        entry = next((e for e in router["groups"] if e["group"] == gid), None)
        case = output / gid
        old, judgment, usage, cost = validate_saved(source, group, frozen, router, case)
        state = {
            **old,
            "state": "validated",
            "route": judgment["decision"],
            "reason": None,
            "cache_hit": True,
            "calls_this_run": 0,
            "input_sha256": tp.sha(
                tp.canonical({"contract": new_contract, "group": group, "local": entry})
            ),
            "validated_sha256": tp.sha(tp.canonical(judgment)),
            "usage": usage,
            "estimated_cost_usd": cost,
            "imported_from": str(source / gid),
            "imported_contract": "v2",
            "revalidated_contract": "v3",
        }
        state["artifact_hashes"] = {
            p.name: tp.sha(p.read_bytes())
            for p in case.iterdir()
            if p.is_file() and p.name != "state.json"
        }
        b.save(case / "state.json", state)
    b.verify(
        original_hashes
        == {
            str(p.relative_to(source)): tp.sha(p.read_bytes())
            for p in source.rglob("*")
            if p.is_file()
        },
        "source evidence changed",
    )
    b.save(
        output / "continuation.json",
        {
            "contract": "v3",
            "source": str(source),
            "source_hashes": original_hashes,
            "imported_groups": list(CONSUMED),
            "external_calls_this_run": 0,
            "inherited_external_calls": 3,
            "validated": 3,
            "raw_unchanged": True,
        },
    )
    return b.run(frozen_path, router_path, output, selected_groups=[])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--frozen", type=Path, default=ROOT / "artifacts/cloud-judge-batch/frozen.json"
    )
    p.add_argument(
        "--scene-router",
        type=Path,
        default=ROOT / "artifacts/cloud-judge-batch/authorized-input/scene-router.json",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v2",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3",
    )
    args = p.parse_args()
    result = run(args.frozen, args.scene_router, args.source, args.output)
    print(
        {
            k: result[k]
            for k in ("states", "external_calls_total", "external_calls_this_run")
        }
    )
