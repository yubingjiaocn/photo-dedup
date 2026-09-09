"""Prepare an offline blinded AI-audit packet; does NOT run a model or network.

Inputs are already-exported, fingerprint-verified human-audit cache bundles.
Outputs are disjoint from human JSONL and must never enter selective-risk stats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.human_audit import canonical, digest, load_json, validate_bundle  # noqa: E402
from src.human_audit_bundle import verify_media  # noqa: E402


def prepare(bundle_dir, output, seed="model-audit-v1", strata_count=3):
    bundle_dir, output = Path(bundle_dir), Path(output).absolute()
    if (type(strata_count) is not int or not 0 <= strata_count <= 10
            or not isinstance(seed, str) or not seed):
        raise ValueError("nonempty seed and 0..10 strata required")
    bundle = validate_bundle(load_json(bundle_dir / "manifest.json"))
    verify_media(bundle, bundle_dir / "reviewer")
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output must be a new non-symlink directory")
    if bundle_dir.resolve() in output.parents:
        raise ValueError("AI packet must be separate from immutable human bundle")
    selected = {t["task_id"]: t for t in bundle["tasks"] if t["narrative_recheck"]}
    strata = sorted({t["stratum"] for t in bundle["tasks"] if t["kind"] == "probability"},
                    key=lambda s: digest([seed, s]))[:strata_count]
    for stratum in strata:
        options = [t for t in bundle["tasks"] if t["kind"] == "probability" and t["stratum"] == stratum]
        chosen = min(options, key=lambda t: digest([seed, t["task_id"]]))
        selected[chosen["task_id"]] = chosen
    public, private, media = [], [], {}
    ordered = sorted(selected.values(), key=lambda t: digest([seed, "order", t["task_id"]]))
    for index, task in enumerate(ordered, 1):
        case_id = f"case-{index:02}"
        members = []
        for m in task["media"]:
            name = f'{case_id}-{m["alias"]}.jpg' if m["file"] else None
            members.append({"alias": m["alias"], "file": name})
            if name:
                data = (bundle_dir / "reviewer" / m["file"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != m["sha256"]:
                    raise ValueError("media changed during preparation")
                media[name] = data
        public.append({"case_id": case_id, "members": members,
                       "media_complete": task["media_complete"]})
        private.append({"case_id": case_id, "task_id": task["task_id"],
                        "audit_id": task["audit_id"], "stratum": task["stratum"],
                        "candidate_aliases": [task["aliases"][task["member_ids"].index(m)]
                                              for m in task["candidate_keeper_ids"]]})
    packet = {"schema_version": 1, "annotation_kind": "model_provisional",
              "seed": seed, "source_bundle_id": bundle["bundle_id"],
              "status": "pending_local_inference", "human_annotation": False,
              "selective_risk_eligible": False, "deletion_authority": "none", "cases": public}
    packet["packet_id"] = digest(packet)
    output.mkdir(parents=True, exist_ok=False)
    (output / "blind-input").mkdir()
    for name, data in media.items():
        (output / "blind-input" / name).write_bytes(data)
    (output / "blind-input/tasks.json").write_text(canonical(packet) + "\n", encoding="utf-8")
    (output / "analyst-unblind.json").write_text(canonical({"packet_id": packet["packet_id"],
                                                          "mapping": private}) + "\n", encoding="utf-8")
    instruction = """# Local-only provisional AI visual audit

Use only an already available local vision model with networking disabled.
Do not upload these pixels to Astra or any remote API. Do not open the analyst
mapping until the model's blind output has been saved with its model/version,
prompt, packet ID and inference settings. Never open originals or other data.

For each case, inspect members in supplied temporal order. Return:
annotation_kind=model_provisional; case_id; phases=[{members,acceptable_keepers}];
group_impure=true/false/null; status=assessed/uncertain/unassessable; rationale.
No default phases/keepers. Missing or insufficient thumbnails => unassessable.
Different valuable action/tableau/subject states should remain separate phases;
a phase may have several acceptable keepers. Explain uncertainty explicitly.

Save AI results separately as model-provisional.json (NOT human JSONL). Include
human_annotation=false, selective_risk_eligible=false, deletion_authority=none.
Then compare with analyst-unblind.json and record concrete failures/hypotheses.
Do not compute a safety probability/CI from AI judgments or claim human review.
This preparation script supplies no visual conclusions and performs no inference.
"""
    (output / "blind-input/INSTRUCTIONS.md").write_text(instruction, encoding="utf-8")
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", default="model-audit-v1")
    parser.add_argument("--strata-count", type=int, default=3)
    args = parser.parse_args()
    try:
        result = prepare(args.bundle, args.output, args.seed, args.strata_count)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"cases": len(result["cases"]), "status": result["status"],
                      "packet_id": result["packet_id"]}))


if __name__ == "__main__":
    main()
