"""Compare current review-only selection to frozen v0 on all existing groups.

No model inference or report regeneration; uses prior copied feature DBs read-only.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import phase_selection as current  # noqa: E402
import evaluate_local_quality as baseline_loader  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    output = root / "dense-regression.json"
    if output.exists():
        raise ValueError("Regression already recorded; inspect it")
    baseline_loader.PIN = "f79cc0d"
    baseline = baseline_loader.pinned()
    dense = {}
    for path in (root / "dense-review").glob("*/result.json"):
        r = json.loads(path.read_text())
        packet = r["dense_packet"]
        dense[r["day"], frozenset(packet["group_member_ids"])] = packet
    totals = Counter()
    strata = {}
    rows = []
    for oldpath in sorted((root / "stage2-3").glob("*/result.json")):
        old = json.loads(oldpath.read_text())
        day = old["day"]
        origin = root.parent / (
            "astra-local-quality-20260910"
            if day.startswith("2026-07-2")
            else "astra-real-events-20260910"
        )
        mapping = json.loads((origin / f"panels/{day}/mapping.json").read_text())
        oldrows = {r["group_alias"]: r for r in old["rows"]}
        c = sqlite3.connect(
            f"file:{root}/stage2-3/{day}/review_only/inventory.sqlite?mode=ro", uri=True
        )
        c.row_factory = sqlite3.Row
        for group in mapping:
            ids = [f["id"] for f in group["frames"]]
            aliases = [f["alias"] for f in group["frames"]]
            fetched = {
                r["id"]: dict(r)
                for r in c.execute(
                    "SELECT f.*,fe.* FROM files f JOIN features fe ON f.id=fe.file_id WHERE f.id IN ("
                    + ",".join("?" for _ in ids)
                    + ")",
                    ids,
                )
            }
            members = [fetched[i] for i in ids]
            packet = dense.get((day, frozenset(ids)))
            if packet:
                for member in members:
                    meta = json.loads(member["quality_meta"])
                    meta["dense_instance_recovery"] = packet
                    member["quality_meta"] = json.dumps(meta)
            phases = baseline.segment_phases(members, group_type=group["group_type"])
            frozen = baseline.select_phase_keepers(members, phases)
            candidate = current.select_phase_keepers(
                members, phases, instance_recovery_policy="review_only"
            )
            for key in ["keepers", "utility_scores", "group_keeper_budget", "phases"]:
                assert frozen[key] == candidate[key], (day, group["group_alias"], key)
            selected = [aliases[i] for i in candidate["keepers"]]
            assert (
                selected == oldrows[group["group_alias"]]["variants"]["off"]["keepers"]
            )
            proposed = bool(candidate["instance_recovery_context"]["proposed_groups"])
            regime = oldrows[group["group_alias"]]["regime"]
            strata.setdefault(regime, Counter()).update(
                groups=1, photos=len(ids), proposed_groups=int(proposed)
            )
            totals.update(
                groups=1,
                photos=len(ids),
                keepers=len(selected),
                proposed_groups=int(proposed),
            )
            rows.append(
                {
                    "day": day,
                    "group_alias": group["group_alias"],
                    "regime": regime,
                    "keepers": selected,
                    "proposed": proposed,
                }
            )
        c.close()
    assert (
        totals["groups"] == 188
        and totals["photos"] == 498
        and totals["proposed_groups"] == 2
    )
    result = {
        "baseline_pin": "f79cc0d",
        "totals": totals,
        "strata": strata,
        "rows": rows,
        "keeper_order_scores_budget_phases_exact_parity": True,
        "model_inference": False,
        "actual_stage2_stage3_new_input_scope": "52 Jul25 groups separately in dense-stage2-3",
        "previous_all188_stage2_stage3_scope": "stage2-3/ remains the frozen v0 proof",
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
