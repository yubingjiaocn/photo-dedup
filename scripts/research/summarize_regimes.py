"""Evaluation sidecars only: never imported by runtime or feature producers."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/ubuntu/photo-dedup-eval/astra-local-quality-20260910/region-set")


def main():
    regimes = {}
    for p in (ROOT / "evaluation").glob("*-regimes.json"):
        for g in json.loads(p.read_text())["groups"]:
            key = (g["day"], g["group_alias"])
            if key in regimes:
                raise ValueError("Duplicate sidecar key")
            regimes[key] = g
    totals = defaultdict(lambda: defaultdict(Counter))
    details = []
    seen = set()
    for p in sorted(ROOT.glob("FINAL-*.json")):
        day = p.stem.removeprefix("FINAL-")
        ab = json.loads(p.read_text())
        for row in ab["rows"]:
            key = (day, row["group_alias"])
            side = regimes[key]
            seen.add(key)
            regime = side["regime"]
            phase_by_id = {x["phase_id"]: x for x in row["phases"]}
            interactions = set(side["interaction_phase_ids"])
            if not interactions <= set(phase_by_id):
                raise ValueError(f"Unknown interaction phase {key}")
            aliases = {a for phase in row["phases"] for a in phase["member_aliases"]}
            known = side["subject_coverage_status"] == "assessed" and regime in {
                "single",
                "multi",
            }
            subjects = side["frame_subjects"]
            critical = set(side["critical_subjects"])
            if known and (set(subjects) != aliases or not critical):
                raise ValueError(f"Invalid subject sidecar {key}")
            row_out = {
                "day": day,
                "group_alias": row["group_alias"],
                "regime": regime,
                "metrics": {},
            }
            for variant, result in row["variants"].items():
                kept = set(result["keepers"])
                m = Counter(result["metrics"])
                m.update(
                    groups=1,
                    photos=row["members"],
                    subject_groups_assessed=int(known),
                    subject_groups_unknown=int(
                        not known and regime in {"single", "multi"}
                    ),
                )
                m["interaction_phases"] = len(interactions)
                m["interaction_misses"] = sum(
                    not kept.intersection(phase_by_id[i]["member_aliases"])
                    for i in interactions
                )
                quality_known = all(
                    not ph["required"] or ph["quality_status"] == "adjudicable"
                    for ph in row["phases"]
                )
                m["whole_group_quality_assessed"] = int(quality_known)
                m["whole_group_quality_unknown"] = int(not quality_known)
                m["whole_group_phase_quality_usable"] = int(
                    quality_known
                    and all(
                        not ph["required"]
                        or bool(
                            kept.intersection(ph["acceptable_keeper_aliases"] or [])
                        )
                        for ph in row["phases"]
                    )
                )
                m["subject_coverage_groups_wrong"] = 0
                m["subject_phase_coverage_misses"] = 0
                if known:
                    covered = (
                        set().union(*(set(subjects[a]) for a in kept))
                        if kept
                        else set()
                    )
                    m["subject_coverage_groups_wrong"] = int(not critical <= covered)
                    for phase in row["phases"]:
                        if not phase["required"]:
                            continue
                        expected = (
                            set().union(
                                *(set(subjects[a]) for a in phase["member_aliases"])
                            )
                            & critical
                        )
                        selected = kept.intersection(phase["member_aliases"])
                        covered = (
                            set().union(*(set(subjects[a]) for a in selected))
                            if selected
                            else set()
                        )
                        m["subject_phase_coverage_misses"] += int(
                            not expected <= covered
                        )
                m["eye_related_bad_keeper_upper_bound"] = 0
                m["blur_related_bad_keeper_upper_bound"] = 0
                for phase in row["phases"]:
                    if phase["quality_status"] != "adjudicable":
                        continue
                    bad = kept.intersection(phase["member_aliases"]) - set(
                        phase["acceptable_keeper_aliases"]
                    )
                    reason = str(phase.get("quality_reason") or "").lower()
                    if re.search(
                        r"闭眼|眯眼|眼睛.*闭|eyes?.*(?:closed|narrow)|blink", reason
                    ):
                        m["eye_related_bad_keeper_upper_bound"] += len(bad)
                    if re.search(r"模糊|失焦|糊|blur|soft.focus|soft facial", reason):
                        m["blur_related_bad_keeper_upper_bound"] += len(bad)
                # Same native observations for both selections, never human-sidecar features.
                ctx = row["variants"]["region_set"].get("local_quality_context") or {}
                frames = ctx.get("frames", [])
                order = [a for ph in row["phases"] for a in ph["member_aliases"]]
                # Frame indices in runtime are original F-number order, not phase order.
                order = sorted(order, key=lambda a: int(a[1:]))
                selected_frames = [
                    f
                    for f in frames
                    if order[f["member"]] in kept and f["region_count"]
                ]
                m["selected_frames_with_native_region_quality"] = len(selected_frames)
                m["sum_selected_worst_region_musiq"] = sum(
                    f["worst_musiq"] for f in selected_frames
                )
                m["sum_selected_worst_region_clipiqa"] = sum(
                    f["worst_clipiqa"] for f in selected_frames
                )
                for category in [regime, "overall"]:
                    totals[category][variant].update(m)
                row_out["metrics"][variant] = dict(m)
            details.append(row_out)
    if seen != set(regimes):
        raise ValueError("Sidecars/regression coverage mismatch")
    for variants in totals.values():
        for m in variants.values():
            m["retention"] = m["keepers"] / m["photos"]
            n = m["selected_frames_with_native_region_quality"]
            m["mean_selected_worst_region_musiq"] = (
                m["sum_selected_worst_region_musiq"] / n if n else None
            )
            m["mean_selected_worst_region_clipiqa"] = (
                m["sum_selected_worst_region_clipiqa"] / n if n else None
            )
    output = {
        "status": "exposed_regression_model_provisional",
        "regime_labels_runtime_input": False,
        "metrics": totals,
        "rows": details,
        "eye_and_interaction_semantics_runtime": "unassessed",
        "eye_blur_counts": "upper bounds in phases with annotated defect; not per-actor classifications",
        "native_region_quality": "detector-defined region minima, not semantic critical-subject truth",
    }
    (ROOT / "STRATIFIED.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2)
    )
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
