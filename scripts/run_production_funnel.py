"""Build the development-only production exposure frontier.

Allowed inputs are the 39 development names/labels and their saved v3 case
artifacts.  Holdout names are used only for a cleanliness count.  No provider,
live state, DB, keeper, decision, or delete interface is imported.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src import photo_exposure as router
from src import two_pass_review as tp

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "artifacts/large-group-tuning"
SOURCE = ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3"
TEACHER = TUNING / "candidate/development-run-a-r2"
OUT = TUNING / "production-funnel"
GROUPED_FRACTION_ASSUMPTION = 0.247  # documented frozen-library observation


def read_json(path):
    return json.loads(path.read_text())


def identity_file_id(identity):
    return int(identity.split("@", 1)[0][1:])


def source_hashes(groups):
    records = {}
    for group in groups:
        case = SOURCE / group
        for name in ("request.json", "validated.json", "state.json"):
            path = case / name
            records[str(path.relative_to(ROOT))] = tp.sha(path.read_bytes())
        request = read_json(case / "request.json")
        for asset in request["images"]:
            path = case / asset["path"]
            digest = tp.sha(path.read_bytes())
            if digest != asset["sha256"]:
                raise RuntimeError(f"source asset binding changed: {group}/{asset['path']}")
            records[str(path.relative_to(ROOT))] = digest
    return records


def load_frames(group):
    case = SOURCE / group
    request = read_json(case / "request.json")
    times = {
        tp.identity(item["frame"]): item["capture_time"]
        for item in request["context"]["temporal"]
    }
    assets, frames = {}, []
    for asset in request["images"]:
        if asset["kind"] != "full_group_page":
            continue
        if len(asset["cells"]) != 1:
            raise RuntimeError("local descriptor requires one complete source per page")
        identity = asset["cells"][0]["frame"]
        path = (case / asset["path"]).resolve()
        if not path.is_relative_to(case.resolve()):
            raise RuntimeError("asset path escape")
        data = path.read_bytes()
        if tp.sha(data) != asset["sha256"]:
            raise RuntimeError("asset SHA mismatch")
        with Image.open(path) as image:
            desc = router.descriptor(image)
        frames.append({"identity": identity, "time": times[identity], "descriptor": desc})
        assets[identity] = asset
    if set(times) != {f["identity"] for f in frames}:
        raise RuntimeError("incomplete local frame coverage")
    return frames, assets


def score(predicted, human):
    tp_count = len(predicted & human)
    missing = len(human - predicted)
    extra = len(predicted - human)
    return tp_count, missing, extra


def metrics(total_tp, total_missing, total_extra, exact, groups):
    precision = total_tp / max(total_tp + total_extra, 1)
    recall = total_tp / max(total_tp + total_missing, 1)
    return {
        "groups": groups,
        "exact": exact,
        "exact_rate": exact / groups,
        "true_positive": total_tp,
        "missing": total_missing,
        "extra": total_extra,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def q(values):
    if not values:
        return None
    return {str(p): float(np.quantile(values, p)) for p in (0.1, 0.5, 0.9)}


def main():
    split = read_json(TUNING / "split.json")
    labels = read_json(TUNING / "development-labels.json")["groups"]
    groups = split["development_groups"]
    if len(groups) != 39 or set(groups) != set(labels):
        raise RuntimeError("development allowlist mismatch")
    # Holdout content is never opened.  Only split names/count and disjointness are checked.
    holdout_names = split["holdout_groups"]
    if len(holdout_names) != 21 or set(groups) & set(holdout_names):
        raise RuntimeError("holdout split metadata mismatch")

    before = source_hashes(groups)
    loaded = {group: load_frames(group) for group in groups}
    teacher_pairs = {"same_cluster": [], "different_cluster": []}
    teacher_style = {"clusters": 0, "selected": 0, "uncertain_distinct": 0, "unassessable": 0}
    teacher_selected = {}
    for group in groups:
        answer = read_json(TEACHER / group / "validated.json")
        cluster_of = {}
        for cluster in answer["clusters"]:
            teacher_style["clusters"] += 1
            teacher_style[cluster["relation"]] = teacher_style.get(cluster["relation"], 0) + 1
            for frame in cluster["members"]:
                cluster_of[tp.identity(frame)] = cluster["id"]
        selected = {r["frame"]["file_id"] for r in answer["recommended"]}
        teacher_selected[group] = selected
        teacher_style["selected"] += len(selected)
        frames = loaded[group][0]
        for left, right in itertools.combinations(frames, 2):
            stats = router.pair_metrics(left, right)
            if stats is None:
                continue
            key = "same_cluster" if cluster_of[left["identity"]] == cluster_of[right["identity"]] else "different_cluster"
            teacher_pairs[key].append(stats)

    pair_summary = {}
    for relation, rows in teacher_pairs.items():
        pair_summary[relation] = {
            "pairs": len(rows),
            "quantiles": {key: q([r[key] for r in rows]) for key in ("mae", "tile_p95", "dhash", "seconds", "hist_l1", "edge_mae")},
        }

    profile_results, per_group = [], []
    total_members = sum(len(loaded[g][0]) for g in groups)
    human_total = sum(len(v["human_selected"]) for v in labels.values())
    for profile in router.ROUTER_PROFILES:
        vlm_groups = exposed_count = pixels = parts = local_groups = 0
        coverage_hits = vlm_hits = 0
        total_tp = total_missing = total_extra = exact = 0
        for group in groups:
            frames, assets = loaded[group]
            plan = router.route_group(frames, profile)
            exposed = set(plan["vlm_exposed"])
            local = set(plan["local_selected"])
            human = set(labels[group]["human_selected"])
            covered_ids = {identity_file_id(x) for x in exposed | local}
            vlm_ids = {identity_file_id(x) for x in exposed}
            coverage_hits += len(human & covered_ids)
            vlm_hits += len(human & vlm_ids)
            if plan["vlm_required"]:
                vlm_groups += 1
                exposed_count += len(exposed)
                parts += len(exposed)
                pixels += sum(assets[x]["size"][0] * assets[x]["size"][1] for x in exposed)
                # Development teacher replay is only a capacity/coverage simulation.
                predicted = teacher_selected[group] & vlm_ids
            else:
                local_groups += 1
                predicted = {identity_file_id(x) for x in local}
            t, miss, extra = score(predicted, human)
            total_tp += t
            total_missing += miss
            total_extra += extra
            exact += predicted == human
            per_group.append({
                "group": group,
                "profile": profile,
                "member_count": len(frames),
                "route": plan["route"],
                "local_resolved": sorted(local),
                "vlm_required": plan["vlm_required"],
                "vlm_exposed": sorted(exposed),
                "micro_clusters": plan["micro_clusters"],
                "human_coverage": len(human & covered_ids) / len(human),
                "teacher_replay_selected": sorted(predicted),
                "teacher_replay_missing": sorted(human - predicted),
                "teacher_replay_extra": sorted(predicted - human),
            })
        grouped_per_1000 = 1000 * GROUPED_FRACTION_ASSUMPTION
        scale = grouped_per_1000 / total_members
        profile_results.append({
            "profile": profile,
            "development_members": total_members,
            "local_resolved_groups": local_groups,
            "vlm_groups": vlm_groups,
            "unique_source_photos_exposed": exposed_count,
            "member_to_representative_ratio": total_members / max(exposed_count, 1),
            "high_resolution_parts": parts,
            "pixels": pixels,
            "human_selected_decision_coverage_recall": coverage_hits / human_total,
            "human_selected_vlm_exposure_recall": vlm_hits / human_total,
            "teacher_replay_quality": metrics(total_tp, total_missing, total_extra, exact, len(groups)),
            "production_estimate_per_1000_input_photos": {
                "assumed_fraction_in_large_groups": GROUPED_FRACTION_ASSUMPTION,
                "large_group_member_photos": grouped_per_1000,
                "vlm_groups": vlm_groups * scale,
                "unique_source_photos_exposed": exposed_count * scale,
                "high_resolution_parts": parts * scale,
                "pixels": pixels * scale,
                "scope_warning": "Extrapolation from 39 development groups and a documented frozen-library group-incidence assumption; not a measured production result.",
            },
        })

    after = source_hashes(groups)
    if before != after:
        raise RuntimeError("development source artifacts changed during run")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "router-plans.json").write_text(json.dumps({"router_version": router.ROUTER_VERSION, "groups": per_group}, ensure_ascii=False, indent=2) + "\n")
    experiment = read_json(TUNING / "candidate/development-comparison.json")["actual_experiment"]
    for receipt in experiment["receipts"]:
        receipt_group = Path(receipt["path"]).parent.name
        if receipt_group not in set(groups):
            raise RuntimeError("teacher receipt outside development allowlist")
    teacher_ledger = {
        "model": "global.anthropic.claude-sonnet-5-5",
        "calls": experiment["calls"],
        "input_tokens": experiment["input_tokens"],
        "output_tokens": experiment["output_tokens"],
        "photo_submissions": experiment["photo_submissions"],
        "automatic_retries": 0,
        "receipts": experiment["receipts"],
        "note": "Existing bounded development experiment; this production-funnel run made no model calls.",
    }
    (OUT / "teacher-ledger.json").write_text(json.dumps(teacher_ledger, ensure_ascii=False, indent=2) + "\n")
    report = {
        "scope": "39 development groups only",
        "holdout": {"groups_known_by_name_only": len(holdout_names), "labels_images_old_ai_or_model_calls_read": False},
        "router_version": router.ROUTER_VERSION,
        "profiles": router.ROUTER_PROFILES,
        "frontier": profile_results,
        "teacher_style": teacher_style,
        "teacher_cv_surrogate": pair_summary,
        "quality_interpretation": "Teacher replay intersects existing development v4 selections with locally exposed frames. It is not a fresh VLM run, blinded accuracy, or a final production claim.",
        "two_frame_policy": "The router can compress exposure but pair selection remains v3_unchanged. No two-frame data was read or fitted in this development run.",
        "contact_sheet_accounting": "A sheet containing N source photos counts N unique exposures and N high-resolution parts.",
        "development_teacher": {
            "model": teacher_ledger["model"],
            "calls": teacher_ledger["calls"],
            "input_tokens": teacher_ledger["input_tokens"],
            "output_tokens": teacher_ledger["output_tokens"],
            "photo_submissions": teacher_ledger["photo_submissions"],
            "ledger": "teacher-ledger.json",
        },
        "external_calls_this_run": 0,
        "automatic_retries": 0,
    }
    (OUT / "frontier.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    integrity = {
        "development_source_files": len(before),
        "before_manifest_sha256": hashlib.sha256(tp.canonical(before)).hexdigest(),
        "after_manifest_sha256": hashlib.sha256(tp.canonical(after)).hexdigest(),
        "unchanged": before == after,
        "writes": {"original_photos": 0, "database": 0, "v3_artifacts": 0, "decision": 0, "keeper": 0, "delete": 0},
        "output_root": str(OUT.relative_to(ROOT)),
    }
    (OUT / "source-integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    lock_paths = [
        ROOT / "src/photo_exposure.py",
        ROOT / "scripts/run_production_funnel.py",
        ROOT / "src/large_group_review.py",
        ROOT / "src/large_group_projection.py",
        ROOT / "schemas/large-group-judge-v4-candidate-2.schema.json",
        OUT / "frontier.json",
        OUT / "router-plans.json",
        OUT / "teacher-ledger.json",
        OUT / "source-integrity.json",
    ]
    lock = {
        "status": "development_candidate_locked_for_parent_holdout",
        "router_version": router.ROUTER_VERSION,
        "files": {str(path.relative_to(ROOT)): tp.sha(path.read_bytes()) for path in lock_paths},
        "holdout_touched": False,
        "live_runner_enabled": False,
    }
    (OUT / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"profiles": len(profile_results), "groups": len(groups), "source_unchanged": True}))


if __name__ == "__main__":
    main()
