"""Offline development B-route preflight. No VLM calls; labels score after plans.

Policies are fixed in label-free src.photo_exposure. A failed exposure-recall
preflight rejects B before sending any image, rather than tuning around labels.
"""

from __future__ import annotations

from PIL import Image

from scripts.prepare_large_group_candidate import OUT, SOURCE, TUNING
from scripts.run_large_group_development import allowed_groups
from src import cloud_judge_batch as b
from src import photo_exposure as cv
from src import two_pass_review as tp


def main():
    plans = {name: [] for name in cv.POLICIES}
    for group in allowed_groups():
        case = SOURCE / group
        request = b.read(case / "request.json")
        times = {
            tp.identity(x["frame"]): x["capture_time"]
            for x in request["context"]["temporal"]
        }
        frames = []
        assets = {}
        for asset in request["images"]:
            if asset["kind"] != "full_group_page":
                continue
            b.verify(
                len(asset["cells"]) == 1, "one complete source per local descriptor"
            )
            identity = asset["cells"][0]["frame"]
            path = (case / asset["path"]).resolve()
            b.verify(
                path.is_relative_to(case.resolve())
                and tp.sha(path.read_bytes()) == asset["sha256"],
                "asset binding",
            )
            with Image.open(path) as image:
                desc = cv.descriptor(image)
            frames.append(
                {"identity": identity, "time": times[identity], "descriptor": desc}
            )
            assets[identity] = asset
        b.verify(len(frames) == len(times), "local descriptor coverage")
        for name, policy in cv.POLICIES.items():
            clusters = cv.micro_clusters(frames, policy)
            exposed = [c["representative"] for c in clusters]
            plans[name].append(
                {
                    "group": group,
                    "members": len(frames),
                    "exposed": exposed,
                    "clusters": clusters,
                    "high_resolution_candidates": len(exposed),
                    "image_parts": len(exposed),
                    "pixels": sum(
                        assets[f]["size"][0] * assets[f]["size"][1] for f in exposed
                    ),
                    "packet_policy": "Only full representative pages; no contact sheet/storyboard/crop containing omitted sources.",
                }
            )
    b.save(
        OUT / "exposure-b-plans.json",
        {"policies": cv.POLICIES, "plans": plans, "external_calls": 0},
    )
    # Label access occurs only AFTER every independent CV plan is frozen to disk.
    labels = b.read(TUNING / "development-labels.json")["groups"]
    results = []
    for name, groups in plans.items():
        missing = []
        human_total = 0
        for group in groups:
            human = set(labels[group["group"]]["human_selected"])
            exposed_ids = {int(f.split("@")[0][1:]) for f in group["exposed"]}
            human_total += len(human)
            for file_id in sorted(human - exposed_ids):
                missing.append({"group": group["group"], "file_id": file_id})
        count = sum(len(g["exposed"]) for g in groups)
        members = sum(g["members"] for g in groups)
        results.append(
            {
                "policy": name,
                "status": "rejected_exposure_recall"
                if missing
                else "eligible_for_model_canary"
                if count < members
                else "no_exposure_saving",
                "planned_vlm_groups": len(groups),
                "actual_vlm_groups": 0,
                "members": members,
                "unique_source_photos_exposed_if_run": len(
                    {f for g in groups for f in g["exposed"]}
                ),
                "member_exposures_if_run": count,
                "saved_photos": members - count,
                "high_resolution_candidates": count,
                "image_parts": sum(g["image_parts"] for g in groups),
                "pixels": sum(g["pixels"] for g in groups),
                "human_selected_exposure_recall": (human_total - len(missing))
                / human_total,
                "human_selected_exposure_missing": missing,
                "quality_exact_precision_recall_f1": None,
                "quality_note": "No inference: exposure preflight is a necessary condition, not model quality.",
            }
        )
    b.save(
        OUT / "exposure-b-evaluation.json",
        {
            "routes": results,
            "required_exposure_recall": 1.0,
            "decision": "Any policy dropping a human-selected source is rejected; no label-based reinstatement or threshold repair.",
        },
    )
    for result in results:
        print(
            result["policy"],
            result["status"],
            result["members"],
            "->",
            result["member_exposures_if_run"],
            "human exposure recall",
            result["human_selected_exposure_recall"],
        )


if __name__ == "__main__":
    main()
