import json
from src import actor_set_pose as ap, phase_selection as ps
from tests.test_region_set_quality import frame, phases
from tests.test_pose_evidence import body


def example(fid, moving=False):
    m = frame(fid)
    meta = json.loads(m["quality_meta"])
    people = []
    for region in meta["local_region_set"]["regions"]:
        if region["kind"] != "subject":
            continue
        x1, y1, x2, y2 = region["box_normalized"]
        person = body()
        if moving and region["region_id"] == "S0":
            for i in [7, 8, 9, 10]:
                person["keypoints"][i][1] -= 0.18
        person["keypoints"] = [
            [(x - 0.1) / 0.85 * (x2 - x1) + x1, (y - 0.1) / 0.85 * (y2 - y1) + y1, c]
            for x, y, c in person["keypoints"]
        ]
        person["box"] = region["box_normalized"][:]
        people.append(person)
    meta["pose_evidence"] = {
        "producer": "local-coco17-model",
        "status": "ok",
        "shape": [1000, 1000],
        "people": people,
    }
    m["quality_meta"] = json.dumps(meta)
    return m


def test_actor_pose_is_add_only_and_not_largest_subject_selection():
    ms = [example(1), example(2, True)]
    keep, e = ap.protect_actor_variants(ms, [0], {0: 0.8, 1: 0.8})
    assert keep == [0, 1]
    assert e["added_keeper"] == 1 and not e["largest_subject_priority"]
    assert e["existing_keepers_preserved"] and not e["semantic_phase_authority"]


def test_stable_actor_runtime_preserves_baseline_and_marks_review():
    ms = [example(1), example(2, True)]
    baseline = ps.select_phase_keepers(ms, phases(0, 1))
    out = ps.select_phase_keepers(
        ms,
        phases(0, 1),
        pose_policy="stable_actor",
        local_quality_policy="region_set",
        local_quality_similarity=0.97,
    )
    assert set(baseline["keepers"]) < set(out["keepers"])
    assert out["mandatory_review"] and out["group_keeper_budget"] == 2


def test_missing_subject_or_ambiguous_identity_abstains():
    a, b = example(1), example(2, True)
    meta = json.loads(b["quality_meta"])
    meta["local_region_set"]["status"] = "incomplete"
    b["quality_meta"] = json.dumps(meta)
    assert ap.protect_actor_variants([a, b], [0], {0: 0.8, 1: 0.8})[0] == [0]


def test_quality_drop_of_any_region_blocks_extra():
    a, b = example(1), example(2, True)
    meta = json.loads(b["quality_meta"])
    meta["local_region_set"]["regions"][1]["musiq"] = 20
    b["quality_meta"] = json.dumps(meta)
    assert ap.protect_actor_variants([a, b], [0], {0: 0.8, 1: 0.8})[0] == [0]


def test_missing_or_malformed_pose_is_not_motion():
    a, b = example(1), example(2, True)
    meta = json.loads(b["quality_meta"])
    meta.pop("pose_evidence")
    b["quality_meta"] = json.dumps(meta)
    assert ap.protect_actor_variants([a, b], [0], {0: 0.8, 1: 0.8})[0] == [0]
    rec = json.loads(a["quality_meta"])["pose_evidence"]
    rec["people"][0]["keypoints"][0] = ["bad"]
    region = json.loads(a["quality_meta"])["local_region_set"]["regions"][0]
    assert ap._assigned_pose(region, {region["region_id"]: region}, rec) is None


def test_pose_must_contrast_with_every_existing_keeper():
    ms = [example(1), example(2, True), example(3, True)]
    assert ap.protect_actor_variants(ms, [0, 1], {0: 0.8, 1: 0.8, 2: 0.8})[0] == [0, 1]


def test_no_subject_and_pets_only_never_invent_human_pose():
    ms = [frame(1, classes=(16, 16)), frame(2, classes=(16, 16))]
    assert ap.protect_actor_variants(ms, [0], {0: 0.8, 1: 0.8})[0] == [0]
    for m in ms:
        m["quality_meta"] = "{}"
    assert ap.protect_actor_variants(ms, [0], {0: 0.8, 1: 0.8})[0] == [0]


def test_dispatch_depends_on_native_evidence_not_evaluation_labels():
    from src.region_set_quality import decode, evidence_dispatch

    ms = [example(1), example(2)]
    assert evidence_dispatch([decode(m) for m in ms])["route"] == "multi_set"
    one = frame(3, qualities=((60, 0.4),), classes=(0,))
    assert evidence_dispatch([decode(one)])["route"] == "single_set"
    assert evidence_dispatch([None])["route"] == "global_baseline"
    ms[0]["evaluation_regime"] = "no_subject"
    assert evidence_dispatch([decode(m) for m in ms])["route"] == "multi_set"
