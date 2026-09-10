import copy
import json
import numpy as np
import pytest
from src import phase_selection as ps, region_set_quality as rs
from src.local_quality import observations
from tests.test_phase_selection import member
from tests.test_local_quality import observed


def frame(
    fid, qualities=((60, 0.4), (70, 0.5)), classes=(0, 0), reverse=False, faces=True
):
    m = member(fid, fid, [1, 0], q=80)
    full = np.zeros(768, dtype="<f2")
    full[-1] = 1
    m["dinov2_embedding"] = full.tobytes()
    entries = []
    blob = bytearray()
    indices = list(range(len(qualities)))
    if reverse:
        indices.reverse()
    for j, i in enumerate(indices):
        v = np.zeros(768, dtype="<f2")
        v[i] = 1
        x = 0.05 + i * 0.5
        entries.append(
            {
                "region_id": f"S{j}",
                "parent_id": None,
                "kind": "subject",
                "class_id": classes[i],
                "confidence": 0.99,
                "box_normalized": [x, 0.1, x + 0.4, 0.9],
                "native_size": [400, 800],
                "musiq": qualities[i][0],
                "clipiqa": qualities[i][1],
                "embedding_offset": len(blob),
            }
        )
        blob.extend(v.tobytes())
    meta = json.loads(m["quality_meta"])
    meta["clipiqa"] = 0.5
    meta["local_region_set"] = {
        "schema_version": 2,
        "status": "complete",
        "subject_count": len(entries),
        "regions": entries,
    }
    m["quality_meta"] = json.dumps(meta)
    m["local_quality_embedding"] = bytes(blob)
    if faces:
        for r in entries:
            if r["class_id"] == 0:
                m = with_face(m, r["region_id"], (r["musiq"], r["clipiqa"]))
    return m


def phases(*indices):
    return [ps.Phase(0, tuple(indices), (), False, ())]


def test_complete_set_association_is_symmetric_under_permutation():
    a = rs.decode(frame(1))
    b = rs.decode(frame(2, reverse=True))
    forward, _ = rs.match(a, b)
    reverse, _ = rs.match(b, a)
    assert {(x["region_id"], y["region_id"]) for x, y in forward} == {
        (y["region_id"], x["region_id"]) for x, y in reverse
    }
    assert sum(x["kind"] == "subject" for x, _ in forward) == 2


@pytest.mark.parametrize("classes", [(0, 0), (16, 16), (0, 16)])
def test_multi_people_pets_and_mixed_sets_are_supported(classes):
    ms = [frame(1, classes=classes), frame(2, ((70, 0.6), (80, 0.7)), classes=classes)]
    out = ps.select_phase_keepers(
        ms,
        phases(0, 1),
        local_quality_policy="region_set",
        local_quality_similarity=0.97,
    )
    assert out["keepers"] == [1]
    assert out["local_quality_context"]["selection_applied"]
    assert out["local_quality_context"]["comparisons"][0]["subject_count"] == 2
    assert out["mandatory_review"] and not out["physical_group_split"]


def test_improving_best_actor_cannot_hide_worse_other_actor():
    ms = [frame(1, ((60, 0.4), (90, 0.9))), frame(2, ((80, 0.7), (89, 0.89)))]
    scores, ctx = rs.adjust_scores(ms, phases(0, 1), {0: 0.8, 1: 0.8})
    assert scores == {0: 0.8, 1: 0.8}
    assert ctx["refusals"]["ANY_REGION_WORSENED"] > 0


def test_best_actor_only_gain_does_not_improve_worst_actor():
    ms = [frame(1), frame(2, ((60, 0.4), (90, 0.9)))]
    scores, _ = rs.adjust_scores(ms, phases(0, 1), {0: 0.8, 1: 0.8})
    assert scores == {0: 0.8, 1: 0.8}


def test_missing_actor_or_changed_class_abstains():
    a = rs.decode(frame(1))
    b = rs.decode(frame(2, ((80, 0.7),), classes=(0,)))
    assert rs.match(a, b)[0] is None
    assert rs.match(a, rs.decode(frame(2, classes=(0, 16))))[0] is None


def test_identical_ambiguous_instances_are_not_assigned_arbitrarily():
    a = rs.decode(frame(1))
    b = copy.deepcopy(a)
    for value in [a, b]:
        value["S1"]["vector"] = value["S0"]["vector"].copy()
        value["S1"]["box_normalized"] = value["S0"]["box_normalized"][:]
    assert rs.match(a, b)[0] is None
    assert rs.match(b, a)[0] is None


def test_interaction_context_change_blocks_local_improvement():
    ms = [frame(1), frame(2, ((70, 0.6), (80, 0.7)))]
    v = np.zeros(768, dtype="<f2")
    v[-2] = 1
    ms[1]["dinov2_embedding"] = v.tobytes()
    scores, ctx = rs.adjust_scores(ms, phases(0, 1), {0: 0.8, 1: 0.8})
    assert scores == {0: 0.8, 1: 0.8}
    assert ctx["refusals"]["CONTEXT_OR_INTERACTION_UNCERTAIN"] > 0


def test_whole_frame_quality_is_not_sacrificed():
    ms = [frame(1), frame(2, ((70, 0.6), (80, 0.7)))]
    ms[1]["quality_score"] = 50
    scores, _ = rs.adjust_scores(ms, phases(0, 1), {0: 0.8, 1: 0.8})
    assert scores == {0: 0.8, 1: 0.8}


@pytest.mark.parametrize("status", ["no_supported_regions", "incomplete", "unknown"])
def test_no_subject_or_uncertain_set_preserves_exact_baseline(status):
    ms = [frame(1), frame(2)]
    for m in ms:
        meta = json.loads(m["quality_meta"])
        meta["local_region_set"]["status"] = status
        m["quality_meta"] = json.dumps(meta)
    base = ps.select_phase_keepers(ms, phases(0, 1))
    result = ps.select_phase_keepers(
        ms,
        phases(0, 1),
        local_quality_policy="region_set",
        local_quality_similarity=0.97,
    )
    assert {k: v for k, v in result.items() if k != "local_quality_context"} == base
    assert not result["local_quality_context"]["selection_applied"]


def test_regime_labels_cannot_change_runtime():
    ms = [frame(1), frame(2, ((70, 0.6), (80, 0.7)))]
    a = ps.select_phase_keepers(
        ms,
        phases(0, 1),
        local_quality_policy="region_set",
        local_quality_similarity=0.97,
    )
    for m in ms:
        m["evaluation_regime"] = "no_subject"
        m["critical_subjects"] = ["FAKE"]
    b = ps.select_phase_keepers(
        ms,
        phases(0, 1),
        local_quality_policy="region_set",
        local_quality_similarity=0.97,
    )
    assert a == b


def test_baseline_phase_and_retained_witness_guards():
    ctx = {"changed_members": [0], "comparisons": [{"worse": 0, "better": 1}]}
    base = {"keepers": [0]}
    result = rs.protect_selection(
        {"keepers": [1]},
        base,
        [ps.Phase(0, (0,), (), False, ()), ps.Phase(1, (1,), (), False, ())],
        ctx,
    )
    assert result["keepers"] == [0]
    result = rs.protect_selection({"keepers": [2]}, base, phases(0, 1, 2), ctx)
    assert result["keepers"] == [0]


def test_legacy_single_policy_rejects_unverified_or_multi_catalog():
    m = observed(1, 60, 0.4)
    meta = json.loads(m["quality_meta"])
    meta["local_quality"].pop("catalog_complete_single")
    m["quality_meta"] = json.dumps(meta)
    assert observations(m) == {}
    meta["local_quality"]["catalog_complete_single"] = True
    meta["local_quality"]["detected_subject_count"] = 2
    m["quality_meta"] = json.dumps(meta)
    assert observations(m) == {}


def test_invalid_vectors_and_catalog_fail_closed():
    m = frame(1)
    m["local_quality_embedding"] = b"bad"
    assert rs.decode(m) is None
    assert rs.match({}, {})[0] is None


def with_face(m, parent, quality):
    m = copy.deepcopy(m)
    meta = json.loads(m["quality_meta"])
    regions = meta["local_region_set"]["regions"]
    existing = next((r for r in regions if r.get("parent_id") == parent), None)
    if existing is not None:
        existing["musiq"], existing["clipiqa"] = quality
        m["quality_meta"] = json.dumps(meta)
        return m
    body = next(r for r in regions if r["region_id"] == parent)
    v = np.zeros(768, dtype="<f2")
    v[10 + len(regions)] = 1
    x = body["box_normalized"][0]
    regions.append(
        {
            "region_id": "F" + parent,
            "parent_id": parent,
            "kind": "face",
            "class_id": 0,
            "confidence": 0.99,
            "box_normalized": [x + 0.1, 0.2, x + 0.25, 0.4],
            "native_size": [150, 200],
            "musiq": quality[0],
            "clipiqa": quality[1],
            "embedding_offset": len(m["local_quality_embedding"]),
        }
    )
    m["local_quality_embedding"] += v.tobytes()
    m["quality_meta"] = json.dumps(meta)
    return m


def test_face_coverage_cannot_disappear_behind_good_body_scores():
    a = with_face(frame(1), "S0", (60, 0.4))
    b = frame(2, ((70, 0.6), (80, 0.7)), faces=False)
    assert rs.match(rs.decode(a), rs.decode(b))[1] == "FACE_COVERAGE_UNSTABLE"


def test_one_worse_face_blocks_group_improvement():
    a = with_face(frame(1), "S1", (90, 0.9))
    b = with_face(frame(2, ((70, 0.6), (80, 0.7))), "S1", (89, 0.89))
    scores, ctx = rs.adjust_scores([a, b], phases(0, 1), {0: 0.8, 1: 0.8})
    assert scores == {0: 0.8, 1: 0.8}
    assert ctx["refusals"]["ANY_REGION_WORSENED"] > 0


def test_position_alone_cannot_identify_indistinguishable_actors():
    a = rs.decode(frame(1))
    b = rs.decode(frame(2))
    for s in (a, b):
        s["S1"]["vector"] = s["S0"]["vector"].copy()
    assert rs.match(a, b)[1] == "IDENTITY_NOT_UNIQUE_WITHOUT_POSITION"
    assert rs.match(b, a)[1] == "IDENTITY_NOT_UNIQUE_WITHOUT_POSITION"
