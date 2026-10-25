import copy
import sqlite3

import numpy as np
import pytest
from PIL import Image

from scripts import run_pair_router_development as runner
from src import pair_exposure_resolver as resolver
from src import photo_exposure


def blob(index=0):
    vector = np.zeros(768, dtype="<f2")
    vector[index] = 1
    return vector.tobytes()


def frames():
    return [{"identity": f"I{i}@" + c * 64, "time": i,
             "descriptor": photo_exposure.descriptor(Image.new("RGB", (64, 64), (100, 100, 100)))}
            for i, c in ((1, "a"), (2, "b"))]


def evidence(pair):
    first, second = [f["identity"] for f in pair]
    result = {"raw_dino_blobs": {first: blob(), second: blob()}}
    for name, (criterion, direction) in resolver.CRITERIA.items():
        if name != "gradient":
            result[criterion] = {first: 2 if direction == 1 else 1,
                                 second: 1 if direction == 1 else 2}
    return result


def test_select_one_identity_and_reverse_symmetry():
    pair = frames()
    ev = evidence(pair)
    a = resolver.resolve(pair, ev)
    b = resolver.resolve(pair[::-1], ev)
    assert a["route"] == b["route"] == "local_select_one"
    assert a["local_selected"] == b["local_selected"] == [{"file_id": 1, "content_sha256": "a" * 64}]
    assert a["vlm_exposed"] == []
    assert a["decision_evidence"]["available_votes"] == 7


@pytest.mark.parametrize("bad", [None, b"", bytes(1534), bytes(1538), bytes(3072), bytes(1536),
                                  np.full(768, np.nan, dtype="<f2").tobytes(),
                                  np.full(768, np.inf, dtype="<f2").tobytes()])
def test_strict_dino_decoder_rejects_bad_contract(bad):
    with pytest.raises(ValueError):
        resolver.decode_dino(bad)
    pair = frames()
    ev = evidence(pair)
    ev["raw_dino_blobs"][pair[0]["identity"]] = bad
    assert resolver.resolve(pair, ev)["route"] == "vlm_required"


def test_little_endian_and_exact_raw_distance():
    v = np.zeros(768, dtype="<f2")
    v[:2] = [0.6, 0.8]
    decoded = resolver.decode_dino(v.tobytes())
    expected = v.astype(np.float64)
    expected /= np.linalg.norm(expected)
    np.testing.assert_array_equal(decoded, expected)
    assert resolver.raw_dino_distance(blob(), v.tobytes()) == pytest.approx(1 - expected[0])
    # No endian marker exists in raw bytes; contract is always interpreted <f2.
    assert not np.allclose(resolver.decode_dino(v.astype(">f2").tobytes()), expected)
    unnormalized = np.ones(768, dtype="<f2").tobytes()
    assert np.linalg.norm(resolver.decode_dino(unnormalized)) == pytest.approx(1)


def test_raw_dino_barrier_cannot_use_rounded_or_pixel_similarity():
    pair = frames()
    ev = evidence(pair)
    ev["raw_dino_blobs"][pair[1]["identity"]] = blob(1)
    ev["pair_measurements"] = {"dino_cosine_distance": 0}
    assert resolver.resolve(pair, ev)["route"] == "vlm_required"
    del ev["raw_dino_blobs"]
    assert resolver.resolve(pair, ev)["route"] == "vlm_required"


def test_same_source_aliases_cannot_multiply_votes():
    pair = frames()
    ev = evidence(pair)
    # Only four provenance families remain, despite 11 directional scalars.
    for name in ("face_quality", "edge_margin", "near_edge_faces", "eye_open"):
        ev.pop(resolver.CRITERIA[name][0])
    ev["quality_score"] = {pair[0]["identity"]: 100, pair[1]["identity"]: 0}
    result = resolver.resolve(pair, ev)
    assert result["decision_evidence"]["available_votes"] == 4
    assert result["route"] == "vlm_required"
    del ev["quality_score"]
    assert resolver.resolve(pair, ev) == result


def test_within_family_conflict_abstains_and_missing_is_not_a_vote():
    pair = frames()
    ev = evidence(pair)
    ev[resolver.CRITERIA["local_musiq"][0]] = dict(zip([f["identity"] for f in pair], [1, 2]))
    result = resolver.resolve(pair, ev)
    assert result["feature_audit"]["family_votes"]["musiq"] == 0
    assert result["decision_evidence"]["available_votes"] == 6
    ev.pop(resolver.CRITERIA["face_quality"][0])
    assert resolver.resolve(pair, ev)["route"] == "vlm_required"


def test_available_votes_and_margin_not_winner_vote_count():
    pair = frames()
    ev = evidence(pair)
    ev.pop(resolver.CRITERIA["eye_open"][0])
    criterion = resolver.CRITERIA["face_quality"][0]
    ev[criterion] = dict(zip([f["identity"] for f in pair], [1, 2]))
    result = resolver.resolve(pair, ev)
    assert result["decision_evidence"]["first_votes"] == 5
    assert result["decision_evidence"]["available_votes"] == 6
    assert result["decision_evidence"]["winner_margin"] == 4
    assert result["route"] == "local_select_one"


@pytest.mark.parametrize("mutation", ["time", "identity", "pixel_nan", "aspect_nan", "evidence_nan", "duplicate", "far_rgb", "dhash"])
def test_invalid_unknown_or_non_near_fails_closed(mutation):
    pair = frames()
    ev = evidence(pair)
    if mutation == "time":
        pair[0]["time"] = None
    elif mutation == "identity":
        pair[0]["identity"] = "bad"
    elif mutation == "pixel_nan":
        pair[0]["descriptor"]["rgb"][0, 0, 0] = np.nan
    elif mutation == "aspect_nan":
        pair[0]["descriptor"]["aspect"] = np.nan
    elif mutation == "evidence_nan":
        ev[resolver.CRITERIA["musiq"][0]][pair[0]["identity"]] = float("nan")
    elif mutation == "duplicate":
        pair[1] = copy.deepcopy(pair[0])
    elif mutation == "far_rgb":
        pair[0]["descriptor"]["rgb"][:] = 1
    else:
        pair[0]["descriptor"]["dhash"][:] = True
    result = resolver.resolve(pair, ev)
    assert result["route"] == "vlm_required"
    assert result["local_selected"] == []


def test_no_local_keep_both_or_label_dependency():
    pair = frames()
    ev = evidence(pair)
    expected = resolver.resolve(pair, ev)
    ev.update(group_id="ignored", human_selected=[1, 2], keeper=2)
    assert resolver.resolve(pair, ev) == expected
    assert resolver.resolve(pair, ev, "unknown")["route"] == "vlm_required"
    for profile in resolver.PROFILES:
        assert resolver.resolve(pair, ev, profile)["route"] in {"vlm_required", "local_select_one"}


def test_profile_A_exact_thresholds():
    assert resolver.PROFILES["A"] == {"max_raw_dino_distance": .045, "max_rgb_mae": .09,
                                     "max_dhash": 8, "min_available_votes": 6, "min_vote_margin": 4}


def test_read_only_database_selects_only_allowlisted_ids(tmp_path):
    path = tmp_path / "inventory.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE files (id INTEGER, path TEXT)")
        conn.execute("CREATE TABLE features (file_id INTEGER, content_sha256 TEXT, dinov2_embedding BLOB)")
        conn.executemany("INSERT INTO files VALUES (?, ?)", [(1, "a"), (2, "holdout-do-not-read")])
        conn.executemany("INSERT INTO features VALUES (?, ?, ?)", [(1, "a" * 64, blob()), (2, "b" * 64, b"invalid holdout")])
    before = runner.digest(path)
    records = runner.fetch_development_rows(path, {1: "a" * 64})
    assert len(records) == 1 and records[0][0] == 1
    assert runner.digest(path) == before
    with pytest.raises(RuntimeError, match="SHA"):
        runner.fetch_development_rows(path, {1: "b" * 64})
    with pytest.raises(RuntimeError, match="missing"):
        runner.fetch_development_rows(path, {3: "a" * 64})
    assert runner.digest(path) == before


def test_zero_events_without_trials_has_no_precision_claim():
    assert runner.zero_error_bound(0) is None
    assert runner.zero_error_bound(15) == pytest.approx(.1810362725)
    assert runner.aggregate([])["exact_rate"] is None
