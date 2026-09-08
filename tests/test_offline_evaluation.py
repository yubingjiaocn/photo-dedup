import pytest

from src.dataset_contract import DatasetManifest, Split
from src.offline_evaluation import compare, evaluate_predictions, validate_annotation

ANNOTATIONS = [{
    "schema_version": 1,
    "event_id": "synthetic-event",
    "group_id": "g1",
    "member_ids": ["a", "b", "c"],
    "phase_ids": {"p1": ["a", "b"], "p2": ["c"]},
    "acceptable_keepers": {"p1": ["a"], "p2": ["c"]},
    "group_impure": True,
    "phase_undersegmented": True,
    "risk_member_ids": ["c"],
}]

BASELINE = {"g1": {
    "keepers": ["a"], "phases": {"one": ["a", "b", "c"]},
    "review_member_ids": [],
}}
CANDIDATE = {"g1": {
    "keepers": ["a", "c"], "phases": {"p1": ["a", "b"], "p2": ["c"]},
    "review_member_ids": ["a", "b", "c"],
}}


def test_metrics_cover_promised_risk_retention_impurity_and_undersegmentation():
    baseline = evaluate_predictions(ANNOTATIONS, BASELINE)
    candidate = evaluate_predictions(ANNOTATIONS, CANDIDATE)
    assert baseline["phase_recall"] == 0.5
    assert baseline["phase_undersegmentation_count"] == 1
    assert candidate["phase_recall"] == 1.0
    assert candidate["retention"] == pytest.approx(2 / 3)
    assert candidate["risk_review_count"] == 3
    assert candidate["risk_member_recall"] == 1.0
    assert candidate["group_impurity_review_recall"] == 1.0
    manifest = DatasetManifest("held-fixture", Split.HELD_OUT, "synthetic", ("synthetic-event",))
    report = compare(manifest, ANNOTATIONS, BASELINE, CANDIDATE)
    assert report["delta"]["phase_recall"] == 0.5
    assert report["writeback"] == "forbidden"


@pytest.mark.parametrize("field,value,match", [
    ("member_ids", ["a", "a", "c"], "duplicate"),
    ("phase_ids", {"p1": ["a", "b"], "p2": ["b", "c"]}, "overlap"),
    ("phase_ids", {"p1": ["a", "b"]}, "cover"),
    ("acceptable_keepers", {"p1": ["z"], "p2": ["c"]}, "outside"),
    ("risk_member_ids", ["z"], "within"),
])
def test_annotation_rejects_invalid_ids_partitions_and_bounds(field, value, match):
    broken = {**ANNOTATIONS[0], field: value}
    with pytest.raises(ValueError, match=match):
        validate_annotation(broken)


def test_annotation_event_must_belong_to_manifest_scope():
    with pytest.raises(ValueError, match="outside the manifest"):
        evaluate_predictions(ANNOTATIONS, BASELINE, expected_events={"other-event"})


def test_prediction_rejects_keeper_outside_group_and_extra_group():
    broken = {"g1": {**BASELINE["g1"], "keepers": ["z"]}}
    with pytest.raises(ValueError, match="outside group"):
        evaluate_predictions(ANNOTATIONS, broken)
    with pytest.raises(ValueError, match="unannotated"):
        evaluate_predictions(ANNOTATIONS, {**BASELINE, "g2": BASELINE["g1"]})


def test_compare_refuses_non_held_out_final_report():
    manifest = DatasetManifest("dev", Split.TUNE, "synthetic", ("synthetic-event",))
    with pytest.raises(PermissionError, match="held_out"):
        compare(manifest, ANNOTATIONS, BASELINE, CANDIDATE)
