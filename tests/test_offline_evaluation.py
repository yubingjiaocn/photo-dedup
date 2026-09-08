import pytest

from src.dataset_contract import DatasetManifest, Split
from src.offline_evaluation import compare, evaluate_predictions

ANNOTATIONS = [{
    "schema_version": 1,
    "event_id": "synthetic-event",
    "group_id": "g1",
    "phase_ids": {"p1": ["a", "b"], "p2": ["c"]},
    "acceptable_keepers": {"p1": ["a"], "p2": ["c"]},
}]


def test_metrics_compare_baseline_and_candidate():
    baseline = evaluate_predictions(ANNOTATIONS, {"g1": ["a"]})
    candidate = evaluate_predictions(ANNOTATIONS, {"g1": ["a", "c"]})
    assert baseline["phase_recall"] == 0.5
    assert candidate["phase_recall"] == 1.0
    manifest = DatasetManifest("held-fixture", Split.HELD_OUT, "synthetic", ("synthetic-event",))
    report = compare(manifest, ANNOTATIONS, {"g1": ["a"]}, {"g1": ["a", "c"]})
    assert report["delta"]["phase_recall"] == 0.5
    assert report["writeback"] == "forbidden"
    assert report["contract"]["delete_trash_authority"] == "none"


def test_compare_refuses_non_held_out_final_report():
    manifest = DatasetManifest("dev", Split.TUNE, "synthetic", ("synthetic-event",))
    with pytest.raises(PermissionError, match="held_out"):
        compare(manifest, ANNOTATIONS, {}, {})
