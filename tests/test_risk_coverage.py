import copy

import pytest

from src.risk_coverage import (
    DIAGNOSTIC_SAMPLE,
    REVIEW_PRIMARY,
    SAFE_SILENT,
    add_deciles,
    assign_primary_budget,
    frontier,
    one_sided_wilson_upper,
    risk_score,
    stratified_audit_sample,
)


def evidence(**changes):
    value = {
        "group_type": "phash_near", "member_count": 3, "time_span_seconds": 2,
        "min_pair_similarity": 0.99, "min_adjacent_similarity": 0.99,
        "face_count_states": 1, "position_conflict": False,
        "quality_margin": 0.2, "critical_missing_count": 0,
        "v2_v3_disagreement": False, "keeper_count": 1,
        "logical_phase_count": 1, "changed": False,
        "baseline_keeper_ids": [1], "candidate_keeper_ids": [1],
    }
    value.update(changes)
    return value


def row(group_id, risk, changed=False, dataset="d"):
    return {
        "dataset": dataset, "group_id": group_id, "group_type": "burst",
        "member_count": 2 + group_id % 5, "member_ids": [group_id * 10, group_id * 10 + 1],
        "candidate_keeper_ids": [group_id * 10], "changed": changed,
        "risk_score": risk, "expected_review_value": risk * 2,
        "risk_reason": "test risk", "reason_codes": ["LOCAL_CHANGE_SIGNAL"],
    }


def test_risk_is_deterministic_explainable_and_monotone():
    low = risk_score(evidence())
    high = risk_score(evidence(
        min_adjacent_similarity=0.8, quality_margin=0.01,
        critical_missing_count=2, changed=True,
    ))
    assert low == risk_score(evidence())
    assert high["risk_score"] >= low["risk_score"]
    assert "LOCAL_CHANGE_SIGNAL" in high["reason_codes"]
    assert "low quality margin" in high["risk_reason"]


def test_primary_budget_is_hard_capped_and_frontier_deterministic():
    rows = [row(index, index / 20) for index in range(1, 21)]
    ranked = assign_primary_budget(rows, 0.10)
    assert sum(item["selective_output"] == REVIEW_PRIMARY for item in rows) == 2
    assert sum(item["selective_output"] == SAFE_SILENT for item in rows) == 18
    assert ranked[0]["risk_score"] > ranked[-1]["risk_score"]
    curve = frontier(rows, (0.03, 0.05, 0.10, 0.20), {("d", 20)})
    assert [item["review_groups"] for item in curve] == [0, 1, 2, 4]
    assert curve[-1]["error_capture_recall"] == 1.0
    assert curve[-1]["label_scope"] == "human_tune_subset"


def test_stratified_audit_is_deterministic_anonymous_and_has_controls():
    rows = [row(index, index / 30, changed=index % 4 == 0) for index in range(1, 31)]
    add_deciles(rows)
    assign_primary_budget(rows, 0.10)
    first = stratified_audit_sample(copy.deepcopy(rows), 12)
    second = stratified_audit_sample(copy.deepcopy(rows), 12)
    assert first == second
    assert first["sample_count"] == 12
    assert any(not item["changed"] for item in first["groups"])
    assert all(item["label_status"] == "unlabelled" for item in first["groups"])
    assert all(item["inclusion_probability"] > 0 for item in first["groups"])
    assert all("thumbnail_paths" not in item for item in first["groups"])
    assert len({item["risk_decile"] for item in first["groups"]}) >= 4


def test_silent_error_bound_requires_actual_audit_and_zero_error_exact_bound():
    assert one_sided_wilson_upper(0, 0) is None
    assert one_sided_wilson_upper(0, 59) == pytest.approx(0.0495, abs=0.001)
    assert one_sided_wilson_upper(2, 30) > 2 / 30
    with pytest.raises(ValueError):
        one_sided_wilson_upper(2, 1)


def test_output_vocabulary_is_exact():
    assert {SAFE_SILENT, REVIEW_PRIMARY, DIAGNOSTIC_SAMPLE} == {
        "SAFE_SILENT", "REVIEW_PRIMARY", "DIAGNOSTIC_SAMPLE"
    }
