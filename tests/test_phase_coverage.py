"""Contracts use synthetic IDs only; no media or feature extraction."""
import copy

import pytest

from src.phase_coverage import diagnose_phase_coverage, keeper_budget
from src.risk_coverage import assign_primary_budget, frontier, risk_score


CODE = "PHASE_COVERAGE_BUDGET_CONFLICT"


def requirements(partition, source="shadow"):
    return {"source": source, "evidence_ref": "synthetic-contract",
            "confidence": "uncalibrated", "phases": [
                {"phase_id": str(i), "member_ids": ids}
                for i, ids in enumerate(partition)]}


def diagnose(partition, keepers, budget=None, source="shadow"):
    ids = [i for phase in partition for i in phase]
    return diagnose_phase_coverage(ids, keepers, keeper_budget(len(ids)) if budget is None else budget,
                                   requirements(partition, source))


def test_three_singletons_n_minus_one_conflict_no_authority():
    result = diagnose([[1], [2], [3]], [1, 3], source="human")
    assert result["reason_codes"] == [CODE]
    assert result["required_phase_count"] == 3
    assert result["keeper_budget"] == 2
    assert result["minimum_extra_keepers"] == 1
    assert result["uncovered_phase_ids"] == ["1"]
    assert result["mandatory_review"] is True
    assert result["policy"] == "A_BUDGET_UNCHANGED_REVIEW"
    assert result["authority"] == "shadow_review_only"
    assert result["keeper_ids"] == [1, 3]


def test_low_confidence_pseudo_splits_remain_visible_not_auto_merged():
    evidence = requirements([[1], [2], [3]])
    evidence["confidence"] = "low"
    original = copy.deepcopy(evidence)
    result = diagnose_phase_coverage([1, 2, 3], [1, 3], 2, evidence)
    assert result["budget_conflict"] is True
    assert result["evidence"] == original
    assert evidence == original
    assert result["keeper_ids"] == [1, 3]


@pytest.mark.parametrize("partition,keepers", [([[1], [2, 3]], [1, 2]),
                                              ([[1, 2], [3], [4]], [1, 3, 4]),
                                              ([[1, 2, 3]], [1, 3])])
def test_covered_multiphase_and_single_phase_unchanged(partition, keepers):
    result = diagnose(partition, keepers)
    assert not result["budget_conflict"]
    assert not result["mandatory_review"]
    assert result["reason_codes"] == []
    assert result["keeper_ids"] == keepers


def test_face_count_not_a_phase_requirement():
    result = diagnose([[1, 2, 3]], [1, 3])
    assert result["required_phase_count"] == 1
    assert not result["budget_conflict"]


def test_coverage_gap_with_sufficient_budget_not_mislabelled_conflict():
    result = diagnose([[1, 2], [3, 4]], [1, 2])
    assert result["budget_conflict"] is False
    assert result["reason_codes"] == ["PHASE_COVERAGE_GAP"]
    assert result["minimum_extra_keepers"] == 0
    assert result["mandatory_review"] is True


@pytest.mark.parametrize("partition", [[[1], [1, 2]], [[1], []], [[1], [4]], [[1]]])
def test_invalid_partitions_rejected(partition):
    with pytest.raises(ValueError):
        diagnose_phase_coverage([1, 2, 3], [1], 2, requirements(partition))


@pytest.mark.parametrize("keepers,budget", [([9], 2), ([1, 1], 2), ([1], 0),
                                             ([1], True), ([1], 4), ([1, 2, 3], 2)])
def test_invalid_keeper_budget_contract_rejected(keepers, budget):
    with pytest.raises(ValueError):
        diagnose_phase_coverage([1, 2, 3], keepers, budget, requirements([[1, 2, 3]]))


def test_diagnostic_risk_is_mandatory_even_at_zero_budget_without_reweighting():
    evidence = {"group_type": "burst", "member_count": 3, "keeper_count": 2,
                "quality_margin": .9}
    before = risk_score(evidence)
    evidence["phase_coverage_diagnostics"] = [diagnose([[1], [2], [3]], [1, 3])]
    after = risk_score(evidence)
    assert after["risk_score"] == before["risk_score"]
    assert CODE in after["reason_codes"]
    assert after["mandatory_review"] is True
    row = {"dataset": "synthetic", "group_id": 1, "member_count": 3, **after}
    assign_primary_budget([row], 0)
    assert row["selective_output"] == "REVIEW_PRIMARY"
    assert row["review_budget_overflow"] is True
    point = frontier([row], [0])[0]
    assert point["review_groups"] == 1
    assert point["review_budget_overflow_groups"] == 1


def test_mandatory_overflow_does_not_evict_existing_ranked_review():
    rows = [dict(dataset="s", group_id=i, member_count=3, risk_score=1-i*.1,
                 expected_review_value=1-i*.1, mandatory_review=(i == 2)) for i in range(3)]
    assign_primary_budget(rows, .34)
    assert [r["group_id"] for r in rows if r["selective_output"] == "REVIEW_PRIMARY"] == [0, 2]
    assert frontier(rows, [.34])[0]["review_groups"] == 2


def test_selector_diagnoses_without_changing_keeper_output():
    from src.phase_selection import Phase, select_phase_keepers
    members = [{"id": i, "quality_score": 50 + i} for i in (1, 2, 3)]
    phases = [Phase(i, (i,), ("TIME_GAP",), False) for i in range(3)]
    result = select_phase_keepers(members, phases)
    assert result["keepers"] == [2, 1]
    assert result["phase_coverage_diagnostics"][0]["budget_conflict"] is True
    assert CODE in result["reason_codes"]
    assert result["mandatory_review"] is True


def test_external_requirements_do_not_steer_selection():
    from src.phase_selection import Phase, select_phase_keepers
    members = [{"id": i, "quality_score": 50 + i} for i in (1, 2, 3)]
    phases = [Phase(0, (0, 1, 2), (), False)]
    old = select_phase_keepers(members, phases)
    new = select_phase_keepers(members, phases, phase_requirements=requirements([[1], [2], [3]], "human"))
    assert new["keepers"] == old["keepers"]
    assert new["phases"] == old["phases"]
    assert new["mandatory_review"] is True
    assert new["phase_coverage_diagnostics"][1]["budget_conflict"] is True


def frozen_row(group_id, partition, keepers):
    ids = [i for phase in partition for i in phase]
    evidence = {"member_count": len(ids), "keeper_count": len(keepers),
                "group_type": "burst", "quality_margin": .9,
                "logical_phase_count": len(partition), "candidate_keeper_ids": keepers}
    return {"dataset": "synthetic", "group_id": group_id, "member_ids": ids,
            "member_count": len(ids), "candidate_keeper_ids": keepers,
            "logical_phase_count": len(partition),
            "logical_phases": requirements(partition)["phases"],
            "risk_evidence": evidence, **risk_score(evidence),
            "review_required": False, "review_tier": "none", "selective_output": "SAFE_SILENT"}


def test_frozen_replay_external_partition_detects_without_reselection():
    from scripts.phase_coverage_shadow import replay
    rows = [frozen_row(999, [[1, 2, 3]], [1, 3])]
    old = copy.deepcopy(rows)
    shadow = replay(rows, primary_budget=0)
    assert shadow["metrics"]["conflict_groups"] == 0
    result = replay(rows, [{"dataset": "synthetic", "group_id": 999,
                           "requirements": requirements([[1], [2], [3]], "analyst_interpretation")}],
                    primary_budget=0)
    assert result["metrics"]["conflict_groups"] == 1
    assert result["metrics"]["keeper_changed_groups"] == 0
    assert result["metrics"]["retention_delta_keepers"] == 0
    assert result["metrics"]["new_primary_groups"] == 1
    assert rows == old
    assert result["groups"][0]["logical_phases"] == old[0]["logical_phases"]


def test_frozen_replay_low_confidence_does_not_merge_or_suppress():
    from scripts.phase_coverage_shadow import replay
    rows = [frozen_row(17, [[1], [2], [3]], [1, 3])]
    result = replay(rows, primary_budget=0)
    assert result["metrics"]["conflict_groups"] == 1
    assert result["metrics"]["keep_all_conflict_upper_bound_extra_keepers"] == 1
    assert result["groups"][0]["candidate_keeper_ids"] == [1, 3]


def test_empty_and_singleton_contracts():
    from src.phase_selection import select_phase_keepers
    assert select_phase_keepers([], [])["group_keeper_budget"] == 0
    result = diagnose([[7]], [7])
    assert result["keeper_budget"] == 1
    assert not result["mandatory_review"]


@pytest.mark.parametrize("field,value", [("source", "model-as-human"), ("confidence", None),
                                          ("evidence_ref", ""), ("phases", None)])
def test_invalid_evidence_metadata_is_not_silently_accepted(field, value):
    evidence = requirements([[1, 2, 3]])
    evidence[field] = value
    with pytest.raises(ValueError):
        diagnose_phase_coverage([1, 2, 3], [1], 2, evidence)


def test_frozen_exact_group_reconstructs_single_keeper_budget():
    from scripts.phase_coverage_shadow import replay
    row = frozen_row(5, [[1, 2, 3, 4]], [1])
    row["group_type"] = "sha_exact"
    row["risk_evidence"]["group_type"] = "sha_exact"
    result = replay([row], primary_budget=0)["groups"][0]
    assert result["group_keeper_budget"] == 1
    assert not result["mandatory_review"]
    assert result["keeper_budget_source"] == "baseline_default_reconstructed"


def test_replay_changed_budget_does_not_restore_obsolete_primary():
    from scripts.phase_coverage_shadow import replay
    row = frozen_row(7, [[1, 2, 3]], [1])
    row["selective_output"] = "REVIEW_PRIMARY"
    result = replay([row], primary_budget=0)
    assert result["groups"][0]["selective_output"] == "SAFE_SILENT"
    assert result["metrics"]["selective_after"].get("REVIEW_PRIMARY", 0) == 0
    assert result["frontier"][0]["review_groups"] == 0


def test_selector_preserves_metadata_only_input_without_ids():
    from src.phase_selection import Phase, select_phase_keepers
    members = [{"quality_score": 30}, {"quality_score": 90}]
    result = select_phase_keepers(members, [Phase(0, (0, 1), (), False)])
    assert result["keepers"] == [1]
    assert result["phase_coverage_diagnostics"][0]["evidence"]["member_id_space"] == "member_index"


def test_frozen_exact_external_requirements_use_actual_one_budget():
    from scripts.phase_coverage_shadow import replay
    row = frozen_row(5, [[1, 2, 3]], [1])
    row["group_type"] = "sha_exact"
    result = replay([row], [{"dataset": "synthetic", "group_id": 5,
                            "requirements": requirements([[1], [2, 3]], "human")}], primary_budget=0)
    diagnostic = result["groups"][0]["phase_coverage_diagnostics"][1]
    assert diagnostic["keeper_budget"] == 1
    assert diagnostic["budget_conflict"]


def test_mixed_missing_ids_use_one_index_namespace_without_keeper_change():
    from src.phase_selection import Phase, select_phase_keepers
    members = [{"id": 1, "quality_score": 30}, {"quality_score": 90}]
    phases = [Phase(0, (0, 1), (), False)]
    result = select_phase_keepers(members, phases)
    assert result["keepers"] == [1]
    assert result["phase_coverage_diagnostics"][0]["member_ids"] == [0, 1]
    with pytest.raises(ValueError, match="stable member IDs"):
        select_phase_keepers(members, phases, phase_requirements=requirements([[0, 1]]))
