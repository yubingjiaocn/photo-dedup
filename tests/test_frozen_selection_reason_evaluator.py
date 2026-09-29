from __future__ import annotations

import json

from scripts.evaluate_frozen_selection_reasons import _source_evidence_ok
from src.selection_reason import explain_group


def test_source_evidence_checker_binds_margin_to_candidate_row():
    group = {"keep_file_id": 1}
    members = [
        {
            "file_id": 1, "basename": "keeper", "is_keep": 1,
            "decision": "KEEP", "reason": "GROUP_KEEPER",
            "quality_score": 80.0, "quality_meta": "{}",
            "evidence_json": json.dumps({"utility_score": .8, "pair_margin": 0}),
            "width": None, "height": None,
        },
        {
            "file_id": 2, "basename": "candidate", "is_keep": 0,
            "decision": "MAYBE", "reason": "RELATIVE_ONLY",
            "quality_score": 70.0, "quality_meta": "{}",
            "evidence_json": json.dumps({"utility_score": .7, "pair_margin": .1}),
            "width": None, "height": None,
        },
    ]
    reason = explain_group(group, members)
    checked, valid = _source_evidence_ok(reason, members)
    assert checked == valid
    margin = next(item for item in reason["evidence"]
                  if item["code"] == "PERSISTED_PAIR_MARGIN")
    assert margin["subject_file_id"] == 2
    assert margin["compared_to_file_id"] == 1
