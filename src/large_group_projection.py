"""Candidate revision 2: one authoritative shortlist, projected from clusters.

The model reports phase census and cluster representatives, not a second copy of
the shortlist. Local deterministic projection is explicit, never raw repair.
"""

from __future__ import annotations

import copy

import jsonschema

from . import cloud_judge_batch as b
from . import large_group_review as v4
from . import two_pass_review as tp

CONTRACT = "large-group-v4-candidate-2"
REVISION_HYPOTHESIS = (
    "The first 11-frame canary produced a different recommended list from its "
    "cluster representatives. Remove duplicate selection authorship; derive "
    "recommended and multi-reason evidence from the single cluster ledger. "
    "Do not alter semantic phase or representative selection policy."
)


def schema():
    value = copy.deepcopy(v4.response_schema())
    value["$id"] = "large-group-judge-v4-candidate-2.schema.json"
    value["properties"]["contract"]["const"] = CONTRACT
    value["required"].remove("recommended")
    del value["properties"]["recommended"]
    value["properties"]["multi_select_reason"] = {
        "oneOf": [
            {"type": "null"},
            {"type": "string", "minLength": 1, "maxLength": 500},
        ]
    }
    # The base conditional only constrains fields that still exist here.
    for condition in value["allOf"]:
        condition["then"]["properties"].pop("recommended", None)
    return value


def build_request(request, rows):
    if len(rows) == 2:
        return v4.build_request(request, rows)
    result = v4.build_request(request, rows)
    result["judgment_contract"] = CONTRACT
    result["response_schema"] = schema()
    result["prompt"] = v4.PROMPT.replace(
        "recommended equals the cluster representatives exactly.",
        "The application derives recommended from cluster representatives exactly.",
    ).replace(
        "multi_select_reason explains the complementary useful states and\nbinds every selected frame.",
        "multi_select_reason is only a Chinese text string explaining complementary states. The application derives its evidence bindings.",
    )
    result["prompt"] += (
        "\nOutput phase_census and clusters before the summary. DO NOT output a "
        "recommended field: the cluster representative is the ONLY selection "
        "authority. The client copies each representative's medium/high own-frame "
        "census evidence into recommended and multi-select evidence. Never emit a "
        "second selection list in prose. Each cluster must reflect what its member "
        "images actually show, independent of CV rank."
    )
    return result


def validate_response(raw, rows, packet, validation, l2):
    if len(rows) == 2:
        return v4.validate_response(raw, rows, packet, validation, l2)
    jsonschema.validate(raw, schema())
    value = copy.deepcopy(raw)
    value["contract"] = v4.CONTRACT
    census = {tp.identity(c["frame"]): c for c in raw["phase_census"]}
    recommended = []
    if raw["decision"] == "select":
        for cluster in raw["clusters"]:
            rep = cluster["representative"]
            b.verify(
                rep is not None and tp.identity(rep) in census,
                "representative missing from census",
            )
            evidence = census[tp.identity(rep)]["evidence"]
            strong = [
                e
                for e in evidence
                if e["confidence"] != "low"
                and validation["lookup"].get(e["region"], {}).get("confidence")
                in {"medium", "high"}
            ]
            b.verify(bool(strong), "representative requires strong census evidence")
            recommended.append({"frame": rep, "evidence": strong})
    value["recommended"] = recommended
    text = raw["multi_select_reason"]
    if len(recommended) > 1:
        b.verify(text is not None, "multi-select prose required")
        value["multi_select_reason"] = {
            "text": text,
            "evidence": [r["evidence"][0] for r in recommended],
        }
    else:
        b.verify(text is None, "unexpected multi-select prose")
    checked = v4.validate_response(value, rows, packet, validation, l2)
    checked["contract"] = CONTRACT
    checked["derived_fields"] = ["recommended", "multi_select_reason.evidence"]
    return checked
