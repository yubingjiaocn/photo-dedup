"""Incident delivery regression: metadata only, never holdout images or labels."""

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/learning-loop-mvp"
INCIDENT = EVIDENCE / "holdout-incidents.json"
SCHEMA = ROOT / "schemas/holdout-incidents-v1.schema.json"


def validator():
    schema = json.loads(SCHEMA.read_bytes())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )


def test_readme_incident_exists_validates_and_is_indexed():
    assert "holdout-incidents.json" in (EVIDENCE / "README.md").read_text()
    assert INCIDENT.is_file()
    payload = INCIDENT.read_bytes()
    record = json.loads(payload)
    validator().validate(record)
    incident = record["incidents"][0]
    assert incident["group"] == "2026-04-04-G001"
    assert incident["evaluation_policy"]["required_reports"] == [
        {"name": "all_holdout", "groups": 64, "excluded_groups": []},
        {
            "name": "exclude-G001_sensitivity",
            "groups": 63,
            "excluded_groups": ["2026-04-04-G001"],
        },
    ]
    assert (EVIDENCE / incident["audit_trail_ref"].split("#")[0]).is_file()
    for name in ("checkpoint.json", "current.json"):
        receipt = json.loads((EVIDENCE / name).read_bytes())["holdout_incident_evidence"]
        assert ROOT / receipt["path"] == INCIDENT
        assert ROOT / receipt["schema"] == SCHEMA
        assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("mutation", ["missing", "decoded", "flow", "residue", "policy", "time"])
def test_incident_schema_rejects_incomplete_or_conflicting_claims(mutation):
    record = copy.deepcopy(json.loads(INCIDENT.read_bytes()))
    incident = record["incidents"][0]
    if mutation == "missing":
        del incident["source_test"]
    elif mutation == "decoded":
        incident["decoded"] = True
    elif mutation == "flow":
        incident["information_flow"]["shadow"] = True
    elif mutation == "residue":
        incident["residue"]["status"] = "unknown"
    elif mutation == "policy":
        incident["evaluation_policy"]["required_reports"] = []
    else:
        incident["fixed_at"] = "not a timestamp"
    with pytest.raises(jsonschema.ValidationError):
        validator().validate(record)
