"""Tests for the dependency-free, fail-closed SigLIP prompt-score recorder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.routing_schema import SCENE_TAGS, SUBJECT_TAGS
from src.siglip_router import build_shadow_routing_record, load_prompt_bank, prompt_bank_hash


BANK_PATH = Path(__file__).parents[1] / "research" / "siglip_prompt_bank_v1.yaml"


def _bank():
    return load_prompt_bank(BANK_PATH)


def _scores(bank):
    return {
        code: {
            prompt: float(index + 1) / 10
            for index, prompt in enumerate(
                spec["positive_prompts"]
                + spec["hard_negative_prompts"]
                + [bank["generic_null_prompt"]]
            )
        }
        for code, spec in bank["tags"].items()
    }


def test_bank_covers_exact_schema_with_fixed_prompt_counts_and_namespaces():
    bank = _bank()
    assert bank["bank_version"] == 1
    assert set(bank["tags"]) == SCENE_TAGS | SUBJECT_TAGS
    for code, spec in bank["tags"].items():
        assert len(spec["positive_prompts"]) == 3
        assert len(spec["hard_negative_prompts"]) >= 2
        assert spec["namespace"] == ("scene_context" if code in SCENE_TAGS else "subject_protection")


def test_bank_has_explicit_high_risk_hard_negatives():
    text = BANK_PATH.read_text().lower()
    for word in ("printed", "screen", "doll", "statue", "mascot", "city lights", "restaurant", "people"):
        assert word in text


def test_valid_scores_record_auditable_fields_and_stays_unknown():
    bank = _bank()
    record = build_shadow_routing_record(bank, _scores(bank), model_name="siglip-base", model_revision="abc123")
    assert record["model"]["prompt_bank_hash"] == prompt_bank_hash(BANK_PATH)
    assert record["scene_context"]["state"] == "UNKNOWN"
    assert record["subject_protection"]["state"] == "UNKNOWN"
    assert record["scene_context"]["reasons"] == ["OUT_OF_CALIBRATION_DOMAIN"]
    audit = record["model"]["shadow_prompt_audit"]
    assert set(audit["tags"]) == SCENE_TAGS | SUBJECT_TAGS
    one = audit["tags"]["FIREWORKS"]
    assert {"raw_prompt_scores", "positive_mean", "positive_median", "positive_range", "positive_std", "hard_negative_max", "hard_negative_gap", "prompt_consistency"} <= set(one)
    assert json.loads(json.dumps(record)) == record


def test_prompt_bank_hash_normalizes_line_endings_and_loaded_hash_matches_helper(tmp_path):
    raw = BANK_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lf_path = tmp_path / "bank-lf.yaml"
    crlf_path = tmp_path / "bank-crlf.yaml"
    lf_path.write_bytes(raw)
    crlf_path.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert prompt_bank_hash(lf_path) == prompt_bank_hash(crlf_path)
    assert load_prompt_bank(lf_path).file_hash == prompt_bank_hash(lf_path)
    assert load_prompt_bank(crlf_path).file_hash == prompt_bank_hash(crlf_path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_raw_scores_are_rejected(bad):
    bank = _bank()
    scores = _scores(bank)
    scores["INDOOR"][bank["tags"]["INDOOR"]["positive_prompts"][0]] = bad
    with pytest.raises(ValueError, match="finite"):
        build_shadow_routing_record(bank, scores, model_name="siglip", model_revision="r1")


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_missing_extra_and_duplicate_prompt_data_are_rejected(mutation):
    bank = _bank()
    scores = _scores(bank)
    tag = "INDOOR"
    prompt = bank["tags"][tag]["positive_prompts"][0]
    if mutation == "missing":
        del scores[tag][prompt]
    elif mutation == "extra":
        scores[tag]["unknown prompt"] = 0.2
    else:
        bank["tags"][tag]["positive_prompts"][1] = prompt
    with pytest.raises(ValueError):
        build_shadow_routing_record(bank, scores, model_name="siglip", model_revision="r1")


def test_unknown_tag_data_is_rejected():
    bank = _bank()
    scores = _scores(bank)
    scores["UNKNOWN"] = {}
    with pytest.raises(ValueError, match="tags mismatch"):
        build_shadow_routing_record(bank, scores, model_name="siglip", model_revision="r1")


def test_record_and_router_source_do_not_claim_calibration_or_runtime_inference():
    bank = _bank()
    rendered = json.dumps(build_shadow_routing_record(bank, _scores(bank), model_name="siglip", model_revision="r1")).lower()
    assert "probability" not in rendered
    assert "confidence" not in rendered
    assert "decision" not in rendered
    assert "action" not in rendered
    source = (Path(__file__).parents[1] / "src" / "siglip_router.py").read_text().lower()
    assert "import torch" not in source
    assert "import transformers" not in source
