"""Strict prompt-bank validation and shadow score recording for SigLIP.

This module neither loads models nor reads images.  A caller supplies raw scores
already computed elsewhere; this boundary validates them and records auditable,
JSON-safe shadow evidence without claiming calibrated certainty.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import yaml

from .routing_schema import (
    ReasonCode,
    SCENE_TAGS,
    SUBJECT_TAGS,
    safe_default_record,
    validate_routing_record,
)

BANK_VERSION = 1
MODEL_FAMILY = "siglip"


class PromptBank(dict[str, Any]):
    """Validated bank retaining the normalized on-disk hash outside YAML data."""

    def __init__(self, data: Mapping[str, Any], file_hash: str | None = None) -> None:
        super().__init__(data)
        self.file_hash = file_hash


def load_prompt_bank(path: str | Path) -> dict[str, Any]:
    """Load one versioned YAML prompt bank and validate it strictly."""
    bank_path = Path(path)
    try:
        raw = bank_path.read_bytes()
        value = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load prompt bank: {exc}") from exc
    bank = _mapping(value, "prompt bank")
    _keys(bank, {"bank_version", "model_family", "prompt_template_notes", "generic_null_prompt", "tags"}, "prompt bank")
    if bank["bank_version"] != BANK_VERSION:
        raise ValueError(f"prompt bank bank_version must be {BANK_VERSION}")
    if bank["model_family"] != MODEL_FAMILY:
        raise ValueError(f"prompt bank model_family must be {MODEL_FAMILY!r}")
    _nonempty_string(bank["prompt_template_notes"], "prompt_template_notes")
    null_prompt = _nonempty_string(bank["generic_null_prompt"], "generic_null_prompt")
    tags = _mapping(bank["tags"], "prompt bank tags")
    expected = SCENE_TAGS | SUBJECT_TAGS
    if set(tags) != expected:
        raise ValueError(f"prompt bank tags must exactly match schema; missing={sorted(expected - set(tags))!r}, extra={sorted(set(tags) - expected)!r}")
    normalized_tags = {code: _tag_spec(code, value) for code, value in tags.items()}
    if any(null_prompt in spec["positive_prompts"] + spec["hard_negative_prompts"] for spec in normalized_tags.values()):
        raise ValueError("generic_null_prompt must appear only once at bank level")
    return PromptBank({"bank_version": BANK_VERSION, "model_family": MODEL_FAMILY,
            "prompt_template_notes": str(bank["prompt_template_notes"]),
            "generic_null_prompt": null_prompt, "tags": normalized_tags}, _hash_bytes(raw))


def prompt_bank_hash(path: str | Path) -> str:
    """Return SHA-256 of the exact, canonical on-disk prompt-bank bytes."""
    try:
        return _hash_bytes(Path(path).read_bytes())
    except OSError as exc:
        raise ValueError(f"cannot hash prompt bank: {exc}") from exc


def build_shadow_routing_record(
    bank: Mapping[str, Any], raw_scores: Mapping[str, Mapping[str, float]], *,
    model_name: str, model_revision: str,
    model_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record complete external raw scores as a fail-closed routing record.

    ``raw_scores`` must contain exactly every tag and exactly its fixed positive
    and hard-negative prompts plus ``__generic_null__``.  This function never
    decides tag membership: all routing states deliberately remain UNKNOWN.
    """
    normalized_bank = _validate_bank_object(bank)
    name = _nonempty_string(model_name, "model_name")
    revision = _nonempty_string(model_revision, "model_revision")
    audit = _score_audit(normalized_bank, raw_scores)
    model = {
        "name": name,
        "revision": revision,
        "prompt_bank_hash": getattr(bank, "file_hash", None) or _hash_bank_object(normalized_bank),
        "shadow_prompt_audit": audit,
    }
    if model_audit is not None:
        extra = _mapping(model_audit, "model_audit")
        if set(extra) & set(model):
            raise ValueError("model_audit must not override recorder-owned fields")
        model.update(extra)
    record = safe_default_record(reasons=[ReasonCode.OUT_OF_CALIBRATION_DOMAIN], model=model)
    return validate_routing_record(record)


def _validate_bank_object(bank: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an in-memory bank by the same strict rules as a loaded bank."""
    if not isinstance(bank, Mapping):
        raise ValueError("prompt bank must be an object")
    _keys(bank, {"bank_version", "model_family", "prompt_template_notes", "generic_null_prompt", "tags"}, "prompt bank")
    if bank["bank_version"] != BANK_VERSION or bank["model_family"] != MODEL_FAMILY:
        raise ValueError("invalid prompt bank version or model family")
    tags = _mapping(bank["tags"], "prompt bank tags")
    expected = SCENE_TAGS | SUBJECT_TAGS
    if set(tags) != expected:
        raise ValueError("prompt bank tags must exactly match schema")
    output = {"bank_version": BANK_VERSION, "model_family": MODEL_FAMILY,
            "prompt_template_notes": _nonempty_string(bank["prompt_template_notes"], "prompt_template_notes"),
            "generic_null_prompt": _nonempty_string(bank["generic_null_prompt"], "generic_null_prompt"),
            "tags": {code: _tag_spec(code, spec) for code, spec in tags.items()}}
    if any(output["generic_null_prompt"] in spec["positive_prompts"] + spec["hard_negative_prompts"] for spec in output["tags"].values()):
        raise ValueError("generic_null_prompt must appear only once at bank level")
    return output


def _tag_spec(code: str, value: Any) -> dict[str, Any]:
    spec = _mapping(value, f"tag {code}")
    _keys(spec, {"namespace", "exclusive_cluster", "positive_prompts", "hard_negative_prompts"}, f"tag {code}") if "exclusive_cluster" in spec else _keys(spec, {"namespace", "positive_prompts", "hard_negative_prompts"}, f"tag {code}")
    expected_namespace = "scene_context" if code in SCENE_TAGS else "subject_protection"
    if spec["namespace"] != expected_namespace:
        raise ValueError(f"tag {code} has wrong namespace")
    positive = _unique_prompts(spec["positive_prompts"], f"tag {code} positive_prompts")
    negative = _unique_prompts(spec["hard_negative_prompts"], f"tag {code} hard_negative_prompts")
    if len(positive) != 3:
        raise ValueError(f"tag {code} must have exactly 3 positive prompts")
    if len(negative) < 2:
        raise ValueError(f"tag {code} must have at least 2 hard-negative prompts")
    if set(positive) & set(negative):
        raise ValueError(f"tag {code} repeats a prompt across positive and hard-negative prompts")
    output = {"namespace": expected_namespace, "positive_prompts": positive, "hard_negative_prompts": negative}
    if "exclusive_cluster" in spec:
        output["exclusive_cluster"] = _nonempty_string(spec["exclusive_cluster"], f"tag {code} exclusive_cluster")
    return output


def _score_audit(bank: Mapping[str, Any], raw_scores: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    supplied = _mapping(raw_scores, "raw_scores")
    expected_tags = set(bank["tags"])
    if set(supplied) != expected_tags:
        raise ValueError(f"raw_scores tags mismatch; missing={sorted(expected_tags - set(supplied))!r}, extra={sorted(set(supplied) - expected_tags)!r}")
    audit: dict[str, Any] = {"generic_null_prompt": bank["generic_null_prompt"], "tags": {}}
    for code, spec in bank["tags"].items():
        expected_prompts = list(spec["positive_prompts"]) + list(spec["hard_negative_prompts"]) + [bank["generic_null_prompt"]]
        values = _scores_for_tag(supplied[code], expected_prompts, code)
        positives = [values[prompt] for prompt in spec["positive_prompts"]]
        negatives = [values[prompt] for prompt in spec["hard_negative_prompts"]]
        audit["tags"][code] = {
            "namespace": spec["namespace"],
            "raw_prompt_scores": values,
            "positive_mean": mean(positives), "positive_median": median(positives),
            "positive_range": max(positives) - min(positives), "positive_std": pstdev(positives),
            "hard_negative_max": max(negatives), "hard_negative_gap": mean(positives) - max(negatives),
            "prompt_consistency": {"positive_count": len(positives), "positive_sign_agreement": _sign_agreement(positives)},
        }
        if "exclusive_cluster" in spec:
            audit["tags"][code]["exclusive_cluster"] = spec["exclusive_cluster"]
    return audit


def _scores_for_tag(value: Any, prompts: list[str], code: str) -> dict[str, float]:
    scores = _mapping(value, f"raw_scores.{code}")
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"prompt bank has duplicate expected prompt for {code}")
    if set(scores) != set(prompts):
        raise ValueError(f"raw_scores.{code} prompts mismatch; missing={sorted(set(prompts) - set(scores))!r}, extra={sorted(set(scores) - set(prompts))!r}")
    output: dict[str, float] = {}
    for prompt in prompts:
        score = scores[prompt]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"raw_scores.{code} prompt score must be finite")
        output[prompt] = float(score)
    return output


def _hash_bank_object(bank: Mapping[str, Any]) -> str:
    # A loaded bank exposes a file-byte hash through ``prompt_bank_hash``.
    # This deterministic fallback keeps direct in-memory unit use auditable.
    import json
    return hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hash_bytes(raw: bytes) -> str:
    """Hash file bytes after a deterministic newline normalization."""
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _sign_agreement(values: Sequence[float]) -> float:
    signs = [value >= 0.0 for value in values]
    return max(signs.count(True), signs.count(False)) / len(signs)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has invalid keys; missing={sorted(expected - set(value))!r}, extra={sorted(set(value) - expected)!r}")


def _unique_prompts(value: Any, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list")
    prompts = [_nonempty_string(item, label) for item in value]
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"{label} contains duplicate prompts")
    return prompts


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value
