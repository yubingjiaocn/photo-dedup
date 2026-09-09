"""Frozen, blinded human audit bundles. No pipeline DB or original-media access."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .dataset_contract import validate_manifest, validate_split_contract
from .risk_coverage import stratified_audit_sample

VERSION = 1  # Frozen sampling/bundle format; label and report semantics are versioned separately.
LABEL_VERSION = 2
LABEL_KEYS = {"schema_version", "bundle_id", "task_id", "status", "annotator",
              "human_attested", "phases", "group_impure", "note"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("non-finite JSON number")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=constant)


def integer(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("expected positive integer")
    return value


def ids(value):
    if not isinstance(value, list) or not value:
        raise ValueError("expected non-empty ID list")
    for item in value:
        integer(item)
    if len(set(value)) != len(value):
        raise ValueError("duplicate member")
    return value


def development_scope(value):
    """Must run before opening group inputs/cache. Held-out is never authorized."""
    if not isinstance(value, dict) or not value:
        raise ValueError("explicit development dataset scope is required")
    manifests = [validate_manifest(item) for item in value.values()]
    validate_split_contract(manifests)
    for key, manifest in zip(value, manifests, strict=True):
        if (key != manifest.dataset_id or manifest.split.value not in {"train", "tune"}
                or manifest.owner_scope not in {"willy", "synthetic"}):
            raise ValueError("only explicit Willy/synthetic development datasets allowed")
    return value


def freeze_design(groups, audit, scopes):
    """Replay v4 sampling from the entire pre-audit silent population."""
    development_scope(scopes)
    if not isinstance(groups, list) or not groups:
        raise ValueError("empty group population")
    rows = copy.deepcopy(groups)
    keys = set()
    for row in rows:
        if row["dataset"] not in scopes:
            raise ValueError("dataset outside authorized scope")
        integer(row["group_id"])
        key = f'{row["dataset"]}:G{row["group_id"]}'
        if key in keys:
            raise ValueError("duplicate group")
        keys.add(key)
        members = ids(row["member_ids"])
        if integer(row["member_count"]) != len(members):
            raise ValueError("member count mismatch")
        if not set(ids(row["candidate_keeper_ids"])) <= set(members):
            raise ValueError("candidate outside group")
        if type(row["changed"]) is not bool:
            raise ValueError("changed must be boolean")
        if not 1 <= integer(row["risk_decile"]) <= 10:
            raise ValueError("invalid decile")
        if (type(row["risk_score"]) not in {int, float}
                or not math.isfinite(row["risk_score"]) or not 0 <= row["risk_score"] <= 1):
            raise ValueError("invalid risk score")
        if row["selective_output"] not in {"REVIEW_PRIMARY", "SAFE_SILENT", "DIAGNOSTIC_SAMPLE"}:
            raise ValueError("invalid selective output")
        if row["selective_output"] == "DIAGNOSTIC_SAMPLE":
            row["selective_output"] = "SAFE_SILENT"
    integer(audit["sample_count"], 0)
    if not isinstance(audit["seed"], str) or not audit["seed"]:
        raise ValueError("missing sampling seed")
    replay = stratified_audit_sample(rows, audit["sample_count"], audit["seed"])
    for field in ("silent_population", "sample_count", "selected_keys", "strata"):
        if canonical(audit[field]) != canonical(replay[field]):
            raise ValueError(f"sampling replay mismatch: {field}")
    supplied = {item["audit_id"]: item for item in audit["groups"]}
    if len(supplied) != len(audit["groups"]) or set(supplied) != set(replay["selected_keys"]):
        raise ValueError("sample ID mismatch")
    for item in replay["groups"]:
        for field, expected in item.items():
            if canonical(supplied[item["audit_id"]].get(field)) != canonical(expected):
                raise ValueError(f"sample row mismatch: {field}")
    return replay


def make_plan(groups, audit, scopes, provenance, seed, fixture=None):
    design = freeze_design(groups, audit, scopes)
    if not isinstance(seed, str) or not seed:
        raise ValueError("presentation seed required")
    if (provenance.get("opened_original_media") is not False
            or provenance.get("lin_held_out_touched") is not False
            or provenance.get("writeback") is not False
            or provenance.get("authority") != "shadow_review_only"
            or provenance.get("auto_remove_authority") != "BYTE_IDENTICAL_ONLY"):
        raise ValueError("incompatible source safety provenance")
    sources = provenance.get("source_sha256_before_after", {})
    if set(sources) != set(scopes) or any(
        not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
        for h in sources.values()
    ):
        raise ValueError("source fingerprints required for each dataset")
    row_map = {f'{r["dataset"]}:G{r["group_id"]}': r for r in groups}
    selections = [(key, "probability") for key in design["selected_keys"]]
    if fixture is not None:
        if set(fixture) != {"schema_version", "audit_id", "member_ids", "source_sha256",
                            "status", "purpose"}:
            raise ValueError("invalid narrative fixture")
        if (type(fixture["schema_version"]) is not int or fixture["schema_version"] != 1
                or fixture["status"] != "pending"
                or fixture["purpose"] != "narrative_recheck"):
            raise ValueError("fixture must be pending narrative recheck")
        row = row_map.get(fixture["audit_id"])
        if (row is None or row["member_ids"] != ids(fixture["member_ids"])
                or sources[row["dataset"]] != fixture["source_sha256"]):
            raise ValueError("stale narrative fixture")
        if fixture["audit_id"] not in design["selected_keys"]:
            selections.append((fixture["audit_id"], "purposive"))
    identity = digest({"groups": groups, "audit": audit, "scope": scopes,
                       "provenance": provenance, "seed": seed, "fixture": fixture})
    tasks = []
    sample = {item["audit_id"]: item for item in design["groups"]}
    for key, kind in selections:
        row = row_map[key]
        task_id = digest([identity, key])[:24]
        aliases = [f"M{i + 1}" for i in range(len(row["member_ids"]))]
        tasks.append({
            "task_id": task_id, "audit_id": key, "dataset": row["dataset"],
            "kind": kind, "member_ids": row["member_ids"], "aliases": aliases,
            "candidate_keeper_ids": row["candidate_keeper_ids"],
            "member_fingerprint": digest(row["member_ids"]),
            "stratum": sample[key]["stratum"] if key in sample else None,
            "narrative_recheck": fixture is not None and key == fixture["audit_id"],
        })
    tasks.sort(key=lambda item: digest([seed, item["task_id"]]))
    return {"schema_version": VERSION, "kind": "blinded_phase_audit", "seed": seed,
            "source_identity": identity, "source_sha256": sources, "scopes": scopes,
            "sampling_seed": design["seed"], "silent_population": design["silent_population"],
            "strata": design["strata"], "tasks": tasks, "authority": "annotation_only"}


def seal(plan):
    return {**plan, "bundle_id": digest(plan)}


def validate_bundle(bundle):
    value = dict(bundle)
    bundle_id = value.pop("bundle_id", None)
    if bundle_id != digest(value) or value.get("schema_version") != VERSION:
        raise ValueError("bundle integrity/version mismatch")
    if value.get("authority") != "annotation_only" or value.get("kind") != "blinded_phase_audit":
        raise ValueError("incompatible bundle")
    development_scope(value["scopes"])
    return bundle


def validate_labels(bundle, labels):
    validate_bundle(bundle)
    tasks = {task["task_id"]: task for task in bundle["tasks"]}
    seen = set()
    result = {}
    for label in labels:
        if not isinstance(label, dict) or set(label) != LABEL_KEYS:
            raise ValueError("invalid label fields")
        version = label["schema_version"]
        valid_ids = [bundle["bundle_id"]]
        if version == 1 and bundle.get("label_schema_version", 1) == 2:
            valid_ids = [bundle["legacy_bundle_id"]] if bundle.get("legacy_bundle_id") else []
        if (type(version) is not int or version not in {1, LABEL_VERSION}
                or label["bundle_id"] not in valid_ids):
            raise ValueError("stale label bundle/version")
        key = label["task_id"]
        if not isinstance(key, str) or key not in tasks or key in seen:
            raise ValueError("unknown/duplicate label task")
        seen.add(key)
        if (not isinstance(label["annotator"], str) or not label["annotator"].strip()
                or len(label["annotator"]) > 80 or type(label["human_attested"]) is not bool
                or not isinstance(label["note"], str) or len(label["note"]) > 2000):
            raise ValueError("invalid annotation provenance")
        status = label["status"]
        if status not in {"reviewed", "uncertain", "unassessable"}:
            raise ValueError("invalid annotation status")
        if status != "reviewed":
            if label["phases"] != [] or label["group_impure"] is not None:
                raise ValueError("uncertain annotations cannot contain judgments")
        else:
            if not label["human_attested"]:
                raise ValueError("explicit human attestation required")
            if not tasks[key]["media_complete"]:
                raise ValueError("missing thumbnails cannot be assessed")
            if type(label["group_impure"]) is not bool:
                raise ValueError("group impurity must be explicit boolean")
            phases = label["phases"]
            if not isinstance(phases, list) or not phases:
                raise ValueError("phases required")
            members_seen = set()
            for phase in phases:
                if not isinstance(phase, dict):
                    raise ValueError("invalid phase fields")
                keeper_status = phase.get("keeper_status") if version == 2 else "assessed"
                if keeper_status not in {"assessed", "quality_abstain"}:
                    raise ValueError("explicit keeper_status required")
                fields = {"members", "acceptable_keepers"} if version == 1 else {"members", "keeper_status"}
                if version == 2 and keeper_status == "assessed":
                    fields.add("acceptable_keepers")
                if set(phase) != fields:
                    raise ValueError("invalid phase fields")
                members, keepers = phase["members"], phase.get("acceptable_keepers", [])
                for values in ([members, keepers] if keeper_status == "assessed" else [members]):
                    if (not isinstance(values, list) or not values
                            or any(not isinstance(v, str) for v in values)
                            or len(set(values)) != len(values)):
                        raise ValueError("invalid phase IDs")
                if (not set(members) <= set(tasks[key]["aliases"])
                        or members_seen.intersection(members) or not set(keepers) <= set(members)):
                    raise ValueError("invalid phase partition/keepers")
                members_seen.update(members)
            if members_seen != set(tasks[key]["aliases"]):
                raise ValueError("phase partition must cover every member")
        result[key] = label
    return result


def read_labels(path):
    return parse_labels(Path(path).read_text(encoding="utf-8"))


def parse_labels(text):
    # Shared by file replay and byte-preserving migration snapshots.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("non-finite JSON number")
    return [json.loads(line, object_pairs_hook=pairs, parse_constant=constant)
            for line in text.splitlines() if line.strip()]


def labels_v2(bundle, labels):
    """Explicit structural conversion; v1 reviewed means assessed, regardless of notes."""
    parsed = validate_labels(bundle, labels)
    result = copy.deepcopy(list(parsed.values()))
    for label in result:
        if label["schema_version"] == 1:
            for phase in label["phases"]:
                phase["keeper_status"] = "assessed"
        label.update(schema_version=LABEL_VERSION, bundle_id=bundle["bundle_id"])
    validate_labels(bundle, result)
    return result


def report(bundle, labels):
    parsed = validate_labels(bundle, labels)
    details = []
    for task in bundle["tasks"]:
        label = parsed.get(task["task_id"])
        status = label["status"] if label else "pending"
        error = phase_miss = keeper_bad = coverage_error = quality_error = None
        assessed_count = abstain_count = quality_phase_errors = 0
        if status == "reviewed":
            aliases = dict(zip(task["member_ids"], task["aliases"], strict=True))
            chosen = {aliases[item] for item in task["candidate_keeper_ids"]}
            phases = label["phases"]
            coverage_error = any(not chosen.intersection(p["members"]) for p in phases)
            assessed = [p for p in phases if p.get("keeper_status", "assessed") == "assessed"]
            assessed_count, abstain_count = len(assessed), len(phases) - len(assessed)
            # Quality concerns selected candidates, not whether a phase was covered.
            quality_phase_errors = sum(bool(chosen.intersection(p["members"]) -
                                            set(p["acceptable_keepers"])) for p in assessed)
            if not abstain_count:
                quality_error = keeper_bad = bool(quality_phase_errors)
                # Preserve the v1 acceptable-coverage + bad-keeper union exactly.
                phase_miss = any(not chosen.intersection(p["acceptable_keepers"]) for p in phases)
                error = bool(label["group_impure"] or phase_miss or keeper_bad)
        details.append({"task_id": task["task_id"], "audit_id": task["audit_id"],
                        "kind": task["kind"], "stratum": task["stratum"], "status": status,
                        "error": error, "phase_miss": phase_miss, "keeper_bad": keeper_bad,
                        "phase_coverage_error": coverage_error, "keeper_quality_error": quality_error,
                        "phase_coverage_eligible": coverage_error is not None,
                        "keeper_quality_eligible": quality_error is not None,
                        "joint_eligible": error is not None,
                        "quality_assessed_phase_count": assessed_count,
                        "quality_abstain_phase_count": abstain_count,
                        "quality_assessed_phase_errors": quality_phase_errors})
    strata = []
    active = sum(s["population"] > s["sample"] for s in bundle["strata"])
    for s in bundle["strata"]:
        subset = [d for d in details if d["kind"] == "probability" and d["stratum"] == s["stratum"]]
        eligible = [d for d in subset if d["joint_eligible"]]
        covered = [d for d in subset if d["phase_coverage_eligible"]]
        n, population = s["sample"], s["population"]
        errors = sum(d["error"] for d in eligible)
        complete = len(eligible) == n and n > 0
        rate = errors / n if complete else None
        # Conservative SRSWOR Hoeffding + Bonferroni; census strata are exact.
        upper = (rate if n == population else min(1., rate + math.sqrt(math.log(active / .05) / (2*n)))) if complete else None
        strata.append({**s, "reviewed": sum(d["status"] == "reviewed" for d in subset),
                       "joint_eligible_count": len(eligible), "errors": errors,
                       "phase_coverage_eligible_count": len(covered),
                       "phase_coverage_error_rate": sum(d["phase_coverage_error"] for d in covered) / n
                       if len(covered) == n and n > 0 else None,
                       "error_rate": rate, "upper_95": upper})
    complete = bool(strata) and all(s["error_rate"] is not None for s in strata)
    phase_complete = bool(strata) and all(s["phase_coverage_error_rate"] is not None for s in strata)
    total = bundle["silent_population"]

    def evidence_counts(kind):
        subset = [d for d in details if d["kind"] == kind]
        result = {"task_count": len(subset)}
        for metric in ("phase_coverage", "keeper_quality"):
            eligible = [d for d in subset if d[f"{metric}_eligible"]]
            result[f"{metric}_eligible_count"] = len(eligible)
            result[f"{metric}_error_count"] = sum(d[f"{metric}_error"] for d in eligible)
        result.update(joint_eligible_count=sum(d["joint_eligible"] for d in subset),
                      quality_abstain_group_count=sum(d["quality_abstain_phase_count"] > 0 for d in subset),
                      quality_abstain_phase_count=sum(d["quality_abstain_phase_count"] for d in subset),
                      quality_assessed_phase_count=sum(d["quality_assessed_phase_count"] for d in subset),
                      quality_assessed_phase_errors=sum(d["quality_assessed_phase_errors"] for d in subset))
        return result

    return {
        "schema_version": LABEL_VERSION, "bundle_id": bundle["bundle_id"],
        "labels_fingerprint": digest(sorted(labels, key=lambda x: x["task_id"])),
        "authority": "annotation_only", "deletion_authority": "none",
        "estimand": "frozen_development_silent_group_error_rate",
        "status": "complete_probability_sample" if complete else "incomplete_no_risk_estimate",
        "silent_population": total, "task_status_counts": dict(Counter(d["status"] for d in details)),
        "probability_sample_count": sum(s["sample"] for s in strata),
        "purposive_count": sum(d["kind"] == "purposive" for d in details),
        "weighted_error_rate": sum(s["population"] * s["error_rate"] for s in strata) / total if complete else None,
        "upper_95": sum(s["population"] * s["upper_95"] for s in strata) / total if complete else None,
        "evidence_counts": {kind: evidence_counts(kind) for kind in ("probability", "purposive")},
        "phase_only": {
            "descriptive_only": True, "safety_validated": False,
            "complete_probability_phase_labels": phase_complete,
            "weighted_phase_coverage_error_rate": sum(s["population"] * s["phase_coverage_error_rate"] for s in strata) / total if phase_complete else None,
            "upper_95": None,
        },
        "bound_method": "population_weighted_bonferroni_hoeffding_srswor_census_exact",
        "safety_validated": False, "strata": strata, "tasks": details,
        "limitations": ["Human attestations are not proof of reviewer identity or correctness.",
                        "Development events only; not future-event or held-out generalization.",
                        "No policy fitting on these labels before reporting this frozen audit.",
                        "No deletion or pipeline writeback; incomplete/uncertain/quality-abstain labels block joint estimation.",
                        "Phase coverage uses membership only; phase-only results are descriptive, not safety certification.",
                        "Quality abstention is neither success nor error. Mixed groups have only assessed-phase quality counts.",
                        "Legacy v1 reviewed retains fully assessed semantics; notes never infer abstention."],
    }
