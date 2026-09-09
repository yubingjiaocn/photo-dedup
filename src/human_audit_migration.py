"""Explicitly authorized, deterministic note migration. No media reads/inference."""
from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter

from .human_audit import digest, parse_labels, validate_labels

RULES_VERSION = "authorized-note-migration-v1"
CURRENT_BUNDLE = "@current"
ABSTAIN = re.compile(r"缩略图.{0,12}看不清|没法确认留哪张|(?:没法|无法)判断|要看清晰度|动作很快.{0,12}看不清")
ASSESSED = re.compile(r"构图更完整|主体完整|水柱.{0,16}(?:挡住|遮挡)|更水平|正脸优先")
PHASE_UNCERTAIN = re.compile(r"(?:阶段|动作|姿势).{0,8}(?:不确定|无法判断|没法判断)|(?:不确定|无法判断|没法判断).{0,8}(?:阶段|动作|姿势)")


def note_rule(label):
    """Fail closed: punctuation/conflict wins over affirmative phrase matches."""
    note = label["note"]
    abstain, assessed = ABSTAIN.findall(note), ASSESSED.findall(note)
    if label["status"] != "reviewed" or not label["phases"]:
        return "needs_review_nonreviewed", True, []
    if "?" in note or "？" in note:
        return "needs_review_question", True, [c for c in note if c in "?？"]
    if PHASE_UNCERTAIN.search(note):
        return "needs_review_phase_uncertain", True, PHASE_UNCERTAIN.findall(note)
    if abstain and assessed:
        return "needs_review_conflicting_quality", True, abstain + assessed
    if abstain:
        return "quality_abstain_explicit", False, abstain
    if assessed:
        return "assessed_explicit", False, assessed
    return "needs_review_no_explicit_quality_rule", True, []


def authorized_note_migration(bundle, source_bytes, authorization_id):
    if not isinstance(authorization_id, str) or not authorization_id.strip() or len(authorization_id) > 120:
        raise ValueError("explicit note-migration authorization identifier required")
    labels = parse_labels(source_bytes.decode("utf-8"))
    validate_labels(bundle, labels)
    if any(x["schema_version"] != 1 for x in labels):
        raise ValueError("note migration requires original v1 labels")
    migration_id = digest({"rules_version": RULES_VERSION, "authorization_id": authorization_id,
                           "source_bundle_id": bundle["bundle_id"],
                           "source_jsonl_sha256": hashlib.sha256(source_bytes).hexdigest()})
    lines = [(i, line) for i, line in enumerate(source_bytes.splitlines(), 1) if line.strip()]
    rows, records = [], []
    for original, (number, raw_line) in zip(labels, lines, strict=True):
        rule, needs_review, matches = note_rule(original)
        source_hash = hashlib.sha256(raw_line).hexdigest()
        result = copy.deepcopy(original)
        result.update(schema_version=2, bundle_id=CURRENT_BUNDLE)
        for phase in result["phases"]:
            phase["keeper_status"] = "quality_abstain" if rule == "quality_abstain_explicit" else "assessed"
            if phase["keeper_status"] == "quality_abstain":
                # Strict v2 expresses an empty/no keeper judgment by field absence.
                del phase["acceptable_keepers"]
        result["provenance"] = {"kind": "authorized_note_migration", "migration_id": migration_id,
                                "source_row_sha256": source_hash, "needs_review": needs_review,
                                "human_attestation_scope": "original_visual_observation"}
        rows.append(result)
        records.append({"task_id": original["task_id"], "source_line_number": number,
                        "source_row_sha256": source_hash, "note": original["note"],
                        "rule": rule, "matched_terms": matches, "needs_review": needs_review,
                        "provenance": "authorized_note_migration",
                        "human_attestation_scope": "original_visual_observation",
                        "v2_result": result})
    counts = Counter(r["rule"] for r in records)
    manifest = {"schema_version": 1, "provenance": "authorized_note_migration",
                "migration_id": migration_id, "rules_version": RULES_VERSION,
                "authorization_id": authorization_id, "source_bundle_id": bundle["bundle_id"],
                "source_jsonl_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "source_row_hash_definition": "SHA256 of UTF-8 line bytes excluding line terminator; blank lines ignored",
                "target_bundle_id": CURRENT_BUNDLE,
                "counts": {"input": len(rows), "automatic": sum(not r["needs_review"] for r in records),
                           "needs_review": sum(r["needs_review"] for r in records),
                           "quality_abstain": counts["quality_abstain_explicit"],
                           "assessed": counts["assessed_explicit"]},
                "rules": {"priority": ["nonreviewed", "question", "phase_uncertain", "quality_conflict", "abstain", "assessed", "no_match"],
                          "abstain_regex": ABSTAIN.pattern, "assessed_regex": ASSESSED.pattern,
                          "phase_uncertain_regex": PHASE_UNCERTAIN.pattern,
                          "partition": "preserve original; needs_review records excluded from all evidence until human confirmation",
                          "acceptable_keepers_on_abstain": "absent (strict v2 no-judgment representation)"},
                "records": records}
    sealed = {"migration_id": migration_id, "manifest_sha256": digest(manifest),
              "counts": manifest["counts"],
              "records": {r["task_id"]: {k: v for k, v in r.items() if k != "bundle_id"} for r in rows}}
    return rows, manifest, sealed


def bind_migration_manifest(manifest, bundle_id):
    result = copy.deepcopy(manifest)
    result["target_bundle_id"] = bundle_id
    for record in result["records"]:
        record["v2_result"]["bundle_id"] = bundle_id
    return result
