"""Summarize the bounded CV + VLM semantic-observer experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm_semantic import sanitize_group

PROVISIONAL_EYE_LABELS = {
    ("2026-05-27-G014", "F01"): "CLOSED",
    ("2026-05-27-G014", "F02"): "CLOSED",
    ("2026-05-27-G014", "F03"): "OPEN",
    ("2026-05-27-G015", "F01"): "OPEN",
    ("2026-05-27-G015", "F02"): "OPEN",
    ("2026-05-27-G015", "F03"): "CLOSED",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(cv_path: Path, vlm_paths: list[Path]) -> dict:
    cv = _json(cv_path)
    aliases = {
        f"{group['day']}-{group['group_alias']}": [
            frame["frame_alias"] for frame in group["frames"]
        ]
        for group in cv["groups_detail"]
    }
    cv_eye = {}
    for group in cv["groups_detail"]:
        group_id = f"{group['day']}-{group['group_alias']}"
        for frame in group["frames"]:
            eye = next(item for item in frame["observations"]
                       if item["code"].startswith("EYE_"))
            cv_eye[(group_id, frame["frame_alias"])] = eye["state"]

    eye_counts = {
        "provisional_closed": 0, "closed_detected": 0, "closed_abstained": 0,
        "provisional_open": 0, "open_detected": 0, "open_abstained": 0,
        "false_closed_on_provisional_open": 0,
    }
    for key, label in PROVISIONAL_EYE_LABELS.items():
        state = cv_eye[key]
        if label == "CLOSED":
            eye_counts["provisional_closed"] += 1
            if state == "HAS_CLOSED_EYES":
                eye_counts["closed_detected"] += 1
            else:
                eye_counts["closed_abstained"] += 1
        else:
            eye_counts["provisional_open"] += 1
            if state == "ALL_OPEN":
                eye_counts["open_detected"] += 1
            else:
                eye_counts["open_abstained"] += 1
            if state == "HAS_CLOSED_EYES":
                eye_counts["false_closed_on_provisional_open"] += 1

    sanitized = []
    vlm_metrics = {
        "model_calls": 0, "images": 0, "groups": set(), "image_bytes": 0,
        "input_tokens": 0, "output_tokens": 0, "accepted_claims": 0,
        "rejected_claims": 0,
    }
    for path in vlm_paths:
        report = _json(path)
        vlm_metrics["model_calls"] += report["model_calls"]
        vlm_metrics["images"] += report["image_count"]
        vlm_metrics["image_bytes"] += report["total_image_bytes"]
        vlm_metrics["input_tokens"] += report["usage"]["inputTokens"]
        vlm_metrics["output_tokens"] += report["usage"]["outputTokens"]
        for group in report["observations"]["groups"]:
            group_id = group["group_id"]
            clean = sanitize_group(group, aliases[group_id])
            sanitized.append(clean)
            vlm_metrics["groups"].add(group_id)
            vlm_metrics["accepted_claims"] += len(clean["accepted"])
            vlm_metrics["rejected_claims"] += len(clean["rejected"])

    input_cost = vlm_metrics["input_tokens"] * 2 / 1_000_000
    output_cost = vlm_metrics["output_tokens"] * 10 / 1_000_000
    unique_groups = len(vlm_metrics["groups"])
    vlm_metrics["groups"] = len(vlm_metrics["groups"])
    vlm_metrics["estimated_usd_at_catalog_rates"] = input_cost + output_cost
    return {
        "schema_version": 1,
        "mode": "shadow_review_only",
        "sample": {
            "cv_groups": cv["groups"], "cv_frames": cv["frames"],
            "selection": cv["sample_policy"],
            "label_status": "sealed_blind_model_provisional",
            "human_truth": False,
        },
        "cv": {
            "runtime_seconds": cv["runtime_seconds"],
            "eye_status_counts": cv["eye_status_counts"],
            "provisional_eye_descriptive_overlap": eye_counts,
            "model": cv["model"],
        },
        "vlm": vlm_metrics,
        "resolution_gate": {
            "recommended_minimum": "2048",
            "1024_scope": "broad pose/action only; subtle expression less stable",
            "2048_scope": "broad pose/action/expression with labelled face crops",
            "unique_groups": unique_groups,
        },
        "sanitized_groups": sanitized,
        "known_findings": [
            "YuNet selected illustrated-character regions as faces in one group; VLM noticed the crop mismatch.",
            "One VLM claim cited a detected face crop and was rejected as delegated-CV contamination.",
            "Low-confidence VLM claims are excluded from reason evidence.",
        ],
        "invariants": {
            "changes_keeper": False, "changes_sorting": False,
            "changes_routing": False, "deletion_authority": "none",
        },
    }


def markdown(report: dict) -> str:
    cv, vlm = report["cv"], report["vlm"]
    eye = cv["provisional_eye_descriptive_overlap"]
    return f"""# CV + bounded VLM semantic observer probe

This is a shadow/review-only experiment. Frozen labels are `sealed_blind_model_provisional`, not human truth.

## Bounded sample

- Local CV: {report['sample']['cv_groups']} groups / {report['sample']['cv_frames']} frames, selected as all seven provisional-issue groups plus the only multi group.
- VLM: {vlm['groups']} unique groups, {vlm['images']} contact-sheet images, {vlm['model_calls']} Sonnet calls.
- VLM payload: {vlm['image_bytes']} bytes; {vlm['input_tokens']} input + {vlm['output_tokens']} output tokens; catalog-rate estimate `${vlm['estimated_usd_at_catalog_rates']:.4f}`.
- Resolution gate: use 2048px. 1024px retained broad pose/action but weakened subtle expression comparison.

## CV result

- Eye statuses over the 26 frames: `{json.dumps(cv['eye_status_counts'], sort_keys=True)}`.
- Against the six eye-specific model-provisional frame labels only: CLOSED {eye['closed_detected']}/{eye['provisional_closed']} detected; OPEN {eye['open_detected']}/{eye['provisional_open']} promoted, with {eye['open_abstained']} abstentions and {eye['false_closed_on_provisional_open']} false CLOSED flags.
- This is descriptive overlap, not calibrated accuracy. The detector is deliberately asymmetric: it can flag likely closed eyes while abstaining on many open-eye frames.

## VLM result

- Accepted medium/high direct-visual claims: {vlm['accepted_claims']}.
- Rejected low-confidence, invalid, or delegated-CV claims: {vlm['rejected_claims']}.
- Sonnet recovered useful pose/action and scene-information differences on the five CV-unresolved groups.
- It also caught a real routing defect: in one group YuNet cropped an illustrated character instead of the photographed person. Resolution cannot repair the wrong subject crop.

## Decision

Keep the hybrid direction. Local CV should own eye state, face/local quality, exposure, and edge-proximity observations. A bounded VLM can add broad pose, expression, camera-engagement, interaction, and scene-information observations after CV routing. Do not send every photo: use one 2048px labelled sheet per unresolved group, include face crops, accept only medium/high schema-valid claims, and preserve human review. No result here expands deletion or keeper authority.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, action="append", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.cv, args.vlm)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"cv": report["cv"], "vlm": report["vlm"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
