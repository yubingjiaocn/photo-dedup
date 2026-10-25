"""Deterministic, evidence-bound explanations for persisted keeper choices.

This shadow presentation layer consumes only persisted fields, never opens
media, never calls a model, and never proposes a different keeper.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
GENERATOR = "deterministic-selection-reason-v1"
EXPLANATION_CLOSE_MARGIN = 0.08


def _finite(value: Any, low: float | None = None,
            high: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if low is not None and number < low:
        return None
    if high is not None and number > high:
        return None
    return number


def _object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _id(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _name(member: Mapping[str, Any]) -> str:
    basename = member.get("basename")
    if isinstance(basename, str) and basename.strip():
        return basename.strip()
    file_id = _id(member.get("file_id"))
    return f"照片 #{file_id}" if file_id is not None else "未知照片"


def _metrics(member: Mapping[str, Any]) -> dict[str, float | str | None]:
    meta = _object(member.get("quality_meta"))
    evidence = _object(member.get("evidence_json"))
    eye = _object(meta.get("eye_detection"))
    eye_status = eye.get("status")
    if eye_status not in {
        "NO_FACE", "ALL_OPEN", "HAS_CLOSED_EYES", "HAS_MAYBE", "UNKNOWN",
    }:
        eye_status = None
    width, height = _id(member.get("width")), _id(member.get("height"))
    return {
        "quality_score": _finite(member.get("quality_score"), 0, 100),
        "clipiqa": _finite(meta.get("clipiqa"), 0, 1),
        "face_quality": _finite(meta.get("face_quality"), 0, 1),
        "resolution_pixels": float(width * height) if width and height else None,
        "utility_score": _finite(evidence.get("utility_score")),
        "pair_margin": _finite(evidence.get("pair_margin")),
        "eye_status": eye_status,
    }


class _Evidence:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, code: str, source: str, *, subject: int | None = None,
            compared_to: int | None = None, value: Any = None) -> str:
        evidence_id = f"E{len(self.items) + 1:03d}"
        item = {"id": evidence_id, "code": code, "source": source}
        if subject is not None:
            item["subject_file_id"] = subject
        if compared_to is not None:
            item["compared_to_file_id"] = compared_to
        if value is not None:
            item["value"] = value
        self.items.append(item)
        return evidence_id


def _claim(code: str, text: str, refs: Sequence[str]) -> dict[str, Any]:
    return {"code": code, "text": text, "evidence_codes": list(refs)}


_FOCUS_PRIORITY = {
    "RIVAL_EYE_STATE_ADVANTAGE": 100,
    "EYE_STATE_ADVANTAGE": 98,
    "RIVAL_CLOSED_EYES_FLAG": 96,
    "RIVAL_FACE_QUALITY_ADVANTAGE": 90,
    "FACE_QUALITY_ADVANTAGE": 88,
    "RIVAL_QUALITY_SCORE_ADVANTAGE": 72,
    "QUALITY_SCORE_ADVANTAGE": 70,
    "RIVAL_CLIPIQA_ADVANTAGE": 68,
    "CLIPIQA_ADVANTAGE": 66,
    "RIVAL_RESOLUTION_ADVANTAGE": 45,
}


def _review_focus(support: Sequence[Mapping[str, Any]],
                  tradeoffs: Sequence[Mapping[str, Any]],
                  evidence_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return at most three concrete places to inspect, not another metric dump."""
    registry = {
        item.get("id"): item for item in evidence_items
        if isinstance(item.get("id"), str)
    }
    candidates: list[dict[str, Any]] = []
    for kind, claims in (("support", support), ("tradeoff", tradeoffs)):
        for claim in claims:
            code = str(claim.get("code") or "")
            priority = _FOCUS_PRIORITY.get(code)
            if priority is None:
                continue
            refs = [registry[ref] for ref in claim.get("evidence_codes", [])
                    if ref in registry]
            if not refs:
                continue
            subjects = sorted({int(item["subject_file_id"]) for item in refs
                               if _id(item.get("subject_file_id")) is not None})
            compared = sorted({int(item["compared_to_file_id"]) for item in refs
                               if _id(item.get("compared_to_file_id")) is not None})
            region = "face" if ("EYE" in code or "FACE" in code) else "full_frame"
            candidates.append({
                "code": code,
                "text": claim["text"],
                "kind": kind,
                "priority": priority,
                "inspect": {
                    "file_ids": subjects,
                    "compare_to_file_ids": compared,
                    "region": region,
                },
                "evidence_codes": list(claim.get("evidence_codes", [])),
            })
    candidates.sort(key=lambda item: (-item["priority"], item["code"],
                                      item["inspect"]["file_ids"]))
    highlights = candidates[:3]
    return {
        "status": "focused" if highlights else "no_clear_difference",
        "summary": (
            "先看下面标出的局部差异；通常无需重新无重点地扫完整组图。"
            if highlights else
            "现有可靠证据没有给出明确区分点；请按整体偏好选择，不必逐项寻找不存在的差异。"
        ),
        "highlights": highlights,
        "max_highlights": 3,
    }


def _base_result(status: str, recommendation: dict[str, Any],
                 support: list[dict[str, Any]], tradeoffs: list[dict[str, Any]],
                 confidence: dict[str, Any], evidence: _Evidence,
                 keeper_ids: list[int]) -> dict[str, Any]:
    focus = _review_focus(support, tradeoffs, evidence.items)
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "mode": "shadow_review_only",
        "status": status,
        "review_focus": focus,
        "recommendation": recommendation,
        "support": support,
        "tradeoffs": tradeoffs,
        "confidence": confidence,
        "human_confirmation": {
            "required": True,
            "code": "HUMAN_CONFIRMATION_REQUIRED",
            "text": "请查看原图复核闭眼检测，并确认表情、遮挡、构图与叙事取舍。",
        },
        "evidence": evidence.items,
        "invariants": {
            "keeper_file_ids": keeper_ids,
            "changes_keeper": False,
            "changes_sorting": False,
            "changes_routing": False,
            "deletion_authority": "none",
            "model_calls": 0,
        },
    }


def explain_group(group: Mapping[str, Any],
                  members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Explain the already-persisted keeper set without changing it."""
    rows = [dict(member) for member in members]
    valid = [row for row in rows if _id(row.get("file_id")) is not None]
    keepers = [row for row in valid if bool(row.get("is_keep"))]
    rivals = [row for row in valid if not bool(row.get("is_keep"))]
    keeper_ids = [int(row["file_id"]) for row in keepers]
    persisted = _id(group.get("keep_file_id"))
    primary = next((row for row in keepers if row["file_id"] == persisted), None)
    if primary is None and persisted is None:
        primary = next((row for row in keepers
                        if row.get("reason") == "GROUP_KEEPER"), None)
    if primary is None and persisted is None and len(keepers) == 1:
        primary = keepers[0]
    primary_is_persisted = bool(
        primary is not None and persisted is not None
        and int(primary["file_id"]) == persisted
    )

    evidence = _Evidence()
    keeper_refs = [evidence.add(
        "PERSISTED_KEEPER", "group_members.is_keep",
        subject=int(row["file_id"]), value=True,
    ) for row in keepers]
    support: list[dict[str, Any]] = []
    tradeoffs: list[dict[str, Any]] = []
    contradiction = False

    protected = {
        "LOGICAL_PHASE_KEEPER": (
            "LOGICAL_PHASE_PROTECTION",
            "是已记录的逻辑阶段额外保留项；该代码只表示保护性保留，不证明叙事语义。",
        ),
        "POSE_VARIANT_KEEPER": (
            "POSE_VARIANT_PROTECTION",
            "是已记录的姿态变化保护项；姿态证据没有语义阶段或删除权限。",
        ),
    }
    for row in keepers:
        reason = str(row.get("reason") or "")
        if reason in protected:
            code, text = protected[reason]
            ref = evidence.add(code, "group_members.reason",
                               subject=int(row["file_id"]), value=reason)
            support.append(_claim(code, f"{_name(row)} {text}", [ref]))

    exact = [row for row in rivals if row.get("reason") == "BYTE_IDENTICAL"]
    comparison_rivals = [row for row in rivals if row.get("reason") != "BYTE_IDENTICAL"]
    primary_metrics = _metrics(primary) if primary is not None else {}
    ranked_rivals = [(row, _metrics(row)) for row in comparison_rivals]
    ranked_rivals.sort(key=lambda pair: (
        pair[1]["utility_score"] is not None,
        pair[1]["utility_score"] if pair[1]["utility_score"] is not None else float("-inf"),
        -int(pair[0]["file_id"]),
    ), reverse=True)

    if primary is not None and primary_is_persisted and ranked_rivals:
        rival, rival_metrics = ranked_rivals[0]
        primary_id, rival_id = int(primary["file_id"]), int(rival["file_id"])
        margin = rival_metrics["pair_margin"]
        if margin is not None:
            ref = evidence.add(
                "PERSISTED_PAIR_MARGIN",
                "group_members.evidence_json.pair_margin",
                subject=rival_id, compared_to=primary_id,
                value=round(float(margin), 6),
            )
            if margin > 1e-9:
                support.append(_claim(
                    "UTILITY_MARGIN",
                    f"相对所选未保留对手 {_name(rival)}，持久化效用差为 {margin:.3f}。",
                    [ref],
                ))
                if margin < EXPLANATION_CLOSE_MARGIN:
                    tradeoffs.append(_claim(
                        "LOW_UTILITY_MARGIN",
                        "解释层将效用差低于 0.080 视为接近；请直接人工比较，"
                        "这不等同于决策层的 LOW_MARGIN 判定。",
                        [ref],
                    ))
            elif margin < 0:
                contradiction = True
                tradeoffs.append(_claim(
                    "NEGATIVE_PERSISTED_MARGIN",
                    f"{_name(rival)} 的持久化效用高出主保留项 {abs(margin):.3f}；理由层拒绝解释这一矛盾。",
                    [ref],
                ))
        # The closest utility rival is not guaranteed to hold every measurable
        # advantage.  Inspect all rivals, but emit only the strongest tradeoff
        # (or one support comparison) per metric to keep the UI bounded.
    if primary is not None and ranked_rivals:
        primary_id = int(primary["file_id"])
        closest, closest_metrics = ranked_rivals[0]
        for key, code, label, threshold in (
            ("quality_score", "QUALITY_SCORE_ADVANTAGE", "缓存质量分", 0.5),
            ("clipiqa", "CLIPIQA_ADVANTAGE", "CLIP-IQA", 0.01),
            ("face_quality", "FACE_QUALITY_ADVANTAGE", "人脸局部质量值", 0.01),
        ):
            left = primary_metrics.get(key)
            comparable = [
                (row, metrics) for row, metrics in ranked_rivals
                if metrics.get(key) is not None
            ]
            if left is None or not comparable:
                continue
            strongest, strongest_metrics = max(
                comparable, key=lambda pair: float(pair[1][key])
            )
            strongest_delta = float(left) - float(strongest_metrics[key])
            if strongest_delta <= -threshold:
                rival, rival_metrics, delta = strongest, strongest_metrics, strongest_delta
                statement_code = f"RIVAL_{code}"
                statement_text = (
                    f"对手 {_name(rival)} 的{label}高 {abs(delta):.3f}，"
                    "这是人工复核时应检查的取舍。"
                )
                destination = tradeoffs
            else:
                right = closest_metrics.get(key)
                if right is None:
                    continue
                delta = float(left) - float(right)
                if delta < threshold:
                    continue
                rival, rival_metrics = closest, closest_metrics
                statement_code = code
                statement_text = f"主保留项的{label}比 {_name(rival)} 高 {delta:.3f}。"
                destination = support
            rival_id = int(rival["file_id"])
            right = float(rival_metrics[key])
            source = ("features.quality_score" if key == "quality_score"
                      else f"features.quality_meta.{key}")
            subject_id, compared_id = (
                (rival_id, primary_id)
                if statement_code.startswith("RIVAL_")
                else (primary_id, rival_id)
            )
            subject_value, compared_value = (
                (right, float(left))
                if subject_id == rival_id
                else (float(left), right)
            )
            ref = evidence.add(
                statement_code, source, subject=subject_id,
                compared_to=compared_id, value={
                    "subject": round(subject_value, 6),
                    "compared_to": round(compared_value, 6),
                    "delta": round(abs(delta), 6),
                },
            )
            destination.append(_claim(statement_code, statement_text, [ref]))

        kp = primary_metrics.get("resolution_pixels")
        resolution_rivals = [
            (row, metrics) for row, metrics in ranked_rivals
            if metrics.get("resolution_pixels") is not None
        ]
        rival, rival_metrics = max(
            resolution_rivals,
            key=lambda pair: float(pair[1]["resolution_pixels"]),
            default=(None, {}),
        )
        rp = rival_metrics.get("resolution_pixels")
        if kp is not None and rp is not None and kp > 0 and rp / kp >= 1.05:
            rival_id = int(rival["file_id"])
            ratio = float(rp / kp)
            ref = evidence.add(
                "RIVAL_RESOLUTION_ADVANTAGE", "files.width*files.height",
                subject=rival_id, compared_to=primary_id,
                value={"subject_pixels": int(rp), "compared_to_pixels": int(kp),
                       "ratio": round(ratio, 6)},
            )
            tradeoffs.append(_claim(
                "RIVAL_RESOLUTION_ADVANTAGE",
                f"对手 {_name(rival)} 的像素数是主保留项的 {ratio:.2f} 倍。",
                [ref],
            ))

    if primary is not None and exact:
        primary_id = int(primary["file_id"])
        refs = [evidence.add(
            "BYTE_IDENTICAL_DECISION_EVIDENCE", "group_members.reason",
            subject=int(row["file_id"]), compared_to=primary_id,
            value="BYTE_IDENTICAL",
        ) for row in exact]
        support.append(_claim(
            "BYTE_IDENTICAL_PEERS",
            f"{len(exact)} 个未保留成员被既有决策层记录为字节完全相同；理由层不新增删除权限。",
            refs,
        ))

    if primary is not None:
        primary_id = int(primary["file_id"])
        primary_eye = primary_metrics.get("eye_status")
        closed_rivals = [
            row for row, metrics in ranked_rivals
            if metrics.get("eye_status") == "HAS_CLOSED_EYES"
        ]
        open_rivals = [
            row for row, metrics in ranked_rivals
            if metrics.get("eye_status") == "ALL_OPEN"
        ]
        if closed_rivals and primary_eye in {"ALL_OPEN", "HAS_MAYBE"}:
            rival = closed_rivals[0]
            rival_id = int(rival["file_id"])
            code = ("EYE_STATE_ADVANTAGE" if primary_eye == "ALL_OPEN"
                    else "RIVAL_CLOSED_EYES_FLAG")
            ref = evidence.add(
                code, "features.quality_meta.eye_detection.status",
                subject=primary_id, compared_to=rival_id,
                value={"subject": primary_eye, "compared_to": "HAS_CLOSED_EYES"},
            )
            support.append(_claim(
                code,
                f"本地闭眼检测器将 {_name(rival)} 标为疑似双眼闭合；"
                f"主保留项状态为 {primary_eye}，仍需查看原图确认。",
                [ref],
            ))
        elif primary_eye == "HAS_CLOSED_EYES" and open_rivals:
            rival = open_rivals[0]
            rival_id = int(rival["file_id"])
            ref = evidence.add(
                "RIVAL_EYE_STATE_ADVANTAGE",
                "features.quality_meta.eye_detection.status",
                subject=rival_id, compared_to=primary_id,
                value={"subject": "ALL_OPEN", "compared_to": primary_eye},
            )
            tradeoffs.append(_claim(
                "RIVAL_EYE_STATE_ADVANTAGE",
                f"本地闭眼检测器将主保留项标为疑似双眼闭合，"
                f"而 {_name(rival)} 为 ALL_OPEN；这是必须人工复核的反向证据。",
                [ref],
            ))

    if not tradeoffs:
        ref = evidence.add(
            "NO_VERIFIABLE_RIVAL_ADVANTAGE", "derived.comparable_rival_metrics",
            value={"rival_count": len(rivals)},
        )
        tradeoffs.append(_claim(
            "RIVAL_ADVANTAGE_ABSTAIN",
            "现有结构化字段不足以确认对手优势；这不等于对手没有优势。",
            [ref],
        ))

    protection_codes = {"LOGICAL_PHASE_PROTECTION", "POSE_VARIANT_PROTECTION"}
    comparison_support = [
        item for item in support if item["code"] not in protection_codes
    ]
    substantive = bool(comparison_support)
    low_margin = any(
        (margin := _metrics(row)["pair_margin"]) is not None
        and 0 <= margin < EXPLANATION_CLOSE_MARGIN
        for row in comparison_rivals
    ) if primary_is_persisted else False
    explained = bool(keepers) and primary is not None and substantive and not contradiction
    if not explained:
        ref = evidence.add(
            "EXPLANATION_EVIDENCE_INSUFFICIENT", "derived.explanation_gate",
            value={"keeper_count": len(keepers),
                   "primary_keeper_resolved": primary is not None,
                   "support_count": len(support), "contradiction": contradiction},
        )
        if not support:
            support.append(_claim(
                "SUPPORT_ABSTAIN", "没有足够的可核验支持项可用于生成选片理由。", [ref],
            ))
        confidence = _claim(
            "EXPLANATION_CONFIDENCE_UNKNOWN", "解释证据置信度：unknown；必须人工确认。", [ref],
        )
        confidence.update(level="unknown", score=None)
        recommendation = _claim(
            "ABSTAIN_INSUFFICIENT_EVIDENCE",
            "结构化证据不足或相互矛盾，理由层弃权；请直接人工比较，不据此改变现有选择。",
            [ref],
        )
        return _base_result(
            "abstained",
            recommendation,
            support, tradeoffs, confidence, evidence, keeper_ids,
        )

    codes = {item["code"] for item in comparison_support}
    margins = ([_metrics(row)["pair_margin"] for row in comparison_rivals]
               if primary_is_persisted else [])
    margins = [value for value in margins if value is not None]
    minimum_margin = min(margins) if margins else None
    if "BYTE_IDENTICAL_PEERS" in codes and not comparison_rivals:
        level, score = "high", 0.90
    elif (len(codes) >= 2 and minimum_margin is not None
          and minimum_margin >= EXPLANATION_CLOSE_MARGIN):
        level, score = "medium", 0.72
    else:
        level, score = "low", 0.45
    confidence_refs = list(dict.fromkeys(
        ref for item in comparison_support for ref in item["evidence_codes"]
    ))
    confidence = _claim(
        f"EXPLANATION_CONFIDENCE_{level.upper()}",
        f"解释证据置信度：{level}（{score:.2f}）；它不是审美正确率或删除安全概率。",
        confidence_refs,
    )
    confidence.update(level=level, score=score)
    names = "、".join(_name(row) for row in keepers)
    close_call = low_margin and "BYTE_IDENTICAL_PEERS" not in codes
    recommendation = _claim(
        "REVIEW_CLOSE_CALL" if close_call else "REVIEW_EXISTING_KEEPERS",
        (
            f"现有选择为：{names}；与对手差距很小，理由层不为该选择背书，请直接人工比较。"
            if close_call else
            f"建议人工优先复核并暂按现有选择保留：{names}。"
        ),
        keeper_refs,
    )
    return _base_result(
        "explained",
        recommendation,
        support, tradeoffs, confidence, evidence, keeper_ids,
    )


def faithfulness(reason: Mapping[str, Any]) -> dict[str, int | float]:
    """Count human-readable claims whose evidence IDs resolve."""
    registry = {
        item.get("id") for item in reason.get("evidence", [])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    claims = [reason.get("recommendation"), *reason.get("support", []),
              *reason.get("tradeoffs", []), reason.get("confidence")]
    claims = [item for item in claims if isinstance(item, Mapping)]
    faithful = sum(bool(item.get("evidence_codes")) and all(
        ref in registry for ref in item["evidence_codes"]
    ) for item in claims)
    total = len(claims)
    return {"claim_count": total, "faithful_claim_count": faithful,
            "faithfulness": faithful / total if total else 1.0}
