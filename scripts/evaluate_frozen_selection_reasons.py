"""Replay selection reasons on the sealed 37-group real-photo baseline.

The frozen labels are model-provisional, not human truth.  This evaluator checks
evidence fidelity and highlights where a faithful explanation can still explain
a provisionally questionable keeper.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.selection_reason import explain_group, faithfulness


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _phase_evidence(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    phases = snapshot["decision"]["phase_selection"]["phases"]
    for phase in phases:
        for index, evidence in phase.get("evidence", {}).items():
            found[int(index)] = evidence
    return found


def _case(snapshot: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    by_id = {int(frame["id"]): frame for frame in mapping["frames"]}
    member_ids = [int(value) for value in snapshot["member_ids"]]
    scores = {
        int(index): float(value) for index, value in
        snapshot["decision"]["phase_selection"]["utility_scores"].items()
    }
    details = _phase_evidence(snapshot)
    keepers = {int(value) for value in snapshot["keepers"]}
    primary = int(snapshot["keep_file_id"])
    primary_index = member_ids.index(primary)
    primary_score = scores[primary_index]
    members = []
    for index, file_id in enumerate(member_ids):
        evidence = details.get(index, {})
        quality = evidence.get("quality")
        quality_meta = {}
        if evidence.get("face_clarity") is not None:
            quality_meta["face_quality"] = evidence["face_clarity"]
        utility = scores[index]
        members.append({
            "file_id": file_id,
            "basename": f"{mapping['date']}-{mapping['group_alias']}-{by_id[file_id]['alias']}",
            "is_keep": int(file_id in keepers),
            "decision": "KEEP" if file_id in keepers else "MAYBE",
            "reason": "GROUP_KEEPER" if file_id == primary else "FROZEN_BASELINE",
            "quality_score": quality * 100 if quality is not None else None,
            "quality_meta": json.dumps(quality_meta),
            "evidence_json": json.dumps({
                "utility_score": utility,
                "pair_margin": primary_score - utility,
            }),
            "width": None,
            "height": None,
            "face_count": 0,
        })
    return {
        "case_id": f"{mapping['date']}:{mapping['group_alias']}",
        "group": {
            "group_id": int(mapping["group_id"]),
            "group_type": mapping["group_type"],
            "keep_file_id": primary,
            "decision_state": snapshot["decision_state"],
        },
        "members": members,
    }


def _source_evidence_ok(reason: dict[str, Any], members: list[dict[str, Any]]) -> tuple[int, int]:
    rows = {int(member["file_id"]): member for member in members}
    checked = valid = 0
    for item in reason["evidence"]:
        source = item["source"]
        if source == "group_members.is_keep":
            checked += 1
            row = rows.get(int(item["subject_file_id"]))
            valid += bool(row) and bool(row["is_keep"]) == bool(item["value"])
        elif source == "group_members.evidence_json.pair_margin":
            checked += 1
            row = rows.get(int(item["subject_file_id"]))
            value = json.loads(row["evidence_json"])["pair_margin"] if row else None
            valid += value is not None and round(float(value), 6) == item["value"]
        elif source == "features.quality_score":
            checked += 1
            left = rows.get(int(item["subject_file_id"]))
            right = rows.get(int(item["compared_to_file_id"]))
            value = item["value"]
            valid += bool(left and right) and all((
                round(float(left["quality_score"]), 6) == value["subject"],
                round(float(right["quality_score"]), 6) == value["compared_to"],
                round(abs(float(left["quality_score"] - right["quality_score"])), 6)
                == value["delta"],
            ))
        elif source == "features.quality_meta.face_quality":
            checked += 1
            left = rows.get(int(item["subject_file_id"]))
            right = rows.get(int(item["compared_to_file_id"]))
            left_value = json.loads(left["quality_meta"]).get("face_quality") if left else None
            right_value = json.loads(right["quality_meta"]).get("face_quality") if right else None
            value = item["value"]
            valid += left_value is not None and right_value is not None and all((
                round(float(left_value), 6) == value["subject"],
                round(float(right_value), 6) == value["compared_to"],
                round(abs(float(left_value - right_value)), 6) == value["delta"],
            ))
    return checked, valid


def evaluate(root: Path) -> dict[str, Any]:
    evaluation = _json(root / "holdout/evaluation/summary.json")
    metric_rows = {
        (row["day"], row["group_alias"]): row for row in evaluation["rows"]
    }
    cases = []
    labels = {}
    for day in ("2026-03-29", "2026-05-27"):
        mappings = _json(root / f"holdout/panels/{day}/mapping.json")
        by_members = {
            frozenset(int(frame["id"]) for frame in item["frames"]): item
            for item in mappings
        }
        day_labels = _json(root / f"holdout/blind-labels/{day}/labels.json")
        labels.update({(day, item["group_alias"]): item for item in day_labels["groups"]})
        snapshots = _json(root / f"holdout/replay/{day}/A_off/snapshot.json")
        for snapshot in snapshots:
            mapping = by_members[frozenset(int(value) for value in snapshot["member_ids"])]
            cases.append(_case(snapshot, mapping))

    decisions = [
        (case["case_id"], member["file_id"], member["is_keep"], member["decision"])
        for case in cases for member in case["members"]
    ]
    before = _digest(decisions)
    explained = abstained = claims = linked_claims = 0
    source_checked = source_valid = issue_groups = explained_issue_groups = 0
    endorsed = close_calls = endorsed_issue_groups = 0
    regime = {}
    examples = []
    for case in cases:
        original = copy.deepcopy(case)
        reason = explain_group(case["group"], case["members"])
        assert case == original
        linked = faithfulness(reason)
        claims += linked["claim_count"]
        linked_claims += linked["faithful_claim_count"]
        checked, valid = _source_evidence_ok(reason, case["members"])
        source_checked += checked
        source_valid += valid
        is_explained = reason["status"] == "explained"
        is_endorsed = reason["recommendation"]["code"] == "REVIEW_EXISTING_KEEPERS"
        is_close_call = reason["recommendation"]["code"] == "REVIEW_CLOSE_CALL"
        explained += is_explained
        abstained += not is_explained
        endorsed += is_endorsed
        close_calls += is_close_call
        day, alias = case["case_id"].split(":")
        row = metric_rows[(day, alias)]
        stratum = row["regime"]
        bucket = regime.setdefault(stratum, {"groups": 0, "explained": 0, "abstained": 0})
        bucket["groups"] += 1
        bucket["explained"] += is_explained
        bucket["abstained"] += not is_explained
        metrics = row["metrics"]["A_off"]
        has_issue = any(metrics[key] for key in (
            "phase_misses", "quality_wrong", "selected_unacceptable",
        ))
        issue_groups += has_issue
        explained_issue_groups += has_issue and is_explained
        endorsed_issue_groups += has_issue and is_endorsed
        if len(examples) < 8 or has_issue:
            examples.append({
                "case_id": case["case_id"],
                "regime": stratum,
                "label_status": row["label_status"],
                "provisional_issue": has_issue,
                "provisional_metrics": {
                    key: metrics[key] for key in (
                        "phase_misses", "quality_wrong", "selected_unacceptable",
                    )
                },
                "reason": reason,
            })
    after = _digest([
        (case["case_id"], member["file_id"], member["is_keep"], member["decision"])
        for case in cases for member in case["members"]
    ])
    return {
        "schema_version": 1,
        "source": "sealed 37-group A_off real-photo replay",
        "source_root": root.name,
        "label_status": evaluation["label_status"],
        "human_truth": evaluation["human_truth"],
        "safety_validated": evaluation["safety_validated"],
        "metrics": {
            "groups": len(cases),
            "photos": sum(len(case["members"]) for case in cases),
            "keepers": sum(member["is_keep"] for case in cases for member in case["members"]),
            "explained": explained,
            "coverage": explained / len(cases),
            "abstained": abstained,
            "abstention_rate": abstained / len(cases),
            "endorsed_existing_keeper_groups": endorsed,
            "close_call_no_endorsement_groups": close_calls,
            "claims": claims,
            "evidence_linked_claims": linked_claims,
            "evidence_link_integrity": linked_claims / claims if claims else 1.0,
            "translated_evidence_items_checked": source_checked,
            "translated_evidence_items_valid": source_valid,
            "translated_evidence_self_consistency": (
                source_valid / source_checked if source_checked else 1.0
            ),
            "input_decision_digest_before": before,
            "input_decision_digest_after": after,
            "function_input_unchanged": before == after,
            "provisional_issue_groups": issue_groups,
            "explained_provisional_issue_groups": explained_issue_groups,
            "endorsed_provisional_issue_groups": endorsed_issue_groups,
        },
        "by_regime": regime,
        "interpretation": (
            "The replay translates frozen snapshots into review-shaped rows, then checks that "
            "generated statements resolve to those translated values. This is conversion "
            "self-consistency, not a direct live-SQLite fidelity proof. The sealed labels are "
            "model-provisional, so issue overlap is a warning signal, not human-truth accuracy."
        ),
        "examples": examples,
    }


def markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    strata = "\n".join(
        f"- `{name}`：{values['explained']}/{values['groups']} 组生成事实说明，"
        f"{values['abstained']} 组弃权。"
        for name, values in report["by_regime"].items()
    )
    return f"""# 选片理由：冻结真实照片回放

数据：`{report['source']}`，{metrics['groups']} 组、{metrics['photos']} 张照片、{metrics['keepers']} 个既有 keeper。
标签状态：`{report['label_status']}`；不是人工真值，不能据此认证偏好准确率或删除安全。

- 事实说明覆盖：{metrics['explained']}/{metrics['groups']} = {metrics['coverage']:.1%}。
- 弃权：{metrics['abstained']}/{metrics['groups']} = {metrics['abstention_rate']:.1%}。
- 近差距说明但不背书：{metrics['close_call_no_endorsement_groups']} 组。
- 对既有 keeper 给出背书：{metrics['endorsed_existing_keeper_groups']} 组。
- 证据引用完整：{metrics['evidence_linked_claims']}/{metrics['claims']}。
- 转换后字段自洽：{metrics['translated_evidence_items_valid']}/{metrics['translated_evidence_items_checked']}；不是 live SQLite 忠实度证明。
- 函数未修改输入中的决策字段：`{metrics['function_input_unchanged']}`；线上只读性另由 SQLite/API 回归测试覆盖。
- provisional 标签提示有问题的组：{metrics['provisional_issue_groups']}；其中 {metrics['explained_provisional_issue_groups']} 组得到事实说明，但背书为 {metrics['endorsed_provisional_issue_groups']} 组。

## 分层

{strata}

## 结论

真实回放表明这层可以低成本地把转换后的结构化信号组织成可追溯说明，但现有信号普遍不足以支持“这张就是最佳”的结论。当前实现对近差距组只展示取舍并明确不背书；矛盾或证据不足时弃权。下一步若要判断这些文字是否真的帮助人选片，需要独立人工标注或真实用户复核，不能继续用 sealed model-provisional 标签自证。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate(args.root)
    args.json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.report:
        args.report.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
