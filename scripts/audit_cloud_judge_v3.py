"""Read-only receipt/hash/decision audit and bounded batch summary; never invokes models."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cloud_judge_batch as b
from src import two_pass_review as tp


def run(output):
    output = Path(output).resolve()
    b.verify(
        (output / "completion.json").exists() or (output / "stopped.json").exists(),
        "finish or stop the batch before final audit",
    )
    report = b.read(output / "report.json")
    frozen = b.read(ROOT / "artifacts/cloud-judge-batch/frozen.json")
    b.verify_snapshot(frozen)
    continuation = b.read(output / "continuation.json")
    source = Path(continuation["source"])
    for name, expected in continuation["source_hashes"].items():
        b.verify(
            tp.sha((source / name).read_bytes()) == expected,
            "v2 evidence changed: " + name,
        )
    indexed = {g["group"]: g for g in frozen["groups"]}
    ledger = []
    for state in report["groups"]:
        if not state["reserved"]:
            continue
        gid = state["group"]
        case = output / gid
        b.verify(state["external_calls"] == 1, "expected one consumed cloud slot")
        for name, expected in state["artifact_hashes"].items():
            b.verify(
                tp.sha((case / name).read_bytes()) == expected,
                "artifact changed: " + gid + "/" + name,
            )
        request = b.read(case / "request.json")
        packet = b.read(case / "visual-packet.json")
        validation = b.read(case / "view-validation.json")
        b.verify(packet == request["context"]["visual_packet"], "packet mismatch")
        b.verify(
            validation["lookup"] == {v["id"]: v for v in packet["views"]},
            "view binding mismatch",
        )
        b.verify(
            set(packet["complete_frames"])
            == {tp.identity(r) for r in indexed[gid]["frames"]},
            "incomplete group",
        )
        b.verify(
            len([a for a in request["images"] if a["kind"] == "full_group_page"])
            == len(indexed[gid]["frames"]),
            "full-page coverage mismatch",
        )
        retry = None
        model_decision = "unavailable"
        recommended_count = None
        weak_observations = 0
        if (case / "provider-response.json").exists():
            response = b.read(case / "provider-response.json")
            retry = response.get("ResponseMetadata", {}).get("RetryAttempts")
            b.verify(
                retry == 0, "provider receipt indicates retries or missing metadata"
            )
            response_text = "\n".join(
                block["text"]
                for block in response.get("output", {})
                .get("message", {})
                .get("content", [])
                if "text" in block
            )
            try:
                reported = tp.strict_json(response_text)
                if isinstance(reported, dict) and reported.get("decision") in {
                    "select",
                    "attention_only",
                    "abstain",
                }:
                    model_decision = reported["decision"]
            except (tp.ReviewError, ValueError):
                pass  # Informational only: malformed responses keep unavailable.
            if state["state"] != "validated":
                rejected = output / "rejected-text"
                rejected.mkdir(exist_ok=True)
                (rejected / (gid + ".txt")).write_text(
                    "\n".join(
                        block["text"]
                        for block in response.get("output", {})
                        .get("message", {})
                        .get("content", [])
                        if "text" in block
                    )
                )
            if state["state"] == "validated":
                parsed = tp.strict_json((case / "raw.txt").read_text())
                b.verify(
                    parsed
                    == b.read(case / "raw.json")
                    == b.read(case / "validated.json"),
                    "raw altered",
                )
                judgment = tp.validate_judgment(
                    parsed,
                    indexed[gid]["frames"],
                    packet,
                    validation,
                    request["context"]["evidence"],
                    allow_visual_views=True,
                )
                b.verify(judgment == parsed, "judgment changed")
                recommended_count = len(parsed["recommended"])
                weak_observations = sum(
                    any(e["confidence"] == "low" for e in item["evidence"])
                    for item in parsed["discriminators"]
                )
        ledger.append(
            {
                "group": gid,
                "new_call": not bool(state.get("imported_from")),
                "state": state["state"],
                "decision": state["route"]
                if state["state"] == "validated"
                else "unvalidated",
                "reported_model_decision": model_decision,
                "recommended_count": recommended_count,
                "observations_with_low_evidence": weak_observations,
                "usage": state["usage"],
                "estimated_cost_usd": state["estimated_cost_usd"],
                "sdk_retries": retry,
                "reason": state["reason"],
            }
        )
    b.verify(
        len({s["group"] for s in ledger})
        == len(ledger)
        == report["external_calls_total"],
        "dispatch accounting mismatch",
    )
    b.verify(
        not any(s["decision_table_writes"] for s in report["groups"]),
        "decision write detected",
    )
    b.verify(
        round(sum(s["estimated_cost_usd"] or 0 for s in ledger), 8)
        == report["estimated_cost_usd_known"],
        "cost accounting mismatch",
    )
    for token_key in ("inputTokens", "outputTokens", "totalTokens"):
        b.verify(
            sum((s["usage"] or {}).get(token_key, 0) for s in ledger)
            == report["usage"][token_key],
            "usage accounting mismatch",
        )
    new = [s for s in ledger if s["new_call"]]
    summary = b.read(output / "summary.json")
    decision_buckets = {}
    for state in ledger:
        bucket = decision_buckets.setdefault(
            state["decision"], {"groups": 0, "validated": 0, "estimated_cost_usd": 0}
        )
        bucket["groups"] += 1
        bucket["validated"] += state["state"] == "validated"
        bucket["estimated_cost_usd"] = round(
            bucket["estimated_cost_usd"] + (state["estimated_cost_usd"] or 0), 8
        )
    summary["strata"]["decision"] = decision_buckets
    summary.update(
        {
            "new_usage": {
                k: sum((s["usage"] or {}).get(k, 0) for s in new)
                for k in ("inputTokens", "outputTokens", "totalTokens")
            },
            "new_estimated_cost_usd": round(
                sum(s["estimated_cost_usd"] or 0 for s in new), 8
            ),
            "states": report["states"],
            "validated_multi_select_groups": sum(
                (s["recommended_count"] or 0) > 1 for s in ledger
            ),
            "validated_groups_retaining_weak_observations": sum(
                s["observations_with_low_evidence"] > 0 for s in ledger
            ),
            "decisions": dict(collections.Counter(s["decision"] for s in ledger)),
            "reported_model_decisions_including_unvalidated": dict(
                collections.Counter(s["reported_model_decision"] for s in ledger)
            ),
            "max_estimated_group_cost_usd": max(
                (s["estimated_cost_usd"] or 0 for s in ledger), default=0
            ),
            "provider_receipts": report["provider_receipts"],
            "complete_frame_pages_verified": sum(
                len(indexed[s["group"]]["frames"]) for s in ledger
            ),
            "failures": [s for s in ledger if s["state"] != "validated"],
            "uncalled_groups": [
                s["group"] for s in report["groups"] if not s["reserved"]
            ],
            "cost_basis": report["cost_basis"],
            "actual_aws_bill": None,
            "canary_gate": b.read(output / "canary-gate.json"),
            "v2_evidence_unchanged": True,
            "all_consumed_artifact_hashes_verified": True,
            "source_inventories_unchanged": True,
            "sdk_retries": 0,
            "scene_strata_basis": "input scene-route hint, not human-labeled or cloud-confirmed scene",
            "visual_preference_accuracy": None,
            "visual_consistency_sample": "astra-visual-check.json",
            "independent_parent_acceptance_pending": True,
        }
    )
    summary["execution_policy"] = "execution-policy.json"
    if (output / "stopped.json").exists():
        summary["phase"] = "stopped"
    b.save(output / "summary.json", summary)
    b.save(output / "progress.json", summary)
    b.save(
        output / "deterministic-audit.json",
        {
            "passed": True,
            "consumed_groups": len(ledger),
            "v2_evidence_unchanged": True,
            "source_inventories_unchanged": True,
            "decision_table_writes": 0,
            "receipts": ledger,
        },
    )
    lines = [
        "# Contract v3 冻结批次结果",
        "",
        f"- 状态：{summary['phase']}；已消费 {len(ledger)}/194 个云端槽位；{report['states'].get('validated', 0)} 组 validated。",
        f"- 新增调用 {len(new)}；继承 v2 调用 {len(ledger) - len(new)}；SDK 重试 0。",
        f"- 总 usage：{report['usage']}；新增 usage：{summary['new_usage']}。",
        f"- 估算 USD：总计 {report['estimated_cost_usd_known']:.6f}；新增 {summary['new_estimated_cost_usd']:.6f}；最大单组 {summary['max_estimated_group_cost_usd']:.6f}。",
        "- 费率假设：input $3/M、output $15/M；不是核实过的官方费率或 AWS 账单。",
        "- keeper/delete/decision 写入均为 0；冻结 inventory 和 v2 原始证据未改。",
        "- 没有 Willy 人标，不报告偏好准确率。等待 parent 独立验收。",
        "- 三组 Astra 视觉/证据抽查见 astra-visual-check.json；G007 背景鞋子裁切、G004 舞台位置描述有误，raw 未改。结构化校验不等于语义正确。",
        "",
        "## 分层汇总",
        "",
        "scene 为输入的粗路由标签；unknown 不代表真实场景类别，也不用于计算 scene 准确率。",
        "",
    ]
    for field, values in summary["strata"].items():
        lines += [
            f"### {field}",
            "",
            "| 分层 | 组数 | Validated | 估算 USD |",
            "|---|---:|---:|---:|",
        ]
        for value, bucket in sorted(values.items()):
            lines.append(
                f"| {value} | {bucket['groups']} | {bucket['validated']} | {bucket['estimated_cost_usd']:.6f} |"
            )
        lines.append("")
    lines += [
        "## 失败组",
        "",
        json.dumps(summary["failures"], ensure_ascii=False, indent=2),
        "",
        "## 未调用组",
        "",
        ", ".join(summary["uncalled_groups"]) or "无",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3",
    )
    print(json.dumps(run(parser.parse_args().output), indent=2))
