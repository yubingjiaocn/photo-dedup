"""Compare identity-bound human cloud-review selections with VLM recommendations."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def member_identity(value: dict[str, Any]) -> tuple[int, str]:
    return int(value["file_id"]), str(value["content_sha256"])


def packet_members(packet: dict[str, Any]) -> frozenset[tuple[int, str]]:
    members = []
    for value in packet["complete_frames"]:
        alias, digest = value.split("@", 1)
        members.append((int(alias.removeprefix("I")), digest))
    return frozenset(members)


def relation(decision: str, ai: set[int], human: set[int]) -> str:
    if decision == "abstain":
        return "manual_after_abstain"
    if decision == "failed":
        return "manual_after_failed"
    if human == ai:
        return "exact"
    if ai < human:
        return "human_added"
    if human < ai:
        return "human_removed"
    return "changed"


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selectable = [row for row in rows if row["decision"] == "select"]
    exact = sum(row["relation"] == "exact" for row in selectable)
    intersection = sum(len(set(row["ai"]) & set(row["human"])) for row in selectable)
    ai_photos = sum(len(row["ai"]) for row in selectable)
    human_photos = sum(len(row["human"]) for row in selectable)
    any_overlap = sum(bool(set(row["ai"]) & set(row["human"])) for row in selectable)
    summary = {
        "labelled_groups": len(rows),
        "decision_counts": dict(Counter(row["decision"] for row in rows)),
        "relation_counts": dict(Counter(row["relation"] for row in rows)),
        "select_groups": len(selectable),
        "exact_groups": exact,
        "exact_set_rate_select_groups": rate(exact, len(selectable)),
        "exact_set_rate_all_groups": rate(exact, len(rows)),
        "any_overlap_groups": any_overlap,
        "any_overlap_rate_select_groups": rate(any_overlap, len(selectable)),
        "ai_selected_photos": ai_photos,
        "human_selected_photos": human_photos,
        "intersection_photos": intersection,
        "photo_micro_precision": rate(intersection, ai_photos),
        "photo_micro_recall": rate(intersection, human_photos),
        "photo_micro_f1": rate(2 * intersection, ai_photos + human_photos),
        "human_added_photos": sum(len(set(row["human"]) - set(row["ai"])) for row in selectable),
        "human_removed_photos": sum(len(set(row["ai"]) - set(row["human"])) for row in selectable),
    }
    for field, name in (("date", "by_date"), ("member_count", "by_group_size"),
                        ("scene", "by_scene"), ("ai_count", "by_ai_recommendation_count")):
        buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in selectable:
            buckets[row[field]].append(row)
        summary[name] = {
            str(key): {
                "groups": len(values),
                "exact": sum(value["relation"] == "exact" for value in values),
                "exact_rate": rate(sum(value["relation"] == "exact" for value in values), len(values)),
                "relations": dict(Counter(value["relation"] for value in values)),
            }
            for key, values in sorted(buckets.items(), key=lambda item: str(item[0]))
        }
    small = [row for row in selectable if row["member_count"] == 2]
    large = [row for row in selectable if row["member_count"] >= 3]
    summary["two_photo_groups"] = {
        "groups": len(small),
        "exact": sum(row["relation"] == "exact" for row in small),
        "exact_rate": rate(sum(row["relation"] == "exact" for row in small), len(small)),
    }
    summary["three_plus_photo_groups"] = {
        "groups": len(large),
        "exact": sum(row["relation"] == "exact" for row in large),
        "exact_rate": rate(sum(row["relation"] == "exact" for row in large), len(large)),
    }
    return summary


def render(summary: dict[str, Any], rows: list[dict[str, Any]], label_sha: str) -> str:
    pct = lambda value: "n/a" if value is None else f"{value:.1%}"
    lines = [
        "# Willy 人标 vs AI 建议",
        "",
        f"- 人标快照 SHA-256：`{label_sha}`",
        f"- 完成人标：{summary['labelled_groups']} 组",
        f"- AI 可执行建议：{summary['select_groups']} 组；完全一致 {summary['exact_groups']} 组（{pct(summary['exact_set_rate_select_groups'])}）",
        f"- 全部 194 组口径完全一致：{pct(summary['exact_set_rate_all_groups'])}",
        f"- AI建议与人标至少有一张重合：{summary['any_overlap_groups']}/{summary['select_groups']}（{pct(summary['any_overlap_rate_select_groups'])}）",
        f"- 照片级 micro precision / recall / F1：{pct(summary['photo_micro_precision'])} / {pct(summary['photo_micro_recall'])} / {pct(summary['photo_micro_f1'])}",
        f"- AI共建议 {summary['ai_selected_photos']} 张，Willy共保留 {summary['human_selected_photos']} 张；Willy加回 {summary['human_added_photos']} 张、删去AI多留 {summary['human_removed_photos']} 张。",
        "",
        "## 组级差异",
        "",
        "| 分类 | 组数 |",
        "|---|---:|",
    ]
    for key in ("exact", "human_added", "human_removed", "changed", "manual_after_abstain", "manual_after_failed"):
        lines.append(f"| {key} | {summary['relation_counts'].get(key, 0)} |")
    lines += [
        "",
        "## 组大小",
        "",
        "| 组大小 | 组数 | 完全一致 | 一致率 |",
        "|---:|---:|---:|---:|",
    ]
    for key, value in summary["by_group_size"].items():
        lines.append(f"| {key} | {value['groups']} | {value['exact']} | {pct(value['exact_rate'])} |")
    lines += [
        "",
        f"两张组：{summary['two_photo_groups']['exact']}/{summary['two_photo_groups']['groups']}（{pct(summary['two_photo_groups']['exact_rate'])}）；三张及以上：{summary['three_plus_photo_groups']['exact']}/{summary['three_plus_photo_groups']['groups']}（{pct(summary['three_plus_photo_groups']['exact_rate'])}）。",
        "",
        "## 日期",
        "",
        "| 日期 | AI select组 | 完全一致 | 一致率 |",
        "|---|---:|---:|---:|",
    ]
    for key, value in summary["by_date"].items():
        lines.append(f"| {key} | {value['groups']} | {value['exact']} | {pct(value['exact_rate'])} |")
    lines += [
        "",
        "## 非完全一致明细",
        "",
        "| 组 | 组大小 | 分类 | AI | Willy |",
        "|---|---:|---|---|---|",
    ]
    for row in rows:
        if row["relation"] == "exact":
            continue
        lines.append(
            f"| {row['group']} | {row['member_count']} | {row['relation']} | "
            f"{','.join(map(str, row['ai'])) or '—'} | {','.join(map(str, row['human']))} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- `exact`：AI建议集合与Willy选择完全一致。",
        "- `human_added`：Willy额外保留，优先视为AI可能漏掉阶段/表情。",
        "- `human_removed`：Willy删去AI建议，优先视为AI过度多选。",
        "- `changed`：双方各有独占照片，包括换选最佳帧或部分重合的组合变化。",
        "- failed / abstain 不进入AI建议准确率分母，但进入全流程覆盖率。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    label_bytes = args.labels.read_bytes()
    label_sha = hashlib.sha256(label_bytes).hexdigest()
    labels = json.loads(label_bytes)["groups"]
    batch_index: dict[frozenset[tuple[int, str]], Path] = {}
    for directory in args.batch.iterdir():
        packet_path = directory / "visual-packet.json"
        if not directory.is_dir() or not packet_path.exists():
            continue
        members = packet_members(load(packet_path))
        if members in batch_index:
            raise ValueError(f"duplicate group identity: {directory}")
        batch_index[members] = directory

    rows = []
    for saved in labels.values():
        members = frozenset(member_identity(value) for value in saved["members"])
        directory = batch_index.get(members)
        if directory is None:
            raise ValueError("saved group identity is absent from batch")
        human = {int(value["file_id"]) for value in saved["selected"]}
        if not human or not human.issubset({value[0] for value in members}):
            raise ValueError(f"invalid human selection: {directory.name}")
        validated_path = directory / "validated.json"
        if validated_path.exists():
            result = load(validated_path)
            decision = str(result["decision"])
            ai = {int(value["frame"]["file_id"]) for value in result.get("recommended", [])}
        else:
            decision, ai = "failed", set()
        if not ai.issubset({value[0] for value in members}):
            raise ValueError(f"invalid AI selection: {directory.name}")
        state = load(directory / "state.json")
        rows.append({
            "group": directory.name,
            "date": directory.name[:10],
            "member_count": len(members),
            "decision": decision,
            "scene": state.get("scene_route", "unknown"),
            "ai": sorted(ai),
            "human": sorted(human),
            "ai_count": len(ai),
            "relation": relation(decision, ai, human),
        })
    rows.sort(key=lambda row: row["group"])
    if len(rows) != len(batch_index):
        raise ValueError(f"incomplete labels: {len(rows)}/{len(batch_index)}")

    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps({
        "label_sha256": label_sha,
        "summary": summary,
        "groups": rows,
    }, ensure_ascii=False, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(render(summary, rows, label_sha))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
