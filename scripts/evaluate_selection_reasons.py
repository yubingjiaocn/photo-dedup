"""Generate the checked-in shadow explanation sample and smoke evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.selection_reason import GENERATOR, explain_group, faithfulness


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evaluate(fixture, loops=5000):
    cases = fixture["cases"]
    before = [{"group": copy.deepcopy(case["group"]), "members": copy.deepcopy(case["members"])} for case in cases]
    decision_before = digest([[m["file_id"], m["is_keep"], m["decision"]] for case in cases for m in case["members"]])
    results = []
    claims = faithful = 0
    explained = abstained = 0
    for case in cases:
        reason = explain_group(case["group"], case["members"])
        score = faithfulness(reason)
        claims += score["claim_count"]
        faithful += score["faithful_claim_count"]
        explained += reason["status"] == "explained"
        abstained += reason["status"] == "abstained"
        results.append({"case_id": case["case_id"], "reason": reason})
    decision_after = digest([[m["file_id"], m["is_keep"], m["decision"]] for case in cases for m in case["members"]])
    assert before == [{"group": case["group"], "members": case["members"]} for case in cases]
    start = time.perf_counter()
    for _ in range(loops):
        for case in cases:
            explain_group(case["group"], case["members"])
    seconds = time.perf_counter() - start
    calls = loops * len(cases)
    return {
        "schema_version": 1,
        "generator": GENERATOR,
        "fixture_kind": fixture["kind"],
        "source_basis": fixture["source_basis"],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "metrics": {
            "cases": len(cases),
            "explained": explained,
            "coverage": explained / len(cases),
            "abstained": abstained,
            "abstention_rate": abstained / len(cases),
            "claims": claims,
            "faithful_claims": faithful,
            "faithfulness": faithful / claims if claims else 1.0,
            "decision_digest_before": decision_before,
            "decision_digest_after": decision_after,
            "decision_unchanged": decision_before == decision_after,
            "benchmark_calls": calls,
            "benchmark_seconds": seconds,
            "microseconds_per_group": seconds * 1_000_000 / calls,
            "model_calls": 0
        },
        "limitations": [
            "Checked-in production-shaped synthetic smoke cases; not a real-photo accuracy estimate.",
            "Human preference, expression, eye state, occlusion, composition and narrative remain unverified.",
            "Confidence is explanation sufficiency, not keeper correctness or deletion safety."
        ],
        "cases": results
    }


def markdown(report):
    m = report["metrics"]
    return f"""# 人类可读选片理由 shadow prototype 评估\n\n日期：2026-09-29。范围：已检入 production-shaped synthetic/annotation smoke fixture；不是外部真实照片准确率。\n\n- 生成器：`{report['generator']}`；纯 Python 确定性模板，模型调用 `{m['model_calls']}`。\n- 证据引用完整性：{m['faithful_claims']}/{m['claims']} = {m['faithfulness']:.1%}（每条事实文本均引用存在的 evidence ID；不等同于真人偏好正确率）。\n- 覆盖率：{m['explained']}/{m['cases']} = {m['coverage']:.1%}。\n- 弃权：{m['abstained']}/{m['cases']} = {m['abstention_rate']:.1%}。\n- 合成输入未被函数修改：`{m['decision_unchanged']}`；前后 digest 均为 `{m['decision_digest_before']}`。\n- 粗略成本：{m['benchmark_calls']} 次生成共 {m['benchmark_seconds']:.3f}s，约 {m['microseconds_per_group']:.1f} µs/组；不读图片、不调用模型。\n\n## 证据边界\n\n可陈述：落库 keeper、utility/pair margin、MUSIQ/缓存质量分、CLIP-IQA、人脸局部质量数值、分辨率、BYTE_IDENTICAL、逻辑阶段/姿态保护代码。不可陈述：更好表情、睁眼、无遮挡、主体完整、构图或叙事更佳；这些一律要求人工确认。\n\n## 外部评测集\n\n实现与 smoke 验证不依赖外部大集。冻结真实照片回放由 `scripts/evaluate_frozen_selection_reasons.py` 单独执行，结果见 `docs/SELECTION_REASON_FROZEN_REPLAY.md` 和 `docs/selection-reason-frozen-replay.json`。现有冻结标签为 `model_provisional`，因此真实回放只评估证据忠实度、覆盖率、弃权和潜在误导风险，不报告人工偏好一致率。\n\n## 限制\n\n""" + "\n".join(f"- {item}" for item in report["limitations"]) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/selection-reason-shadow-v1.json")
    parser.add_argument("--json", default="docs/selection-reason-shadow-sample.json")
    parser.add_argument("--report", default="docs/SELECTION_REASON_SHADOW_EVALUATION.md")
    parser.add_argument("--loops", type=int, default=5000)
    args = parser.parse_args()
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    report = evaluate(fixture, args.loops)
    Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.report).write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
