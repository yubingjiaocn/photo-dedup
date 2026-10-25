"""Local-only final report: preserve invalid raw output and verify all receipts."""

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import two_pass_review as tp


def run():
    root = ROOT / "artifacts/two-pass-real-pilot"
    report = json.loads((root / "revalidate-v2/report.json").read_text())
    ledger = json.loads((root / "call-ledger.json").read_text())
    assert len(ledger) == 4
    verification = {
        "total_calls": 4,
        "cloud_paused": True,
        "groups": [],
        "remaining_calls_authorized": 0,
        "all_raw_responses_preserved": True,
    }
    body = [
        "<h1>G004 / G007 真实两段式审查</h1>",
        "<p>已完成原始A复用 → CV → 真实B调用。两份B均返回attention_only，但严格输出校验拒绝；不产生推荐。云调用已暂停，共4次。</p>",
        "<p>当前产物在 revalidate-v2；原始A及历史拒绝状态在 archive/v1 与原组目录。</p>",
        '<p><a href="call-ledger.json">调用账本</a> · <a href="report.json">总报告</a> · <a href="artifact-verification.json">核验</a></p>',
    ]
    for group in report["groups"]:
        name = group["group"]
        case = root / "revalidate-v2" / name
        sidecar = json.loads((case / "sidecar.json").read_text())
        assert sidecar["sidecar_sha"] == tp.sha(
            tp.canonical({k: v for k, v in sidecar.items() if k != "sidecar_sha"})
        )
        errors = []
        raw = tp.strict_json((case / "pass-B-raw.txt").read_text())
        for i, point in enumerate(raw["discriminators"]):
            if len(point["text"]) > 240:
                errors.append(
                    f"discriminator {i + 1}: text length {len(point['text'])} > 240"
                )
            for e in point["evidence"]:
                region = sidecar["subject_set_evidence"]["lookup"].get(e["region"])
                if not region or region["role"] not in {"key_subject", "supporting"}:
                    errors.append(
                        f"discriminator {i + 1}: non-main evidence {e['region']}"
                    )
        (case / "pass-B-raw.json").write_bytes(tp.canonical(raw))
        (case / "pass-B-validated.json").write_bytes(
            tp.canonical(
                {
                    "status": "not_produced",
                    "reason": errors,
                    "decision_ceiling": "attention_only",
                    "recommended": [],
                }
            )
        )
        (case / "validation-audit.json").write_bytes(
            tp.canonical(
                {
                    "A_structural": "passed",
                    "B_errors": errors,
                    "B_raw_decision": raw["decision"],
                    "raw_prose_not_accepted_recommendation": True,
                }
            )
        )
        group["output_validation_errors"] = errors
        group["raw_decision"] = raw["decision"]
        for asset in sidecar["assets"]:
            assert tp.sha((case / asset["path"]).read_bytes()) == asset["sha256"]
            for cell in asset["cells"]:
                assert not cell["upsampled"]
        for call in sidecar["accounting"]:
            source = root / name if call["pass"] == "A" else case
            for file, field in (
                (f"pass-{call['pass']}-wire-request.json", "request_sha256"),
                (f"pass-{call['pass']}-provider-response.json", "response_sha256"),
                (f"pass-{call['pass']}-raw.txt", "response_text_sha256"),
            ):
                assert tp.sha((source / file).read_bytes()) == call[field]
            provider = json.loads(
                (source / f"pass-{call['pass']}-provider-response.json").read_text()
            )
            assert provider["ResponseMetadata"]["RetryAttempts"] == 0
        assert group["decision_tables_before"] == group["decision_tables_after"]
        verification["groups"].append(
            {
                "group": name,
                "sidecar_assets_receipts": "passed",
                "A": "revalidated",
                "B": "raw_returned_validation_rejected",
                "errors": errors,
                "decision_ceiling": "attention_only",
                "recommended": [],
            }
        )
        body.append(
            f"<h2>{name}</h2><p>decision_ceiling=attention_only</p><ul>"
            + "".join(f"<li>{html.escape(e)}</li>" for e in errors)
            + "</ul>"
        )
        for file in (
            "pass-A-validated.json",
            "revalidation.json",
            "L1.json",
            "L2.json",
            "pass-B-request.json",
            "pass-B-raw.txt",
            "pass-B-validated.json",
            "sidecar.json",
        ):
            body.append(f'<a href="revalidate-v2/{name}/{file}">{file}</a> · ')
        for asset in sidecar["assets"]:
            body.append(
                f'<h3>{asset["path"]}</h3><img loading="lazy" style="max-width:100%" src="revalidate-v2/{name}/{asset["path"]}">'
            )
        body.append(
            "<h3>未经接纳的B原始文本</h3><pre>"
            + html.escape(json.dumps(raw, indent=2, ensure_ascii=False))
            + "</pre>"
        )
    report.update(
        cloud_paused=True,
        maximum_calls=4,
        completion="real_A_CV_B_executed_B_outputs_rejected",
        input_tokens=sum(c["usage"]["inputTokens"] for c in ledger),
        output_tokens=sum(c["usage"]["outputTokens"] for c in ledger),
    )
    for name, expected in report["old_real_json_hashes_unchanged"].items():
        assert (
            tp.sha((ROOT / "artifacts/multi-subject-narrow" / name).read_bytes())
            == expected
        )
    manifest = json.loads(
        (root / "revalidate-v2/implementation-manifest.json").read_text()
    )
    for name, expected in manifest.items():
        assert tp.sha((ROOT / name).read_bytes()) == expected, name
    verification["executed_implementation_unchanged"] = True
    (root / "report.json").write_bytes(tp.canonical(report))
    (root / "artifact-verification.json").write_bytes(tp.canonical(verification))
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>真实两段式小测</title><style>body{max-width:1200px;margin:auto;font:16px sans-serif;padding:24px}pre{white-space:pre-wrap}a{line-height:2}</style>'
        + "\n".join(body)
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    run()
