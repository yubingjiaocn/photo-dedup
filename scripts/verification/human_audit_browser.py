"""Synthetic-only real browser audit export/import/report gate. No real media.

Run with a Python environment containing Playwright + Pillow + numpy.
Uses the installed Chrome, no browser download; creates temporary synthetic caches.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from src.human_audit import read_labels, report  # noqa: E402
from src.human_audit_bundle import export_bundle  # noqa: E402
from tests.test_human_audit import files  # noqa: E402


def main():
    with tempfile.TemporaryDirectory(prefix="photo-human-audit-browser-") as work:
        root = Path(work)
        args = files(root)
        bundle = export_bundle(**args)
        url = (args["output"] / "reviewer/index.html").as_uri()
        requests, errors = [], []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
            page = browser.new_page(accept_downloads=True)
            page.on("request", lambda request: requests.append(request.url))
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url)
            expect(page.locator("section")).to_have_count(5)
            assert page.locator("input[type=number]").evaluate_all("nodes=>nodes.every(n=>n.value==='')")
            assert page.locator("input[type=checkbox]").evaluate_all("nodes=>nodes.every(n=>!n.checked)")
            page.locator("#annotator").fill("synthetic-browser-reviewer")
            page.locator("#attest").check()
            for section in page.locator("section").all():
                for field in section.locator("input[type=number]").all():
                    field.fill("1")
                section.locator('[data-keeper="M1"]').check()
                section.locator("select").select_option("false")
                section.get_by_role("button", name="记录完整判断", exact=True).click()
                expect(section.locator(".saved")).to_contain_text("已记录")
            with page.expect_download() as download:
                page.locator("#export").click()
            path = root / "synthetic-labels.jsonl"
            download.value.save_as(path)
            labels = read_labels(path)
            assert len(labels) == 5
            assert report(bundle, labels)["weighted_error_rate"] == 0
            # Reload starts blank; snapshot import restores actual controls.
            page.reload()
            expect(page.locator(".saved").first).to_have_text("尚未记录")
            page.locator("#import").set_input_files(path)
            expect(page.locator("#status")).to_contain_text("已重放 5 条记录")
            assert page.locator("input[type=number]").evaluate_all("nodes=>nodes.every(n=>n.value==='1')")
            # Editing invalidates prior saved result, never exports stale judgments.
            page.locator("section").first.locator("input[type=number]").first.fill("2")
            expect(page.locator(".saved").first).to_contain_text("重新记录")
            with page.expect_download() as download:
                page.locator("#export").click()
            edited = root / "synthetic-edited.jsonl"
            download.value.save_as(edited)
            assert len(read_labels(edited)) == 4
            # Raw JSONL corpus: browser and CLI agree before any re-export.
            raw = json.dumps(read_labels(path)[0])
            corpus = [raw.replace('"schema_version": 1', '"schema_version": 1.0'),
                      raw.replace('"human_attested": true', '"human_attested": false, "human_attested": true'),
                      raw.replace('"members": [', '"members": [], "members": ['),
                      raw.replace('"schema_version": 1', '"schema_version": NaN')]
            for text in corpus:
                rejected = root / "rejected.jsonl"
                rejected.write_text(text)
                try:
                    report(bundle, read_labels(rejected))
                except ValueError:
                    pass
                else:
                    raise AssertionError("CLI accepted malformed raw JSON")
                page.locator("#import").set_input_files(rejected)
                expect(page.locator("#status")).to_contain_text("导入拒绝")
            # Draft protection survives export; cancel import leaves drafts intact.
            page.once("dialog", lambda dialog: dialog.dismiss())
            page.locator("#import").set_input_files(path)
            expect(page.locator("section").first.locator("input[type=number]").first).to_have_value("2")
            # Wrong-bundle import rejected atomically; preceding state retained.
            bad = root / "bad.jsonl"
            labels[0]["bundle_id"] = "wrong"
            bad.write_text("\n".join(json.dumps(x) for x in labels))
            page.locator("#import").set_input_files(bad)
            expect(page.locator("#status")).to_contain_text("导入拒绝")
            with page.expect_download() as download:
                page.locator("#export").click()
            retained = root / "retained.jsonl"
            download.value.save_as(retained)
            assert read_labels(retained) == read_labels(edited)
            assert not errors, errors
            assert requests and all(request.startswith(args["output"].as_uri() + "/reviewer/") for request in requests), requests
            assert all("manifest" not in request for request in requests)
            # A real attempted navigation still warns about the unexported draft.
            dialogs = []
            def cancel_leave(dialog):
                dialogs.append(dialog.type)
                dialog.dismiss()
            page.once("dialog", cancel_leave)
            page.evaluate("window.location.href='about:blank'")
            expect(page.locator("h1")).to_have_text("盲化阶段标注")
            assert dialogs == ["beforeunload"]
            browser.close()
        print("PASS: synthetic browser phase labels -> JSONL -> weighted report -> reload/import -> edit invalidation; local reviewer assets only")


if __name__ == "__main__":
    main()
