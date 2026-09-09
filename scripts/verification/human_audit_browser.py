"""Synthetic-only Chrome gate: v1 migration, v2 notes/partial labels and lightbox.

Run with Python + Playwright + Pillow + numpy; uses installed Chrome, no downloads.
Only temporary synthetic media/labels are used. No real labels are manufactured.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from src.human_audit import canonical, read_labels, report  # noqa: E402
from src.human_audit_bundle import upgrade_bundle  # noqa: E402
from tests.test_human_audit import files, label, legacy_bundle_files, note_migration_files  # noqa: E402


def download_labels(page, path):
    with page.expect_download() as download:
        page.locator("#export").click()
    download.value.save_as(path)
    return read_labels(path)


def check_note_migration(browser, root):
    root.mkdir()
    _, _, _, output, bundle = note_migration_files(root)
    page = browser.new_page(accept_downloads=True)
    errors, requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(request.url))
    page.goto((output / "reviewer/index.html").as_uri())
    expect(page.locator("#migration-notice")).to_be_visible()
    expect(page.locator("#status")).to_contain_text("授权自动迁移 2 条，需复核 3 条")
    expect(page.locator("section").first.locator("[data-quality]")).to_have_value("quality_abstain")
    expect(page.locator("section").nth(2).locator(".saved")).to_contain_text("需人工复核")
    path = root / "exported.jsonl"
    rows = download_labels(page, path)
    assert rows == read_labels(output / "labels-v2.jsonl")
    assert all(x["provenance"]["human_attestation_scope"] == "original_visual_observation" for x in rows)
    assert report(bundle, rows)["note_migration_counts"]["needs_review"] == 3
    page.locator("#import").set_input_files(path)
    expect(page.locator("#status")).to_contain_text("已重放 5 条")
    page.reload()
    expect(page.locator("#status")).to_contain_text("本机恢复 5 条")
    assert download_labels(page, root / "restored.jsonl") == rows
    bad = json.loads(canonical(rows[2]))
    bad["provenance"]["needs_review"] = False
    rejected = root / "bad-provenance.jsonl"
    rejected.write_text(canonical(bad))
    page.locator("#import").set_input_files(rejected)
    expect(page.locator("#status")).to_contain_text("导入拒绝")
    stripped = json.loads(canonical(rows[2]))
    del stripped["provenance"]
    rejected.write_text(canonical(stripped))
    page.locator("#import").set_input_files(rejected)
    expect(page.locator("#status")).to_contain_text("导入拒绝")
    # Same controls become a new human judgment only after explicit confirmation.
    section = page.locator("section").nth(2)
    section.locator('[data-save="reviewed"]').click()
    expect(page.locator("#status")).to_contain_text("审阅来源无效")
    page.locator("#annotator").fill("synthetic-confirming-human")
    page.locator("#attest").check()
    section.locator('[data-save="reviewed"]').click()
    expect(section.locator(".saved")).not_to_contain_text("需人工复核")
    confirmed = download_labels(page, root / "confirmed.jsonl")
    changed = next(x for x in confirmed if x["task_id"] == rows[2]["task_id"])
    assert changed["provenance"]["kind"] == "human_reassessment"
    assert changed["provenance"]["confirmed"] is True and changed["note"] == rows[2]["note"]
    assert report(bundle, confirmed)["note_migration_counts"] == {"automatic": 2, "needs_review": 2, "human_reassessment": 1, "unmarked_labels": 0}
    assert not errors, errors
    assert all(url.startswith(output.as_uri() + "/reviewer/") for url in requests)
    page.close()


def main():
    with tempfile.TemporaryDirectory(prefix="photo-human-audit-browser-") as work:
        root = Path(work)
        args = files(root)
        old = legacy_bundle_files(args)
        legacy = [label(old, t) for t in old["tasks"][:2]]
        legacy[0]["note"] = "SYNTHETIC: 看不清质量，但旧标签不得自动变更"
        legacy_path = root / "legacy.jsonl"
        legacy_path.write_text("\n".join(canonical(x) for x in legacy))
        output = root / "v2"
        bundle = upgrade_bundle(args["output"], legacy_path, output)
        url = (output / "reviewer/index.html").as_uri()
        requests, errors = [], []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
            page = browser.new_page(accept_downloads=True)
            page.on("request", lambda request: requests.append(request.url))
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url)
            expect(page.locator("section")).to_have_count(5)
            expect(page.locator("#status")).to_contain_text("随包导入 2 条")
            first = page.locator("section").first
            expect(first.locator(".saved")).to_contain_text("旧 v1 标签")
            expect(first.locator("textarea")).to_have_value(legacy[0]["note"])
            expect(first.locator("[data-quality]")).to_have_value("assessed")
            exported = download_labels(page, root / "converted.jsonl")
            assert len(exported) == 2 and all(x["schema_version"] == 2 for x in exported)
            assert exported[0]["note"] == legacy[0]["note"]
            assert report(bundle, exported)["tasks"] == report(old, legacy)["tasks"]
            # Keyboard and mouse lightbox; natural pixels versus actual fit scaling.
            opener = first.get_by_role("button", name="打开 M1 缓存大图", exact=True)
            opener.focus()
            page.keyboard.press("Enter")
            expect(page.locator("#lightbox")).to_be_visible()
            expect(page.locator("#lightbox-status")).to_contain_text("32×24")
            expect(page.locator("#lightbox")).to_contain_text("放大不增加细节")
            page.locator("#native-image").click()
            expect(page.locator("#image-viewport")).to_have_class("native")
            expect(page.locator("#lightbox-status")).to_contain_text("原像素 1:1")
            assert page.locator("#large-image").evaluate("i=>i.getBoundingClientRect().width===i.naturalWidth")
            page.keyboard.press("ArrowRight")
            expect(page.locator("#large-image")).to_have_attribute("alt", "M2")
            expect(page.locator("#next-image")).to_be_disabled()
            page.locator("#previous-image").click()
            expect(page.locator("#large-image")).to_have_attribute("alt", "M1")
            page.locator("#fit-image").click()
            expect(page.locator("#image-viewport")).to_have_class("fit")
            assert page.locator("#large-image").evaluate("i=>i.getBoundingClientRect().width>i.naturalWidth")
            page.keyboard.press("Escape")
            expect(page.locator("#lightbox")).not_to_be_visible()
            expect(opener).to_be_focused()
            opener.click()
            page.locator("#close-image").click()
            # Initial labels are editable; no prechecked attestation.
            expect(page.locator("#attest")).not_to_be_checked()
            page.locator("#annotator").fill("synthetic-browser-reviewer")
            page.locator("#attest").check()
            for section in page.locator("section").all():
                for field in section.locator("input[type=number]").all():
                    field.fill("1")
                section.locator("[data-quality]").select_option("assessed")
                section.locator('[data-keeper="M1"]').check()
                section.locator("[data-impure]").select_option("false")
                section.locator('[data-save="reviewed"]').click()
                expect(section.locator(".saved")).to_contain_text("已保存")
            path = root / "synthetic-labels.jsonl"
            labels = download_labels(page, path)
            assert len(labels) == 5 and report(bundle, labels)["weighted_error_rate"] == 0
            # Uncertain/unassessable notes save despite incomplete phase forms.
            for index, status in enumerate(("uncertain", "unassessable")):
                section = page.locator("section").nth(index)
                section.locator("textarea").fill(f"SYNTHETIC {status} 备注\n第二行：表情看不清")
                section.locator(f'[data-save="{status}"]').click()
                expect(section.locator(".saved")).to_contain_text("备注已保存")
                expect(page.locator("#status")).to_contain_text("本机备份成功")
            # Mixed phase quality: no keeper required in the abstaining phase.
            partial = page.locator("section").nth(2)
            partial.locator('[data-phase="M2"]').fill("2")
            partial.locator('[data-quality="1"]').select_option("assessed")
            partial.locator('[data-quality="2"]').select_option("quality_abstain")
            expect(partial.locator('[data-keeper="M2"]')).to_be_disabled()
            partial.locator('[data-save="reviewed"]').click()
            expect(partial.locator(".saved")).to_contain_text("含质量弃权")
            mixed_path = root / "mixed.jsonl"
            mixed = download_labels(page, mixed_path)
            mixed_report = report(bundle, mixed)
            assert len(mixed) == 5 and mixed_report["weighted_error_rate"] is None
            detail = mixed_report["tasks"][2]
            assert detail["phase_coverage_error"] is True
            assert detail["keeper_bad"] is detail["keeper_quality_error"] is detail["error"] is None
            # Local recovery retains saved notes; file import also restores controls.
            page.reload()
            expect(page.locator("#status")).to_contain_text("本机恢复 5 条")
            for index, status in enumerate(("uncertain", "unassessable")):
                expect(page.locator("section").nth(index).locator("textarea")).to_have_value(f"SYNTHETIC {status} 备注\n第二行：表情看不清")
            page.evaluate("localStorage.clear()")
            page.reload()
            expect(page.locator("#status")).to_contain_text("随包导入 2 条")
            page.locator("#import").set_input_files(mixed_path)
            expect(page.locator("#status")).to_contain_text("已重放 5 条")
            assert download_labels(page, root / "roundtrip.jsonl") == mixed
            expect(page.locator("section").nth(2).locator('[data-quality="2"]')).to_have_value("quality_abstain")
            # Strict raw input parity + atomic rejection; duplicate tasks/fields too.
            raw = canonical(labels[0])
            malformed = [raw.replace('"schema_version":2', '"schema_version":2.0'),
                         raw.replace('"schema_version":2', '"schema_version":NaN'),
                         raw.replace('"human_attested":true', '"human_attested":false,"human_attested":true'),
                         raw.replace('"members":[', '"members":[],"members":['),
                         raw[:-1] + ',"unknown":true}', raw + "\n" + raw]
            forged_legacy = {**legacy[0], "bundle_id": bundle["bundle_id"]}
            malformed.append(canonical(forged_legacy))
            invalid = json.loads(canonical(mixed[-1]))
            invalid["bundle_id"] = "wrong"
            malformed.append(canonical(invalid))
            malformed.append(canonical({**labels[0], "phases": [{"members": ["M1", "M2"], "keeper_status": "quality_abstain", "acceptable_keepers": []}]}))
            for text in malformed:
                rejected = root / "rejected.jsonl"
                rejected.write_text(text)
                try:
                    report(bundle, read_labels(rejected))
                except ValueError:
                    pass
                else:
                    raise AssertionError("CLI accepted malformed label")
                page.locator("#import").set_input_files(rejected)
                expect(page.locator("#status")).to_contain_text("导入拒绝")
            assert download_labels(page, root / "retained.jsonl") == mixed
            # Edits invalidate old labels. Export does not dismiss draft-loss warning.
            page.locator("section").first.locator("textarea").fill("UNSAVED DRAFT")
            expect(page.locator(".saved").first).to_contain_text("重新保存")
            edited = download_labels(page, root / "edited.jsonl")
            assert len(edited) == 4
            with page.expect_event("dialog") as confirmation:
                page.locator("#import").set_input_files(mixed_path)
            assert confirmation.value.type == "confirm"
            confirmation.value.dismiss()
            expect(page.locator("#import")).to_have_value("")
            expect(page.locator("section").first.locator("textarea")).to_have_value("UNSAVED DRAFT")
            dialogs = []
            def cancel_leave(dialog):
                dialogs.append(dialog.type)
                dialog.dismiss()
            page.once("dialog", cancel_leave)
            page.evaluate("window.location.href='about:blank'")
            expect(page.locator("h1")).to_have_text("盲化阶段标注")
            assert dialogs == ["beforeunload"]
            # Storage failures must not falsely claim a durable local backup.
            blocked = browser.new_page(accept_downloads=True)
            blocked.add_init_script("Storage.prototype.setItem=function(){throw Error('synthetic quota failure')}")
            blocked.goto(url)
            blocked.locator("#annotator").fill("synthetic-storage-failure")
            blocked.locator("section").first.locator("textarea").fill("SYNTHETIC note with storage blocked")
            blocked.locator("section").first.locator('[data-save="unassessable"]').click()
            expect(blocked.locator("#status")).to_contain_text("本机备份失败")
            retained = download_labels(blocked, root / "storage-failure.jsonl")
            assert any(x["note"] == "SYNTHETIC note with storage blocked" for x in retained)
            assert not errors, errors
            assert requests and all(request.startswith(output.as_uri() + "/reviewer/") for request in requests), requests
            assert all("manifest" not in request for request in requests)
            check_note_migration(browser, root / "authorized-notes")
            browser.close()
        print("PASS: authorized note migration provenance/needs-review/export/import/recovery/explicit human confirmation; "
              "PASS: synthetic v1 migration -> editable v2 -> note/partial export -> local/file restore -> strict rejection -> lightbox keyboard/zoom -> storage failure/draft warnings; local assets only")


if __name__ == "__main__":
    main()
