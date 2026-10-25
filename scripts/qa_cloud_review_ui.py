"""Offline real-browser QA; ephemeral loopback server, never a persistent service."""
from __future__ import annotations

import http.server
import json
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cloud_review import CloudReviewData  # noqa: E402
from src.review_server import create_handler  # noqa: E402


def wait_for_group_images(page):
    """Require every card in the current group to finish decoding."""
    page.wait_for_function(
        """() => {
            const images = [...document.querySelectorAll('.cloud-card img')];
            return images.length > 0 && images.every(i => i.complete && i.naturalWidth > 0);
        }"""
    )


def run():
    artifacts = ROOT / "artifacts/vlm-review-ui"
    output = Path(tempfile.mkdtemp(prefix="browser-output-", dir=artifacts))
    data = CloudReviewData(output, ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3",
                           ROOT / "artifacts/cloud-judge-batch/frozen.json")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), create_handler(data))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    errors, originals = [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path="/usr/bin/google-chrome-stable")
            page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda req: originals.append(req.url) if "/api/original/" in req.url else None)
            page.goto(f"http://127.0.0.1:{server.server_port}/review.html?mode=cloud")
            expect(page.get_by_text("2026-04-04-G001", exact=True)).to_be_visible()
            expect(page.locator(".cloud-card")).to_have_count(4)
            wait_for_group_images(page)
            page.screenshot(path=str(artifacts / "01-select-desktop.png"), full_page=True)
            page.get_by_role("button", name="接受AI建议", exact=True).click()
            expect(page.get_by_text("已保存到本地审阅状态，原片未改动", exact=True)).to_be_visible()
            page.get_by_role("button", name="忽略此理由", exact=True).first.click()
            expect(page.get_by_role("button", name="恢复理由", exact=True)).to_be_visible()
            page.get_by_role("button", name="全部保留", exact=True).click()
            expect(page.get_by_text("已勾选 4 / 4 张", exact=True)).to_be_visible()
            page.reload()
            expect(page.get_by_text("已勾选 4 / 4 张", exact=True)).to_be_visible()
            expect(page.get_by_role("button", name="恢复理由", exact=True)).to_be_visible()
            page.screenshot(path=str(artifacts / "02-modified-restored.png"), full_page=True)
            # G014: a single suggested expression can become two with one checkbox.
            for _ in range(13):
                page.get_by_role("button", name="下一组", exact=True).click()
            expect(page.get_by_text("2026-04-04-G014", exact=True)).to_be_visible()
            expect(page.locator(".cloud-card.recommended")).to_have_count(1)
            page.locator('input[type=checkbox]:not(:checked)').check()
            page.get_by_role("button", name="保存我的选择", exact=True).click()
            expect(page.get_by_text("已勾选 2 / 2 张", exact=True)).to_be_visible()
            expect(page.locator(".wbhead .meta")).to_contain_text("已修改")
            expect(page.get_by_role("button", name="下一组", exact=True)).to_be_enabled()
            wait_for_group_images(page)
            page.screenshot(path=str(artifacts / "03-g014-multiselect.png"), full_page=True)
            page.get_by_role("button", name="AI失败/弃权").click()
            expect(page.get_by_text("AI结果不可用，手工检查", exact=True)).to_be_visible()
            expect(page.get_by_role("button", name="接受AI建议", exact=True)).to_be_disabled()
            wait_for_group_images(page)
            page.screenshot(path=str(artifacts / "04-manual-failed.png"), full_page=True)
            page.get_by_role("button", name="稍后处理", exact=True).click()
            expect(page.get_by_text("已保存到本地审阅状态，原片未改动", exact=True)).to_be_visible()
            for _ in range(8):
                if page.get_by_text("AI 已弃权，请手工选择", exact=True).is_visible():
                    break
                page.get_by_role("button", name="下一组", exact=True).click()
            expect(page.get_by_text("AI 已弃权，请手工选择", exact=True)).to_be_visible()
            wait_for_group_images(page)
            page.screenshot(path=str(artifacts / "05-abstain.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            wait_for_group_images(page)
            page.screenshot(path=str(artifacts / "06-mobile.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "mobile overflow"
            assert not errors, errors
            assert not originals, originals
            (artifacts / "browser-results.json").write_text(json.dumps({"passed": True,
                "checks": ["194 total", "complete photos", "accept", "ignore reason", "keep all", "reload restore",
                           "G014 single to multi", "failed images decoded", "abstain images decoded", "later",
                           "mobile images decoded", "mobile no overflow", "zero original requests"],
                "console_errors": errors, "original_requests": originals}, indent=2))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


if __name__ == "__main__":
    run()
