"""Real Chromium QA against an explicit four-group development-only manifest."""
from __future__ import annotations

import http.server
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cloud_review import CloudReviewData
from src.learning_io import atomic_json, read_json
from src.review_events import HoldoutPolicy
from src.review_server import create_handler


def main():
    evidence = ROOT / "artifacts/learning-loop-mvp"
    temporary = Path(tempfile.mkdtemp(prefix="browser-", dir=evidence))
    frozen = read_json(ROOT / "artifacts/cloud-judge-batch/frozen.json")
    policy = HoldoutPolicy(frozen)
    frozen["groups"] = [g for g in frozen["groups"] if g["group"] in {"2026-04-04-G002", "2026-04-04-G004"}]
    assert all(not policy.blocked(g["group"], g["date"], g["frames"]) for g in frozen["groups"])
    atomic_json(temporary / "development.json", frozen)
    data = CloudReviewData(temporary / "output", ROOT / "artifacts/cloud-judge-batch/authorized-scene-run-v3", temporary / "development.json")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), create_handler(data))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    errors, original_requests, save_ids = [], [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path="/usr/bin/google-chrome-stable")
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: original_requests.append(request.url) if "/api/original" in request.url else None)
            def fail_first_response(route):
                save_ids.append(route.request.post_data_json["request_id"])
                response = route.fetch()
                if len(save_ids) == 1:
                    # Commit succeeded but HTTP response was lost: the harder retry case.
                    route.fulfill(status=503, content_type="application/json", body='{"error":"simulated lost save response"}')
                else:
                    route.fulfill(response=response)
            page.route("**/api/action", fail_first_response)
            page.goto(f"http://127.0.0.1:{server.server_port}/review.html?mode=cloud")
            expect(page.get_by_text("2026-04-04-G002", exact=True)).to_be_visible()
            page.get_by_role("button", name="接受AI建议", exact=True).click()
            expect(page.get_by_role("alert")).to_contain_text("simulated lost save response")
            assert len(data.journal.events()) == 1
            page.get_by_role("button", name="接受AI建议", exact=True).click()
            expect(page.get_by_text("已保存到本地审阅状态，原片未改动", exact=True)).to_be_visible()
            assert save_ids[0] == save_ids[1]
            assert len(data.journal.events()) == 1
            page.reload()
            expect(page.locator(".wbhead .meta")).to_contain_text("已接受AI")
            assert len(data.journal.events()) == 1
            page.get_by_role("button", name="全部保留", exact=True).click()
            expect(page.locator(".wbhead .meta")).to_contain_text("已修改")
            assert len(data.journal.events()) == 2
            page.reload()
            expect(page.get_by_text("已勾选 3 / 3 张", exact=True)).to_be_visible()
            page.wait_for_function("() => [...document.querySelectorAll('.cloud-card img')].every(i => i.complete && i.naturalWidth > 0)")
            assert len(data.journal.events()) == 2
            page.screenshot(path=str(evidence / "browser-v3-save-retry.png"), full_page=True)
            assert not errors and not original_requests
            atomic_json(evidence / "browser-v3-results.json", {"passed": True, "retry_request_ids": save_ids, "failed_response_committed_once": True, "event_count": 2, "reload_duplicates": 0, "original_requests": original_requests, "console_errors": errors, "groups": len(frozen["groups"]), "holdout_groups_loaded": 0})
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
        data.close()


if __name__ == "__main__":
    main()
