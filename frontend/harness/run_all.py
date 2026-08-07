"""Run the production browser harness against an in-process fixture server.

Starts the real ``src.review_server`` (serving the committed build installed by
Stage 3's code path) on a loopback port in a background thread, then runs
``browser_check.run``/``run_queue_workflow`` for both viewers in the same process.
This avoids spawning a detached server subprocess (which SIGURG-trips some
shells) while exercising exactly the production HTTP surface.

    python3 frontend/harness/run_all.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playwright.sync_api import sync_playwright  # noqa: E402

from scripts.verification.album_viewer_fixture import build  # noqa: E402
from src import review_server  # noqa: E402
import browser_check  # noqa: E402


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="photo-dedup-harness-"))
    output = work / "output"
    build(output, work / "photos")
    server, url = review_server.start_server(output, port=0)
    base = url.rsplit("/review.html", 1)[0] + "/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    browser_check.BASE = base
    failures: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=browser_check.CHROME, headless=True)
            for viewer in ("panzoom", "osd"):
                ctx = browser.new_context(viewport={"width": 1440, "height": 900})
                try:
                    browser_check.run(ctx.new_page(), f"prod-{viewer}", viewer)
                except AssertionError as exc:
                    failures.append(f"prod-{viewer}: {exc}")
                ctx.close()
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            try:
                browser_check.run_queue_workflow(ctx.new_page())
            except AssertionError as exc:
                failures.append(f"prod-workflow: {exc}")
            ctx.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server.review_data.close()
        shutil.rmtree(work, ignore_errors=True)

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("ALL PRODUCTION BROWSER GATES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
