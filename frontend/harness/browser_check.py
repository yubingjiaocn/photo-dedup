"""Real Chromium acceptance test for the production review UI.

Drives the *built* Preact app served by the *real* ``src.review_server`` (see
``fixture_server.py``) and asserts the behaviour a DOM stub cannot express:
which originals actually cross the network, centred contain-fit, wheel/drag,
F/1, dual-pane sync, hold-C blink, the compiled bundle's lifecycle, and the
exact HDD request discipline.

It runs the same gate against **both** viewers -- Panzoom (production) and
OpenSeadragon (``?viewer=osd``) -- so the viewer decision rests on identical
evidence. The strict request-sequence and blink-release gates that decide HDD
correctness run against the production Panzoom build.

    python3 frontend/harness/browser_check.py <base-url> [screenshot-dir] [viewer]

Exits non-zero on the first failed assertion; writes screenshots and
``artifacts/browser-results.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import Page, sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18931/"
SHOTS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "artifacts" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)
CHROME = "/usr/bin/google-chrome-stable"

results: dict[str, dict[str, object]] = {}


def state(page: Page) -> dict:
    return page.evaluate("() => globalThis.__review.states()")


def frame_and_image_centres(page: Page, osd: bool) -> list[float]:
    selector = "#hostA canvas" if osd else "#hostA img"
    return page.evaluate(
        """(sel) => {
          const f = document.querySelector('#hostA').getBoundingClientRect();
          const el = document.querySelector(sel);
          const i = el ? el.getBoundingClientRect() : f;
          return [f.x + f.width/2, f.y + f.height/2, i.x + i.width/2, i.y + i.height/2];
        }""",
        selector,
    )


def assert_fit_centered(page: Page, name: str, phase: str, osd: bool) -> None:
    a = state(page)["a"]
    assert abs(a["scale"] - 1) < 0.01 and abs(a["x"]) < 1 and abs(a["y"]) < 1, (name, phase, "state", a)
    centres = frame_and_image_centres(page, osd)
    assert abs(centres[0] - centres[2]) < 2 and abs(centres[1] - centres[3]) < 2, (name, phase, "centre", centres)


def run(page: Page, name: str, viewer: str) -> None:
    osd = viewer == "osd"
    originals: list[str] = []
    errors: list[str] = []
    page.on("request", lambda r: originals.append(r.url) if "/api/original/" in r.url else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("requestfailed", lambda r: errors.append("requestfailed " + r.url) if "/api/" not in r.url else None)

    suffix = "?viewer=osd" if osd else ""
    page.goto(urljoin(BASE, "review.html" + suffix), wait_until="domcontentloaded")
    page.wait_for_function("() => globalThis.__review && globalThis.__review.state().groups.length > 0", timeout=15000)
    page.wait_for_timeout(400)

    # --- landing: pending queue, one original, three thumbs, no prefetch -----
    assert not errors, (name, "boot", errors)
    st = page.evaluate("() => globalThis.__review.state()")
    assert st["view"] == "GROUPS" and st["queue"] == "PENDING", (name, st)
    assert page.locator("#strip .fs-item").count() == 3, name
    assert page.evaluate("() => globalThis.__review.kind()") == ("osd" if osd else "panzoom"), name
    initial = list(originals)
    assert len(initial) == 1, (name, "initial originals", initial)
    assert_fit_centered(page, name, "initial", osd)
    page.screenshot(path=str(SHOTS / f"{name}-01-landing.png"))

    # --- exact request sequence /2 -> /1 -> /5 -------------------------------
    # Move off the keeper (members[0]=2) to members[1]=1, then to the next group.
    page.keyboard.press("l")
    page.wait_for_timeout(200)
    assert len(originals) == 2, (name, "after L", originals)
    assert_fit_centered(page, name, "switch-photo", osd)

    host = page.locator("#hostA")
    box = host.bounding_box()
    assert box
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    # --- wheel zoom ----------------------------------------------------------
    page.mouse.move(cx, cy)
    for _ in range(3):
        page.mouse.wheel(0, -500)
    page.wait_for_timeout(150)
    zoomed = state(page)
    assert zoomed["a"]["scale"] > 1.2, (name, "wheel", zoomed)
    # A zoomed frame flips to the grab cursor class (Panzoom frame; OSD canvas).
    assert page.locator(".frame.zoomed, .osd.zoomed").count() >= 1, (name, "no zoomed class")
    page.screenshot(path=str(SHOTS / f"{name}-02-zoomed.png"))

    # --- drag pan ------------------------------------------------------------
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx - 100, cy - 55, steps=4)
    page.mouse.up()
    page.wait_for_timeout(150)
    dragged = state(page)
    assert dragged["a"] != zoomed["a"], (name, "drag", zoomed, dragged)

    # --- F resets, 1 goes to 100% -------------------------------------------
    page.keyboard.press("f")
    page.wait_for_timeout(250)
    assert_fit_centered(page, name, "reset", osd)
    page.keyboard.press("1")
    page.wait_for_timeout(200)
    assert state(page)["a"]["scale"] > 1, (name, "hundred", state(page))
    page.keyboard.press("f")
    page.wait_for_timeout(200)

    # --- compare split, synced zoom -----------------------------------------
    before_compare = len(originals)
    page.locator("#compareBtn").click()
    page.wait_for_timeout(250)
    assert page.locator("#paneB").is_visible(), (name, "compare not visible")
    # The AI keeper (file 2) was retained from the active group; opening compare
    # must not fetch it again.
    assert len(originals) == before_compare, (name, "compare fetched", originals[before_compare:])
    box = host.bounding_box()
    assert box
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, -350)
    page.wait_for_timeout(150)
    synced = state(page)
    assert abs(synced["a"]["scale"] - synced["b"]["scale"]) < 0.02, (name, "sync", synced)
    page.screenshot(path=str(SHOTS / f"{name}-03-compare-sync.png"))
    page.locator("#compareBtn").click()
    page.wait_for_timeout(150)
    page.keyboard.press("f")
    page.wait_for_timeout(150)
    assert_fit_centered(page, name, "compare-close", osd)

    # --- three blink cycles add exactly zero originals -----------------------
    before_blink = len(originals)
    for _ in range(3):
        page.keyboard.down("c")
        page.wait_for_timeout(90)
        page.keyboard.up("c")
        page.wait_for_timeout(90)
    assert originals[before_blink:] == [], (name, "blink added requests", originals[before_blink:])
    assert_fit_centered(page, name, "blink-release", osd)
    page.screenshot(path=str(SHOTS / f"{name}-04-after-blink.png"))

    # --- next group: exactly one new original (/5), old group released -------
    before_switch = len(originals)
    old_group_urls = page.evaluate(
        "() => globalThis.__review.state().groups[0].members.map(m => '/api/original/' + m.file_id)"
    )
    page.locator("#wbNext").click()
    page.wait_for_timeout(300)
    delta = originals[before_switch:]
    assert len(delta) == 1, (name, "group switch delta", delta)
    assert_fit_centered(page, name, "switch-group", osd)
    # The previous group's decoded <img> pool is released (Panzoom only; OSD owns
    # its own canvas world and reports 0).
    if not osd:
        remaining = page.evaluate(
            """(urls) => {
              const pathOf = (img) => {
                const src = img.getAttribute('src') || '';
                try { return new URL(src, location.href).pathname; } catch { return ''; }
              };
              const store = document.getElementById('panzoomImageStore');
              const held = store ? [...store.querySelectorAll('img')].map(pathOf) : [];
              const live = [...document.querySelectorAll('.frame img')].map(pathOf);
              const all = new Set([...held, ...live].filter(Boolean));
              return urls.filter(u => all.has(u));
            }""",
            old_group_urls,
        )
        assert remaining == [], (name, "old group not released", remaining)

    # The full sequence the harness drove: /2 (initial), /1 (L), /5 (next group).
    paths = [u.rsplit("/api/original/", 1)[1] for u in originals]
    assert paths == ["2", "1", "5"], (name, "exact sequence", paths)

    assert not errors, (name, "page errors", errors)
    page.screenshot(path=str(SHOTS / f"{name}-05-next-group.png"))
    results[name] = {
        "viewer": viewer,
        "originalRequests": originals,
        "sequence": paths,
        "errors": errors,
        "kind": page.evaluate("() => globalThis.__review.kind()"),
    }
    print(f"PASS {name}: sequence {paths}, {len(originals)} originals, 0 errors")


def run_queue_workflow(page: Page) -> None:
    """A/P/M/U, cross-page advance, reload restore, completion, mutation guard,
    CSRF + static boundary -- the production queue contract, in a real browser."""
    name = "prod-workflow"
    errors: list[str] = []
    posts: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST" else None)
    page.goto(urljoin(BASE, "review.html"), wait_until="domcontentloaded")
    page.wait_for_function("() => globalThis.__review && globalThis.__review.state().groups.length > 0", timeout=15000)
    page.wait_for_timeout(300)

    ev = lambda expr: page.evaluate("() => " + expr)  # noqa: E731
    gid = ev("globalThis.__review.state().groups[globalThis.__review.state().gIndex].group_id")

    # A: accept keeps the AI recommendation, group leaves PENDING, advances.
    page.keyboard.press("a")
    page.wait_for_timeout(600)
    assert ev("globalThis.__review.state().queueCounts.PENDING") == 3, (name, "A pending")
    assert ev("globalThis.__review.state().queueCounts.DONE") == 1, (name, "A done")
    assert ev("globalThis.__review.state().groups[globalThis.__review.state().gIndex].group_id") != gid, (name, "A advance")

    # P: pick the shown photo.
    page.keyboard.press("p")
    page.wait_for_timeout(600)
    assert ev("globalThis.__review.state().queueCounts.DONE") == 2, (name, "P done")

    # M: mark later, then U undoes back to that exact group + photo.
    gid = ev("globalThis.__review.state().groups[globalThis.__review.state().gIndex].group_id")
    marked_photo = page.evaluate("() => { globalThis.__review.showIndex(2); return globalThis.__review.state().groups[globalThis.__review.state().gIndex].members[2].file_id; }")
    page.wait_for_timeout(200)
    page.keyboard.press("m")
    page.wait_for_timeout(600)
    assert ev("globalThis.__review.state().queueCounts.LATER") == 1, (name, "M later")
    page.keyboard.press("u")
    page.wait_for_timeout(800)
    assert ev("globalThis.__review.state().groups[globalThis.__review.state().gIndex].group_id") == gid, (name, "U group")
    assert ev("globalThis.__review.state().queueCounts.LATER") == 0, (name, "U later cleared")
    cur = page.evaluate("() => { const s=globalThis.__review.state(); const g=s.groups[s.gIndex]; return g.members[Math.min(s.mIndex,g.members.length-1)].file_id; }")
    assert cur == marked_photo, (name, "U returns to photo", cur, marked_photo)

    # Mutation guard: a double-tapped decision posts once.
    page.evaluate("() => globalThis.__review.goQueue('PENDING')")
    page.wait_for_selector("#workbench")
    page.wait_for_timeout(400)
    before = ev("globalThis.__review.state().queueCounts.PENDING")
    posts.clear()
    page.keyboard.down("m")
    page.keyboard.press("m")
    page.keyboard.up("m")
    page.wait_for_timeout(900)
    assert len(posts) == 1, (name, "double-tap posted", len(posts))
    assert ev("globalThis.__review.state().queueCounts.PENDING") == before - 1, (name, "double-tap counted twice")
    page.keyboard.press("u")
    page.wait_for_timeout(700)

    # Reload restores queue + group position.
    restore_gid = ev("globalThis.__review.state().groups[globalThis.__review.state().gIndex].group_id")
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("() => globalThis.__review && globalThis.__review.state().groups.length > 0", timeout=15000)
    page.wait_for_timeout(400)
    assert ev("globalThis.__review.state().queue") == "PENDING", (name, "reload queue")
    assert ev("globalThis.__review.state().groups[globalThis.__review.state().gIndex].group_id") == restore_gid, (name, "reload group")

    # CSRF: a form-encoded write is refused; a text/plain write is refused.
    form_status = page.evaluate(
        "async () => { const b=new URLSearchParams({group_id:'1',action:'accept'}); const r=await fetch('/api/action',{method:'POST',body:b}); return r.status; }"
    )
    assert form_status == 403, (name, "form write not refused", form_status)

    # Static boundary: only the UI bundle is served; working files are 404.
    exposure = page.evaluate(
        """async () => {
          const names=['summary.txt','delete_local.txt','review_state.json','review_summary.json','inventory.sqlite','thumbs/1.jpg','review_assets.json'];
          const out={};
          for (const n of names) out[n]=(await fetch('/'+n)).status;
          out['review.html']=(await fetch('/review.html')).status;
          return out;
        }"""
    )
    assert exposure["review.html"] == 200, (name, "page not served", exposure)
    assert all(v == 404 for k, v in exposure.items() if k != "review.html"), (name, "working file exposed", exposure)

    # Empty the queue -> completion panel.
    for _ in range(6):
        if ev("globalThis.__review.state().queueCounts.PENDING") == 0:
            break
        page.keyboard.press("p")
        page.wait_for_timeout(500)
    assert ev("globalThis.__review.state().queueCounts.PENDING") == 0, (name, "queue not emptied")
    page.wait_for_selector("#donePanel")
    assert "本轮审阅完成" in page.inner_text("#donePanel"), (name, "no completion panel")
    page.screenshot(path=str(SHOTS / "prod-workflow-complete.png"))

    assert not errors, (name, "page errors", errors)
    results[name] = {"errors": errors}
    print(f"PASS {name}: A/P/M/U, guard, reload, CSRF, static boundary, completion")


def main() -> int:
    viewers = sys.argv[3:] if len(sys.argv) > 3 else ["panzoom", "osd"]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME, headless=True)
        for viewer in viewers:
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            run(page, f"prod-{viewer}", viewer)
            context.close()
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        run_queue_workflow(context.new_page())
        context.close()
        browser.close()
    (SHOTS.parent / "browser-results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: {"sequence": v.get("sequence"), "errors": v["errors"]} for k, v in results.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
