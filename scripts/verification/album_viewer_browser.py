"""Real-browser verification for the GROUPS album viewer.

Not part of pytest (it needs a Chromium download and a live server); this is the
manual gate described in scripts/verification/README.md. Run the scratch server
first, then::

    .venv/bin/python scripts/verification/album_viewer_browser.py http://127.0.0.1:18911/review.html
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18911/review.html"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + ((" :: " + detail) if detail else ""))


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    requests: list[str] = []
    page.on("request", lambda r: requests.append(r.url))
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL, wait_until="networkidle")

    # --- 1. enlarged, responsive list grid ---------------------------------
    page.wait_for_selector(".card img")
    columns = """()=>getComputedStyle(document.getElementById('content'))
        .gridTemplateColumns.split(' ').length"""
    wide = page.eval_on_selector(".card img", "el=>el.getBoundingClientRect().width")
    wide_cols = page.evaluate(columns)
    check("list thumbnails are much larger than the old 190px", wide > 230, f"{wide:.0f}px")
    check("a wide viewport lays out several columns", wide_cols >= 3, f"{wide_cols} columns")
    page.set_viewport_size({"width": 640, "height": 900})
    page.wait_for_timeout(300)
    narrow = page.eval_on_selector(".card img", "el=>el.getBoundingClientRect().width")
    narrow_cols = page.evaluate(columns)
    check("grid reflows to fewer columns when narrow", narrow_cols < wide_cols,
          f"{narrow_cols} vs {wide_cols} columns")
    check("narrow tiles still fit the viewport", narrow <= 640, f"{narrow:.0f}px")
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.wait_for_timeout(300)

    # --- 2. GROUPS album layout -------------------------------------------
    page.click('[data-view="GROUPS"]')
    page.wait_for_selector(".group.focused")
    requests.clear()
    page.keyboard.press("Enter")
    page.wait_for_selector("#lightbox.open")
    page.wait_for_timeout(400)
    frames = page.locator("#filmstrip .fs-item").count()
    check("filmstrip shows every group member", frames == 3, f"{frames} frames")
    check("exactly one main photo pane is visible",
          page.locator("#leftPane").is_visible() and not page.locator("#rightPane").is_visible())
    check("current frame is marked",
          page.locator("#filmstrip .fs-item.current").count() == 1)
    check("AI keeper is labelled in the filmstrip",
          page.locator("#filmstrip .badge.ai").count() == 1)
    originals = [u for u in requests if "/api/original/" in u]
    check("viewer opens exactly one original", len(originals) == 1, str(originals))
    # Asserted on the DOM, not on the network: the grid already warmed the HTTP
    # cache, so a correct filmstrip legitimately issues zero new requests.
    strip_srcs = page.eval_on_selector_all(
        "#filmstrip img", "els=>els.map(e=>new URL(e.src).pathname)")
    check("every filmstrip frame is served by /api/thumb",
          bool(strip_srcs) and all(s.startswith("/api/thumb/") for s in strip_srcs),
          str(strip_srcs))
    check("the filmstrip never references an original",
          not any("/api/original" in s for s in strip_srcs))

    # --- 3. keyboard in the open viewer -----------------------------------
    first = page.locator("#filmstrip .fs-item.current").get_attribute("data-file-id")
    page.keyboard.press("l")
    page.wait_for_timeout(300)
    second = page.locator("#filmstrip .fs-item.current").get_attribute("data-file-id")
    check("L advances within the group", first != second, f"{first} -> {second}")
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(300)
    check("ArrowLeft goes back",
          page.locator("#filmstrip .fs-item.current").get_attribute("data-file-id") == first)
    page.keyboard.press("h")
    page.wait_for_timeout(300)
    check("H also steps",
          page.locator("#filmstrip .fs-item.current").get_attribute("data-file-id") != first)

    # clicking a frame jumps to it
    page.locator("#filmstrip .fs-item").first.click()
    page.wait_for_timeout(300)
    check("clicking a filmstrip frame selects it",
          page.locator("#filmstrip .fs-item").first.get_attribute("aria-current") == "true")

    # C compare: temporary two-pane split, toggles back
    page.keyboard.press("c")
    page.wait_for_timeout(400)
    check("C opens the temporary compare split",
          page.locator("#rightPane").is_visible()
          and "compare" in (page.get_attribute("#viewer", "class") or ""))
    page.keyboard.press("c")
    page.wait_for_timeout(400)
    check("C toggles back to one main photo", not page.locator("#rightPane").is_visible())

    # P: human keeper on the shown photo, viewer stays open on the next group
    gid_before = page.locator(".group.focused").get_attribute("data-group-id")
    shown = page.locator("#filmstrip .fs-item.current").get_attribute("data-file-id")
    page.keyboard.press("p")
    page.wait_for_timeout(900)
    check("P keeps the album viewer open", page.locator("#lightbox.open").count() == 1)
    gid_after = page.locator(".group.focused").get_attribute("data-group-id")
    check("P auto-advances to the next group", gid_before != gid_after,
          f"{gid_before} -> {gid_after}")
    state = page.evaluate("()=>JSON.stringify(reviewState)")
    check("P recorded a human keeper", f'"file_id":{shown}' in state.replace(" ", ""), state)
    prev_group = page.locator(f'.group[data-group-id="{gid_before}"]')
    check("previous group shows the human keeper marker",
          prev_group.locator(".badge.human").count() == 1
          and "已审-人工" in (prev_group.locator(".tag").first.inner_text()))

    # A: accept AI keeper, still open, advances again
    gid_before = page.locator(".group.focused").get_attribute("data-group-id")
    page.keyboard.press("a")
    page.wait_for_timeout(900)
    check("A keeps the viewer open and advances",
          page.locator("#lightbox.open").count() == 1
          and page.locator(".group.focused").get_attribute("data-group-id") != gid_before)
    check("A marked the group reviewed-by-AI",
          "已审-AI" in page.locator(f'.group[data-group-id="{gid_before}"] .tag')
          .first.inner_text())

    # M: mark for later, still open, advances
    gid_before = page.locator(".group.focused").get_attribute("data-group-id")
    page.keyboard.press("m")
    page.wait_for_timeout(900)
    check("M keeps the viewer open and advances",
          page.locator("#lightbox.open").count() == 1
          and page.locator(".group.focused").get_attribute("data-group-id") != gid_before)
    check("M marked the group for later",
          "[稍后]" in page.locator(f'.group[data-group-id="{gid_before}"] .tag')
          .first.inner_text())

    # U: clear, viewer stays on the same group/photo
    gid_now = page.locator(".group.focused").get_attribute("data-group-id")
    page.keyboard.press("u")
    page.wait_for_timeout(900)
    check("U stays on the same group with the viewer open",
          page.locator("#lightbox.open").count() == 1
          and page.locator(".group.focused").get_attribute("data-group-id") == gid_now)

    # Esc closes and releases the original
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    check("Esc closes the viewer", page.locator("#lightbox.open").count() == 0)
    check("closing drops the original src",
          page.evaluate("()=>!document.getElementById('leftImage').getAttribute('src')"))

    # --- 5. non-GROUPS viewer still works ---------------------------------
    # ALL is the flat timeline; the last two photos there are ungrouped, which
    # exercises the single-photo viewer path (no filmstrip, no group actions).
    page.click('[data-view="ALL"]')
    page.wait_for_function(
        "()=>view==='ALL'&&!document.getElementById('content').classList.contains('groups')"
        "&&document.querySelectorAll('.card img').length>0")
    page.wait_for_timeout(600)  # let the smooth-scroll settle before clicking
    requests.clear()
    ungrouped = page.evaluate(
        "()=>{const u=visibleItems.filter(r=>r.group_id==null);return u.length?u[0].file_id:null}")
    check("the flat timeline contains ungrouped photos", ungrouped is not None, str(ungrouped))
    page.evaluate(f"()=>openViewer({ungrouped})")
    page.wait_for_selector("#lightbox.open")
    page.wait_for_timeout(600)
    check("non-GROUPS viewer opens one original",
          len([u for u in requests if "/api/original/" in u]) == 1,
          str([u for u in requests if "/api/original/" in u]))
    check("an ungrouped photo hides the filmstrip",
          not page.locator("#filmstrip").is_visible())
    check("group review buttons are hidden outside GROUPS",
          not page.locator("#vAccept").is_visible()
          and not page.locator("#vPick").is_visible())
    page.keyboard.press("p")
    page.wait_for_timeout(400)
    check("P is inert outside GROUPS (no action, viewer unchanged)",
          page.locator("#lightbox.open").count() == 1
          and page.evaluate("()=>Object.keys(reviewState).length") == 0)
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    check("Esc closes the non-GROUPS viewer", page.locator("#lightbox.open").count() == 0)

    # A grouped photo reached from the flat timeline still gets its group
    # filmstrip (members fetched via /api/group), but no review buttons.
    grouped = page.evaluate(
        "()=>{const g=visibleItems.filter(r=>r.group_id!=null);return g.length?g[0].file_id:null}")
    page.evaluate(f"()=>openViewer({grouped})")
    page.wait_for_selector("#lightbox.open")
    page.wait_for_timeout(600)
    check("a grouped photo opened from ALL still gets a filmstrip",
          page.locator("#filmstrip .fs-item").count() == 3)
    check("the AI keeper is labelled in the flat timeline too",
          page.locator(".card .badge.ai").count() >= 1,
          str(page.locator(".card .badge.ai").count()))
    check("but still no group review buttons outside GROUPS",
          not page.locator("#vAccept").is_visible())
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)

    # --- 6. boundary -------------------------------------------------------
    all_urls = set(requests)
    check("no page/path leakage in requests",
          all(("/api/" in u or u.endswith("review.html") or u.endswith("/favicon.ico"))
              for u in all_urls), str(sorted(all_urls))[:400])
    check("no uncaught page errors", not errors, str(errors))

    browser.close()

failed = [name for name, ok, _ in checks if not ok]
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED: " + "; ".join(failed))
    sys.exit(1)
