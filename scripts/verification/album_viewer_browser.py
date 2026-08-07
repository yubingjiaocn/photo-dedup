"""Real-browser verification for the review workbench.

Not part of pytest (it needs a Chromium download and a live server); this is the
manual gate described in scripts/verification/README.md. Run the scratch server
first, then::

    python3 scripts/verification/album_viewer_browser.py http://127.0.0.1:18911/review.html

It drives the behaviour a DOM stub cannot express: which URLs the browser really
requests, keyboard focus, CSS transforms after a wheel gesture, and the queue
transitions that follow each decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18911/review.html"
SHOTS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/review-workbench-shots")
GROUPS_IN_FIXTURE = 4          # album_viewer_fixture.py builds this many groups

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + ((" :: " + detail) if detail else ""))


SHOTS.mkdir(parents=True, exist_ok=True)

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    requests: list[str] = []
    page.on("request", lambda r: requests.append(r.url))
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL, wait_until="networkidle")

    # --- 1. the pending queue workbench is the landing page ----------------
    page.wait_for_selector("#workbench.show")
    page.wait_for_timeout(400)
    page.screenshot(path=str(SHOTS / "01-pending-workbench.png"))
    check("opens on the GROUPS pending queue without a click",
          page.evaluate("()=>view") == "GROUPS" and page.evaluate("()=>queue") == "PENDING")
    check("the workbench is visible immediately (no modal to open)",
          page.locator("#workbench.show").count() == 1
          and page.locator("#imgA").is_visible())
    check("the group header states position in the queue",
          "未审 1/4" in page.inner_text("#wbPos"), page.inner_text("#wbPos"))
    check("the group type is shown in Chinese with the internal value beside it",
          "高度相似" in page.inner_text("#wbMeta") and "phash_near" in page.inner_text("#wbMeta"),
          page.inner_text("#wbMeta"))
    check("the action bar is on screen with Chinese wording",
          "保留 AI 推荐" in page.inner_text("#bAccept")
          and "保留当前照片" in page.inner_text("#bPick")
          and "稍后处理" in page.inner_text("#bMark")
          and "撤销上一步" in page.inner_text("#bUndo"))
    check("the shortcut hint is always visible", page.locator(".actionbar .hint").is_visible())
    frames = page.locator("#strip .fs-item").count()
    check("the filmstrip shows every group member", frames == 3, f"{frames} frames")
    strip_fit = page.eval_on_selector_all(
        "#strip img", "els=>els.map(e=>getComputedStyle(e).objectFit)")
    check("filmstrip frames use object-fit: contain",
          bool(strip_fit) and all(f == "contain" for f in strip_fit), str(strip_fit))
    strip_srcs = page.eval_on_selector_all(
        "#strip img", "els=>els.map(e=>new URL(e.src).pathname)")
    check("every filmstrip frame is served by /api/thumb",
          bool(strip_srcs) and all(s.startswith("/api/thumb/") for s in strip_srcs),
          str(strip_srcs))
    originals = [u for u in requests if "/api/original/" in u]
    check("exactly one original is read for the photo on stage",
          len(originals) == 1, str(originals))
    tabs = page.inner_text(".tabs")
    check("queue tabs show their own counts", "未审4" in tabs.replace(" ", ""),
          tabs.replace("\n", " "))

    # --- 2. evidence panel is opt-in ---------------------------------------
    check("推荐依据 starts closed", page.locator("#evidence.open").count() == 0)
    check("no diagnostics leak into the header",
          "43.0" not in page.inner_text("#wbHead"))
    page.keyboard.press("e")
    page.wait_for_timeout(250)
    evidence = page.inner_text("#evidence")
    check("E opens 推荐依据 with Chinese labels",
          page.locator("#evidence.open").count() == 1 and "清晰度评分" in evidence,
          evidence.replace("\n", " ")[:120])
    check("推荐依据 keeps the internal enum in small type",
          "GROUP_KEEPER" in evidence)
    page.screenshot(path=str(SHOTS / "02-evidence-open.png"))
    page.keyboard.press("e")
    page.wait_for_timeout(200)
    check("E closes it again", page.locator("#evidence.open").count() == 0)

    # --- 3. zoom / pan, synced across panes --------------------------------
    box = page.locator("#frameA").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, -400)
    page.wait_for_timeout(300)
    scale_a = page.evaluate("()=>zoom.scale")
    check("wheel zooms in", scale_a > 1.2, f"scale={scale_a:.2f}")
    check("the zoom state is displayed", "%" in page.inner_text("#zoomVal"),
          page.inner_text("#zoomVal"))
    check("the zoomed frame shows a grab cursor",
          page.locator("#frameA.zoomed").count() == 1)
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 - 120, box["y"] + box["height"] / 2 - 60)
    page.mouse.up()
    page.wait_for_timeout(250)
    panned = page.evaluate("()=>[zoom.tx,zoom.ty]")
    check("dragging pans", panned[0] != 0 or panned[1] != 0, str(panned))
    page.mouse.wheel(0, -6000)
    page.wait_for_timeout(300)
    bounded = page.evaluate("()=>[zoom.scale,zoom.tx,zoom.ty]")
    limit = page.evaluate(
        "()=>{const r=document.getElementById('frameA').getBoundingClientRect();"
        "return [(zoom.scale-1)*r.width/2,(zoom.scale-1)*r.height/2]}")
    check("scale is capped and pan stays inside bounds",
          bounded[0] <= 8.001 and abs(bounded[1]) <= limit[0] + 1
          and abs(bounded[2]) <= limit[1] + 1, str(bounded))
    page.screenshot(path=str(SHOTS / "03-zoomed.png"))
    page.keyboard.press("f")
    page.wait_for_timeout(250)
    check("F returns to fit", page.evaluate("()=>zoom.scale") == 1)
    page.keyboard.press("1")
    page.wait_for_timeout(250)
    check("1 goes to 100%", page.evaluate("()=>zoom.scale") > 1)
    page.dblclick("#frameA")
    page.wait_for_timeout(250)
    check("double click returns to fit", page.evaluate("()=>zoom.scale") == 1)

    # --- 4. hold C blinks, Shift+C splits ---------------------------------
    ai_id = page.evaluate("()=>aiKeeperId(stageItems())")
    page.evaluate("()=>showIndex(1)")  # move off the AI keeper
    page.wait_for_timeout(300)
    before = page.evaluate("()=>document.getElementById('imgA').getAttribute('src')")
    page.keyboard.down("c")
    page.wait_for_timeout(350)
    during = page.evaluate("()=>document.getElementById('imgA').getAttribute('src')")
    check("holding C blinks the main photo to the AI recommendation",
          during == f"/api/original/{ai_id}" and during != before, f"{before} -> {during}")
    check("holding C does not open the split pane",
          page.locator("#paneB.hide").count() == 1)
    page.screenshot(path=str(SHOTS / "04-blink.png"))
    page.keyboard.up("c")
    page.wait_for_timeout(350)
    check("releasing C restores the current photo",
          page.evaluate("()=>document.getElementById('imgA').getAttribute('src')") == before)
    page.keyboard.press("Shift+C")
    page.wait_for_timeout(400)
    check("Shift+C opens the two-pane compare",
          page.locator("#paneB").is_visible()
          and "compare" in (page.get_attribute("#stage", "class") or ""))
    box = page.locator("#frameA").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, -400)
    page.wait_for_timeout(300)
    transforms = page.evaluate(
        "()=>[getComputedStyle(document.getElementById('imgA')).transform,"
        "getComputedStyle(document.getElementById('imgB')).transform]")
    check("zoom is synchronised across both compare panes",
          transforms[0] == transforms[1] and "matrix" in transforms[0], str(transforms))
    page.screenshot(path=str(SHOTS / "05-compare-synced-zoom.png"))
    page.keyboard.press("Shift+C")
    page.wait_for_timeout(350)
    check("Shift+C toggles back to one photo", page.locator("#paneB.hide").count() == 1)

    # --- 5. A/P/M leave the pending queue, U undoes ------------------------
    gid = page.evaluate("()=>Number(currentGroup().group_id)")
    page.keyboard.press("a")
    page.wait_for_timeout(700)
    check("A shows a short confirmation toast",
          page.locator("#toast.show").count() == 1, page.inner_text("#toast"))
    check("A takes the group out of the pending queue and advances",
          page.evaluate("()=>Number(currentGroup().group_id)") != gid
          and page.evaluate("()=>queueCounts.PENDING") == 3
          and page.evaluate("()=>queueCounts.DONE") == 1)
    accepted = gid

    gid = page.evaluate("()=>Number(currentGroup().group_id)")
    picked = page.evaluate("()=>Number(currentItem().file_id)")
    page.keyboard.press("p")
    page.wait_for_timeout(700)
    check("P records the shown photo and advances",
          page.evaluate("()=>queueCounts.PENDING") == 2
          and page.evaluate("()=>queueCounts.DONE") == 2
          and page.evaluate("()=>Number(currentGroup().group_id)") != gid)

    gid = page.evaluate("()=>Number(currentGroup().group_id)")
    marked_photo = page.evaluate(
        "()=>{showIndex(2);return Number(currentItem().file_id)}")
    page.wait_for_timeout(300)
    first_member = page.evaluate("()=>Number(stageItems()[0].file_id)")
    check("the test is sitting on a photo that is not the group's first",
          marked_photo != first_member, f"{marked_photo} vs {first_member}")
    page.keyboard.press("m")
    page.wait_for_timeout(700)
    check("M moves the group to the later queue",
          page.evaluate("()=>queueCounts.LATER") == 1
          and page.evaluate("()=>queueCounts.PENDING") == 1
          and page.evaluate("()=>Number(currentGroup().group_id)") != gid)
    marked = gid

    page.keyboard.press("u")
    page.wait_for_timeout(900)
    check("U undoes the mark and jumps back to that very group",
          page.evaluate("()=>Number(currentGroup().group_id)") == marked
          and page.evaluate("()=>queue") == "PENDING"
          and page.evaluate("()=>queueCounts.LATER") == 0
          and page.evaluate("()=>queueCounts.PENDING") == 2,
          f"marked={marked} now={page.evaluate('()=>Number(currentGroup().group_id)')}")
    check("U returns to the photo that was on screen, not the group's first",
          page.evaluate("()=>Number(currentItem().file_id)") == marked_photo,
          f"want {marked_photo}, got "
          f"{page.evaluate('()=>Number(currentItem().file_id)')}")
    check("the main pane really shows that photo again",
          page.evaluate("()=>document.getElementById('imgA').getAttribute('src')")
          == f"/api/original/{marked_photo}")
    page.screenshot(path=str(SHOTS / "06-after-undo.png"))

    # The same has to hold for accept, which likewise names no photo of its own.
    gid = page.evaluate("()=>Number(currentGroup().group_id)")
    accepted_photo = page.evaluate(
        "()=>{showIndex(2);return Number(currentItem().file_id)}")
    page.wait_for_timeout(300)
    page.keyboard.press("a")
    page.wait_for_timeout(700)
    page.keyboard.press("u")
    page.wait_for_timeout(900)
    check("undoing an accept also comes back to the photo that was on screen",
          page.evaluate("()=>Number(currentGroup().group_id)") == gid
          and page.evaluate("()=>Number(currentItem().file_id)") == accepted_photo,
          f"want {accepted_photo}, got "
          f"{page.evaluate('()=>Number(currentItem().file_id)')}")

    # --- 5b. one write at a time, and A needs a recommendation -------------
    # Back to the pending queue with a fresh, undecided library state.
    page.evaluate("()=>goQueue('PENDING')")
    page.wait_for_selector("#workbench.show")
    page.wait_for_timeout(500)
    posts: list[str] = []
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST" else None)

    before = page.evaluate("()=>queueCounts.PENDING")
    gid = page.evaluate("()=>Number(currentGroup().group_id)")
    posts.clear()
    # Two presses inside one animation frame: a real double-tap.
    page.keyboard.down("m")
    page.keyboard.press("m")
    page.keyboard.up("m")
    page.wait_for_timeout(1200)
    check("a double-tapped decision posts once, not twice",
          len(posts) == 1 and page.evaluate("()=>queueCounts.PENDING") == before - 1,
          f"{len(posts)} POSTs, PENDING {before} -> "
          f"{page.evaluate('()=>queueCounts.PENDING')}")
    check("the double tap decided one group only",
          page.evaluate("()=>queueCounts.LATER") == 1)
    check("the action guard is released afterwards",
          page.evaluate("()=>mutationBusy") is False
          and not page.locator("#bMark").is_disabled())

    posts.clear()
    page.keyboard.press("u")
    page.wait_for_timeout(900)
    check("U restored the group it just marked",
          len(posts) == 1
          and page.evaluate("()=>Number(currentGroup().group_id)") == gid
          and page.evaluate("()=>queueCounts.LATER") == 0)

    # The fixture's last group has no AI keeper in this scope.
    no_keeper = page.evaluate(
        "()=>fetch('/api/page?view=GROUPS&queue=PENDING&page=1&page_size=200')"
        ".then(r=>r.json()).then(d=>{const g=d.items.find(g=>"
        "!g.members.some(m=>m.is_keep));return g?g.group_id:null})")
    check("the fixture provides a group with no AI keeper", no_keeper is not None,
          str(no_keeper))
    page.evaluate(f"()=>focusGroup({no_keeper},null)")
    page.wait_for_timeout(700)
    posts.clear()
    check("A is disabled for a group with no AI recommendation",
          page.locator("#bAccept").is_disabled())
    page.keyboard.press("a")
    page.wait_for_timeout(600)
    check("pressing A there posts nothing and says why",
          not posts and "没有 AI 推荐" in page.inner_text("#toast"),
          f"{len(posts)} POSTs :: {page.inner_text('#toast')}")
    check("P still works on that group",
          page.evaluate("()=>!document.getElementById('bPick').disabled"))
    page.keyboard.press("u")
    page.wait_for_timeout(700)

    # --- 6. reload restores queue and position ----------------------------
    restore_gid = page.evaluate("()=>Number(currentGroup().group_id)")
    page.reload(wait_until="networkidle")
    page.wait_for_selector("#workbench.show")
    page.wait_for_timeout(500)
    check("a reload restores the queue and lands on the same group",
          page.evaluate("()=>queue") == "PENDING"
          and page.evaluate("()=>Number(currentGroup().group_id)") == restore_gid,
          f"{restore_gid} -> {page.evaluate('()=>Number(currentGroup().group_id)')}")
    finished_before = page.evaluate("()=>queueCounts.DONE")
    state_file_intact = page.evaluate(
        "()=>fetch('/api/status').then(r=>r.json()).then(s=>s.review_state.reviewed)")
    check("the human state survived the reload",
          state_file_intact == finished_before and finished_before > 0,
          f"{state_file_intact} vs {finished_before}")

    # --- 7. completion state ----------------------------------------------
    # P works for every group; A does not, because one group has no AI keeper.
    for _ in range(GROUPS_IN_FIXTURE + 1):
        if page.evaluate("()=>queueCounts.PENDING") == 0:
            break
        page.keyboard.press("p")
        page.wait_for_timeout(700)
    check("the pending queue can be emptied", page.evaluate("()=>queueCounts.PENDING") == 0)
    page.wait_for_selector("#donePanel.show")
    check("emptying the queue shows the completion panel, not the last group again",
          "本轮审阅完成" in page.inner_text("#donePanel")
          and page.locator("#workbench.show").count() == 0)
    check("the last decision says the round is finished, not 'next group'",
          page.inner_text("#toast") == "本轮审阅完成", page.inner_text("#toast"))
    check("the completion panel counts the other queues and offers jumps",
          f"已完成 {GROUPS_IN_FIXTURE} 组" in page.inner_text("#doneText"),
          page.inner_text("#doneText"))
    check("no original is on screen once the queue is empty",
          not page.evaluate("()=>!!document.getElementById('imgA').getAttribute('src')"))
    page.screenshot(path=str(SHOTS / "07-round-complete.png"))
    page.click("#goDone")
    page.wait_for_selector("#workbench.show")
    page.wait_for_timeout(400)
    check("the done queue is re-enterable from the completion panel",
          page.evaluate("()=>queue") == "DONE"
          and page.evaluate("()=>groups.length") == GROUPS_IN_FIXTURE)
    check("a finished group states its human decision",
          "已完成" in page.inner_text("#wbMeta"), page.inner_text("#wbMeta"))
    page.keyboard.press("u")
    page.wait_for_timeout(900)
    check("U from the done list walks the same history back",
          page.evaluate("()=>queueCounts.DONE") == GROUPS_IN_FIXTURE - 1)

    # --- 8. density + browse lists ----------------------------------------
    page.click('[data-view="ALL"]')
    page.wait_for_selector("#listWrap.show .card")
    page.wait_for_timeout(400)
    comfy = page.eval_on_selector(".card img", "el=>el.getBoundingClientRect().width")
    comfy_cols = page.evaluate(
        "()=>getComputedStyle(document.getElementById('content'))"
        ".gridTemplateColumns.split(' ').length")
    page.screenshot(path=str(SHOTS / "08-browse-comfy.png"))
    page.click("#densityBtn")
    page.wait_for_timeout(400)
    dense = page.eval_on_selector(".card img", "el=>el.getBoundingClientRect().width")
    dense_cols = page.evaluate(
        "()=>getComputedStyle(document.getElementById('content'))"
        ".gridTemplateColumns.split(' ').length")
    check("compact density fits more photos per screen",
          dense < comfy and dense_cols > comfy_cols,
          f"{comfy:.0f}px/{comfy_cols}col -> {dense:.0f}px/{dense_cols}col")
    page.screenshot(path=str(SHOTS / "09-browse-dense.png"))
    check("compact tiles stay clickable", page.locator(".card").first.is_visible())
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(500)
    check("density is remembered across reloads",
          page.evaluate("()=>density") == "dense")
    page.click("#densityBtn")
    page.wait_for_timeout(200)

    # a browse tile is one click, with no duplicate "open large photo" button
    page.click('[data-view="ALL"]')
    page.wait_for_selector("#listWrap.show .card")
    page.wait_for_timeout(400)
    check("browse cards carry no duplicate 查看高清大图 button",
          page.locator(".card button").count() == 0
          and "查看高清大图" not in page.inner_text("#content"))
    requests.clear()
    page.locator(".card").first.click()
    page.wait_for_selector("#workbench.show")
    page.wait_for_timeout(600)
    check("clicking anywhere on a card opens the photo",
          page.locator("#imgA").is_visible()
          and len([u for u in requests if "/api/original/" in u]) == 1,
          str([u for u in requests if "/api/original/" in u]))
    check("review actions are hidden outside the group queues",
          not page.locator("#bAccept").is_visible()
          and not page.locator("#bPick").is_visible())
    before_state = page.evaluate("()=>JSON.stringify(queueCounts)")
    page.keyboard.press("a")
    page.wait_for_timeout(400)
    check("A is inert while browsing",
          page.evaluate("()=>JSON.stringify(queueCounts)") == before_state)
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    check("Esc closes the browse viewer and drops the original",
          page.locator("#workbench.show").count() == 0
          and not page.evaluate(
              "()=>!!document.getElementById('imgA').getAttribute('src')"))

    # --- 9. the write and static boundary, from a real browser -------------
    # Snapshot what the page itself asked for, before this section deliberately
    # probes URLs the page would never touch.
    page_urls = set(requests)
    # A form post is what a hostile page can actually issue cross-origin: it can
    # only send the safelisted content types, so it must be refused.
    form_status = page.evaluate("""async()=>{
     const body=new URLSearchParams({group_id:'1',action:'accept'});
     const r=await fetch('/api/action',{method:'POST',body});
     return r.status}""")
    check("a form-encoded write is refused", form_status == 403, str(form_status))
    text_status = page.evaluate("""async()=>{
     const r=await fetch('/api/action',{method:'POST',
      headers:{'Content-Type':'text/plain'},
      body:JSON.stringify({group_id:1,action:'accept'})});
     return r.status}""")
    check("a text/plain write is refused", text_status == 403, str(text_status))
    check("neither refused write changed anything",
          page.evaluate("()=>fetch('/api/status').then(r=>r.json())"
                        ".then(s=>s.undo_depth)")
          == page.evaluate("()=>undoDepth"))

    # The output directory is not a web root.
    exposure = page.evaluate("""async()=>{
     const names=['summary.txt','delete_local.txt','delete_cloud.json',
      'review_state.json','review_summary.json','performance.txt',
      'inventory.sqlite','thumbs/1.jpg'];
     const out={};
     for(const n of names){out[n]=(await fetch('/'+n)).status}
     out['review.html']=(await fetch('/review.html')).status;
     return out}""")
    check("only the review page is served statically",
          exposure["review.html"] == 200
          and all(status == 404 for name, status in exposure.items()
                  if name != "review.html"),
          str(exposure))

    # --- 10. boundary ------------------------------------------------------
    all_urls = page_urls
    check("no unexpected endpoint is used",
          all(("/api/" in u or u.endswith("review.html") or u.endswith("/favicon.ico"))
              for u in all_urls), str(sorted(all_urls))[:400])
    check("no uncaught page errors", not errors, str(errors))

    check("no state warning banner on a healthy library",
          page.locator("#stateWarn").count() == 1
          and not page.locator("#stateWarn").is_visible())

    browser.close()

# --- 11. the unreadable-state banner --------------------------------------
# A second, throwaway server whose review_state.json cannot be read. Skipped
# unless it is running: `album_viewer_fixture.py 18914 --corrupt-state`.
CORRUPT_URL = URL.replace("18911", "18914").replace("18913", "18914")
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    try:
        page.goto(CORRUPT_URL, wait_until="networkidle")
    except Exception as exc:            # noqa: BLE001 - optional companion server
        print(f"SKIP unreadable-state banner (no server at {CORRUPT_URL}): "
              f"{type(exc).__name__}")
    else:
        page.wait_for_selector("#stateWarn:not([hidden])")
        banner = page.inner_text("#stateWarn")
        check("the reviewer is told their saved decisions could not be read",
              "审阅状态文件无法读取" in banner, banner.replace("\n", " "))
        check("the banner names the file that was kept",
              "review_state.corrupt-" in banner or "原文件已保留" in banner, banner)
        check("the banner is above the workbench and actually visible",
              page.locator("#stateWarn").is_visible()
              and page.locator("#stateWarn").bounding_box()["y"]
              < page.locator("#workbench").bounding_box()["y"])
        check("it does not block review: the workbench is usable",
              page.locator("#workbench.show").count() == 1
              and page.locator("#imgA").is_visible())
        page.screenshot(path=str(SHOTS / "10-state-warning.png"))
        gid = page.evaluate("()=>Number(currentGroup().group_id)")
        page.keyboard.press("m")
        page.wait_for_timeout(900)
        check("a decision still records normally with the banner up",
              page.evaluate("()=>queueCounts.LATER") == 1
              and page.evaluate("()=>Number(currentGroup().group_id)") != gid)
        check("the banner carries no algorithm or runtime diagnostics",
              not any(token in banner for token in
                      ("quality", "face", "缩略图缓存", "GiB", "ms")), banner)
    browser.close()

failed = [name for name, ok, _ in checks if not ok]
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
print(f"screenshots: {SHOTS}")
if failed:
    print("FAILED: " + "; ".join(failed))
    sys.exit(1)
