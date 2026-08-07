"""Behavioural gates for the GROUPS album viewer and the enlarged list grid.

The review UI is a template string, so these tests do two things CI can express
without a browser:

* run the embedded script in ``node`` behind a tiny DOM stub and assert on the
  *output of its pure render helpers* (filmstrip markup, tile markup, keyboard
  action mapping) instead of grepping for substrings;
* assert the security boundary that matters here: the filmstrip and the grid
  address ``/api/thumb/<id>.jpg`` only, ``/api/original/<id>`` appears only on
  the explicitly opened viewer panes, and no source path is ever rendered.

Section 7 additionally pins the *absence* of algorithm and runtime diagnostics
from the visible UI: quality scores, face counts, grouping reasons, per-page
remainders, thumbnail-cache sizes and which physical disk a read comes from are
not the reviewer's business, even though the underlying API still returns them
and ``review.html`` still carries the (hidden) performance panel.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from src import review_page, review_template

_SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.DOTALL)

# Minimal DOM/fetch stub: enough for the script's top-level wiring to run so the
# render helpers become callable. Deliberately dumb -- it must not emulate the
# browser, only stop the module scope from throwing.
_DOM_STUB = """
function makeEl(id){
 const el={id,children:[],style:{},dataset:{},textContent:'',innerHTML:'',value:'',
  disabled:false,onclick:null,onchange:null,
  classList:{_s:new Set(),add(...c){c.forEach(x=>el.classList._s.add(x))},
   remove(...c){c.forEach(x=>el.classList._s.delete(x))},
   contains(c){return el.classList._s.has(c)},
   toggle(c,on){on?el.classList._s.add(c):el.classList._s.delete(c)}},
  querySelector(){return null},querySelectorAll(){return []},
  scrollIntoView(){},getAttribute(){return null},setAttribute(){},removeAttribute(){}};
 return el;
}
const _els=new Map();
globalThis.document={
 getElementById(id){if(!_els.has(id))_els.set(id,makeEl(id));return _els.get(id)},
 querySelectorAll(){return []},
 addEventListener(type,fn){globalThis.__handlers=globalThis.__handlers||{};
  globalThis.__handlers[type]=fn},
};
globalThis.alert=msg=>{globalThis.__alerts=(globalThis.__alerts||[]).concat([msg])};
globalThis.fetch=async()=>{throw new Error('network disabled in test harness')};
"""


def _embedded_js() -> str:
    data = {"groups": [], "queues": {"MAYBE": [], "UNKNOWN": []},
            "delete_paths": [], "total_delete_bytes": 0}
    html = review_page.render_html(data, Path("/tmp"), review_limit=0)
    scripts = _SCRIPT_RE.findall(html)
    assert scripts, "no <script> block in the rendered review page"
    return "\n".join(scripts)


def _run_js(probe: str) -> dict:
    """Execute the page script plus ``probe`` in node; return the JSON it prints."""
    if shutil.which("node") is None:  # pragma: no cover - node is a declared gate
        pytest.skip("node is required for the embedded-JS gates")
    source = f"{_DOM_STUB}\n{_embedded_js()}\n{probe}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as handle:
        handle.write(source)
        path = Path(handle.name)
    try:
        result = subprocess.run(["node", str(path)], capture_output=True,
                                text=True, timeout=20)
        assert result.returncode == 0, (
            f"node failed: {result.returncode}\nSTDOUT:{result.stdout}\n"
            f"STDERR:{result.stderr}"
        )
        return json.loads(result.stdout.strip().splitlines()[-1])
    finally:
        path.unlink(missing_ok=True)


_MEMBERS = """
const members=[
 {file_id:11,group_id:7,basename:'IMG_0001.jpg',decision:'AUTO_REMOVE',thumb:'ok',
  width:4000,height:3000,quality_score:41.2,face_count:1,is_keep:false,reason:'dup'},
 {file_id:12,group_id:7,basename:'IMG_0002.jpg',decision:'KEEP',thumb:'ok',
  width:4000,height:3000,quality_score:71.5,face_count:1,is_keep:true,reason:'best'},
 {file_id:13,group_id:7,basename:'IMG_0003.jpg',decision:'MAYBE',thumb:'missing',
  thumb_error:'decode failed',width:4000,height:3000,quality_score:55.0,
  face_count:1,is_keep:false,reason:'low margin'}];
"""


def test_embedded_js_is_syntactically_valid_and_runs():
    """The whole script must parse *and* execute its top-level wiring."""
    out = _run_js("console.log(JSON.stringify({view,page,size,ok:true}))")
    assert out == {"view": "ALL", "page": 1, "size": 100, "ok": True}


def test_review_template_module_is_the_single_source_of_the_page():
    assert review_page.PAGE_TEMPLATE is review_template.PAGE_TEMPLATE
    assert review_page._PAGE_TEMPLATE is review_template.PAGE_TEMPLATE


# --- 1. the list grid is large and responsive ------------------------------

def test_list_tiles_are_large_and_responsive_not_fixed_190px():
    css = review_template.PAGE_TEMPLATE
    assert "grid-template-columns:repeat(auto-fill,minmax(var(--tile),1fr))" in css
    assert "--tile:clamp(" in css and "--tileh:clamp(" in css
    assert "@media(max-width:700px)" in css
    # The old fixed 190x160 thumbnail box must be gone from every tile rule.
    assert "width:190px" not in css and "height:160px" not in css


def test_grid_tiles_and_missing_placeholders_only_use_the_thumb_api():
    probe = _MEMBERS + """
const html=members.map((m,i)=>photoTile(m,i,12,11)).join('');
console.log(JSON.stringify({
 html,
 thumbUrls:(html.match(/\\/api\\/thumb\\/\\d+\\.jpg/g)||[]),
 originals:(html.match(/\\/api\\/original/g)||[]).length}));
"""
    out = _run_js(probe)
    assert out["thumbUrls"] == ["/api/thumb/11.jpg", "/api/thumb/12.jpg"]
    assert out["originals"] == 0
    # The un-thumbnailed member degrades to a placeholder, never a source read.
    assert "缩略图不可用" in out["html"]
    assert "width:190px" not in out["html"]
    # The placeholder stays short and human: no decoder exception detail, no
    # lecture about which disk is or is not being read.
    assert "decode failed" not in out["html"]
    assert "机械盘" not in out["html"] and "HDD" not in out["html"]


# --- 2. album layout: one main photo + clickable filmstrip -----------------

def test_filmstrip_marks_current_ai_keeper_and_human_keeper():
    probe = _MEMBERS + """
const html=filmstripHtml(members,13,12,11);
console.log(JSON.stringify({html,
 items:(html.match(/class="fs-item[^"]*"/g)||[]),
 thumbs:(html.match(/\\/api\\/thumb\\/\\d+\\.jpg/g)||[]),
 originals:(html.match(/\\/api\\/original/g)||[]).length,
 current:(html.match(/aria-current="true"/g)||[]).length}));
"""
    out = _run_js(probe)
    html = out["html"]
    # One frame per group member, exactly one marked current (file 13).
    assert len(out["items"]) == 3
    assert out["current"] == 1
    assert 'data-file-id="13"' in html and "fs-item current" in html
    # AI keeper (12) and human keeper (11) are both labelled, distinguishably.
    assert 'class="badge ai">🤖AI</span>' in html
    assert 'class="badge human">👤人工</span>' in html
    assert 'class="badge cur">当前</span>' in html
    # Every frame is clickable and jumps by index.
    assert html.count("onclick=\"showViewerIndex(") == 3
    # Filmstrip is SSD-cache only: two thumbs, no original, placeholder for the third.
    assert out["thumbs"] == ["/api/thumb/11.jpg", "/api/thumb/12.jpg"]
    assert out["originals"] == 0
    assert "fs-miss" in html


def test_filmstrip_escapes_basenames_and_never_renders_a_path():
    probe = """
const evil=[{file_id:9,group_id:1,basename:'<img src=x onerror=alert(1)>.jpg',
 decision:'MAYBE',thumb:'ok',is_keep:false,path:'C:/Photos/secret/IMG.jpg'}];\nconsole.log(JSON.stringify({html:filmstripHtml(evil,9,null,null)}));
"""
    out = _run_js(probe)
    html = out["html"]
    # The payload survives only as inert escaped text: no second tag is opened and
    # no attribute is closed early, so the handler can never run.
    assert "<img src=x" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;.jpg" in html
    assert html.count("<img ") == 1  # only the thumbnail element itself
    # A source path present on the record is never rendered.
    assert "C:/Photos" not in html and "secret" not in html


def test_viewer_shows_one_main_pane_and_a_filmstrip_by_default():
    css = review_template.PAGE_TEMPLATE
    assert ".viewer{display:grid;grid-template-columns:1fr" in css
    assert ".filmstrip{" in css and "overflow-x:auto" in css
    # Centred when the group fits, but 'safe' so an overflowing strip still
    # starts at frame 1 instead of clipping it off the left edge.
    assert "justify-content:safe center" in css
    assert '<div class="filmstrip" id="filmstrip"></div>' in css


def test_flat_view_query_selects_is_keep_so_the_ai_keeper_is_labelled(tmp_path):
    """ALL/MAYBE/UNKNOWN rows must carry is_keep, or the badge silently vanishes."""
    from src import db

    conn = db.open_db(tmp_path / "inventory.sqlite")
    ids = []
    for i in range(3):
        ids.append(db.insert_file(conn, {
            "path": f"/photos/IMG_{i}.jpg", "basename": f"IMG_{i}.jpg",
            "size_bytes": 10 + i, "mtime_ns": 1, "exif_datetime": f"2026-01-01 12:00:0{i}",
            "exif_timestamp": 100 + i, "width": 10, "height": 10,
            "file_kind": "jpg", "scan_status": "done"}))
    keeper, dup, lone = ids
    db.insert_group(conn, "phash_near", keeper,
                    [(keeper, True, "keep"), (dup, False, "dup")], 1)
    db.build_all_view_index(conn)
    conn.commit()

    rows = {r["file_id"]: r for r in db.review_page(conn, "ALL", 0, 50)}
    conn.close()
    assert "is_keep" in rows[keeper].keys()
    assert rows[keeper]["is_keep"] == 1
    assert rows[dup]["is_keep"] == 0
    # An ungrouped photo has no group membership, so no AI keeper claim.
    assert rows[lone]["is_keep"] is None


def test_flat_tile_labels_the_ai_keeper_and_leaves_others_unbadged():
    probe = """
const keeper={file_id:2,group_id:1,basename:'A.jpg',decision:'KEEP',thumb:'ok',is_keep:true};
const dup={file_id:1,group_id:1,basename:'B.jpg',decision:'MAYBE',thumb:'ok',is_keep:false};
const lone={file_id:9,group_id:null,basename:'C.jpg',decision:'MAYBE',thumb:'ok',is_keep:null};
console.log(JSON.stringify({keeper:tile(keeper),dup:tile(dup),lone:tile(lone)}));
"""
    out = _run_js(probe)
    assert 'class="badge ai"' in out["keeper"]
    assert "badge" not in out["dup"]
    assert "badge" not in out["lone"]
    # The flat view has no group to decide on, so it must not offer a human keeper.
    assert "badge human" not in out["keeper"]


# --- 3. viewer keyboard map ------------------------------------------------

def test_viewer_action_key_mapping_covers_p_a_m_u():
    probe = """
const keys=['p','P','a','A','m','M','u','U','x','Escape',''];
console.log(JSON.stringify(Object.fromEntries(keys.map(k=>[k,viewerActionKey(k)]))));
"""
    out = _run_js(probe)
    assert out["p"] == "pick" and out["P"] == "pick"
    assert out["a"] == "accept" and out["A"] == "accept"
    assert out["m"] == "mark" and out["M"] == "mark"
    assert out["u"] == "clear" and out["U"] == "clear"
    assert out["x"] is None and out["Escape"] is None and out[""] is None


def test_open_viewer_keyboard_branch_handles_navigation_and_actions():
    """With the lightbox open: H/L/arrows step, P/A/M/U act, C compares, Esc closes."""
    js = _embedded_js()
    # The lightbox branch is the block guarded by viewerOpen().
    branch = js.split("if(viewerOpen()){", 1)[1].split("\n if(e.key==='?'", 1)[0]
    for token in ("ArrowLeft", "'h'", "'H'", "ArrowRight", "'l'", "'L'",
                  "stepViewer(-1)", "stepViewer(1)", "Escape", "closeViewer()",
                  "viewerActionKey(e.key)", "reviewAction(viewerGroupId,action)",
                  "compare"):
        assert token in branch, f"viewer keyboard branch is missing {token}"


def test_viewer_keeps_open_and_advances_after_accept_pick_mark():
    js = _embedded_js()
    action = js.split("async function reviewAction(", 1)[1].split("\nfunction pickKeeper(", 1)[0]
    assert "const keepViewer=viewerOpen()" in action
    # clear (U) restores the same photo; A/P/M advance then re-open on the next group.
    assert "if(keepViewer&&viewerFid)await openViewer(viewerFid)" in action
    assert "if(keepViewer&&targetIndex>=0)await reopenViewerAt(targetIndex)" in action
    # The existing auto-advance semantics (next group, next page at the end) stay.
    assert "if(page<pages){page++;await load();targetIndex=0}" in action
    assert "targetIndex=savedIndex+1" in action


def test_pick_uses_the_photo_shown_in_the_viewer_when_it_is_open():
    js = _embedded_js()
    target = js.split("function pickTargetFileId(){", 1)[1].split("\n}", 1)[0]
    assert "if(viewerOpen()){const fid=currentViewerFileId();if(fid)return fid}" in target
    assert "focusedCardIndex" in target  # list-mode fallback preserved


def test_help_documents_the_album_shortcuts():
    page = review_template.PAGE_TEMPLATE
    assert "高清相册模式" in page
    assert "设当前照片为人工 keeper" in page
    assert "接受 AI keeper" in page
    assert "标记稍后再看" in page
    assert "清除本组人工状态" in page


# --- 4. C compare is not broken -------------------------------------------

def test_compare_mode_still_renders_two_panes_from_originals():
    js = _embedded_js()
    assert "viewer.classList.add('compare')" in js
    assert "leftImage.src=`/api/original/${Number(aiKeeper.file_id)}`" in js
    assert "rightImage.src=`/api/original/${Number(selected.file_id)}`" in js
    # Toggling back leaves compare mode and restores the single main photo.
    assert "if(compareMode){showOne(viewerItems[viewerIndex]);return}" in js
    assert "compareMode=false;\n viewer.classList.remove('compare')" in js
    assert ".viewer.compare{grid-template-columns:1fr 1fr}" in review_template.PAGE_TEMPLATE


# --- 5. non-GROUPS viewer still works -------------------------------------

def test_non_groups_viewer_hides_group_actions_but_keeps_navigation():
    js = _embedded_js()
    show = js.split("function showOne(r){", 1)[1].split("\nfunction showViewerIndex", 1)[0]
    assert "const inGroups=view==='GROUPS'&&viewerGroupId!=null" in show
    assert "inGroups?'inline-block':'none'" in show
    # Group-scoped review actions are refused outside GROUPS, from key and button.
    assert "if(action&&view==='GROUPS'&&viewerGroupId!=null)" in js
    assert "if(view!=='GROUPS'||viewerGroupId==null)return;" in js
    # Single-item viewer: filmstrip hidden, but prev/next and Esc keep working.
    assert "strip.style.display=viewerItems.length>1?'flex':'none'" in js


def test_tile_renders_for_flat_views_without_group_state():
    probe = """
const r={file_id:21,basename:'IMG_9.jpg',decision:'MAYBE',thumb:'ok',width:100,
 height:80,quality_score:50,face_count:0,is_keep:false,reason:'low margin'};
const html=tile(r);
console.log(JSON.stringify({html,pick:html.includes('pickKeeper'),
 thumb:html.includes('/api/thumb/21.jpg'),original:html.includes('/api/original')}));
"""
    out = _run_js(probe)
    assert out["thumb"] is True
    assert out["original"] is False
    # No per-card keeper button outside GROUPS: there is no group to decide on.
    assert out["pick"] is False
    assert 'data-card-index' not in out["html"]


# --- 6. security boundary -------------------------------------------------

def test_page_script_uses_only_the_read_only_local_api():
    js = _embedded_js()
    endpoints = sorted(set(re.findall(r"/api/[a-z]+", js)))
    assert endpoints == ["/api/action", "/api/group", "/api/original",
                         "/api/page", "/api/status", "/api/thumb"]
    # Originals are only ever assigned to the two viewer panes.
    for match in re.finditer(r"[^\n]*?/api/original[^\n]*", js):
        line = match.group(0)
        assert ("leftImage.src=" in line or "rightImage.src=" in line
                or "只在" in line), f"unexpected original fetch: {line.strip()}"


def test_closing_the_viewer_drops_original_srcs_and_the_filmstrip():
    js = _embedded_js()
    close = js.split("function closeViewer(){", 1)[1].split("\n}", 1)[0]
    assert "leftImage.removeAttribute('src');rightImage.removeAttribute('src')" in close
    assert "strip.innerHTML='';strip.style.display='none'" in close
    assert "viewerGroupId=null" in close


def test_page_never_renders_a_source_path_field():
    page = review_template.PAGE_TEMPLATE
    for forbidden in ("r.path", "m.path", "item.path", "record.path", "file://"):
        assert forbidden not in page, f"template references {forbidden}"


# --- 7. no diagnostics in the visible UI ----------------------------------

# Substrings that must not appear anywhere in the template or in the markup the
# render helpers produce. Algorithm internals (left) and runtime/storage
# internals (right) are equally out of scope for a reviewer who only has to pick
# a photo.
_DIAGNOSTIC_STRINGS = (
    "r.quality_score", "r.face_count", "r.reason", "r.thumb_error",
    "质量=", "人脸=",
    "d.shown", "d.remaining_after_page",
    "s.thumb_cache_files", "s.thumb_cache_bytes", "缩略图缓存", "GiB",
    "HDD", "机械盘", "回源", "SSD 缓存", "本地服务器模式",
)


def test_template_contains_no_diagnostic_strings():
    page = review_template.PAGE_TEMPLATE
    for forbidden in _DIAGNOSTIC_STRINGS:
        assert forbidden not in page, f"template still shows diagnostic {forbidden!r}"


def test_rendered_page_contains_no_diagnostic_strings():
    """Same gate on the assembled page, so a placeholder cannot smuggle them back.

    ``GiB`` is checked against the template only: the assembled page legitimately
    reports how much space the AUTO_REMOVE manifest would reclaim, which is a
    number the user acts on, unlike the thumbnail-cache size that used to sit in
    the stats line.
    """
    data = {"groups": [], "queues": {"MAYBE": [], "UNKNOWN": []},
            "delete_paths": [], "total_delete_bytes": 0}
    page = review_page.render_html(data, Path("/tmp"), review_limit=0)
    for forbidden in ("质量=", "人脸=", "缩略图缓存", "HDD", "机械盘", "回源"):
        assert forbidden not in page, f"rendered page still shows {forbidden!r}"


def test_photo_tile_shows_identity_not_scores():
    """A card keeps decision, keeper badges, name, size and capture time only."""
    probe = _MEMBERS + """
console.log(JSON.stringify({
 keeper:photoTile(members[1],1,12,11),
 dup:photoTile(members[0],0,12,11)}));
"""
    out = _run_js(probe)
    keeper, dup = out["keeper"], out["dup"]
    # Kept: what the photo is and what the machine/human decided about it.
    assert "KEEP" in keeper and "IMG_0002.jpg" in keeper
    assert "4000x3000" in keeper
    assert 'class="badge ai"' in keeper
    assert 'class="badge human"' in dup  # human keeper is file 11
    assert "查看高清大图" in keeper and "pickKeeper(" in keeper
    # Dropped: the numbers only the algorithm cares about.
    for html_out in (keeper, dup):
        assert "71.5" not in html_out and "41.2" not in html_out
        assert "质量" not in html_out and "人脸" not in html_out
        assert "best" not in html_out and "low margin" not in html_out


def test_stats_line_reports_progress_only():
    """updateStats must show totals/paging/review progress and nothing else."""
    probe = """
const d={page:2,pages:5,total:120,page_size:10,shown:10,remaining_after_page:100};
const s={review_state:{reviewed:7,marked:2,total:120},
 thumb_cache_files:250,thumb_cache_bytes:1073741824};
view='GROUPS';updateStats(d,s);
const groupsText=document.getElementById('stats').textContent;
view='ALL';updateStats(d,s);
console.log(JSON.stringify({groupsText,allText:document.getElementById('stats').textContent}));
"""
    out = _run_js(probe)
    for text in (out["groupsText"], out["allText"]):
        assert "120" in text and "2/5" in text          # total and page position
        assert "缩略图缓存" not in text and "GiB" not in text
        assert "本页之后还剩" not in text and "100" not in text
        assert "250" not in text
    # GROUPS adds review progress; the DOM stub reports no .group nodes, so the
    # 'current' offset is only appended when groups actually exist.
    assert "已审" not in out["allText"]


def test_stats_line_adds_review_progress_when_groups_are_present():
    probe = """
const fake=[{},{},{}];
document.querySelectorAll=sel=>sel==='.group'?fake:[];
const d={page:1,pages:1,total:3,page_size:10,shown:3,remaining_after_page:0};
const s={review_state:{reviewed:1,marked:1,total:3},
 thumb_cache_files:9,thumb_cache_bytes:1073741824};
view='GROUPS';focusedGroupIndex=1;updateStats(d,s);
console.log(JSON.stringify({text:document.getElementById('stats').textContent}));
"""
    out = _run_js(probe)
    text = out["text"]
    assert "已审 1/3" in text and "稍后 1" in text and "当前 2/3" in text
    assert "GiB" not in text and "缩略图缓存" not in text


def test_viewer_title_shows_size_and_position_not_the_storage_path():
    probe = _MEMBERS + """
viewerItems=members;viewerIndex=2;viewerGroupId=7;view='GROUPS';
showOne(members[2]);
console.log(JSON.stringify({title:document.getElementById('viewerTitle').textContent,
 label:document.getElementById('leftLabel').innerHTML}));
"""
    out = _run_js(probe)
    title = out["title"]
    assert "4000×3000" in title
    assert "组内第 3/3 张" in title
    assert "分组 #7" in title
    # No implementation detail about where the pixels came from.
    assert "HDD" not in title and "按需读取" not in title and "原图" not in title
    # The pane label still identifies the photo and its keeper status.
    assert "IMG_0003.jpg" in out["label"] and "MAYBE" in out["label"]


def test_diagnostic_containers_exist_but_are_hidden():
    """#perf/#mode stay in the document so the pipeline can still rewrite them."""
    page = review_template.PAGE_TEMPLATE
    assert 'id="perf"' in page and 'id="mode"' in page
    for block in ('<div class="notice diag" id="perf" hidden>',
                  '<div class="notice diag" id="mode" hidden>'):
        assert block in page, f"missing hidden diagnostic container: {block}"
    assert ".diag{display:none!important}" in page
    # The mode notice is no longer re-filled with storage prose on every load.
    assert "document.getElementById('mode').textContent=" not in page


def test_performance_panel_is_still_generated_and_rewritable(tmp_path):
    """Hiding the panel must not break the report: content and rewrite survive."""
    data = {"groups": [], "queues": {"MAYBE": [], "UNKNOWN": []},
            "delete_paths": [], "total_delete_bytes": 0}
    review = tmp_path / "review.html"
    review.write_text(
        review_page.render_html(data, tmp_path, review_limit=0,
                                performance_panel="stage1 12.5s"),
        encoding="utf-8")
    assert "stage1 12.5s" in review.read_text(encoding="utf-8")
    assert review_page.rewrite_performance_panel(review, "stage1 9.0s") is True
    text = review.read_text(encoding="utf-8")
    assert "stage1 9.0s" in text and "stage1 12.5s" not in text


def test_static_fallback_tile_drops_scores_and_keeps_identity(tmp_path):
    """The file:// fallback follows the same rule as the served tiles."""
    rec = {"file_id": 3, "basename": "IMG_0003.jpg", "decision": "MAYBE",
           "width": 4000, "height": 3000, "quality_score": 55.0, "face_count": 2,
           "reason": "low margin", "exif_datetime": "2026-01-01 09:30:00"}
    tile = review_page._tile(tmp_path, rec)
    assert "IMG_0003.jpg" in tile and "MAYBE" in tile
    assert "4000x3000" in tile and "2026-01-01 09:30:00" in tile
    assert "55.0" not in tile and "质量" not in tile
    assert "人脸" not in tile and "low margin" not in tile


def test_public_api_still_returns_the_diagnostic_fields(tmp_path):
    """Only the UI is trimmed: the read-only API contract is unchanged."""
    from src import review_server

    row = {"file_id": 5, "group_id": 1, "decision": "MAYBE", "basename": "a.jpg",
           "width": 10, "height": 8, "size_bytes": 99, "exif_datetime": None,
           "file_kind": "jpg", "quality_score": 55.5, "face_count": 2,
           "reason": "low margin", "thumb_status": "ok", "thumb_error": None,
           "is_keep": 0}
    item = review_server._public_item(row)
    assert item["quality_score"] == 55.5
    assert item["face_count"] == 2
    assert item["reason"] == "low margin"
    assert "path" not in item
