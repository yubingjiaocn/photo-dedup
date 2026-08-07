"""Behavioural gates for the review workbench front-end.

The UI is a template string, so these tests do two things CI can express without
a browser:

* run the embedded script in ``node`` behind a small DOM stub and assert on the
  *behaviour of its functions* (queue vocabulary, render helpers, zoom clamping,
  keyboard mapping) instead of grepping for substrings;
* assert the security boundary that matters here: lists and filmstrips address
  ``/api/thumb/<id>.jpg`` only, ``/api/original/<id>`` appears only on the two
  stage panes, no source path is ever rendered, and nothing is prefetched.

Real browser behaviour (wheel gestures, keyboard focus, grid reflow, which URLs
are actually requested, queue transitions end to end) is covered by
``scripts/verification/album_viewer_browser.py`` against a live server.

Section 7 additionally pins where diagnostics may appear: quality scores, face
counts and grouping reasons belong in the collapsible 推荐依据 panel for the
group on screen and nowhere else -- not on cards, not in the header, not in the
progress line -- and runtime/storage internals (stage timings, thumbnail-cache
size, which physical disk a read comes from) stay out of the UI entirely.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from src import review_page, review_script, review_styles, review_template

_SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.DOTALL)

# Minimal DOM/fetch stub: enough for the script's top-level wiring to run so its
# functions become callable. Deliberately dumb -- it must not emulate a browser,
# only stop the module scope from throwing.
_DOM_STUB = """
function makeEl(id){
 const el={id,children:[],style:{},dataset:{},textContent:'',innerHTML:'',value:'',
  disabled:false,onclick:null,onchange:null,naturalWidth:0,clientWidth:0,
  _attrs:{},
  classList:{_s:new Set(),add(...c){c.forEach(x=>el.classList._s.add(x))},
   remove(...c){c.forEach(x=>el.classList._s.delete(x))},
   contains(c){return el.classList._s.has(c)},
   toggle(c,on){on?el.classList._s.add(c):el.classList._s.delete(c)}},
  querySelector(){return null},querySelectorAll(){return []},
  scrollIntoView(){},addEventListener(){},removeEventListener(){},
  getBoundingClientRect(){return {left:0,top:0,width:1000,height:700}},
  getAttribute(k){return el._attrs[k]===undefined?null:el._attrs[k]},
  setAttribute(k,v){el._attrs[k]=String(v)},
  removeAttribute(k){delete el._attrs[k]},
  get src(){return el._attrs.src===undefined?'':el._attrs.src},
  set src(v){el._attrs.src=String(v)}};
 return el;
}
const _els=new Map();
globalThis.document={
 body:makeEl('body'),
 getElementById(id){if(!_els.has(id))_els.set(id,makeEl(id));return _els.get(id)},
 querySelectorAll(){return []},
 addEventListener(type,fn){globalThis.__handlers=globalThis.__handlers||{};
  globalThis.__handlers[type]=fn},
};
globalThis.window={addEventListener(){}};
globalThis.localStorage={_v:{},getItem(k){return this._v[k]===undefined?null:this._v[k]},
 setItem(k,v){this._v[k]=String(v)}};
globalThis.alert=msg=>{globalThis.__alerts=(globalThis.__alerts||[]).concat([msg])};
globalThis.fetch=async url=>{
 globalThis.__fetched=(globalThis.__fetched||[]).concat([String(url)]);
 throw new Error('network disabled in test harness');
};
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
 {file_id:12,group_id:7,basename:'IMG_0002.jpg',decision:'KEEP',thumb:'ok',
  width:4000,height:3000,quality_score:71.5,face_count:1,is_keep:true,reason:'GROUP_KEEPER'},
 {file_id:11,group_id:7,basename:'IMG_0001.jpg',decision:'AUTO_REMOVE',thumb:'ok',
  width:4000,height:3000,quality_score:41.2,face_count:1,is_keep:false,reason:'BYTE_IDENTICAL'},
 {file_id:13,group_id:7,basename:'IMG_0003.jpg',decision:'MAYBE',thumb:'missing',
  thumb_error:'decode failed',width:4000,height:3000,quality_score:55.0,
  face_count:1,is_keep:false,reason:'LOW_MARGIN'}];
const group={group_id:7,group_type:'phash_near',member_count:3,members};
function useGroup(){view='GROUPS';queue='PENDING';groups=[group];gIndex=0;mIndex=0;total=1}
"""


def test_embedded_js_is_syntactically_valid_and_runs():
    """The whole script must parse *and* execute its top-level wiring."""
    out = _run_js("console.log(JSON.stringify({view,queue,page,size,ok:true}))")
    assert out == {"view": "GROUPS", "queue": "PENDING", "page": 1, "size": 100,
                   "ok": True}


def test_front_end_is_split_into_markup_styles_and_script():
    """Three modules, one page: no single front-end file carries everything."""
    assert review_page.PAGE_TEMPLATE is review_template.PAGE_TEMPLATE
    assert review_page._PAGE_TEMPLATE is review_template.PAGE_TEMPLATE
    assert review_styles.STYLES in review_template.PAGE_TEMPLATE
    assert review_script.SCRIPT in review_template.PAGE_TEMPLATE
    # The behaviour and the stylesheet do not leak into each other.
    assert "addEventListener" not in review_styles.STYLES
    assert "grid-template-columns" not in review_script.SCRIPT
    # No framework and no build step sneaked in.
    for forbidden in ("import ", "require(", "src=\"http", "cdn."):
        assert forbidden not in review_script.SCRIPT


# --- 1. the pending queue is the default entry point -----------------------

def test_default_view_is_the_pending_group_queue():
    out = _run_js("console.log(JSON.stringify({view,queue,mode:mode()}))")
    assert out == {"view": "GROUPS", "queue": "PENDING", "mode": "queue"}


def test_the_workbench_markup_is_present_without_a_modal():
    page = review_template.PAGE_TEMPLATE
    # Header, one main pane, filmstrip and a fixed action bar, all in the shell.
    for block in ('<div class="wbhead" id="wbHead">', '<span class="pos" id="wbPos">',
                  '<div class="stage" id="stage">', '<div class="pane" id="paneA">',
                  '<div class="filmstrip" id="strip"></div>',
                  '<div class="actionbar" id="actionbar">'):
        assert block in page, f"workbench shell is missing {block}"
    # It is a persistent region of the page, not an overlay that must be opened.
    assert "#workbench{display:none;flex-direction:column" in page
    assert "#workbench.show{display:flex}" in page
    assert ".lightbox" not in page


def test_the_workbench_fits_the_viewport_so_the_action_bar_stays_visible():
    css = review_styles.STYLES
    shell = css.split("#workbench{", 1)[1].split("}", 1)[0]
    # The whole workbench is bounded by the viewport and laid out with flex...
    assert "height:calc(100vh" in shell and "flex-direction:column" in shell
    # ...the stage is the only part allowed to shrink...
    stage = css.split(".stage{", 1)[1].split("}", 1)[0]
    assert "flex:1 1 auto" in stage
    # ...so the header, filmstrip, zoom bar and action bar never scroll away.
    for rule in (".wbhead{", ".filmstrip{", ".zoombar{", ".actionbar{"):
        block = css.split(rule, 1)[1].split("}", 1)[0]
        assert "flex:0 0 auto" in block, f"{rule} can be squeezed off screen"


def test_page_requests_the_queue_from_the_server_not_the_whole_group_list():
    """The queue must be a server-side filter, with the queue's own total."""
    js = _embedded_js()
    assert "`view=${view}&page=${page}&page_size=${size}`" in js
    assert "`&queue=${queue}`" in js
    # Nothing filters a downloaded group list by its human state in the browser.
    assert "items.filter" not in js
    assert ".filter(g=>" not in js


# --- 2. queue vocabulary and Chinese display labels ------------------------

def test_queue_labels_and_action_keys():
    probe = """
console.log(JSON.stringify({
 queues:QUEUES,labels:QUEUES.map(queueLabel),
 keys:Object.fromEntries(['a','A','p','P','m','M','u','U','x','',
  'Escape'].map(k=>[k,actionKey(k)]))}));
"""
    out = _run_js(probe)
    assert out["queues"] == ["PENDING", "LATER", "DONE"]
    assert out["labels"] == ["未审", "稍后", "已完成"]
    keys = out["keys"]
    assert keys["a"] == keys["A"] == "accept"
    assert keys["p"] == keys["P"] == "pick"
    assert keys["m"] == keys["M"] == "mark"
    # U is undo, not "clear this group".
    assert keys["u"] == keys["U"] == "undo"
    assert keys["x"] is None and keys[""] is None and keys["Escape"] is None


def test_group_types_and_decisions_are_translated_for_display():
    probe = """
console.log(JSON.stringify({
 types:['sha_exact','phash_near','burst','similar_scene','weird_new_type',''
  ].map(groupTypeText),
 decisions:['KEEP','AUTO_REMOVE','MAYBE','UNKNOWN','UNGROUPED'].map(decisionText),
 reasons:['GROUP_KEEPER','LOW_MARGIN','EXPOSURE_BORDERLINE: dark','MYSTERY'
  ].map(reasonText)}));
"""
    out = _run_js(probe)
    assert out["types"][:4] == ["完全相同", "高度相似", "连拍", "相似场景"]
    # An unknown value is shown verbatim rather than hidden.
    assert out["types"][4] == "weird_new_type"
    assert out["types"][5] == "未知类型"
    assert out["decisions"] == ["保留", "建议删除", "待确认", "未知", "未分组"]
    assert out["reasons"][:3] == ["组内最佳", "与最佳差距很小", "曝光处于临界"]
    assert out["reasons"][3] == "MYSTERY"


def test_action_bar_and_help_use_the_agreed_chinese_wording():
    page = review_template.PAGE_TEMPLATE
    assert "A 保留 AI 推荐" in page
    assert "P 保留当前照片" in page
    assert "M 稍后处理" in page
    assert "U 撤销上一步" in page
    # The old "清除本组人工状态" semantics for U is gone from the UI.
    assert "清除" not in page
    assert "撤销上一步，回到刚才操作的那一组与那张照片" in page
    # The shortcut hint is part of the always-visible action bar.
    assert '<span class="hint">' in page and "按住 <kbd>C</kbd> 闪切" in page


def test_queue_tabs_render_their_counts():
    probe = """
const buttons=[{dataset:{queue:'PENDING'},classList:{toggle(){}},
  querySelector(){return this._n||(this._n={textContent:''})}},
 {dataset:{queue:'LATER'},classList:{toggle(){}},
  querySelector(){return this._n||(this._n={textContent:''})}},
 {dataset:{queue:'DONE'},classList:{toggle(){}},
  querySelector(){return this._n||(this._n={textContent:''})}}];
document.querySelectorAll=sel=>sel==='[data-queue]'?buttons:[];
queueCounts={PENDING:7,LATER:2,DONE:11};
renderTabs();
console.log(JSON.stringify(buttons.map(b=>b.querySelector().textContent)));
"""
    assert _run_js(probe) == ["7", "2", "11"]


# --- 3. stage rendering: thumbs for lists, originals only for the stage ----

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
    assert len(out["items"]) == 3
    assert out["current"] == 1
    assert 'data-file-id="13"' in html and "fs-item current" in html
    assert 'class="badge ai">🤖AI</span>' in html
    assert 'class="badge human">👤已选</span>' in html
    assert 'class="badge cur">当前</span>' in html
    assert html.count('onclick="showIndex(') == 3
    # SSD cache only: two thumbs, no original, placeholder for the third.
    assert out["thumbs"] == ["/api/thumb/12.jpg", "/api/thumb/11.jpg"]
    assert out["originals"] == 0
    assert "fs-miss" in html


def test_filmstrip_frames_are_contained_not_cropped():
    css = review_styles.STYLES
    assert ".fs-item img{" in css
    frame_rule = css.split(".fs-item img{", 1)[1].split("}", 1)[0]
    assert "object-fit:contain" in frame_rule
    assert "object-fit:cover" not in css


def test_filmstrip_escapes_basenames_and_never_renders_a_path():
    probe = """
const evil=[{file_id:9,group_id:1,basename:'<img src=x onerror=alert(1)>.jpg',
 decision:'MAYBE',thumb:'ok',is_keep:false,path:'C:/Photos/secret/IMG.jpg'}];\nconsole.log(JSON.stringify({html:filmstripHtml(evil,9,null,null)}));
"""
    out = _run_js(probe)
    html = out["html"]
    assert "<img src=x" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;.jpg" in html
    assert html.count("<img ") == 1  # only the thumbnail element itself
    assert "C:/Photos" not in html and "secret" not in html


def test_list_cards_are_one_click_with_no_duplicate_open_button():
    probe = _MEMBERS + """
const html=members.map(m=>photoTile(m,12,11)).join('');
console.log(JSON.stringify({html,
 thumbUrls:(html.match(/\\/api\\/thumb\\/\\d+\\.jpg/g)||[]),
 originals:(html.match(/\\/api\\/original/g)||[]).length,
 cardClicks:(html.match(/<div class="card[^"]*"[^>]*onclick="openBrowsePhoto\\(/g)||[]).length,
 buttons:(html.match(/<button/g)||[]).length}));
"""
    out = _run_js(probe)
    # The whole tile is the target; there is no second "查看高清大图" button.
    assert out["cardClicks"] == 3
    assert out["buttons"] == 0
    assert "查看高清大图" not in out["html"]
    assert out["thumbUrls"] == ["/api/thumb/12.jpg", "/api/thumb/11.jpg"]
    assert out["originals"] == 0
    assert "缩略图不可用" in out["html"]
    assert "decode failed" not in out["html"]


def test_stage_reads_one_original_per_visible_pane_and_releases_it():
    probe = _MEMBERS + """
useGroup();mIndex=1;
renderStage();
const single=document.getElementById('imgA').getAttribute('src');
const singleB=document.getElementById('imgB').getAttribute('src');
compareMode=true;renderStage();
const compareA=document.getElementById('imgA').getAttribute('src');
const compareB=document.getElementById('imgB').getAttribute('src');
compareMode=false;renderStage();
const backB=document.getElementById('imgB').getAttribute('src');
console.log(JSON.stringify({single,singleB,compareA,compareB,backB,
 strip:(document.getElementById('strip').innerHTML.match(/\\/api\\/original/g)||[]).length}));
"""
    out = _run_js(probe)
    # One photo on stage -> one original, second pane empty.
    assert out["single"] == "/api/original/11"
    assert out["singleB"] in (None, "")
    # Comparing shows the AI keeper beside it; leaving compare drops it again.
    assert out["compareA"] == "/api/original/11"
    assert out["compareB"] == "/api/original/12"
    assert out["backB"] in (None, "")
    assert out["strip"] == 0


def test_closing_the_browse_stage_drops_both_originals():
    probe = _MEMBERS + """
view='ALL';browseGroup=group;browseOpen=true;mIndex=0;compareMode=true;
renderStage();
const before=document.getElementById('imgA').getAttribute('src');
closeStage();
console.log(JSON.stringify({before,
 a:document.getElementById('imgA').getAttribute('src'),
 b:document.getElementById('imgB').getAttribute('src'),
 strip:document.getElementById('strip').innerHTML,open:browseOpen}));
"""
    out = _run_js(probe)
    assert out["before"] == "/api/original/12"
    assert out["a"] in (None, "") and out["b"] in (None, "")
    assert out["strip"] == "" and out["open"] is False


def test_nothing_prefetches_an_original():
    js = _embedded_js()
    # Originals are only ever assigned to the two stage panes, and never
    # requested through fetch/Image/link-prefetch.
    for match in re.finditer(r"[^\n]*originalSrc\([^\n]*", js):
        line = match.group(0)
        assert ("function originalSrc" in line or "const next=originalSrc" in line
                or "`/api/original/${" in line), f"unexpected original use: {line.strip()}"
    for forbidden in ("new Image(", "rel=\"prefetch\"", "rel=\"preload\"",
                      "fetch(`/api/original", "fetch('/api/original"):
        assert forbidden not in js, f"page prefetches originals via {forbidden}"


# --- 4. zoom / pan --------------------------------------------------------

def test_pan_is_clamped_to_the_scaled_image():
    probe = """
console.log(JSON.stringify({
 fit:clampPan(1,500,500,1000,700),
 inside:clampPan(2,100,50,1000,700),
 runaway:clampPan(2,99999,-99999,1000,700),
 label:[zoomStateText(1),zoomStateText(1.0004),zoomStateText(2.5)]}));
"""
    out = _run_js(probe)
    # At fit there is nothing to pan, so any offset collapses to zero.
    assert out["fit"] == {"tx": 0, "ty": 0}
    assert out["inside"] == {"tx": 100, "ty": 50}
    # A wild offset is bounded by half the overflow, so the photo cannot fly off.
    assert out["runaway"] == {"tx": 500.0, "ty": -350.0}
    assert out["label"] == ["适应窗口", "适应窗口", "250%"]


def test_zoom_is_bounded_and_wheel_steps_are_bounded_too():
    probe = """
zoomTo(99,0,0);const high=zoom.scale;
zoomTo(0.01,0,0);const low=[zoom.scale,zoom.tx,zoom.ty];
console.log(JSON.stringify({high,low,limits:[MIN_SCALE,MAX_SCALE],
 factors:[wheelFactor(-100),wheelFactor(100),wheelFactor(-99999),wheelFactor(99999)]}));
"""
    out = _run_js(probe)
    assert out["limits"] == [1, 8]
    assert out["high"] == 8
    # Zooming below fit snaps back to fit and re-centres.
    assert out["low"] == [1, 0, 0]
    zoom_in, zoom_out, huge_in, huge_out = out["factors"]
    assert zoom_in > 1 and zoom_out < 1
    assert huge_in == 3 and huge_out == pytest.approx(1 / 3)


def test_both_panes_share_one_transform_so_compare_stays_in_sync():
    probe = """
zoom.scale=2;zoom.tx=30;zoom.ty=-20;applyZoom();
console.log(JSON.stringify({a:document.getElementById('imgA').style.transform,
 b:document.getElementById('imgB').style.transform,
 zoomedA:document.getElementById('frameA').classList.contains('zoomed'),
 zoomedB:document.getElementById('frameB').classList.contains('zoomed'),
 label:document.getElementById('zoomVal').textContent}));
"""
    out = _run_js(probe)
    assert out["a"] == out["b"] == "translate(30px,-20px) scale(2)"
    assert out["zoomedA"] is True and out["zoomedB"] is True
    assert out["label"] == "200%"


def test_zoom_cursor_and_reset_are_declared_in_css():
    css = review_styles.STYLES
    assert ".pane .frame{" in css and "cursor:zoom-in" in css
    assert ".pane .frame.zoomed{cursor:grab}" in css
    assert ".pane .frame.grabbing{cursor:grabbing}" in css
    assert "transform-origin:center center" in css


# --- 5. hold C blinks, Shift+C splits -------------------------------------

def test_holding_c_swaps_the_main_photo_without_opening_the_split():
    probe = _MEMBERS + """
useGroup();mIndex=1;renderStage();
const before=document.getElementById('imgA').getAttribute('src');
setBlink(true);
const blinked=document.getElementById('imgA').getAttribute('src');
const label=document.getElementById('labelA').innerHTML;
const splitHidden=document.getElementById('paneB').classList.contains('hide');
setBlink(false);
console.log(JSON.stringify({before,blinked,label,splitHidden,
 after:document.getElementById('imgA').getAttribute('src'),compareMode}));
"""
    out = _run_js(probe)
    assert out["before"] == "/api/original/11"
    assert out["blinked"] == "/api/original/12"      # the AI keeper
    assert "按住 C" in out["label"]
    assert out["splitHidden"] is True                # blink is not the split
    assert out["after"] == "/api/original/11"        # release restores
    assert out["compareMode"] is False


def test_the_keyboard_gives_c_to_blink_and_shift_c_to_the_split():
    js = _embedded_js()
    branch = js.split("if(key==='c'||key==='C'){", 1)[1].split("\n }", 1)[0]
    assert "if(event.shiftKey)toggleCompare()" in branch
    # Auto-repeat must not restart the blink on every repeat event.
    assert "else if(!event.repeat)setBlink(true)" in branch
    assert "if(key==='c'||key==='C')setBlink(false)" in js  # keyup releases it
    assert "window.addEventListener('blur',()=>setBlink(false))" in js
    page = review_template.PAGE_TEMPLATE
    assert "双栏对比 Shift+C" in page
    assert "闪切到 AI 推荐照片，松开回到当前照片" in page


def test_compare_refuses_when_there_is_nothing_to_compare_with():
    probe = _MEMBERS + """
useGroup();
const noKeeper={group_id:8,group_type:'burst',member_count:1,
 members:[{file_id:31,group_id:8,basename:'X.jpg',decision:'MAYBE',thumb:'ok',
  is_keep:false}]};
groups=[noKeeper];gIndex=0;mIndex=0;
toggleCompare();const withoutKeeper=compareMode;
groups=[group];mIndex=0;   // sitting on the AI keeper itself
toggleCompare();const onKeeper=compareMode;
mIndex=1;toggleCompare();const offKeeper=compareMode;
console.log(JSON.stringify({withoutKeeper,onKeeper,offKeeper,
 toast:document.getElementById('toast').textContent}));
"""
    out = _run_js(probe)
    assert out["withoutKeeper"] is False   # no AI keeper -> nothing to compare
    assert out["onKeeper"] is False        # already showing it
    assert out["offKeeper"] is True
    assert out["toast"]


# --- 6. queue movement, undo, completion ---------------------------------

def test_decisions_post_the_right_payload_and_leave_the_queue():
    js = _embedded_js()
    decide = js.split("async function decide(action){", 1)[1].split(
        "\nasync function advanceAfter(", 1)[0]
    # Every decision carries the photo on screen, so undo can come back to it.
    assert "if(item)payload.context_file_id=Number(item.file_id)" in decide
    # Only pick names it as the keeper.
    assert "if(action==='pick'){" in decide
    assert "payload.file_id=Number(item.file_id)" in decide
    advance = js.split("async function advanceAfter(decidedGroupId){", 1)[1].split(
        "\nasync function undoLast(", 1)[0]
    # The server says what follows the decided group, in the queue as it now
    # stands. Guessing from the old page index breaks at a page boundary.
    assert "`/api/next?queue=${queue}`" in advance
    assert "`&after=${Number(decidedGroupId)}&page_size=${size}`" in advance
    assert "page=next.page||1" in advance
    assert "groups.findIndex(g=>Number(g.group_id)===wanted)" in advance


def test_accept_and_mark_send_the_on_screen_photo_without_claiming_it():
    probe = _MEMBERS + """
const sent=[];
globalThis.fetch=async(url,options)=>{
 sent.push(JSON.parse(options.body));
 return {ok:true,json:async()=>({ok:true,queues:queueCounts,undo_depth:1})};
};
// Each decision auto-advances, so the group and the photo are re-seated before
// the next one. Member index 2 is deliberately neither the first member nor the
// AI keeper.
async function decideOn(action){useGroup();mIndex=2;await decide(action)}
decideOn('accept').then(()=>decideOn('mark')).then(()=>decideOn('pick')).then(()=>{
 console.log(JSON.stringify({sent}));
});
"""
    out = _run_js(probe)
    accept, mark, pick = out["sent"]
    # accept/mark report where the reviewer was, and choose nothing.
    assert accept == {"group_id": 7, "action": "accept", "context_file_id": 13}
    assert mark == {"group_id": 7, "action": "mark", "context_file_id": 13}
    # pick sends both: the same photo, as the keeper *and* as the context.
    assert pick == {"group_id": 7, "action": "pick", "context_file_id": 13,
                    "file_id": 13}


def test_undo_asks_the_server_and_follows_it_back_to_the_group():
    js = _embedded_js()
    undo = js.split("async function undoLast(){", 1)[1].split(
        "\n// --- browse viewer", 1)[0]
    assert "postAction({action:'undo'})" in undo
    # The server says which queue and group the undone step belongs to; the UI
    # switches to that queue and lands on that group and photo.
    assert "if(QUEUES.indexOf(undone.queue)>=0)queue=undone.queue" in undo
    # The photo to return to is the one that was on screen, not the group's first.
    assert "await focusGroup(undone.group_id,undone.focus_file_id)" in undo
    assert "已撤销上一步" in undo
    # U is not "clear the current group" any more.
    assert "'clear'" not in js


def test_reload_position_is_restored_through_the_locate_endpoint():
    js = _embedded_js()
    assert "localStorage.setItem(LS_KEY" in js
    focus = js.split("async function focusGroup(groupId,fileId){", 1)[1].split(
        "\n// --- actions", 1)[0]
    assert "`/api/locate?queue=${queue}&group_id=${Number(groupId)}" in focus
    # When the saved group has left the queue, the server's fallback id is used.
    assert "const wanted=found.group_id!=null?Number(found.group_id):null" in focus
    assert "gIndex=at>=0?at:0" in focus
    boot = js.split("async function boot(){", 1)[1].split("\n}", 1)[0]
    assert "restoreLocal()" in boot
    assert "if(mode()==='queue'&&restoreGroupId!=null)await focusGroup(" in boot


def test_restore_only_accepts_known_views_queues_and_sizes():
    probe = """
localStorage.setItem(LS_KEY,JSON.stringify({view:'EVIL',queue:'EVIL',size:9999,
 density:'EVIL',evidenceOpen:'yes',group_id:5,file_id:6}));
view='GROUPS';queue='PENDING';size=100;density='comfy';
restoreLocal();
console.log(JSON.stringify({view,queue,size,density,evidenceOpen,
 restoreGroupId,restoreFileId}));
"""
    out = _run_js(probe)
    assert out["view"] == "GROUPS" and out["queue"] == "PENDING"
    assert out["size"] == 100 and out["density"] == "comfy"
    assert out["evidenceOpen"] is False
    assert out["restoreGroupId"] == 5 and out["restoreFileId"] == 6


def test_saved_state_never_contains_a_source_path():
    probe = _MEMBERS + """
useGroup();members[0].path='C:/Photos/secret/IMG.jpg';\nsaveLocal();\nconsole.log(JSON.stringify({saved:localStorage.getItem(LS_KEY)}));
"""
    out = _run_js(probe)
    assert "C:/Photos" not in out["saved"] and "secret" not in out["saved"]
    assert json.loads(out["saved"])["group_id"] == 7


def test_the_completion_panel_replaces_the_workbench_when_the_queue_empties():
    probe = """
view='GROUPS';queue='PENDING';groups=[];total=0;
queueCounts={PENDING:0,LATER:3,DONE:9};
setChrome();updateStats();
console.log(JSON.stringify({
 workbench:document.getElementById('workbench').classList.contains('show'),
 donePanel:document.getElementById('donePanel').classList.contains('show'),
 doneText:document.getElementById('doneText').textContent,
 later:document.getElementById('goLater').textContent,
 done:document.getElementById('goDone').textContent}));
"""
    out = _run_js(probe)
    assert out["workbench"] is False
    assert out["donePanel"] is True
    assert out["doneText"] == "已完成 9 组 · 稍后 3 组。"
    assert out["later"] == "查看稍后（3）" and out["done"] == "查看已完成（9）"
    assert "本轮审阅完成" in review_template.PAGE_TEMPLATE


def test_the_last_decision_of_a_round_says_so_instead_of_next_group():
    js = _embedded_js()
    decide = js.split("async function decide(action){", 1)[1].split(
        "\nasync function advanceAfter(", 1)[0]
    assert ("if(mode()==='queue'&&groups.length===0){showToast('本轮审阅完成');return}"
            in decide)


def test_stepping_past_the_last_group_says_so_instead_of_looping():
    probe = _MEMBERS + """
useGroup();page=1;pages=1;
stepGroup(1).then(()=>{
 const forward=document.getElementById('toast').textContent;
 return stepGroup(-1).then(()=>{
  console.log(JSON.stringify({forward,back:document.getElementById('toast').textContent,
   gIndex}));
 });
});
"""
    out = _run_js(probe)
    assert out["forward"] == "已经是本队列最后一组"
    assert out["back"] == "已经是本队列第一组"
    assert out["gIndex"] == 0


# --- 6b. one write at a time -----------------------------------------------

def test_a_second_decision_is_ignored_while_the_first_is_in_flight():
    """A fast double-tap must post the group once, not twice."""
    probe = _MEMBERS + """
useGroup();
const sent=[];
let release;
globalThis.fetch=async(url,options)=>{
 if(String(url).indexOf('/api/action')<0)
  return {ok:true,json:async()=>({queue:'PENDING',group_id:7,page:1,
   queue_counts:queueCounts})};
 sent.push(JSON.parse(options.body));
 // A real request takes time; both key presses land inside that window.
 await new Promise(resolve=>{release=resolve});
 return {ok:true,json:async()=>({ok:true,queues:queueCounts,undo_depth:1})};
};
const first=decide('accept');
const second=decide('accept');       // same group, immediately after
const third=decide('mark');          // a different action, still busy
const busyDuringFlight=mutationBusy;
const disabledDuringFlight=document.getElementById('bAccept').disabled;
release();
Promise.all([first,second,third]).then(()=>{
 console.log(JSON.stringify({sent,busyDuringFlight,disabledDuringFlight,
  busyAfter:mutationBusy}));
});
"""
    out = _run_js(probe)
    assert len(out["sent"]) == 1, out["sent"]
    assert out["sent"][0]["action"] == "accept"
    assert out["busyDuringFlight"] is True
    assert out["disabledDuringFlight"] is True
    # The guard is released even though the page could not reload in the stub.
    assert out["busyAfter"] is False


def test_undo_cannot_be_repeated_while_it_is_in_flight():
    """Otherwise holding U walks back the whole session in one go."""
    probe = _MEMBERS + """
useGroup();undoDepth=5;
const sent=[];
let release;
globalThis.fetch=async(url,options)=>{
 if(String(url).indexOf('/api/action')<0)
  return {ok:true,json:async()=>({queue:'PENDING',group_id:7,page:1,
   queue_counts:queueCounts,total:1})};
 sent.push(JSON.parse(options.body));
 await new Promise(resolve=>{release=resolve});
 return {ok:true,json:async()=>({ok:true,queues:queueCounts,undo_depth:4,
  undo:{group_id:7,queue:'PENDING',focus_file_id:13}})};
};
const calls=[undoLast(),undoLast(),undoLast()];
release();
Promise.all(calls).then(()=>{console.log(JSON.stringify({sent}))});
"""
    out = _run_js(probe)
    assert len(out["sent"]) == 1
    assert out["sent"][0] == {"action": "undo"}


def test_the_guard_is_released_even_when_the_request_fails():
    probe = _MEMBERS + """
useGroup();
let attempts=0;
globalThis.fetch=async(url,options)=>{
 if(String(url).indexOf('/api/action')<0)throw new Error('no page load in stub');
 attempts++;
 return {ok:false,json:async()=>({error:'boom'})};
};
decide('mark').then(()=>{
 const afterFailure=mutationBusy;
 return decide('mark').then(()=>{
  console.log(JSON.stringify({attempts,afterFailure,
   toast:document.getElementById('toast').textContent}));
 });
});
"""
    out = _run_js(probe)
    assert out["afterFailure"] is False
    assert out["attempts"] == 2        # a failure does not lock the UI out
    assert "操作失败" in out["toast"]


def test_an_auto_repeating_key_does_not_fire_a_decision_per_event():
    js = _embedded_js()
    handler = js.split("document.addEventListener('keydown',async event=>{", 1)[1]
    branch = handler.split("const action=actionKey(key);", 1)[1]
    assert "if(event.repeat||mutationBusy)return" in branch
    # The repeat guard must sit after the action is identified but before any
    # request is made, i.e. before either call.
    guard = branch.index("if(event.repeat||mutationBusy)return")
    assert guard < branch.index("await undoLast()")
    assert guard < branch.index("await decide(action)")


def test_holding_a_decision_key_posts_once():
    """The keydown path itself, driven with repeat=true like a browser does."""
    probe = _MEMBERS + """
useGroup();
const sent=[];
globalThis.fetch=async(url,options)=>{
 if(String(url).indexOf('/api/action')<0)
  return {ok:true,json:async()=>({queue:'PENDING',group_id:7,page:1,
   queue_counts:queueCounts,total:1,items:[],review_state:{}})};
 sent.push(JSON.parse(options.body));
 return {ok:true,json:async()=>({ok:true,queues:queueCounts,undo_depth:1})};
};
const press=repeat=>globalThis.__handlers.keydown(
 {key:'m',repeat,shiftKey:false,preventDefault(){}});
(async()=>{
 await press(false);
 await press(true);await press(true);await press(true);
 console.log(JSON.stringify({sent}));
})();
"""
    out = _run_js(probe)
    assert len(out["sent"]) == 1, out["sent"]


# --- 6c. accept needs an AI recommendation ---------------------------------

def test_accept_is_refused_in_the_ui_when_the_group_has_no_ai_keeper():
    probe = """
const members=[
 {file_id:31,group_id:8,basename:'A.jpg',decision:'MAYBE',thumb:'ok',is_keep:false},
 {file_id:32,group_id:8,basename:'B.jpg',decision:'MAYBE',thumb:'ok',is_keep:false}];
view='GROUPS';queue='PENDING';total=1;
groups=[{group_id:8,group_type:'burst',member_count:2,members}];gIndex=0;mIndex=0;
const sent=[];
globalThis.fetch=async(url,options)=>{sent.push(String(url));
 return {ok:true,json:async()=>({ok:true})}};
renderStage();
const disabled=document.getElementById('bAccept').disabled;
decide('accept').then(()=>{
 console.log(JSON.stringify({sent,disabled,
  toast:document.getElementById('toast').textContent}));
});
"""
    out = _run_js(probe)
    # Nothing was posted, the button is disabled, and the reason is stated.
    assert out["sent"] == []
    assert out["disabled"] is True
    assert "没有 AI 推荐" in out["toast"]


def test_accept_is_enabled_again_for_a_group_that_has_one():
    probe = _MEMBERS + """
useGroup();renderStage();
const withKeeper=document.getElementById('bAccept').disabled;
console.log(JSON.stringify({withKeeper,
 undo:document.getElementById('bUndo').disabled}));
"""
    out = _run_js(probe)
    assert out["withKeeper"] is False
    # Undo stays disabled until there is something to undo.
    assert out["undo"] is True

# --- 6d. the unreadable-state warning reaches the reviewer -----------------

def test_the_state_warning_banner_is_hidden_until_there_is_a_warning():
    probe = """
const banner=document.getElementById('stateWarn');
renderStateWarning();
const clean=[banner.hidden,banner.innerHTML];
setStateWarning('review_state.json could not be read (JSONDecodeError) (saved as review_state.corrupt-1770000000.json)');
const warned=[banner.hidden,banner.innerHTML];
setStateWarning(null);
console.log(JSON.stringify({clean,warned,cleared:[banner.hidden,banner.innerHTML]}));
"""
    out = _run_js(probe)
    assert out["clean"] == [True, ""]
    hidden, html = out["warned"]
    assert hidden is False
    # Says what happened, in Chinese, and names the file that was kept.
    assert "审阅状态文件无法读取" in html
    assert "review_state.corrupt-1770000000.json" in html
    assert "之前的记录没有丢失" in html
    # Goes away again once the state is healthy.
    assert out["cleared"] == [True, ""]


def test_the_warning_banner_escapes_the_file_name_it_shows():
    """The name comes from the server, but it is still rendered as inert text."""
    probe = """
setStateWarning('broken (saved as <script>alert(1)</script>.json)');
const withScript=document.getElementById('stateWarn').innerHTML;
setStateWarning('broken (saved as <img src=x onerror=alert(1)>.json)');
console.log(JSON.stringify({withScript,
 withImg:document.getElementById('stateWarn').innerHTML}));
"""
    out = _run_js(probe)
    # Whatever survives the extraction is escaped text, so no tag is opened and
    # no attribute is closed early -- the payload cannot execute.
    for html in (out["withScript"], out["withImg"]):
        assert "<script" not in html and "<img" not in html
        assert "onerror" not in html or "&lt;img" in html
        assert "&lt;" in html
        # Only the banner's own markup is present.
        assert sorted(re.findall(r"<(/?[a-z]+)", html)) == ["/b", "/code", "b",
                                                            "br", "code"]


def test_a_warning_without_a_backup_name_still_explains_itself():
    probe = """
setStateWarning('review_state.json is version 99, this build reads version 1; file kept');
console.log(JSON.stringify({html:document.getElementById('stateWarn').innerHTML}));
"""
    out = _run_js(probe)
    assert "审阅状态文件无法读取" in out["html"]
    assert "原文件已保留" in out["html"]


def test_both_load_paths_pick_the_warning_up():
    js = _embedded_js()
    # The queue path reads pages, not status, so the envelope carries it -- with a
    # one-shot status fetch as the fallback for an older response.
    assert "if(data.state_warning!==undefined)setStateWarning(data.state_warning)" in js
    assert "else if(stateWarning===null)await refreshStateWarning()" in js
    # The browse path already fetches status; it must use the field.
    assert "setStateWarning(status.state_warning)" in js


def test_the_banner_is_not_a_way_back_in_for_diagnostics():
    """It is the one non-progress message allowed, and only that one."""
    banner = review_styles.STYLES.split(".statewarn{", 1)[1].split("}", 1)[0]
    assert "display:none" not in banner          # hidden via the [hidden] attribute
    assert ".statewarn[hidden]{display:none}" in review_styles.STYLES
    template = review_template.PAGE_TEMPLATE
    assert '<div class="statewarn" id="stateWarn" role="alert" hidden></div>' in template
    js = _embedded_js()
    render = js.split("function renderStateWarning(){", 1)[1].split("\nfunction setStateWarning", 1)[0]
    for forbidden in ("quality_score", "face_count", "thumb_cache", "reason",
                      "undo_depth", "queueCounts"):
        assert forbidden not in render, f"the banner leaked {forbidden}"

# --- 7. diagnostics: on demand, for the current group only ----------------

# Substrings that must not appear anywhere in the template or in the markup the
# render helpers produce. Runtime/storage internals are not the reviewer's
# business, and algorithm numbers belong in the collapsible panel only.
_DIAGNOSTIC_STRINGS = (
    "d.shown", "d.remaining_after_page",
    "s.thumb_cache_files", "s.thumb_cache_bytes", "缩略图缓存", "GiB",
    "HDD", "机械盘", "回源", "SSD 缓存", "本地服务器模式",
)


def test_template_contains_no_runtime_diagnostics():
    page = review_template.PAGE_TEMPLATE
    for forbidden in _DIAGNOSTIC_STRINGS:
        assert forbidden not in page, f"template still shows diagnostic {forbidden!r}"


def test_rendered_page_contains_no_runtime_diagnostics():
    data = {"groups": [], "queues": {"MAYBE": [], "UNKNOWN": []},
            "delete_paths": [], "total_delete_bytes": 0}
    page = review_page.render_html(data, Path("/tmp"), review_limit=0)
    for forbidden in ("缩略图缓存", "HDD", "机械盘", "回源"):
        assert forbidden not in page, f"rendered page still shows {forbidden!r}"


def test_list_cards_show_identity_not_scores():
    probe = _MEMBERS + """
console.log(JSON.stringify({keeper:photoTile(members[0],12,11),
 dup:photoTile(members[1],12,11)}));
"""
    out = _run_js(probe)
    keeper, dup = out["keeper"], out["dup"]
    assert "保留" in keeper and "IMG_0002.jpg" in keeper and "4000x3000" in keeper
    assert 'class="badge ai"' in keeper
    assert 'class="badge human"' in dup    # human keeper is file 11
    for html_out in (keeper, dup):
        assert "71.5" not in html_out and "41.2" not in html_out
        assert "清晰度" not in html_out and "人脸" not in html_out
        assert "GROUP_KEEPER" not in html_out and "BYTE_IDENTICAL" not in html_out


def test_the_group_header_carries_no_scores():
    probe = _MEMBERS + """
useGroup();evidenceOpen=false;renderStage();
console.log(JSON.stringify({meta:document.getElementById('wbMeta').innerHTML,
 evidence:document.getElementById('evidence').innerHTML,
 open:document.getElementById('evidence').classList.contains('open')}));
"""
    out = _run_js(probe)
    meta = out["meta"]
    # What the group is, how far in it you are, what was decided.
    assert "高度相似" in meta and "phash_near" in meta
    assert "3 张" in meta and "第 1/3 张" in meta and "待审" in meta
    # No numbers only the algorithm cares about, and the panel starts closed.
    assert "71.5" not in meta and "41.2" not in meta and "GROUP_KEEPER" not in meta
    assert out["open"] is False and out["evidence"] == ""


def test_the_evidence_panel_shows_diagnostics_on_demand_with_chinese_labels():
    probe = _MEMBERS + """
useGroup();evidenceOpen=true;renderStage();
console.log(JSON.stringify({html:document.getElementById('evidence').innerHTML,
 open:document.getElementById('evidence').classList.contains('open')}));
"""
    out = _run_js(probe)
    html = out["html"]
    assert out["open"] is True
    # Chinese column labels...
    assert "清晰度评分" in html and "人脸数" in html and "系统判断" in html
    # ...the actual diagnostics for this one group...
    assert "71.5" in html and "41.2" in html and "55.0" in html
    assert "组内最佳" in html and "与最佳差距很小" in html
    # ...and the internal enum retained in small type next to the translation.
    assert '<span class="enum">GROUP_KEEPER</span>' in html
    assert '<span class="enum">KEEP</span>' in html
    # It is a group-scoped panel, so no path and no cross-page data.
    assert "/api/original" not in html and "path" not in html


def test_the_progress_line_reports_queues_and_position_only():
    probe = """
view='GROUPS';queue='PENDING';page=2;size=10;gIndex=3;total=45;
queueCounts={PENDING:45,LATER:2,DONE:7};
updateStats();const queueText=document.getElementById('stats').textContent;
view='ALL';page=2;pages=5;total=120;
updateStats();const browseText=document.getElementById('stats').textContent;
console.log(JSON.stringify({queueText,browseText}));
"""
    out = _run_js(probe)
    queue_text = out["queueText"]
    assert "未审 45" in queue_text and "稍后 2" in queue_text and "已完成 7" in queue_text
    assert "当前 14/45" in queue_text
    for text in (queue_text, out["browseText"]):
        assert "缩略图缓存" not in text and "GiB" not in text
        assert "本页之后还剩" not in text
    # The browse line names the list in Chinese too, not the internal view name.
    assert out["browseText"].startswith("全部时间线：")
    assert "共 120 项" in out["browseText"] and "第 2/5 页" in out["browseText"]


def test_diagnostic_containers_exist_but_are_hidden():
    """#perf/#mode stay in the document so the pipeline can still rewrite them."""
    page = review_template.PAGE_TEMPLATE
    for block in ('<div class="notice diag" id="perf" hidden>',
                  '<div class="notice diag" id="mode" hidden>'):
        assert block in page, f"missing hidden diagnostic container: {block}"
    assert ".diag{display:none!important}" in page
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
           "reason": "LOW_MARGIN", "exif_datetime": "2026-01-01 09:30:00"}
    tile = review_page._tile(tmp_path, rec)
    assert "IMG_0003.jpg" in tile and "待确认" in tile and "MAYBE" in tile
    assert "4000x3000" in tile and "2026-01-01 09:30:00" in tile
    assert "55.0" not in tile and "清晰度" not in tile
    assert "人脸" not in tile and "LOW_MARGIN" not in tile


# --- 8. list density -----------------------------------------------------

def test_density_toggles_persist_and_only_change_tile_size():
    probe = """
const before=density;
toggleDensity();
const dense=[density,document.body.classList.contains('dense'),
 document.getElementById('densityBtn').textContent];
const saved=JSON.parse(localStorage.getItem(LS_KEY)).density;
toggleDensity();
console.log(JSON.stringify({before,dense,saved,
 back:[density,document.body.classList.contains('dense'),
  document.getElementById('densityBtn').textContent]}));
"""
    out = _run_js(probe)
    assert out["before"] == "comfy"
    assert out["dense"] == ["dense", True, "密度：紧凑"]
    assert out["saved"] == "dense"
    assert out["back"] == ["comfy", False, "密度：舒适"]


def test_compact_density_only_shrinks_the_tile_variables():
    css = review_styles.STYLES
    assert "body.dense{--tile:clamp(" in css
    dense_rule = css.split("body.dense{", 1)[1].split("}", 1)[0]
    # Density is purely a size decision; it must not hide anything.
    for forbidden in ("display:none", "visibility", "pointer-events"):
        assert forbidden not in dense_rule
    assert "grid-template-columns:repeat(auto-fill,minmax(var(--tile),1fr))" in css


# --- 9. status colours ---------------------------------------------------

def test_state_colours_are_declared_once_and_red_is_danger_only():
    css = review_styles.STYLES
    palette = css.split(":root{", 1)[1].split("}", 1)[0]
    for token in ("--focus:#fc3", "--done:#3c6", "--ai:#8ab4ff", "--later:#f93",
                  "--danger:#d15"):
        assert token in palette, f"palette is missing {token}"
    # Red is reserved for the destructive decision.
    assert ".auto_remove{border-color:var(--danger)}" in css
    for rule in (".fs-item.current{", ".wbhead{", ".wbhead.done{", ".wbhead.later{"):
        assert rule in css
    assert "border-color:var(--focus)" in css.split(".fs-item.current{", 1)[1]
    assert "var(--done)" in css.split(".wbhead.done{", 1)[1].split("}", 1)[0]
    assert "var(--later)" in css.split(".wbhead.later{", 1)[1].split("}", 1)[0]
    # Keeper badges follow the same language: AI blue, human-confirmed green.
    assert ".badge.ai{background:#24365c" in css
    assert ".badge.human{background:#1e4232" in css


# --- 10. non-GROUPS browsing ---------------------------------------------

def test_browse_views_keep_navigation_but_offer_no_group_decision():
    probe = _MEMBERS + """
view='ALL';browseGroup=group;browseOpen=true;total=3;pages=1;
setChrome();
const chrome={list:document.getElementById('listWrap').classList.contains('show'),
 pager:document.getElementById('pager').style.display,
 overlay:document.getElementById('workbench').classList.contains('overlay'),
 accept:document.getElementById('bAccept').style.display,
 pick:document.getElementById('bPick').style.display,
 undo:document.getElementById('bUndo').style.display,
 close:document.getElementById('closeStage').style.display};
const counts=JSON.stringify(queueCounts);
globalThis.__fetched=[];   // ignore the page's own initial load
decide('accept').then(()=>undoLast()).then(()=>{
 console.log(JSON.stringify({chrome,inert:JSON.stringify(queueCounts)===counts,
  fetched:(globalThis.__fetched||[]).length}));
});
"""
    out = _run_js(probe)
    chrome = out["chrome"]
    assert chrome["list"] is True and chrome["pager"] == "flex"
    assert chrome["overlay"] is True and chrome["close"] == "inline-block"
    assert chrome["accept"] == chrome["pick"] == chrome["undo"] == "none"
    # A/U cannot change any state while browsing, and never even reach the API.
    assert out["inert"] is True
    assert out["fetched"] == 0


def test_a_flat_tile_labels_the_ai_keeper_and_leaves_others_unbadged():
    probe = """
const keeper={file_id:2,group_id:1,basename:'A.jpg',decision:'KEEP',thumb:'ok',is_keep:true};
const dup={file_id:1,group_id:1,basename:'B.jpg',decision:'MAYBE',thumb:'ok',is_keep:false};
const lone={file_id:9,group_id:null,basename:'C.jpg',decision:'MAYBE',thumb:'ok',is_keep:null};
console.log(JSON.stringify({keeper:tile(keeper),dup:tile(dup),lone:tile(lone)}));
"""
    out = _run_js(probe)
    assert 'class="badge ai"' in out["keeper"]
    assert "badge" not in out["dup"] and "badge" not in out["lone"]
    # There is no group decision to make here, so no human keeper is claimed.
    assert "badge human" not in out["keeper"]


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
    assert rows[lone]["is_keep"] is None


# --- 11. security boundary ----------------------------------------------

def test_page_script_uses_only_the_read_only_local_api():
    js = _embedded_js()
    endpoints = sorted(set(re.findall(r"/api/[a-z]+", js)))
    assert endpoints == ["/api/action", "/api/group", "/api/locate", "/api/next",
                         "/api/original", "/api/page", "/api/status", "/api/thumb"]


def test_page_never_renders_a_source_path_field():
    page = review_template.PAGE_TEMPLATE
    for forbidden in ("r.path", "m.path", "item.path", "record.path", "file://"):
        assert forbidden not in page, f"template references {forbidden}"


def test_public_api_still_returns_the_diagnostic_fields():
    """Only the UI decides where to show these: the API contract is unchanged."""
    from src import review_server

    row = {"file_id": 5, "group_id": 1, "decision": "MAYBE", "basename": "a.jpg",
           "width": 10, "height": 8, "size_bytes": 99, "exif_datetime": None,
           "file_kind": "jpg", "quality_score": 55.5, "face_count": 2,
           "reason": "LOW_MARGIN", "thumb_status": "ok", "thumb_error": None,
           "is_keep": 0}
    item = review_server._public_item(row)
    assert item["quality_score"] == 55.5
    assert item["face_count"] == 2
    assert item["reason"] == "LOW_MARGIN"
    assert "path" not in item
