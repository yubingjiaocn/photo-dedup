"""The review page shell: markup that composes the stylesheet and the script.

The three front-end pieces live in three modules so none of them grows past
being reviewable in one sitting:

* :mod:`src.review_styles` -- the stylesheet (``STYLES``)
* :mod:`src.review_script` -- the browser behaviour (``SCRIPT``)
* this module -- the HTML shell plus the four placeholders
  (``PERFORMANCE_PANEL``, ``STATIC_NOTICE``, ``SUMMARY_TEXT``,
  ``STATIC_FALLBACK``) that :mod:`src.review_page` substitutes.

Nothing here touches the filesystem, and there is no build step: the page is one
self-contained ``review.html``.

What the page deliberately does *not* show: the reviewer only needs to decide
which photo to keep, so the visible UI carries no runtime/storage internals
(stage timings, thumbnail-cache size, which disk a read comes from). The two
diagnostic containers (``#perf``, ``#mode``) are still rendered, and still
rewritten by the pipeline, but they are hidden -- ``performance.txt`` and the
generated report stay the place where those numbers are read. Algorithm
diagnostics (quality score, face count, grouping reason) are available on demand
in the collapsible 推荐依据 panel for the group on screen, and nowhere else.

Front-end boundaries this template must keep:

* Every list tile and every filmstrip frame is served by ``/api/thumb/<id>.jpg``,
  i.e. the SSD cache Stage 1 wrote. Paging and browsing never wake the HDD.
* ``/api/original/<id>`` is requested only for the photo the stage is showing
  (plus the AI keeper while comparing or blinking). Leaving a photo drops the
  ``src`` again, and nothing is prefetched.
* The browser only ever sees ``file_id`` and ``basename``; no source path is
  rendered, and no endpoint other than the read-only API is used.

The UI text is Chinese. Internal values (KEEP/MAYBE/UNKNOWN/AUTO_REMOVE, group
types, the accept/pick/mark/undo actions) are translated for display and kept
verbatim in small type next to the translation, so the page and the database stay
easy to line up.
"""

from __future__ import annotations

from .review_script import SCRIPT
from .review_styles import STYLES

_BODY = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>照片去重 - 审阅</title><style>__STYLES__</style></head><body>
<h1>照片去重 - 审阅<span class="sub">连续审片工作台 · 原图只读</span></h1>
<div class="statewarn" id="stateWarn" role="alert" hidden></div>
<div class="notice diag" id="perf" hidden>PERFORMANCE_PANEL</div>
<div class="notice diag" id="mode" hidden>STATIC_NOTICE</div>
<nav class="tabs">
 <button data-queue="PENDING">未审<span class="n">0</span></button>
 <button data-queue="LATER">稍后<span class="n">0</span></button>
 <button data-queue="DONE">已完成<span class="n">0</span></button>
 <span class="sep">|</span>
 <span class="grouplabel">浏览</span>
 <button data-view="ALL">全部（时间线）</button>
 <button data-view="MAYBE">待确认</button>
 <button data-view="UNKNOWN">未知</button>
 <span class="sep">|</span>
 <button id="densityBtn">密度：舒适</button>
 <button id="evidenceBtn">推荐依据 E</button>
 <button id="helpBtn">快捷键 ?</button>
</nav>
<div class="stats" id="stats">SUMMARY_TEXT</div>
<div id="workbench">
 <div class="wbhead" id="wbHead">
  <span class="pos" id="wbPos"></span>
  <span class="meta" id="wbMeta"></span>
  <button id="wbPrev">&lsaquo; 上一组 K</button>
  <button id="wbNext">下一组 J &rsaquo;</button>
  <button id="closeStage">关闭 Esc</button>
 </div>
 <div class="evidence" id="evidence"></div>
 <div class="stage" id="stage">
  <div class="pane" id="paneA"><div class="label" id="labelA"></div>
   <div class="frame" id="frameA"><img id="imgA" alt=""></div></div>
  <div class="pane hide" id="paneB"><div class="label" id="labelB"></div>
   <div class="frame" id="frameB"><img id="imgB" alt=""></div></div>
 </div>
 <div class="filmstrip" id="strip"></div>
 <div class="zoombar">
  <button id="compareBtn">双栏对比 Shift+C</button>
  <button id="fitBtn">适应窗口 F</button>
  <span>缩放 <span class="val" id="zoomVal">适应窗口</span></span>
  <span>滚轮缩放 · 拖动平移 · 双击回到适应窗口 · 1 = 100%</span>
 </div>
 <div class="actionbar" id="actionbar">
  <button id="bAccept">A 保留 AI 推荐</button>
  <button id="bPick">P 保留当前照片</button>
  <button id="bMark">M 稍后处理</button>
  <button id="bUndo">U 撤销上一步</button>
  <span class="hint"><kbd>J</kbd>/<kbd>K</kbd> 换组 · <kbd>H</kbd>/<kbd>L</kbd> 换照片
   · 按住 <kbd>C</kbd> 闪切 AI 推荐 · <kbd>Shift</kbd>+<kbd>C</kbd> 双栏
   · <kbd>E</kbd> 推荐依据 · <kbd>D</kbd> 密度 · <kbd>?</kbd> 帮助</span>
 </div>
</div>
<div class="donepanel" id="donePanel">
 <h2>本轮审阅完成</h2>
 <p id="doneText"></p>
 <button id="goLater">查看稍后（0）</button>
 <button id="goDone">查看已完成（0）</button>
</div>
<div id="listWrap">
 <nav id="pager">
  <button id="first">&laquo; 首页</button>
  <button id="prev">&lsaquo; 上一页</button>
  <button id="next">下一页 &rsaquo;</button>
  <button id="last">尾页 &raquo;</button>
  <label>每页 <select id="size">
   <option value="50">50</option><option value="100" selected>100</option>
   <option value="200">200</option></select> 张</label>
 </nav>
 <div class="row" id="content"></div>
</div>
<div class="help" id="help">
 <div class="help-box">
  <h2>快捷键说明</h2>
  <table>
   <tr><td>? 或 F1</td><td>显示/隐藏此帮助</td></tr>
   <tr><td colspan="2" class="grouphead">审阅动作</td></tr>
   <tr><td>A</td><td>保留 AI 推荐，本组完成并进入下一组</td></tr>
   <tr><td>P</td><td>保留当前照片，本组完成并进入下一组</td></tr>
   <tr><td>M</td><td>稍后处理，本组移入「稍后」队列</td></tr>
   <tr><td>U</td><td>撤销上一步，回到刚才操作的那一组与那张照片</td></tr>
   <tr><td colspan="2" class="grouphead">导航</td></tr>
   <tr><td>J / ↓&nbsp;&nbsp;K / ↑</td><td>下一组 / 上一组</td></tr>
   <tr><td>L / →&nbsp;&nbsp;H / ←</td><td>组内下一张 / 上一张（点击底部缩略图同效）</td></tr>
   <tr><td>Esc</td><td>关闭浏览查看器 / 关闭帮助</td></tr>
   <tr><td colspan="2" class="grouphead">看图</td></tr>
   <tr><td>按住 C</td><td>闪切到 AI 推荐照片，松开回到当前照片</td></tr>
   <tr><td>Shift + C</td><td>双栏对比（当前照片 vs. AI 推荐），再按恢复单张</td></tr>
   <tr><td>滚轮 / 拖动</td><td>缩放 / 平移；双栏时两侧同步</td></tr>
   <tr><td>F 或双击</td><td>回到适应窗口</td></tr>
   <tr><td>1</td><td>100%（原始像素）</td></tr>
   <tr><td colspan="2" class="grouphead">界面</td></tr>
   <tr><td>E</td><td>展开/收起「推荐依据」</td></tr>
   <tr><td>D</td><td>列表密度：舒适 / 紧凑</td></tr>
  </table>
  <p style="color:#999;font-size:12px;margin-top:12px">未审队列为默认入口：A/P 决定后该组退出未审队列，
  M 移入稍后队列，队列清空后显示「本轮审阅完成」。所有状态仅写入 output 目录，原图永远只读。</p>
  <button onclick="closeHelp()" style="width:100%;margin-top:12px">关闭 Esc</button>
 </div>
</div>
<div id="toast"></div>
<script>__SCRIPT__</script>
<noscript>该审阅页面需要 JavaScript 和本地服务器才能分页浏览大型图库。以下为静态后备内容。</noscript>
<div class="row">STATIC_FALLBACK</div>
</body></html>"""

PAGE_TEMPLATE = _BODY.replace("__STYLES__", STYLES).replace("__SCRIPT__", SCRIPT)
