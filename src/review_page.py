"""Review page rendering: the paged UI shell plus a cached-thumbnail fallback.

Split out of :mod:`src.stage3_report` to keep both files small. Two rules govern
everything here:

* **No original photo is ever opened.** Tiles reference ``output/thumbs/<id>.jpg``
  written by Stage 1, or say plainly that the thumbnail is unavailable.
* **The full views are not embedded.** The template ships an empty grid that the
  local server fills page by page from SQLite, so a 100k-photo library does not
  become a 100k-entry HTML document. The static block is only a small
  ``file://`` fallback.

The UI text is Chinese; internal state names (MAYBE/UNKNOWN/KEEP/AUTO_REMOVE and
group types) stay English to match the database and the manifests.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict

DEFAULT_REVIEW_LIMIT = 200


def cached_thumb_uri(output_dir: Path, file_id: int | None) -> str | None:
    """Relative URL of an existing SSD thumbnail, or None.

    Deliberately never opens an original photo: if Stage 1 did not manage to
    write the thumbnail, the review shows an explicit placeholder instead.
    """
    from . import thumbnails

    if file_id is None:
        return None
    if not thumbnails.thumb_path(thumbnails.thumbs_dir(output_dir), int(file_id)).is_file():
        return None
    return f"{thumbnails.DIR_NAME}/{int(file_id)}{thumbnails.SUFFIX}"


# --- HTML ------------------------------------------------------------------

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>照片去重 - 审阅</title><style>
body{font-family:system-ui,"Microsoft YaHei",Arial,sans-serif;margin:18px;background:#111;color:#eee}
h1{font-size:20px;margin:0 0 8px}
.stats,.notice{color:#9cf;margin:8px 0;font-size:13px;line-height:1.6}
.notice{padding:9px 11px;background:#18232b;border-radius:6px}
nav{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:12px 0}
button,select{padding:6px 11px;background:#222;color:#eee;border:1px solid #555;border-radius:5px}
button.on{background:#2b4;color:#000;font-weight:bold}
button:disabled{opacity:.4}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}
.card,.group{border:1px solid #555;padding:7px;border-radius:6px;background:#1a1a1a}
.group{margin-bottom:12px;width:100%;box-sizing:border-box}
.keep{border-color:#4c9}.maybe{border-color:#fc3}.unknown{border-color:#999}
.auto_remove{border-color:#c55}.ungrouped{border-color:#456}
img{display:block;width:190px;height:160px;object-fit:contain;background:#000;border-radius:3px}
.miss{width:190px;height:160px;background:#221c1c;color:#c88;font-size:12px;
 display:flex;align-items:center;justify-content:center;text-align:center;border-radius:3px}
.cap{font-size:11px;color:#bbb;max-width:190px;word-break:break-word;margin-top:4px}
.tag{font-weight:bold;color:#fc9}
</style></head><body>
<h1>照片去重 - 审阅</h1>
<div class="notice" id="perf">PERFORMANCE_PANEL</div>
<div class="notice" id="mode">STATIC_NOTICE</div>
<nav>
 <button data-view="ALL">全部（时间线）</button>
 <button data-view="MAYBE">待确认 MAYBE</button>
 <button data-view="UNKNOWN">未知 UNKNOWN</button>
 <button data-view="GROUPS">分组建议 GROUPS</button>
 <span>&nbsp;|&nbsp;</span>
 <button id="first">&laquo; 首页</button>
 <button id="prev">&lsaquo; 上一页</button>
 <button id="next">下一页 &rsaquo;</button>
 <button id="last">尾页 &raquo;</button>
 <label>每页 <select id="size">
  <option value="50">50</option><option value="100" selected>100</option>
  <option value="200">200</option></select> 张</label>
</nav>
<div class="stats" id="stats">SUMMARY_TEXT</div>
<div class="row" id="content"></div>
<script>
const content=document.getElementById('content'),stats=document.getElementById('stats');
let view='ALL',page=1,pages=1,size=100;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function tile(r){
 const cls=esc(String(r.decision||'ungrouped').toLowerCase());
 const img=r.thumb==='ok'
  ?`<img loading="lazy" src="/api/thumb/${r.file_id}.jpg" onerror="this.outerHTML='<div class=miss>缩略图缺失<br>（不会回源读取机械盘）</div>'">`
  :`<div class=miss>缩略图不可用<br>${esc(r.thumb_error||r.thumb||'未生成')}</div>`;
 return `<div class="card ${cls}">${img}<div class=cap><span class=tag>${esc(r.decision)}</span>`
  +`${r.is_keep?'（保留项）':''}<br>${esc(r.basename)}<br>${r.width||'?'}x${r.height||'?'}`
  +` &middot; 质量=${r.quality_score??'?'} &middot; 人脸=${r.face_count??'?'}<br>`
  +`${esc(r.exif_datetime||'无拍摄时间')}<br>${esc(r.reason||'')}</div></div>`;
}
function groupBlock(g){
 return `<div class=group><div class=tag>分组 #${g.group_id} [${esc(g.group_type)}] &middot; `
  +`${g.member_count} 张</div><div class=row>${(g.members||[]).map(tile).join('')}</div></div>`;
}
function setButtons(){
 document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('on',b.dataset.view===view));
 document.getElementById('prev').disabled=page<=1;
 document.getElementById('first').disabled=page<=1;
 document.getElementById('next').disabled=page>=pages;
 document.getElementById('last').disabled=page>=pages;
}
async function load(){
 try{
  const r=await fetch(`/api/page?view=${view}&page=${page}&page_size=${size}`);
  if(!r.ok)throw new Error('HTTP '+r.status);
  const d=await r.json();
  if(d.error)throw new Error(d.error);
  page=d.page;pages=d.pages;
  document.getElementById('mode').textContent=
   '本地服务器模式。缩略图仅从 Stage 1 写入的 SSD 缓存读取，翻页绝不会回源读取机械盘上的原图。'
   +'本页面只读：不会移动、删除或修改任何原图。';
  stats.textContent=`${view}：共 ${d.total} 项 · 第 ${d.page}/${d.pages} 页 · 本页 ${d.shown} 项`
   +` · 本页之后还剩 ${d.remaining_after_page} 项`;
  content.innerHTML=d.items.map(view==='GROUPS'?groupBlock:tile).join('')||'<div>本视图没有条目。</div>';
  const s=await (await fetch('/api/status')).json();
  stats.textContent+=` · 缩略图缓存 ${s.thumb_cache_files} 个文件 / `
   +`${(s.thumb_cache_bytes/1073741824).toFixed(2)} GiB`;
 }catch(e){
  stats.textContent='分页浏览需要本地服务器（不要加 --no-serve）。'+e.message;
 }
 setButtons();
}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;page=1;load()});
document.getElementById('prev').onclick=()=>{if(page>1){page--;load()}};
document.getElementById('next').onclick=()=>{if(page<pages){page++;load()}};
document.getElementById('first').onclick=()=>{page=1;load()};
document.getElementById('last').onclick=()=>{page=pages;load()};
document.getElementById('size').onchange=e=>{size=Number(e.target.value);page=1;load()};
load();
</script>
<noscript>该审阅页面需要 JavaScript 和本地服务器才能分页浏览大型图库。以下为静态后备内容。</noscript>
<div class="row">STATIC_FALLBACK</div>
</body></html>"""

_STATIC_NOTICE = (
    "静态后备页面。请启动本地服务器（默认开启）以分页浏览完整的全部时间线、"
    "MAYBE、UNKNOWN 和 GROUPS 视图。"
)


def _tile(output_dir: Path, rec: Dict[str, Any]) -> str:
    """Static-fallback tile using only an already-cached SSD thumbnail."""
    decision = str(rec.get("decision") or "UNKNOWN")
    uri = cached_thumb_uri(output_dir, rec.get("file_id"))
    if uri:
        image = f'<img loading="lazy" src="{html.escape(uri)}">'
    else:
        image = '<div class="miss">缩略图不可用<br>thumbnail unavailable</div>'
    score = rec.get("quality_score")
    score_txt = f"{score:.1f}" if isinstance(score, (int, float)) else "?"
    return (
        f'<div class="card {html.escape(decision.lower())}">{image}<div class="cap">'
        f'<span class="tag">{html.escape(decision)}</span><br>'
        f'{html.escape(str(rec.get("basename") or ""))}<br>'
        f'{rec.get("width") or "?"}x{rec.get("height") or "?"} &middot; 质量={score_txt} '
        f'&middot; 人脸={rec.get("face_count") if rec.get("face_count") is not None else "?"}<br>'
        f'{html.escape(str(rec.get("reason") or ""))}</div></div>'
    )


def render_html(
    data: Dict[str, Any],
    output_dir: Path,
    review_limit: int = DEFAULT_REVIEW_LIMIT,
    performance_panel: str = "",
) -> str:
    """Render the paged review page plus a small cached-thumbnail fallback.

    ``review_limit`` bounds only the static ``file://`` fallback tiles; 0 omits
    them entirely. The paged views are always complete because they are served
    from the DB index, not embedded here.
    """
    if review_limit < 0:
        raise ValueError("review_limit must be 0 or greater")
    queue = data["queues"]["MAYBE"] or data["queues"]["UNKNOWN"]
    blocks = [_tile(Path(output_dir), rec) for rec in queue[:review_limit]]
    return (
        _PAGE_TEMPLATE
        .replace("PERFORMANCE_PANEL", performance_panel or
                 "性能指标由一键流水线写入（参见 performance.txt）。")
        .replace("STATIC_NOTICE", _STATIC_NOTICE)
        .replace(
            "SUMMARY_TEXT",
            f'分组 {len(data["groups"])} 个 &middot; '
            f'建议自动删除（AUTO_REMOVE）{len(data["delete_paths"])} 个文件 &middot; '
            f'MAYBE {len(data["queues"]["MAYBE"])} &middot; '
            f'UNKNOWN {len(data["queues"]["UNKNOWN"])} &middot; '
            f'约可释放 {data["total_delete_bytes"] / (1024 ** 3):.2f} GiB',
        )
        .replace("STATIC_FALLBACK", "".join(blocks))
    )


_PANEL_START = '<div class="notice" id="perf">'
_PANEL_END = "</div>"


def rewrite_performance_panel(review_html: Path, panel: str) -> bool:
    """Replace the performance panel once final stage timings are known."""
    review_html = Path(review_html)
    try:
        text = review_html.read_text(encoding="utf-8")
    except OSError:
        return False
    start = text.find(_PANEL_START)
    if start < 0:
        return False
    body_start = start + len(_PANEL_START)
    end = text.find(_PANEL_END, body_start)
    if end < 0:
        return False
    review_html.write_text(text[:body_start] + panel + text[end:], encoding="utf-8")
    return True
