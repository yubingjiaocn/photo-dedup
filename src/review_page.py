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
button,select{padding:6px 11px;background:#222;color:#eee;border:1px solid #555;border-radius:5px;cursor:pointer}
button.on{background:#2b4;color:#000;font-weight:bold}
button:disabled{opacity:.4;cursor:not-allowed}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}
.card,.group{border:1px solid #555;padding:7px;border-radius:6px;background:#1a1a1a}
.group{margin-bottom:12px;width:100%;box-sizing:border-box;position:relative}
.group.focused{border-color:#fc3;box-shadow:0 0 12px rgba(255,204,51,0.5)}
.group.reviewed{border-left:4px solid #2b4}
.group.marked{border-left:4px solid #f93}
.keep{border-color:#4c9}.maybe{border-color:#fc3}.unknown{border-color:#999}
.auto_remove{border-color:#c55}.ungrouped{border-color:#456}
img{display:block;width:190px;height:160px;object-fit:contain;background:#000;border-radius:3px}
.card{position:relative}.card.focused{border-color:#fc3;box-shadow:0 0 8px rgba(255,204,51,0.4)}
.card img{cursor:zoom-in}.card button{font-size:11px;margin-top:5px;width:100%}
.miss{width:190px;height:160px;background:#221c1c;color:#c88;font-size:12px;
 display:flex;align-items:center;justify-content:center;text-align:center;border-radius:3px}
.cap{font-size:11px;color:#bbb;max-width:190px;word-break:break-word;margin-top:4px}
.tag{font-weight:bold;color:#fc9}
.lightbox{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.94);display:none;
 flex-direction:column;padding:12px;box-sizing:border-box}.lightbox.open{display:flex}
.viewerbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.viewerbar .title{flex:1;color:#ddd}.viewer{display:grid;grid-template-columns:1fr;gap:10px;
 min-height:0;flex:1}.viewer.compare{grid-template-columns:1fr 1fr}
.pane{min-width:0;min-height:0;display:flex;flex-direction:column;align-items:center}
.pane img{width:100%;height:calc(100vh - 100px);object-fit:contain;background:#000;cursor:default}
.pane .label{font-size:12px;color:#fc9;margin-bottom:4px}
.help{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.9);display:none;
 align-items:center;justify-content:center;padding:20px}.help.open{display:flex}
.help-box{background:#1a1a1a;border:1px solid #555;border-radius:8px;padding:20px;max-width:600px;
 max-height:90vh;overflow-y:auto}
.help-box h2{margin-top:0;color:#fc3}
.help-box table{width:100%;border-collapse:collapse;margin:10px 0}
.help-box td{padding:8px;border-bottom:1px solid #333}
.help-box td:first-child{color:#fc3;font-family:monospace;white-space:nowrap}
.review-actions{position:absolute;top:7px;right:7px;display:none;gap:4px}
.group.focused .review-actions{display:flex}
.review-actions button{padding:4px 8px;font-size:11px}
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
 <span>&nbsp;|&nbsp;</span>
 <button id="helpBtn">快捷键 ?</button>
</nav>
<div class="stats" id="stats">SUMMARY_TEXT</div>
<div class="row" id="content"></div>
<div class="lightbox" id="lightbox">
 <div class="viewerbar"><span class="title" id="viewerTitle"></span>
  <button id="compare">与组内 KEEP 对比</button><button id="vprev">&lsaquo; 上一张</button>
  <button id="vnext">下一张 &rsaquo;</button><button id="closeViewer">关闭 Esc</button></div>
 <div class="viewer" id="viewer">
  <div class="pane" id="leftPane"><div class="label" id="leftLabel"></div><img id="leftImage"></div>
  <div class="pane" id="rightPane"><div class="label" id="rightLabel"></div><img id="rightImage"></div>
 </div>
</div>
<div class="help" id="help">
 <div class="help-box">
  <h2>快捷键说明</h2>
  <table>
   <tr><td>? 或 F1</td><td>显示/隐藏此帮助</td></tr>
   <tr><td>Esc</td><td>关闭查看器/帮助</td></tr>
   <tr><td colspan="2" style="color:#fc3;font-weight:bold">GROUPS 模式导航</td></tr>
   <tr><td>↑ / K</td><td>上一组</td></tr>
   <tr><td>↓ / J</td><td>下一组</td></tr>
   <tr><td>← / H</td><td>组内上一张</td></tr>
   <tr><td>→ / L</td><td>组内下一张</td></tr>
   <tr><td colspan="2" style="color:#fc3;font-weight:bold">查看</td></tr>
   <tr><td>Enter / Space</td><td>打开/关闭高清图</td></tr>
   <tr><td>C</td><td>切换对比模式（当前照片 vs. AI keeper）</td></tr>
   <tr><td colspan="2" style="color:#fc3;font-weight:bold">审阅动作</td></tr>
   <tr><td>A</td><td>接受 AI keeper，标记已审</td></tr>
   <tr><td>P</td><td>设当前照片为人工 keeper，标记已审</td></tr>
   <tr><td>M</td><td>标记稍后再看</td></tr>
   <tr><td>U</td><td>清除本组人工状态</td></tr>
  </table>
  <p style="color:#999;font-size:12px;margin-top:12px">动作后自动前进至下一组（U 除外）。所有状态仅写入 output 目录，原图永远只读。</p>
  <button onclick="closeHelp()" style="width:100%;margin-top:12px">关闭 Esc</button>
 </div>
</div>
<script>
const content=document.getElementById('content'),stats=document.getElementById('stats');
let view='ALL',page=1,pages=1,size=100,loadGeneration=0;
let visibleItems=[],groupMembers=new Map(),viewerItems=[],viewerIndex=0;
let focusedGroupIndex=-1,focusedCardIndex=-1,reviewState={},compareMode=false;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function tile(r){
 const cls=esc(String(r.decision||'ungrouped').toLowerCase());
 const img=r.thumb==='ok'
  ?`<img loading="lazy" src="/api/thumb/${r.file_id}.jpg" onclick="openViewer(${r.file_id})" title="点击按需从原图库加载高清大图" onerror="this.outerHTML='<div class=miss>缩略图缺失<br>（不会自动回源读取机械盘）</div>'">`
  :`<div class=miss>缩略图不可用<br>${esc(r.thumb_error||r.thumb||'未生成')}</div>`;
 return `<div class="card ${cls}">${img}<div class=cap><span class=tag>${esc(r.decision)}</span>`
  +`${r.is_keep?'（保留项）':''}<br>${esc(r.basename)}<br>${r.width||'?'}x${r.height||'?'}`
  +` &middot; 质量=${r.quality_score??'?'} &middot; 人脸=${r.face_count??'?'}<br>`
  +`${esc(r.exif_datetime||'无拍摄时间')}<br>${esc(r.reason||'')}`
  +`<button onclick="openViewer(${r.file_id})">查看高清大图</button></div></div>`;
}
function groupBlock(g){
 const state=reviewState[g.group_id]||{};
 const stateClass=state.action==='accept'||state.action==='pick'?'reviewed':state.action==='mark'?'marked':'';
 let keeperName='';
 if(state.action==='pick'&&state.file_id){
  const keeperMember=(g.members||[]).find(m=>Number(m.file_id)===Number(state.file_id));
  keeperName=keeperMember?` ${esc(keeperMember.basename||'')}`:' ?'
 }
 const stateText=state.action==='accept'?'[已审-AI]':state.action==='pick'?`[已审-人工${keeperName}]`:state.action==='mark'?'[稍后]':'';
 return `<div class="group ${stateClass}" data-group-id="${g.group_id}">`
  +`<div class=tag>分组 #${g.group_id} [${esc(g.group_type)}] &middot; ${g.member_count} 张 ${stateText}</div>`
  +`<div class="review-actions">`
  +`<button onclick="reviewAction(${g.group_id},'accept')">A 接受AI</button>`
  +`<button onclick="reviewAction(${g.group_id},'mark')">M 稍后</button>`
  +`<button onclick="reviewAction(${g.group_id},'clear')">U 清除</button>`
  +`</div>`
  +`<div class=row>${(g.members||[]).map((m,i)=>tileWithIndex(m,i,state)).join('')}</div></div>`;
}
function tileWithIndex(r,i,state){
 const cls=esc(String(r.decision||'ungrouped').toLowerCase());
 const isHumanKeeper=state&&state.action==='pick'&&Number(state.file_id)===Number(r.file_id);
 const img=r.thumb==='ok'
  ?`<img loading="lazy" src="/api/thumb/${r.file_id}.jpg" onclick="openViewer(${r.file_id})" title="点击按需从原图库加载高清大图" onerror="this.outerHTML='<div class=miss>缩略图缺失<br>（不会自动回源读取机械盘）</div>'">`
  :`<div class=miss>缩略图不可用<br>${esc(r.thumb_error||r.thumb||'未生成')}</div>`;
 return `<div class="card ${cls}${isHumanKeeper?' keep':''}" data-file-id="${r.file_id}" data-card-index="${i}">${img}<div class=cap><span class=tag>${esc(r.decision)}</span>`
  +`${r.is_keep?' 🤖AI保留':isHumanKeeper?' 👤人工保留':''}<br>${esc(r.basename)}<br>${r.width||'?'}x${r.height||'?'}`
  +` &middot; 质量=${r.quality_score??'?'} &middot; 人脸=${r.face_count??'?'}<br>`
  +`${esc(r.exif_datetime||'无拍摄时间')}<br>${esc(r.reason||'')}`
  +`<button onclick="openViewer(${r.file_id})">查看高清大图</button>`
  +`<button onclick="pickKeeper(${r.file_id})">P 设为keeper</button></div></div>`;
}
function setButtons(){
 document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('on',b.dataset.view===view));
 document.getElementById('prev').disabled=page<=1;
 document.getElementById('first').disabled=page<=1;
 document.getElementById('next').disabled=page>=pages;
 document.getElementById('last').disabled=page>=pages;
}
async function load(preserveFocus){
 const generation=++loadGeneration;
 const savedGroupIndex=focusedGroupIndex,savedPage=page;
 try{
  closeViewer();
  const r=await fetch(`/api/page?view=${view}&page=${page}&page_size=${size}`);
  if(!r.ok)throw new Error('HTTP '+r.status);
  const d=await r.json();
  if(generation!==loadGeneration)return;
  if(d.error)throw new Error(d.error);
  page=d.page;pages=d.pages;
  document.getElementById('mode').textContent=
   '本地服务器模式。缩略图仅从 Stage 1 写入的 SSD 缓存读取，翻页绝不会回源读取机械盘上的原图。'
   +'本页面只读：不会移动、删除或修改任何原图。';
  groupMembers=new Map();
  if(view==='GROUPS')d.items.forEach(g=>groupMembers.set(Number(g.group_id),g.members||[]));
  visibleItems=view==='GROUPS'?d.items.flatMap(g=>g.members||[]):d.items;
  reviewState=view==='GROUPS'&&d.review_state?d.review_state:{};
  content.innerHTML=d.items.map(view==='GROUPS'?groupBlock:tile).join('')||'<div>本视图没有条目。</div>';
  const s=await (await fetch('/api/status')).json();
  if(generation!==loadGeneration)return;
  const rs=s.review_state||{reviewed:0,marked:0,total:0};
  const globalOffset=(d.page-1)*d.page_size+focusedGroupIndex+1;
  stats.textContent=`${view}：共 ${d.total} 组 · 第 ${d.page}/${d.pages} 页 · 本页 ${d.shown} 组`
   +` · 本页之后还剩 ${d.remaining_after_page} 组`;
  if(view==='GROUPS')stats.textContent+=` · 已审 ${rs.reviewed}/${d.total} · 稍后 ${rs.marked} · 当前 ${globalOffset}/${d.total}`;
  stats.textContent+=` · 缩略图缓存 ${s.thumb_cache_files} 个文件 / `
   +`${(s.thumb_cache_bytes/1073741824).toFixed(2)} GiB`;
  focusedGroupIndex=-1;focusedCardIndex=-1;
  if(view==='GROUPS'&&d.items.length>0){
   if(preserveFocus&&savedPage===page&&savedGroupIndex>=0&&savedGroupIndex<d.items.length){
    setGroupFocus(savedGroupIndex)
   }else{
    setGroupFocus(0)
   }
  }
 }catch(e){
  stats.textContent='分页浏览需要本地服务器（不要加 --no-serve）。'+e.message;
 }
 setButtons();
}
const lightbox=document.getElementById('lightbox'),viewer=document.getElementById('viewer');
const leftImage=document.getElementById('leftImage'),rightImage=document.getElementById('rightImage');
function showOne(r){
 viewer.classList.remove('compare');document.getElementById('rightPane').style.display='none';
 leftImage.src=`/api/original/${r.file_id}`;document.getElementById('leftLabel').textContent=`${r.decision} · ${r.basename}`;
 document.getElementById('viewerTitle').textContent=`高清原图（按需读取 HDD） ${r.width||'?'}×${r.height||'?'}`;
 const keeper=viewerItems.find(item=>item.is_keep);
 document.getElementById('compare').style.display=viewerItems.length>1&&keeper&&Number(keeper.file_id)!==Number(r.file_id)?'inline-block':'none';
}
async function openViewer(id){
 const chosen=visibleItems.find(r=>Number(r.file_id)===Number(id));if(!chosen)return;
 viewerItems=groupMembers.get(Number(chosen.group_id))||[];
 if(!viewerItems.length&&chosen.group_id!=null){try{const r=await fetch(`/api/group/${chosen.group_id}`);if(r.ok)viewerItems=(await r.json()).members||[]}catch(_e){viewerItems=[]}}
 if(!viewerItems.length)viewerItems=[chosen];
 viewerIndex=Math.max(0,viewerItems.findIndex(r=>Number(r.file_id)===Number(id)));
 showOne(viewerItems[viewerIndex]);lightbox.classList.add('open');
}
function stepViewer(delta){if(!viewerItems.length)return;viewerIndex=(viewerIndex+delta+viewerItems.length)%viewerItems.length;showOne(viewerItems[viewerIndex])}
document.getElementById('compare').onclick=()=>{
 if(compareMode){compareMode=false;showOne(viewerItems[viewerIndex]);return}
 compareMode=true;
 const selected=viewerItems[viewerIndex],keeper=viewerItems.find(r=>r.is_keep)||viewerItems[0];
 viewer.classList.add('compare');document.getElementById('rightPane').style.display='flex';
 leftImage.src=`/api/original/${keeper.file_id}`;rightImage.src=`/api/original/${selected.file_id}`;
 document.getElementById('leftLabel').textContent=`KEEP · ${keeper.basename}`;
 document.getElementById('rightLabel').textContent=`${selected.decision} · ${selected.basename}`;
};
document.getElementById('vprev').onclick=()=>stepViewer(-1);document.getElementById('vnext').onclick=()=>stepViewer(1);
function closeViewer(){lightbox.classList.remove('open');leftImage.removeAttribute('src');rightImage.removeAttribute('src');viewerItems=[];viewerIndex=0;compareMode=false}
document.getElementById('closeViewer').onclick=closeViewer;
function openHelp(){document.getElementById('help').classList.add('open')}
function closeHelp(){document.getElementById('help').classList.remove('open')}
document.getElementById('helpBtn').onclick=openHelp;
function setGroupFocus(idx){
 document.querySelectorAll('.group').forEach(g=>g.classList.remove('focused'));
 const groups=document.querySelectorAll('.group');
 if(idx>=0&&idx<groups.length){focusedGroupIndex=idx;groups[idx].classList.add('focused');groups[idx].scrollIntoView({block:'nearest',behavior:'smooth'});focusedCardIndex=0;setCardFocus(0)}
}
function setCardFocus(idx){
 const group=document.querySelectorAll('.group')[focusedGroupIndex];
 if(!group)return;
 const cards=group.querySelectorAll('.card');
 cards.forEach(c=>c.classList.remove('focused'));
 if(idx>=0&&idx<cards.length){focusedCardIndex=idx;cards[idx].classList.add('focused');cards[idx].scrollIntoView({block:'nearest',behavior:'smooth'})}
}
async function reviewAction(gid,action){
 const groups=document.querySelectorAll('.group');
 const group=groups[focusedGroupIndex];
 let fid=null;
 if(action==='pick'&&group){
  const card=group.querySelectorAll('.card')[focusedCardIndex];
  fid=card?Number(card.dataset.fileId):null;
  if(!fid){alert('请先选择一张照片');return}
 }
 const savedIndex=focusedGroupIndex,savedPage=page;
 try{
  const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({group_id:gid,file_id:fid,action})});
  const d=await r.json();
  if(!r.ok||d.error){alert('操作失败: '+(d.error||'未知错误'));return}
  if(action!=='clear'){
   // Auto-advance logic
   if(savedIndex>=groups.length-1){
    // Last group on page - go to next page first group
    if(page<pages){page++;await load();setGroupFocus(0)}
    else{await load(true)}
   }else{
    // Not last - advance to next on same page
    await load(true);setGroupFocus(savedIndex+1)
   }
  }else{
   await load(true)
  }
 }catch(e){alert('操作失败: '+e.message)}
}
function pickKeeper(fid){
 const groups=document.querySelectorAll('.group');
 if(focusedGroupIndex<0||focusedGroupIndex>=groups.length)return;
 const group=groups[focusedGroupIndex];
 const cards=group.querySelectorAll('.card');
 // Find card matching this fid
 for(let i=0;i<cards.length;i++){
  if(Number(cards[i].dataset.fileId)===Number(fid)){
   focusedCardIndex=i;break
  }
 }
 const gid=Number(group.dataset.groupId);
 reviewAction(gid,'pick')
}
document.addEventListener('keydown',e=>{
 const help=document.getElementById('help');
 if(help.classList.contains('open')){if(e.key==='Escape'||e.key==='?'||e.key==='F1'){e.preventDefault();closeHelp()}return}
 if(lightbox.classList.contains('open')){
  if(e.key==='Escape'){e.preventDefault();closeViewer()}
  else if(e.key==='Enter'||e.key===' '){e.preventDefault();closeViewer()}
  else if(e.key==='ArrowLeft'||e.key==='h'||e.key==='H'){e.preventDefault();stepViewer(-1)}
  else if(e.key==='ArrowRight'||e.key==='l'||e.key==='L'){e.preventDefault();stepViewer(1)}
  else if(e.key==='c'||e.key==='C'){e.preventDefault();document.getElementById('compare').click()}
  return
 }
 if(e.key==='?'||e.key==='F1'){e.preventDefault();openHelp();return}
 if(view!=='GROUPS')return;
 const groups=document.querySelectorAll('.group');
 if(groups.length===0)return;
 if(e.key==='ArrowUp'||e.key==='k'||e.key==='K'){
  e.preventDefault();
  if(focusedGroupIndex>0){setGroupFocus(focusedGroupIndex-1)}
  else if(page>1){page--;load().then(()=>{const g=document.querySelectorAll('.group');if(g.length>0)setGroupFocus(g.length-1)})}
 }
 else if(e.key==='ArrowDown'||e.key==='j'||e.key==='J'){
  e.preventDefault();
  if(focusedGroupIndex<groups.length-1){setGroupFocus(focusedGroupIndex+1)}
  else if(page<pages){page++;load().then(()=>setGroupFocus(0))}
 }
 else if(e.key==='ArrowLeft'||e.key==='h'||e.key==='H'){e.preventDefault();if(focusedCardIndex>0)setCardFocus(focusedCardIndex-1)}
 else if(e.key==='ArrowRight'||e.key==='l'||e.key==='L'){
  e.preventDefault();
  const group=groups[focusedGroupIndex];
  const cards=group?group.querySelectorAll('.card'):[];
  if(focusedCardIndex<cards.length-1)setCardFocus(focusedCardIndex+1)
 }
 else if(e.key==='Enter'||e.key===' '){
  e.preventDefault();
  const group=groups[focusedGroupIndex];
  const card=group?group.querySelectorAll('.card')[focusedCardIndex]:null;
  if(card){const fid=Number(card.dataset.fileId);if(fid)openViewer(fid)}
 }
 else if(e.key==='a'||e.key==='A'){e.preventDefault();const gid=Number(groups[focusedGroupIndex].dataset.groupId);reviewAction(gid,'accept')}
 else if(e.key==='p'||e.key==='P'){e.preventDefault();const gid=Number(groups[focusedGroupIndex].dataset.groupId);reviewAction(gid,'pick')}
 else if(e.key==='m'||e.key==='M'){e.preventDefault();const gid=Number(groups[focusedGroupIndex].dataset.groupId);reviewAction(gid,'mark')}
 else if(e.key==='u'||e.key==='U'){e.preventDefault();const gid=Number(groups[focusedGroupIndex].dataset.groupId);reviewAction(gid,'clear')}
 else if(e.key==='c'||e.key==='C'){
  e.preventDefault();
  const group=groups[focusedGroupIndex];
  const card=group?group.querySelectorAll('.card')[focusedCardIndex]:null;
  if(card){const fid=Number(card.dataset.fileId);const keeper=Array.from(group.querySelectorAll('.card')).find(c=>c.classList.contains('keep'));if(!keeper){alert('本组无AI keeper，无法对比');return}await openViewer(fid);document.getElementById('compare').click()}
 }
});
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
