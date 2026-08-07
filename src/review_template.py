"""The single-page review shell: markup, CSS and the embedded browser script.

Split out of :mod:`src.review_page` so that the Python rendering logic and the
(much larger) front-end asset stay separately reviewable. Nothing here touches
the filesystem; :mod:`src.review_page` substitutes the four placeholders
(``PERFORMANCE_PANEL``, ``STATIC_NOTICE``, ``SUMMARY_TEXT``, ``STATIC_FALLBACK``)
and writes the result.

What the page deliberately does *not* show: the reviewer only needs to decide
which photo to keep, so the visible UI carries no algorithm diagnostics
(quality score, face count, per-member grouping reason) and no runtime/storage
internals (stage timings, thumbnail-cache size, which disk a read comes from).
The two diagnostic containers (``#perf``, ``#mode``) are still rendered, and
still rewritten by the pipeline, but they are hidden -- ``performance.txt`` and
the generated report stay the place where those numbers are read.

Front-end boundaries this template must keep:

* Every grid tile and every filmstrip frame is served by ``/api/thumb/<id>.jpg``,
  i.e. the SSD cache Stage 1 wrote. Paging and browsing never wake the HDD.
* ``/api/original/<id>`` is requested only while the lightbox is open, for the
  one photo being inspected (plus the AI keeper in the temporary compare split).
  Closing the viewer drops the ``src`` again.
* The browser only ever sees ``file_id`` and ``basename``; no source path is
  rendered, and no endpoint other than the read-only API is used.

The UI text is Chinese; state names (KEEP/MAYBE/UNKNOWN/AUTO_REMOVE, group
types, and the accept/pick/mark/clear actions) stay English to match the
database and ``review_state.json``.
"""

from __future__ import annotations

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>照片去重 - 审阅</title><style>
:root{--tile:clamp(240px,26vw,460px);--tileh:clamp(200px,21vw,380px);
 --fsw:clamp(80px,8.5vw,150px);--fsh:clamp(60px,6.4vw,112px)}
body{font-family:system-ui,"Microsoft YaHei",Arial,sans-serif;margin:18px;background:#111;color:#eee}
h1{font-size:20px;margin:0 0 8px}
.stats,.notice{color:#9cf;margin:8px 0;font-size:13px;line-height:1.6}
.notice{padding:9px 11px;background:#18232b;border-radius:6px}
/* Diagnostics stay in the document (the pipeline rewrites #perf) but out of sight. */
.diag{display:none!important}
nav{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:12px 0}
button,select{padding:6px 11px;background:#222;color:#eee;border:1px solid #555;border-radius:5px;cursor:pointer}
button.on{background:#2b4;color:#000;font-weight:bold}
button:disabled{opacity:.4;cursor:not-allowed}
.row{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--tile),1fr));
 gap:12px;align-items:start}
.row.groups{display:block}
.card,.group{border:1px solid #555;padding:7px;border-radius:6px;background:#1a1a1a}
.group{margin-bottom:14px;width:100%;box-sizing:border-box;position:relative}
.group.focused{border-color:#fc3;box-shadow:0 0 12px rgba(255,204,51,0.5)}
.group.reviewed{border-left:4px solid #2b4}
.group.marked{border-left:4px solid #f93}
.keep{border-color:#4c9}.maybe{border-color:#fc3}.unknown{border-color:#999}
.auto_remove{border-color:#c55}.ungrouped{border-color:#456}
.card{position:relative;min-width:0}
.card.focused{border-color:#fc3;box-shadow:0 0 8px rgba(255,204,51,0.4)}
.card img{display:block;width:100%;height:var(--tileh);object-fit:contain;background:#000;
 border-radius:3px;cursor:zoom-in}
.card button{font-size:11px;margin-top:5px;width:100%}
.miss{width:100%;height:var(--tileh);box-sizing:border-box;background:#221c1c;color:#c88;font-size:12px;
 display:flex;align-items:center;justify-content:center;text-align:center;border-radius:3px}
.cap{font-size:12px;color:#bbb;max-width:100%;word-break:break-word;margin-top:5px;line-height:1.5}
.tag{font-weight:bold;color:#fc9}
.badge{display:inline-block;font-size:10px;padding:1px 5px;margin-left:4px;border-radius:8px;
 background:#333;color:#eee;white-space:nowrap;vertical-align:middle}
.badge.ai{background:#1d4a37;color:#8f9}
.badge.human{background:#4a3419;color:#fc9}
.badge.cur{background:#fc3;color:#000;font-weight:bold}
.lightbox{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.94);display:none;
 flex-direction:column;padding:12px;box-sizing:border-box}.lightbox.open{display:flex}
.viewerbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;flex:0 0 auto}
.viewerbar .title{flex:1;color:#ddd;font-size:13px}
.viewerbar button{font-size:12px;padding:5px 9px}
.viewer{display:grid;grid-template-columns:1fr;gap:10px;min-height:0;flex:1 1 auto}
.viewer.compare{grid-template-columns:1fr 1fr}
.pane{min-width:0;min-height:0;display:flex;flex-direction:column;align-items:center}
.pane img{width:100%;flex:1 1 auto;min-height:0;height:100%;object-fit:contain;background:#000;cursor:default}
.pane .label{font-size:12px;color:#fc9;margin-bottom:4px;flex:0 0 auto;text-align:center}
.filmstrip{flex:0 0 auto;display:none;gap:6px;overflow-x:auto;overflow-y:hidden;
 padding:8px 2px 2px;margin-top:8px;border-top:1px solid #333;justify-content:safe center}
.fs-item{flex:0 0 auto;border:2px solid #444;border-radius:5px;padding:2px;background:#161616;
 cursor:pointer;text-align:center}
.fs-item.current{border-color:#fc3;box-shadow:0 0 10px rgba(255,204,51,.6)}
.fs-item img{display:block;width:var(--fsw);height:var(--fsh);object-fit:cover;background:#000;
 border-radius:3px}
.fs-miss{width:var(--fsw);height:var(--fsh);box-sizing:border-box;background:#221c1c;color:#c88;
 font-size:10px;display:flex;align-items:center;justify-content:center;border-radius:3px}
.fs-meta{font-size:10px;color:#bbb;margin-top:2px;white-space:nowrap}
.fs-idx{color:#9cf}
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
@media(max-width:700px){:root{--tile:min(100%,300px);--tileh:clamp(180px,45vw,300px)}}
</style></head><body>
<h1>照片去重 - 审阅</h1>
<div class="notice diag" id="perf" hidden>PERFORMANCE_PANEL</div>
<div class="notice diag" id="mode" hidden>STATIC_NOTICE</div>
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
  <button id="vAccept">A 接受AI</button><button id="vPick">P 设为keeper</button>
  <button id="vMark">M 稍后</button><button id="vClear">U 清除</button>
  <button id="compare">C 与组内 KEEP 对比</button><button id="vprev">&lsaquo; 上一张</button>
  <button id="vnext">下一张 &rsaquo;</button><button id="closeViewer">关闭 Esc</button></div>
 <div class="viewer" id="viewer">
  <div class="pane" id="leftPane"><div class="label" id="leftLabel"></div><img id="leftImage"></div>
  <div class="pane" id="rightPane"><div class="label" id="rightLabel"></div><img id="rightImage"></div>
 </div>
 <div class="filmstrip" id="filmstrip"></div>
</div>
<div class="help" id="help">
 <div class="help-box">
  <h2>快捷键说明</h2>
  <table>
   <tr><td>? 或 F1</td><td>显示/隐藏此帮助</td></tr>
   <tr><td>Esc</td><td>关闭查看器/帮助</td></tr>
   <tr><td colspan="2" style="color:#fc3;font-weight:bold">GROUPS 列表导航</td></tr>
   <tr><td>↑ / K</td><td>上一组</td></tr>
   <tr><td>↓ / J</td><td>下一组</td></tr>
   <tr><td>← / H</td><td>组内上一张</td></tr>
   <tr><td>→ / L</td><td>组内下一张</td></tr>
   <tr><td>Enter / Space</td><td>打开/关闭高清相册模式</td></tr>
   <tr><td colspan="2" style="color:#fc3;font-weight:bold">高清相册模式（大图打开时）</td></tr>
   <tr><td>← / H&nbsp;&nbsp;→ / L</td><td>切换组内上一张/下一张（点击底部缩略图同效）</td></tr>
   <tr><td>C</td><td>临时双栏对比（当前照片 vs. AI keeper），再按恢复单张</td></tr>
   <tr><td colspan="2" style="color:#fc3;font-weight:bold">审阅动作（列表与相册模式通用）</td></tr>
   <tr><td>A</td><td>接受 AI keeper，标记已审</td></tr>
   <tr><td>P</td><td>设当前照片为人工 keeper，标记已审</td></tr>
   <tr><td>M</td><td>标记稍后再看</td></tr>
   <tr><td>U</td><td>清除本组人工状态</td></tr>
  </table>
  <p style="color:#999;font-size:12px;margin-top:12px">A/P/M 后自动前进至下一组；若在相册模式按下，
  大图保持打开并直接显示下一组，便于连续审阅（U 停在本组）。
  所有状态仅写入 output 目录，原图永远只读。</p>
  <button onclick="closeHelp()" style="width:100%;margin-top:12px">关闭 Esc</button>
 </div>
</div>
<script>
const content=document.getElementById('content'),stats=document.getElementById('stats');
let view='ALL',page=1,pages=1,size=100,loadGeneration=0;
let visibleItems=[],groupMembers=new Map(),viewerItems=[],viewerIndex=0,viewerGroupId=null;
let focusedGroupIndex=-1,focusedCardIndex=-1,reviewState={},compareMode=false;
let lastPageData=null,lastStatus=null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>v==null?null:Number(v);
function viewerActionKey(key){
 const k=String(key||'').toLowerCase();
 return k==='a'?'accept':k==='p'?'pick':k==='m'?'mark':k==='u'?'clear':null;
}
function humanKeeperId(gid){
 const st=(gid==null?null:reviewState[gid])||{};
 return st.action==='pick'&&st.file_id!=null?Number(st.file_id):null;
}
function aiKeeperId(items){
 const found=(items||[]).find(m=>m.is_keep===true);
 return found?Number(found.file_id):null;
}
function badges(r,aiId,humanId,short){
 const fid=Number(r.file_id);let out='';
 if(aiId!=null&&Number(aiId)===fid)out+=`<span class="badge ai">🤖AI${short?'':' keeper'}</span>`;
 if(humanId!=null&&Number(humanId)===fid)out+=`<span class="badge human">👤人工${short?'':' keeper'}</span>`;
 return out;
}
function thumbTag(r,attrs){
 const fid=Number(r.file_id);
 // Short, plain wording on failure: the reviewer needs to know the preview is
 // missing, not which decoder raised what.
 return r.thumb==='ok'
  ?`<img loading="lazy" src="/api/thumb/${fid}.jpg" ${attrs||''}`
   +` onerror="this.outerHTML='<div class=miss>缩略图缺失</div>'">`
  :`<div class=miss>缩略图不可用</div>`;
}
function photoTile(r,index,aiId,humanId){
 const cls=esc(String(r.decision||'ungrouped').toLowerCase());
 const fid=Number(r.file_id);
 const human=humanId!=null&&Number(humanId)===fid;
 const idxAttr=index==null?'':` data-card-index="${index}"`;
 return `<div class="card ${cls}${human?' keep':''}" data-file-id="${fid}"${idxAttr}>`
  +thumbTag(r,`onclick="openViewer(${fid})" title="点击查看高清大图"`)
  +`<div class=cap><span class=tag>${esc(r.decision)}</span>${badges(r,aiId,humanId)}<br>`
  +`${esc(r.basename)}<br>${r.width||'?'}x${r.height||'?'}<br>`
  +`${esc(r.exif_datetime||'无拍摄时间')}`
  +`<button onclick="openViewer(${fid})">查看高清大图</button>`
  +(index==null?'':`<button onclick="pickKeeper(${fid})">P 设为keeper</button>`)
  +`</div></div>`;
}
function tile(r){return photoTile(r,null,r.is_keep===true?Number(r.file_id):null,null)}
function groupBlock(g){
 const state=reviewState[g.group_id]||{};
 const stateClass=state.action==='accept'||state.action==='pick'?'reviewed':state.action==='mark'?'marked':'';
 const members=g.members||[];
 const humanId=humanKeeperId(g.group_id),aiId=aiKeeperId(members);
 let keeperName='';
 if(humanId!=null){
  const keeperMember=members.find(m=>Number(m.file_id)===humanId);
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
  +`<div class=row>${members.map((m,i)=>photoTile(m,i,aiId,humanId)).join('')}</div></div>`;
}
function filmstripHtml(items,currentId,aiId,humanId){
 const list=items||[];
 return list.map((r,i)=>{
  const fid=Number(r.file_id),cur=Number(currentId)===fid;
  const marks=badges(r,aiId,humanId,true)+(cur?'<span class="badge cur">当前</span>':'');
  return `<div class="fs-item${cur?' current':''}" data-file-id="${fid}"`
   +` title="${esc(r.basename||'')}" aria-current="${cur?'true':'false'}"`
   +` onclick="showViewerIndex(${i})">`
   +(r.thumb==='ok'?`<img loading="lazy" src="/api/thumb/${fid}.jpg" alt="">`
     :'<div class="fs-miss">无缩略图</div>')
   +`<div class="fs-meta"><span class="fs-idx">${i+1}/${list.length}</span>${marks}</div></div>`;
 }).join('');
}
function setButtons(){
 document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('on',b.dataset.view===view));
 document.getElementById('prev').disabled=page<=1;
 document.getElementById('first').disabled=page<=1;
 document.getElementById('next').disabled=page>=pages;
 document.getElementById('last').disabled=page>=pages;
}
function updateStats(d,s){
 const rs=s.review_state||{reviewed:0,marked:0,total:0};
 const groups=document.querySelectorAll('.group');
 const globalOffset=(d.page-1)*d.page_size+(focusedGroupIndex>=0?focusedGroupIndex+1:1);
 const itemLabel=view==='GROUPS'?'组':'项';
 // Progress only: how much there is, where you are, how much you have decided.
 stats.textContent=`${view}：共 ${d.total} ${itemLabel} · 第 ${d.page}/${d.pages} 页`;
 if(view==='GROUPS'&&groups.length>0)stats.textContent+=` · 当前 ${globalOffset}/${d.total}`
  +` · 已审 ${rs.reviewed}/${d.total} · 稍后 ${rs.marked}`;
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
  groupMembers=new Map();
  if(view==='GROUPS')d.items.forEach(g=>groupMembers.set(Number(g.group_id),g.members||[]));
  visibleItems=view==='GROUPS'?d.items.flatMap(g=>g.members||[]):d.items;
  reviewState=view==='GROUPS'&&d.review_state?d.review_state:{};
  content.className=view==='GROUPS'?'row groups':'row';
  content.innerHTML=d.items.map(view==='GROUPS'?groupBlock:tile).join('')||'<div>本视图没有条目。</div>';
  const s=await (await fetch('/api/status')).json();
  if(generation!==loadGeneration)return;
  lastPageData=d;lastStatus=s;
  focusedGroupIndex=-1;focusedCardIndex=-1;
  updateStats(d,s);
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
function viewerOpen(){return lightbox.classList.contains('open')}
function currentViewerItem(){return viewerItems.length?viewerItems[viewerIndex]:null}
function currentViewerFileId(){const r=currentViewerItem();return r?Number(r.file_id):null}
function showOne(r){
 compareMode=false;
 viewer.classList.remove('compare');document.getElementById('rightPane').style.display='none';
 rightImage.removeAttribute('src');
 leftImage.src=`/api/original/${Number(r.file_id)}`;
 const aiId=aiKeeperId(viewerItems),humanId=humanKeeperId(viewerGroupId);
 document.getElementById('leftLabel').innerHTML=
  `${esc(r.decision)} · ${esc(r.basename)}${badges(r,aiId,humanId)}`;
 document.getElementById('viewerTitle').textContent=
  `${r.width||'?'}×${r.height||'?'}`
  +` · 组内第 ${viewerIndex+1}/${viewerItems.length} 张`
  +(viewerGroupId!=null?` · 分组 #${viewerGroupId}`:'');
 document.getElementById('compare').style.display=
  viewerItems.length>1&&aiId!=null&&aiId!==Number(r.file_id)?'inline-block':'none';
 const inGroups=view==='GROUPS'&&viewerGroupId!=null;
 ['vAccept','vPick','vMark','vClear'].forEach(id=>{
  document.getElementById(id).style.display=inGroups?'inline-block':'none'});
 const strip=document.getElementById('filmstrip');
 strip.innerHTML=filmstripHtml(viewerItems,r.file_id,aiId,humanId);
 strip.style.display=viewerItems.length>1?'flex':'none';
 const current=strip.querySelector?strip.querySelector('.fs-item.current'):null;
 if(current&&current.scrollIntoView)current.scrollIntoView({block:'nearest',inline:'center',behavior:'smooth'});
 syncCardFocusToViewer(r.file_id);
}
function showViewerIndex(index){
 if(!viewerItems.length)return;
 viewerIndex=(index+viewerItems.length)%viewerItems.length;
 showOne(viewerItems[viewerIndex]);
}
function stepViewer(delta){showViewerIndex(viewerIndex+delta)}
async function openViewer(id){
 const fid=Number(id);
 const chosen=visibleItems.find(r=>Number(r.file_id)===fid);if(!chosen)return;
 viewerGroupId=num(chosen.group_id);
 viewerItems=viewerGroupId!=null?(groupMembers.get(viewerGroupId)||[]):[];
 if(!viewerItems.length&&viewerGroupId!=null){
  try{const r=await fetch(`/api/group/${viewerGroupId}`);if(r.ok)viewerItems=(await r.json()).members||[]}
  catch(_e){viewerItems=[]}
 }
 if(!viewerItems.length)viewerItems=[chosen];
 focusGroupByFileId(fid);
 viewerIndex=Math.max(0,viewerItems.findIndex(r=>Number(r.file_id)===fid));
 lightbox.classList.add('open');
 showOne(viewerItems[viewerIndex]);
}
function closeViewer(){
 lightbox.classList.remove('open');
 leftImage.removeAttribute('src');rightImage.removeAttribute('src');
 const strip=document.getElementById('filmstrip');strip.innerHTML='';strip.style.display='none';
 viewerItems=[];viewerIndex=0;viewerGroupId=null;compareMode=false;
}
document.getElementById('compare').onclick=()=>{
 if(compareMode){showOne(viewerItems[viewerIndex]);return}
 const selected=currentViewerItem();if(!selected)return;
 const aiKeeper=viewerItems.find(r=>r.is_keep===true)||viewerItems[0];
 compareMode=true;
 viewer.classList.add('compare');document.getElementById('rightPane').style.display='flex';
 leftImage.src=`/api/original/${Number(aiKeeper.file_id)}`;
 rightImage.src=`/api/original/${Number(selected.file_id)}`;
 document.getElementById('leftLabel').textContent=`AI KEEP · ${aiKeeper.basename}`;
 document.getElementById('rightLabel').textContent=`${selected.decision} · ${selected.basename}`;
};
document.getElementById('vprev').onclick=()=>stepViewer(-1);
document.getElementById('vnext').onclick=()=>stepViewer(1);
document.getElementById('closeViewer').onclick=closeViewer;
['vAccept','vPick','vMark','vClear'].forEach((id,i)=>{
 const action=['accept','pick','mark','clear'][i];
 document.getElementById(id).onclick=()=>{
  if(view!=='GROUPS'||viewerGroupId==null)return;
  reviewAction(viewerGroupId,action)
 };
});
function openHelp(){document.getElementById('help').classList.add('open')}
function closeHelp(){document.getElementById('help').classList.remove('open')}
document.getElementById('helpBtn').onclick=openHelp;
function setGroupFocus(idx){
 document.querySelectorAll('.group').forEach(g=>g.classList.remove('focused'));
 const groups=document.querySelectorAll('.group');
 if(idx>=0&&idx<groups.length){focusedGroupIndex=idx;groups[idx].classList.add('focused');groups[idx].scrollIntoView({block:'nearest',behavior:'smooth'});focusedCardIndex=0;setCardFocus(0)}
 if(lastPageData&&lastStatus)updateStats(lastPageData,lastStatus);
}
function setCardFocus(idx){
 const group=document.querySelectorAll('.group')[focusedGroupIndex];
 if(!group)return;
 const cards=group.querySelectorAll('.card');
 cards.forEach(c=>c.classList.remove('focused'));
 if(idx>=0&&idx<cards.length){focusedCardIndex=idx;cards[idx].classList.add('focused');cards[idx].scrollIntoView({block:'nearest',behavior:'smooth'})}
}
function locateCard(fid){
 const groups=document.querySelectorAll('.group');
 for(let i=0;i<groups.length;i++){
  const cards=groups[i].querySelectorAll('.card');
  for(let j=0;j<cards.length;j++)if(Number(cards[j].dataset.fileId)===Number(fid))return{groupIndex:i,cardIndex:j};
 }
 return null;
}
function focusGroupByFileId(fid){
 if(view!=='GROUPS')return;
 const at=locateCard(fid);if(!at)return;
 if(at.groupIndex!==focusedGroupIndex)setGroupFocus(at.groupIndex);
 setCardFocus(at.cardIndex);
}
function syncCardFocusToViewer(fid){
 if(view!=='GROUPS')return;
 const group=document.querySelectorAll('.group')[focusedGroupIndex];
 if(!group)return;
 if(viewerGroupId!=null&&Number(group.dataset.groupId)!==Number(viewerGroupId))return;
 const cards=group.querySelectorAll('.card');
 for(let i=0;i<cards.length;i++)if(Number(cards[i].dataset.fileId)===Number(fid)){setCardFocus(i);return}
}
function pickTargetFileId(){
 if(viewerOpen()){const fid=currentViewerFileId();if(fid)return fid}
 const group=document.querySelectorAll('.group')[focusedGroupIndex];
 const card=group?group.querySelectorAll('.card')[focusedCardIndex]:null;
 return card?Number(card.dataset.fileId):null;
}
async function reopenViewerAt(groupIndex){
 const group=document.querySelectorAll('.group')[groupIndex];
 if(!group)return;
 const card=group.querySelectorAll('.card')[0];
 const fid=card?Number(card.dataset.fileId):null;
 if(fid)await openViewer(fid);
}
async function reviewAction(gid,action,explicitFileId){
 let fid=null;
 if(action==='pick'){
  fid=explicitFileId!=null?Number(explicitFileId):pickTargetFileId();
  if(!fid){alert('请先选择一张照片');return}
 }
 const keepViewer=viewerOpen(),viewerFid=keepViewer?currentViewerFileId():null;
 const savedIndex=focusedGroupIndex,groupCount=document.querySelectorAll('.group').length;
 try{
  const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({group_id:gid,file_id:fid,action})});
  const d=await r.json();
  if(!r.ok||d.error){alert('操作失败: '+(d.error||'未知错误'));return}
  if(action==='clear'){
   await load(true);
   if(keepViewer&&viewerFid)await openViewer(viewerFid);
   return
  }
  // A/P/M advance to the next group; the album view stays open on it so a long
  // review run never needs the mouse.
  let targetIndex=savedIndex;
  if(savedIndex>=groupCount-1){
   if(page<pages){page++;await load();targetIndex=0}
   else{await load(true);targetIndex=focusedGroupIndex}
  }else{
   await load(true);targetIndex=savedIndex+1;setGroupFocus(targetIndex)
  }
  if(keepViewer&&targetIndex>=0)await reopenViewerAt(targetIndex);
 }catch(e){alert('操作失败: '+e.message)}
}
function pickKeeper(fid){
 const at=locateCard(fid);if(!at)return;
 const groups=document.querySelectorAll('.group');
 if(at.groupIndex!==focusedGroupIndex)setGroupFocus(at.groupIndex);
 setCardFocus(at.cardIndex);
 reviewAction(Number(groups[at.groupIndex].dataset.groupId),'pick',Number(fid));
}
document.addEventListener('keydown',async e=>{
 const help=document.getElementById('help');
 if(help.classList.contains('open')){if(e.key==='Escape'||e.key==='?'||e.key==='F1'){e.preventDefault();closeHelp()}return}
 if(viewerOpen()){
  if(e.key==='Escape'){e.preventDefault();closeViewer();return}
  if(e.key==='Enter'||e.key===' '){e.preventDefault();closeViewer();return}
  if(e.key==='ArrowLeft'||e.key==='h'||e.key==='H'){e.preventDefault();stepViewer(-1);return}
  if(e.key==='ArrowRight'||e.key==='l'||e.key==='L'){e.preventDefault();stepViewer(1);return}
  if(e.key==='c'||e.key==='C'){e.preventDefault();document.getElementById('compare').click();return}
  const action=viewerActionKey(e.key);
  if(action&&view==='GROUPS'&&viewerGroupId!=null){e.preventDefault();await reviewAction(viewerGroupId,action)}
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
 else if(e.key==='c'||e.key==='C'){
  e.preventDefault();
  const group=groups[focusedGroupIndex];
  const card=group?group.querySelectorAll('.card')[focusedCardIndex]:null;
  if(card){
   const fid=Number(card.dataset.fileId);
   const gid=Number(group.dataset.groupId);
   const members=groupMembers.get(gid)||[];
   const aiKeeper=members.find(m=>m.is_keep);
   if(!aiKeeper){alert('本组无AI keeper，无法对比');return}
   if(Number(aiKeeper.file_id)===fid){alert('已选中AI keeper，无需对比');return}
   await openViewer(fid);document.getElementById('compare').click()
  }
 }
 else{
  const action=viewerActionKey(e.key);
  if(!action)return;
  const group=groups[focusedGroupIndex];
  if(!group)return;
  e.preventDefault();
  await reviewAction(Number(group.dataset.groupId),action)
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
