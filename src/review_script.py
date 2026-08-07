"""The browser script behind the review workbench.

Split out of :mod:`src.review_template` so the page markup, the stylesheet and
the behaviour stay separately reviewable. It is plain ES2020 inlined into
``review.html`` -- no framework, no bundler, no build step.

What it does, in the order the reviewer meets it:

* **Opens on the pending queue.** ``GROUPS`` + queue ``PENDING`` is the default
  entry point and the workbench (group header, one main photo, filmstrip, action
  bar) is on screen immediately -- nothing has to be clicked open first.
* **Pages server-side.** Every list comes from ``/api/page`` with the queue
  applied by the server, so the browser holds one page, never the whole library.
  ``/api/locate`` answers "which page holds this group" for reload restore and
  for following an undo back to its group.
* **Advances by itself.** A/P/M take the decided group out of the pending queue
  and land on the next one; U asks the server to undo the newest decision and
  jumps back to exactly that group and photo.
* **Reads originals only for what is on screen.** Lists and filmstrips use
  ``/api/thumb/<id>.jpg`` (the SSD cache); ``/api/original/<id>`` is requested
  for the photo the stage is showing, plus the AI keeper while comparing or
  blinking. Nothing is prefetched.
"""

from __future__ import annotations

SCRIPT = r"""
// --- constants -------------------------------------------------------------
const LS_KEY='photo-dedup-review-v2';
const QUEUES=['PENDING','LATER','DONE'];
const QUEUE_LABEL={PENDING:'未审',LATER:'稍后',DONE:'已完成'};
const BROWSE_VIEWS=['ALL','MAYBE','UNKNOWN'];
const BROWSE_LABEL={ALL:'全部时间线',MAYBE:'待确认',UNKNOWN:'未知'};
const GROUP_TYPE_LABEL={sha_exact:'完全相同',phash_near:'高度相似',burst:'连拍',
 similar_scene:'相似场景'};
const DECISION_LABEL={KEEP:'保留',AUTO_REMOVE:'建议删除',MAYBE:'待确认',UNKNOWN:'未知',
 UNGROUPED:'未分组'};
const REASON_LABEL={GROUP_KEEPER:'组内最佳',BYTE_IDENTICAL:'字节完全相同',
 LOW_MARGIN:'与最佳差距很小',PHASH_NEAR_UNTRUSTED:'相似判断不够可靠',
 SIMILAR_SCENE_UNTRUSTED:'仅场景相似',GROUP_IMPURE:'同组内容不一致',
 SHA_GROUP_NO_MATCH:'哈希组内无匹配',PHASH_NEAR_DUP_ONLY:'仅像素级近重复',
 FACE_COUNT_MISMATCH:'人脸数量不一致',RELATIVE_ONLY:'仅相对比较',
 FEATURE_MISSING:'缺少分析特征',SUBJECTIVE_ONLY:'主观取舍',
 EXPOSURE_BORDERLINE:'曝光处于临界',dup:'近重复',keep:'保留',best:'组内最佳'};
const MAX_SCALE=8,MIN_SCALE=1;

// --- state -----------------------------------------------------------------
let view='GROUPS',queue='PENDING',page=1,pages=1,size=100,total=0;
let queueCounts={PENDING:0,LATER:0,DONE:0},undoDepth=0;
let groups=[],gIndex=0,mIndex=0,reviewState={},loadGeneration=0;
let browseItems=[],browseGroup=null,browseOpen=false;
let compareMode=false,blinkOn=false,evidenceOpen=false,density='comfy';
let mutationBusy=false;
let stateWarning=null;
let restoreGroupId=null,restoreFileId=null;

const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',
 '"':'&quot;',"'":'&#39;'}[c]));
const num=v=>v==null?null:Number(v);
const mode=()=>view==='GROUPS'?'queue':'browse';

// --- display vocabulary ----------------------------------------------------
function groupTypeText(value){
 const key=String(value||'');
 return GROUP_TYPE_LABEL[key]||key||'未知类型';
}
function decisionText(value){
 const key=String(value||'UNGROUPED').toUpperCase();
 return DECISION_LABEL[key]||key;
}
function reasonText(value){
 const raw=String(value||'');
 if(!raw)return '';
 const head=raw.split(':')[0].trim();
 return REASON_LABEL[head]||REASON_LABEL[raw]||raw;
}
function queueLabel(name){return QUEUE_LABEL[name]||name}
function actionKey(key){
 const k=String(key||'').toLowerCase();
 return k==='a'?'accept':k==='p'?'pick':k==='m'?'mark':k==='u'?'undo':null;
}

// --- persistence -----------------------------------------------------------
function saveLocal(){
 const item=currentItem(),group=currentGroup();
 const payload={view,queue,size,density,evidenceOpen,
  group_id:group?Number(group.group_id):null,
  file_id:item?Number(item.file_id):null};
 try{localStorage.setItem(LS_KEY,JSON.stringify(payload))}catch(_e){}
}
function readLocal(){
 try{
  const raw=localStorage.getItem(LS_KEY);
  if(!raw)return null;
  const value=JSON.parse(raw);
  return value&&typeof value==='object'?value:null;
 }catch(_e){return null}
}
function restoreLocal(){
 const saved=readLocal();
 if(!saved)return;
 if(saved.view==='GROUPS'||BROWSE_VIEWS.indexOf(saved.view)>=0)view=saved.view;
 if(QUEUES.indexOf(saved.queue)>=0)queue=saved.queue;
 if([50,100,200].indexOf(Number(saved.size))>=0)size=Number(saved.size);
 if(saved.density==='dense')density='dense';
 evidenceOpen=saved.evidenceOpen===true;
 restoreGroupId=saved.group_id!=null?Number(saved.group_id):null;
 restoreFileId=saved.file_id!=null?Number(saved.file_id):null;
}

// --- current selection -----------------------------------------------------
function currentGroup(){
 return mode()==='queue'?(groups[gIndex]||null):browseGroup;
}
function stageItems(){
 const group=currentGroup();
 return group&&group.members?group.members:[];
}
function currentItem(){
 const items=stageItems();
 if(!items.length)return null;
 mIndex=Math.max(0,Math.min(mIndex,items.length-1));
 return items[mIndex];
}
function humanKeeperId(gid){
 const st=(gid==null?null:reviewState[gid])||{};
 return st.action==='pick'&&st.file_id!=null?Number(st.file_id):null;
}
function aiKeeperId(items){
 const found=(items||[]).find(m=>m.is_keep===true);
 return found?Number(found.file_id):null;
}
function groupStateOf(gid){
 return (gid==null?null:reviewState[gid])||null;
}

// --- shared markup helpers -------------------------------------------------
function badges(r,aiId,humanId,short){
 const fid=Number(r.file_id);let out='';
 if(aiId!=null&&Number(aiId)===fid)out+=`<span class="badge ai">🤖AI${short?'':'推荐'}</span>`;
 if(humanId!=null&&Number(humanId)===fid)out+=`<span class="badge human">👤已选${short?'':'保留'}</span>`;
 return out;
}
function thumbTag(r,attrs){
 const fid=Number(r.file_id);
 return r.thumb==='ok'
  ?`<img loading="lazy" src="/api/thumb/${fid}.jpg" ${attrs||''}`
   +` onerror="this.outerHTML='<div class=miss>缩略图缺失</div>'">`
  :`<div class=miss>缩略图不可用</div>`;
}
function photoTile(r,aiId,humanId){
 const cls=esc(String(r.decision||'ungrouped').toLowerCase());
 const fid=Number(r.file_id);
 return `<div class="card ${cls}" data-file-id="${fid}" tabindex="0"`
  +` title="点击查看大图" onclick="openBrowsePhoto(${fid})">`
  +thumbTag(r,'alt=""')
  +`<div class=cap><span class=tag>${esc(decisionText(r.decision))}</span>`
  +`${badges(r,aiId,humanId)}<br>${esc(r.basename)}<br>`
  +`${r.width||'?'}x${r.height||'?'}<br>${esc(r.exif_datetime||'无拍摄时间')}`
  +`</div></div>`;
}
function tile(r){return photoTile(r,r.is_keep===true?Number(r.file_id):null,null)}
function filmstripHtml(items,currentId,aiId,humanId){
 const list=items||[];
 return list.map((r,i)=>{
  const fid=Number(r.file_id),cur=Number(currentId)===fid;
  const marks=badges(r,aiId,humanId,true)+(cur?'<span class="badge cur">当前</span>':'');
  return `<div class="fs-item${cur?' current':''}" data-file-id="${fid}"`
   +` title="${esc(r.basename||'')}" aria-current="${cur?'true':'false'}"`
   +` onclick="showIndex(${i})">`
   +(r.thumb==='ok'?`<img loading="lazy" src="/api/thumb/${fid}.jpg" alt="">`
     :'<div class="fs-miss">无缩略图</div>')
   +`<div class="fs-meta"><span class="fs-idx">${i+1}/${list.length}</span>${marks}</div></div>`;
 }).join('');
}
// Diagnostics live here and nowhere else: one collapsible panel for the group on
// screen, closed by default, Chinese labels with the internal enum in small type.
function evidenceHtml(group,aiId){
 const members=(group&&group.members)||[];
 if(!members.length)return '<div>本组没有可显示的推荐依据。</div>';
 const rows=members.map(m=>{
  const fid=Number(m.file_id);
  const score=m.quality_score==null?'—':Number(m.quality_score).toFixed(1);
  const faces=m.face_count==null?'—':Number(m.face_count);
  const why=m.reason?`${esc(reasonText(m.reason))}<span class="enum">${esc(m.reason)}</span>`:'—';
  return `<tr><td class="name">${esc(m.basename||'')}`
   +(aiId!=null&&aiId===fid?'<span class="badge ai">🤖AI</span>':'')
   +`</td><td>${score}</td><td>${faces}</td><td>${esc(decisionText(m.decision))}`
   +`<span class="enum">${esc(m.decision||'')}</span></td><td>${why}</td></tr>`;
 }).join('');
 return '<table><tr><th>照片</th><th>清晰度评分</th><th>人脸数</th><th>系统判断</th>'
  +'<th>依据</th></tr>'+rows+'</table>';
}

// --- zoom / pan ------------------------------------------------------------
function clampPan(scale,tx,ty,width,height){
 const limitX=Math.max(0,(scale-1)*width/2),limitY=Math.max(0,(scale-1)*height/2);
 return {tx:Math.max(-limitX,Math.min(limitX,tx)),
         ty:Math.max(-limitY,Math.min(limitY,ty))};
}
function zoomStateText(scale){
 return Number(scale)<=MIN_SCALE+0.001?'适应窗口':`${Math.round(Number(scale)*100)}%`;
}
const zoom={scale:1,tx:0,ty:0};
function paneBox(){
 const frame=$('frameA');
 const rect=frame&&frame.getBoundingClientRect?frame.getBoundingClientRect():null;
 return {width:rect&&rect.width?rect.width:1000,height:rect&&rect.height?rect.height:700};
}
function applyZoom(){
 const box=paneBox();
 const fixed=clampPan(zoom.scale,zoom.tx,zoom.ty,box.width,box.height);
 zoom.tx=fixed.tx;zoom.ty=fixed.ty;
 const css=`translate(${zoom.tx}px,${zoom.ty}px) scale(${zoom.scale})`;
 ['imgA','imgB'].forEach(id=>{const el=$(id);if(el)el.style.transform=css});
 ['frameA','frameB'].forEach(id=>{
  const el=$(id);
  if(el&&el.classList)el.classList.toggle('zoomed',zoom.scale>MIN_SCALE+0.001);
 });
 const label=$('zoomVal');
 if(label)label.textContent=zoomStateText(zoom.scale);
}
function resetZoom(){zoom.scale=1;zoom.tx=0;zoom.ty=0;applyZoom()}
function zoomTo(scale,anchorX,anchorY){
 const next=Math.max(MIN_SCALE,Math.min(MAX_SCALE,scale));
 if(next===MIN_SCALE){resetZoom();return}
 const ratio=next/zoom.scale;
 // Keep the pixel under the cursor put: shift the offset by the scale change.
 zoom.tx=(zoom.tx-(anchorX||0))*ratio+(anchorX||0);
 zoom.ty=(zoom.ty-(anchorY||0))*ratio+(anchorY||0);
 zoom.scale=next;
 applyZoom();
}
function actualPixelScale(){
 const img=$('imgA');
 if(!img||!img.naturalWidth||!img.clientWidth)return 2;
 return Math.max(MIN_SCALE,Math.min(MAX_SCALE,img.naturalWidth/img.clientWidth));
}
// One wheel notch is ~100px of deltaY in a browser but trackpads and Playwright
// send much larger deltas, so the step follows the delta instead of counting
// events. Bounded per event so one violent scroll cannot jump the whole range.
function wheelFactor(deltaY){
 const step=Math.exp(-Number(deltaY||0)*0.0015);
 return Math.max(1/3,Math.min(3,step));
}
function wireStageGestures(){
 ['frameA','frameB'].forEach(id=>{
  const frame=$(id);
  if(!frame||!frame.addEventListener)return;
  frame.addEventListener('wheel',event=>{
   event.preventDefault();
   const rect=frame.getBoundingClientRect();
   const anchorX=event.clientX-(rect.left+rect.width/2);
   const anchorY=event.clientY-(rect.top+rect.height/2);
   zoomTo(zoom.scale*wheelFactor(event.deltaY),anchorX,anchorY);
  },{passive:false});
  frame.addEventListener('dblclick',event=>{event.preventDefault();resetZoom()});
  frame.addEventListener('pointerdown',event=>{
   if(zoom.scale<=MIN_SCALE+0.001)return;
   frame.setPointerCapture&&frame.setPointerCapture(event.pointerId);
   frame.classList.add('grabbing');
   const startX=event.clientX,startY=event.clientY,baseX=zoom.tx,baseY=zoom.ty;
   const move=ev=>{zoom.tx=baseX+(ev.clientX-startX);zoom.ty=baseY+(ev.clientY-startY);
    applyZoom()};
   const up=()=>{frame.classList.remove('grabbing');
    frame.removeEventListener('pointermove',move);
    frame.removeEventListener('pointerup',up);
    frame.removeEventListener('pointercancel',up)};
   frame.addEventListener('pointermove',move);
   frame.addEventListener('pointerup',up);
   frame.addEventListener('pointercancel',up);
  });
 });
}

// --- stage rendering -------------------------------------------------------
function originalSrc(fileId){return `/api/original/${Number(fileId)}`}
function setPane(which,item,labelText,isAi){
 const img=$(which==='A'?'imgA':'imgB'),label=$(which==='A'?'labelA':'labelB');
 const pane=$(which==='A'?'paneA':'paneB');
 if(label)label.innerHTML=labelText;
 if(pane&&pane.classList)pane.classList.toggle('ai',!!isAi);
 if(!img)return;
 if(item==null){img.removeAttribute('src');return}
 const next=originalSrc(item.file_id);
 if(img.getAttribute('src')!==next)img.src=next;
}
function itemLabel(item,aiId,humanId){
 return `${esc(decisionText(item.decision))} · ${esc(item.basename||'')}`
  +badges(item,aiId,humanId)
  +` <span class="enum">${item.width||'?'}×${item.height||'?'}</span>`;
}
function renderStage(){
 const group=currentGroup(),items=stageItems(),item=currentItem();
 const stage=$('stage'),strip=$('strip');
 if(!item){
  setPane('A',null,'');setPane('B',null,'');
  if(strip)strip.innerHTML='';
  return;
 }
 const gid=group?num(group.group_id):null;
 const aiId=aiKeeperId(items),humanId=humanKeeperId(gid);
 const aiItem=aiId!=null?items.find(m=>Number(m.file_id)===aiId):null;
 const showCompare=compareMode&&aiItem!=null&&Number(aiItem.file_id)!==Number(item.file_id);
 const blinkItem=blinkOn&&aiItem!=null?aiItem:null;
 if(stage&&stage.classList)stage.classList.toggle('compare',showCompare);
 const paneB=$('paneB');
 if(paneB&&paneB.classList)paneB.classList.toggle('hide',!showCompare);
 const shown=blinkItem||item;
 setPane('A',shown,(blinkItem?'<span class="badge ai">🤖AI推荐（按住 C）</span> ':'')
  +itemLabel(shown,aiId,humanId),blinkItem!=null);
 if(showCompare)setPane('B',aiItem,'<span class="badge ai">🤖AI推荐</span> '
  +itemLabel(aiItem,aiId,humanId),true);
 else setPane('B',null,'');
 if(strip){
  strip.innerHTML=filmstripHtml(items,item.file_id,aiId,humanId);
  const current=strip.querySelector?strip.querySelector('.fs-item.current'):null;
  if(current&&current.scrollIntoView)
   current.scrollIntoView({block:'nearest',inline:'center',behavior:'smooth'});
 }
 renderHead(group,items,item,aiId);
 applyZoom();
}
function renderHead(group,items,item,aiId){
 const head=$('wbHead'),pos=$('wbPos'),meta=$('wbMeta');
 const gid=group?num(group.group_id):null;
 const state=groupStateOf(gid);
 if(head&&head.classList){
  head.classList.toggle('done',state!=null&&(state.action==='accept'||state.action==='pick'));
  head.classList.toggle('later',state!=null&&state.action==='mark');
 }
 if(pos){
  pos.textContent=mode()==='queue'
   ?`${queueLabel(queue)} ${total?(page-1)*size+gIndex+1:0}/${total}`
   :'查看器';
 }
 if(meta&&group){
  const stateText=state==null?'待审'
   :state.action==='accept'?'已完成 · 保留AI推荐'
   :state.action==='pick'?'已完成 · 保留人工所选'
   :'稍后处理';
  meta.innerHTML=`<span class="gtype">${esc(groupTypeText(group.group_type))}</span>`
   +`<span class="enum">${esc(group.group_type||'')}</span>`
   +` · ${group.member_count||items.length} 张 · 第 ${mIndex+1}/${items.length} 张`
   +` · ${esc(stateText)}`
   +(aiId==null?' · 本组无AI推荐':'');
 }else if(meta){meta.innerHTML=''}
 const evidence=$('evidence');
 if(evidence){
  if(evidence.classList)evidence.classList.toggle('open',evidenceOpen);
  evidence.innerHTML=evidenceOpen?evidenceHtml(group,aiId):'';
 }
 const evidenceBtn=$('evidenceBtn');
 if(evidenceBtn&&evidenceBtn.classList)evidenceBtn.classList.toggle('on',evidenceOpen);
 setBusy(mutationBusy);
}
function showIndex(index){
 const items=stageItems();
 if(!items.length)return;
 mIndex=(index+items.length)%items.length;
 compareMode=false;
 resetZoom();
 renderStage();
 saveLocal();
}
function stepPhoto(delta){showIndex(mIndex+delta)}

// --- chrome ----------------------------------------------------------------
function applyDensity(){
 if(document.body&&document.body.classList)
  document.body.classList.toggle('dense',density==='dense');
 const button=$('densityBtn');
 if(button)button.textContent=density==='dense'?'密度：紧凑':'密度：舒适';
}
function renderTabs(){
 document.querySelectorAll('[data-queue]').forEach(button=>{
  const name=button.dataset.queue;
  button.classList.toggle('on',view==='GROUPS'&&name===queue);
  const slot=button.querySelector?button.querySelector('.n'):null;
  if(slot)slot.textContent=String(queueCounts[name]||0);
 });
 document.querySelectorAll('[data-view]').forEach(button=>{
  button.classList.toggle('on',button.dataset.view===view);
 });
}
function setChrome(){
 const inQueue=mode()==='queue';
 const workbench=$('workbench'),listWrap=$('listWrap'),pager=$('pager');
 const showStage=inQueue?total>0:browseOpen;
 if(workbench&&workbench.classList){
  workbench.classList.toggle('show',showStage);
  workbench.classList.toggle('overlay',!inQueue&&browseOpen);
 }
 if(listWrap&&listWrap.classList)listWrap.classList.toggle('show',!inQueue);
 if(pager&&pager.style)pager.style.display=inQueue?'none':'flex';
 const donePanel=$('donePanel');
 if(donePanel&&donePanel.classList)
  donePanel.classList.toggle('show',inQueue&&total===0);
 ['bAccept','bPick','bMark','bUndo'].forEach(id=>{
  const button=$(id);
  if(button&&button.style)button.style.display=inQueue?'inline-block':'none';
 });
 const bar=$('actionbar');
 if(bar&&bar.style)bar.style.display=showStage?'flex':'none';
 const closer=$('closeStage');
 if(closer&&closer.style)closer.style.display=inQueue?'none':'inline-block';
 $('prev').disabled=page<=1;$('first').disabled=page<=1;
 $('next').disabled=page>=pages;$('last').disabled=page>=pages;
 renderTabs();
}
function updateStats(){
 const stats=$('stats');
 if(!stats)return;
 if(mode()==='queue'){
  stats.textContent=`未审 ${queueCounts.PENDING||0} · 稍后 ${queueCounts.LATER||0}`
   +` · 已完成 ${queueCounts.DONE||0}`
   +(total?` · 当前 ${(page-1)*size+gIndex+1}/${total}`:'');
 }else{
  stats.textContent=`${BROWSE_LABEL[view]||view}：共 ${total} 项 · 第 ${page}/${pages} 页`;
 }
 const doneText=$('doneText');
 if(doneText)doneText.textContent=`已完成 ${queueCounts.DONE||0} 组 · 稍后 ${queueCounts.LATER||0} 组。`;
 const laterBtn=$('goLater');
 if(laterBtn)laterBtn.textContent=`查看稍后（${queueCounts.LATER||0}）`;
 const doneBtn=$('goDone');
 if(doneBtn)doneBtn.textContent=`查看已完成（${queueCounts.DONE||0}）`;
}
// The reviewer's saved decisions could not be read: say so, name the kept file,
// and get out of the way. This is the only non-progress message the page shows --
// algorithm and runtime diagnostics stay out of the UI.
function renderStateWarning(){
 const banner=$('stateWarn');
 if(!banner)return;
 if(!stateWarning){
  banner.hidden=true;banner.innerHTML='';return;
 }
 const kept=/\(saved as ([^)]+)\)/.exec(String(stateWarning));
 banner.innerHTML='<b>审阅状态文件无法读取</b>：本次审阅从空白开始，之前的记录没有丢失。'
  +(kept?`原文件已另存为 <code>${esc(kept[1])}</code>。`:'原文件已保留。')
  +'<br>如需恢复，请退出后检查该文件，或直接继续审阅（新的决定会正常保存）。';
 banner.hidden=false;
}
function setStateWarning(value){
 const next=value||null;
 if(next===stateWarning)return;
 stateWarning=next;
 renderStateWarning();
}
let toastTimer=null;
function showToast(text){
 const toast=$('toast');
 if(!toast)return;
 toast.textContent=text;
 if(toast.classList)toast.classList.add('show');
 if(toastTimer)clearTimeout(toastTimer);
 toastTimer=setTimeout(()=>{if(toast.classList)toast.classList.remove('show')},1900);
}

// --- loading ---------------------------------------------------------------
async function getJson(url){
 const response=await fetch(url);
 if(!response.ok)throw new Error('HTTP '+response.status);
 const value=await response.json();
 if(value&&value.error)throw new Error(value.error);
 return value;
}
async function load(keepIndex){
 const generation=++loadGeneration;
 const savedIndex=gIndex;
 try{
  const query=`view=${view}&page=${page}&page_size=${size}`
   +(mode()==='queue'?`&queue=${queue}`:'');
  const data=await getJson('/api/page?'+query);
  if(generation!==loadGeneration)return;
  page=data.page;pages=data.pages;total=data.total;
  if(mode()==='queue'){
   groups=data.items||[];
   reviewState=data.review_state||{};
   queueCounts=data.queue_counts||queueCounts;
   undoDepth=Number(data.undo_depth||0);
   gIndex=keepIndex?Math.min(savedIndex,Math.max(0,groups.length-1)):0;
   if(!keepIndex)mIndex=0;
   compareMode=false;blinkOn=false;
   resetZoom();
   renderStage();
   if(data.state_warning!==undefined)setStateWarning(data.state_warning);
   else if(stateWarning===null)await refreshStateWarning();
  }else{
   browseItems=data.items||[];
   const content=$('content');
   if(content)content.innerHTML=browseItems.map(tile).join('')
    ||'<div>本视图没有条目。</div>';
   const status=await getJson('/api/status');
   if(generation!==loadGeneration)return;
   queueCounts=status.queues||queueCounts;
   undoDepth=Number(status.undo_depth||0);
   setStateWarning(status.state_warning);
  }
  setChrome();updateStats();saveLocal();
 }catch(error){
  const stats=$('stats');
  if(stats)stats.textContent='分页浏览需要本地服务器（不要加 --no-serve）。'+error.message;
 }
}
async function refreshStateWarning(){
 try{setStateWarning((await getJson('/api/status')).state_warning)}
 catch(_error){}
}
async function goQueue(name){
 if(QUEUES.indexOf(name)<0)return;
 queue=name;view='GROUPS';page=1;gIndex=0;mIndex=0;browseOpen=false;
 await load();
}
async function goView(name){
 view=name;page=1;browseOpen=false;
 if(name==='GROUPS'){gIndex=0;mIndex=0}
 await load();
}
async function focusGroup(groupId,fileId){
 // Ask the server which page of this queue holds the group, then land on it.
 // When the group has left the queue, /api/locate points at the next survivor.
 try{
  const found=await getJson(
   `/api/locate?queue=${queue}&group_id=${Number(groupId)}&page_size=${size}`);
  page=found.page||1;
  await load();
  const wanted=found.group_id!=null?Number(found.group_id):null;
  const at=groups.findIndex(g=>Number(g.group_id)===wanted);
  gIndex=at>=0?at:0;
  mIndex=0;
  if(fileId!=null){
   const items=stageItems();
   const hit=items.findIndex(m=>Number(m.file_id)===Number(fileId));
   if(hit>=0)mIndex=hit;
  }
  renderStage();setChrome();updateStats();saveLocal();
 }catch(_error){await load()}
}

// --- actions ---------------------------------------------------------------
async function postAction(payload){
 const response=await fetch('/api/action',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 const data=await response.json();
 if(!response.ok||data.error)throw new Error(data.error||('HTTP '+response.status));
 return data;
}
// One write at a time. A/P/M/U are single keystrokes on a page that reloads after
// each one, so a fast double-tap (or a held key) would otherwise post the same
// group twice, or walk U back through the whole session in one go.
function setBusy(on){
 mutationBusy=!!on;
 ['bAccept','bPick','bMark','bUndo'].forEach(id=>{
  const button=$(id);
  if(button)button.disabled=mutationBusy||actionDisabled(id);
 });
}
function actionDisabled(id){
 if(id==='bUndo')return undoDepth<=0;
 // "Keep the AI recommendation" is meaningless when this scope has none.
 if(id==='bAccept')return aiKeeperId(stageItems())==null;
 return false;
}
async function decide(action){
 if(mode()!=='queue'||mutationBusy)return;
 const group=currentGroup();
 if(!group){showToast('本队列已经没有待处理的分组');return}
 const gid=Number(group.group_id);
 const item=currentItem();
 if(action==='accept'&&aiKeeperId(stageItems())==null){
  showToast('本组没有 AI 推荐，请用 P 选一张');return;
 }
 const payload={group_id:gid,action};
 // The photo on screen travels with every decision, so undo can come back to
 // it. accept/mark carry no file_id (they choose nothing), which is why this is
 // a separate field the server validates on its own.
 if(item)payload.context_file_id=Number(item.file_id);
 if(action==='pick'){
  if(!item){showToast('请先选择一张照片');return}
  payload.file_id=Number(item.file_id);
 }
 setBusy(true);
 try{
  const data=await postAction(payload);
  queueCounts=data.queues||queueCounts;
  undoDepth=Number(data.undo_depth||0);
  await advanceAfter(gid);
  if(mode()==='queue'&&groups.length===0){showToast('本轮审阅完成');return}
  showToast(action==='accept'?'已保留 AI 推荐，进入下一组'
   :action==='pick'?'已保留当前照片，进入下一组':'已标记稍后处理');
 }catch(error){showToast('操作失败：'+error.message)}
 finally{setBusy(false)}
}
async function advanceAfter(decidedGroupId){
 // Ask the server which group follows the one just decided, in the queue as it
 // now stands. Guessing from the old page index breaks at a page boundary: the
 // queue can lose a whole page, the server clamps the request back, and the UI
 // lands on an earlier group it has already reviewed.
 try{
  const next=await getJson(`/api/next?queue=${queue}`
   +`&after=${Number(decidedGroupId)}&page_size=${size}`);
  queueCounts=next.queue_counts||queueCounts;
  page=next.page||1;
  await load();
  const wanted=next.group_id!=null?Number(next.group_id):null;
  const at=groups.findIndex(g=>Number(g.group_id)===wanted);
  gIndex=at>=0?at:0;
 }catch(_error){
  await load(true);
  gIndex=Math.min(gIndex,Math.max(0,groups.length-1));
 }
 mIndex=0;compareMode=false;blinkOn=false;resetZoom();
 renderStage();setChrome();updateStats();saveLocal();
}
async function undoLast(){
 if(mode()!=='queue'||mutationBusy)return;
 setBusy(true);
 try{
  const data=await postAction({action:'undo'});
  queueCounts=data.queues||queueCounts;
  undoDepth=Number(data.undo_depth||0);
  const undone=data.undo||{};
  if(QUEUES.indexOf(undone.queue)>=0)queue=undone.queue;
  await focusGroup(undone.group_id,undone.focus_file_id);
  showToast(`已撤销上一步，回到分组 #${undone.group_id}`);
 }catch(error){showToast(error.message==='nothing to undo'?'没有可撤销的操作'
  :'撤销失败：'+error.message)}
 finally{setBusy(false)}
}
// --- browse viewer ---------------------------------------------------------
async function openBrowsePhoto(fileId){
 const fid=Number(fileId);
 const chosen=browseItems.find(r=>Number(r.file_id)===fid);
 if(!chosen)return;
 const gid=num(chosen.group_id);
 let members=[];
 if(gid!=null){
  try{const group=await getJson(`/api/group/${gid}`);members=group.members||[]}
  catch(_error){members=[]}
 }
 if(!members.length)members=[chosen];
 browseGroup={group_id:gid,group_type:gid!=null?'':'',member_count:members.length,members};
 mIndex=Math.max(0,members.findIndex(r=>Number(r.file_id)===fid));
 browseOpen=true;compareMode=false;blinkOn=false;resetZoom();
 setChrome();renderStage();updateStats();
}
function closeStage(){
 browseOpen=false;browseGroup=null;compareMode=false;blinkOn=false;
 setPane('A',null,'');setPane('B',null,'');
 const strip=$('strip');if(strip)strip.innerHTML='';
 setChrome();
}

// --- toggles ---------------------------------------------------------------
function toggleCompare(){
 const items=stageItems(),item=currentItem();
 const aiId=aiKeeperId(items);
 if(!compareMode){
  if(aiId==null){showToast('本组没有 AI 推荐，无法对比');return}
  if(item&&Number(item.file_id)===aiId){showToast('当前就是 AI 推荐');return}
 }
 compareMode=!compareMode;
 resetZoom();renderStage();
}
function setBlink(on){
 const aiId=aiKeeperId(stageItems());
 if(on&&aiId==null){showToast('本组没有 AI 推荐');return}
 if(blinkOn===on)return;
 blinkOn=on;
 renderStage();
}
function toggleEvidence(){
 evidenceOpen=!evidenceOpen;
 renderStage();saveLocal();
}
function toggleDensity(){
 density=density==='dense'?'comfy':'dense';
 applyDensity();saveLocal();
}
function openHelp(){$('help').classList.add('open')}
function closeHelp(){$('help').classList.remove('open')}

// --- keyboard --------------------------------------------------------------
document.addEventListener('keydown',async event=>{
 const help=$('help');
 if(help&&help.classList&&help.classList.contains('open')){
  if(event.key==='Escape'||event.key==='?'||event.key==='F1'){event.preventDefault();closeHelp()}
  return;
 }
 if(event.key==='?'||event.key==='F1'){event.preventDefault();openHelp();return}
 if(event.key==='Escape'){
  if(mode()!=='queue'&&browseOpen){event.preventDefault();closeStage()}
  return;
 }
 const key=String(event.key||'');
 if(key==='e'||key==='E'){event.preventDefault();toggleEvidence();return}
 if(key==='d'||key==='D'){event.preventDefault();toggleDensity();return}
 if(key==='f'||key==='F'){event.preventDefault();resetZoom();return}
 if(key==='1'){event.preventDefault();zoomTo(actualPixelScale(),0,0);return}
 if(key==='c'||key==='C'){
  event.preventDefault();
  // Shift+C toggles the persistent split; holding C only blinks to the AI photo.
  if(event.shiftKey)toggleCompare();
  else if(!event.repeat)setBlink(true);
  return;
 }
 if(key==='ArrowLeft'||key==='h'||key==='H'){event.preventDefault();stepPhoto(-1);return}
 if(key==='ArrowRight'||key==='l'||key==='L'){event.preventDefault();stepPhoto(1);return}
 if(mode()!=='queue')return;
 if(key==='ArrowUp'||key==='k'||key==='K'){event.preventDefault();await stepGroup(-1);return}
 if(key==='ArrowDown'||key==='j'||key==='J'){event.preventDefault();await stepGroup(1);return}
 const action=actionKey(key);
 if(!action)return;
 event.preventDefault();
 // A held key must not fire a decision per repeat event, and a second press is
 // ignored while the first is still in flight.
 if(event.repeat||mutationBusy)return;
 if(action==='undo')await undoLast();
 else await decide(action);
});
document.addEventListener('keyup',event=>{
 const key=String(event.key||'');
 if(key==='c'||key==='C')setBlink(false);
});
window.addEventListener('blur',()=>setBlink(false));
async function stepGroup(delta){
 if(mode()!=='queue')return;
 const next=gIndex+delta;
 if(next>=0&&next<groups.length){
  gIndex=next;mIndex=0;compareMode=false;blinkOn=false;resetZoom();
  renderStage();updateStats();saveLocal();
  return;
 }
 if(delta<0&&page>1){page--;await load();gIndex=Math.max(0,groups.length-1);
  renderStage();updateStats();saveLocal();return}
 if(delta>0&&page<pages){page++;await load();gIndex=0;
  renderStage();updateStats();saveLocal();return}
 showToast(delta>0?'已经是本队列最后一组':'已经是本队列第一组');
}

// --- wiring ----------------------------------------------------------------
function wire(){
 document.querySelectorAll('[data-queue]').forEach(button=>{
  button.onclick=()=>goQueue(button.dataset.queue);
 });
 document.querySelectorAll('[data-view]').forEach(button=>{
  button.onclick=()=>goView(button.dataset.view);
 });
 const bind=(id,fn)=>{const el=$(id);if(el)el.onclick=fn};
 bind('helpBtn',openHelp);
 bind('densityBtn',toggleDensity);
 bind('evidenceBtn',toggleEvidence);
 bind('wbPrev',()=>stepGroup(-1));
 bind('wbNext',()=>stepGroup(1));
 bind('bAccept',()=>decide('accept'));
 bind('bPick',()=>decide('pick'));
 bind('bMark',()=>decide('mark'));
 bind('bUndo',undoLast);
 bind('compareBtn',toggleCompare);
 bind('fitBtn',resetZoom);
 bind('closeStage',closeStage);
 bind('goLater',()=>goQueue('LATER'));
 bind('goDone',()=>goQueue('DONE'));
 bind('first',()=>{page=1;load()});
 bind('prev',()=>{if(page>1){page--;load()}});
 bind('next',()=>{if(page<pages){page++;load()}});
 bind('last',()=>{page=pages;load()});
 const sizeSelect=$('size');
 if(sizeSelect)sizeSelect.onchange=event=>{size=Number(event.target.value);page=1;load()};
 wireStageGestures();
}
async function boot(){
 restoreLocal();
 applyDensity();
 const sizeSelect=$('size');
 if(sizeSelect)sizeSelect.value=String(size);
 wire();
 if(mode()==='queue'&&restoreGroupId!=null)await focusGroup(restoreGroupId,restoreFileId);
 else await load();
}
boot();
"""
