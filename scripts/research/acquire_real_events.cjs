/* Local R&D: existing-session Google Photos official Download only. */
'use strict';
const {chromium}=require('/home/ubuntu/.nvm/versions/node/v24.18.0/lib/node_modules/openclaw/node_modules/playwright-core');
const fs=require('node:fs'); const path=require('node:path');
const {atomicJson,safeError,terminal,retryOperation,preservePayload}=require('./acquisition_state.cjs');
let checkpoint=()=>{};
for(const [signal,code] of [['SIGTERM',143],['SIGINT',130]])process.once(signal,()=>{checkpoint(signal);process.exit(code);});
const OUT='/home/ubuntu/photo-dedup-eval/astra-real-events-20260910/acquisition';
const TARGET=(process.argv.find(a=>a.startsWith('--target='))||'--target=B344106AE822C093792854A484ACAEA8').slice(9);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const stamp=()=>new Date().toISOString();
fs.mkdirSync(OUT,{recursive:true});
function save(name,data){fs.writeFileSync(path.join(OUT,name),JSON.stringify(data,null,2));}
function log(s){console.log(stamp(),s);fs.appendFileSync(path.join(OUT,'run.log'),`${stamp()} ${s}\n`);}
function status(s){fs.writeFileSync(path.join(OUT,'STATUS.md'),`# Acquisition\n\n${stamp()} ${s}\n\nOfficial UI Download only; no cloud writes. Original payloads retained.\n`);}
async function main(){
 const browser=await chromium.connectOverCDP('http://127.0.0.1:18800');
 const ctx=browser.contexts()[0];let page;
 const browserCDP=await browser.newBrowserCDPSession();
 const downloadCache=path.join(OUT,'browser-download-cache');fs.mkdirSync(downloadCache,{recursive:true});
 await browserCDP.send('Browser.setDownloadBehavior',{behavior:'allowAndName',downloadPath:downloadCache,eventsEnabled:true});
 for(const p of ctx.pages()){const c=await ctx.newCDPSession(p);const r=await c.send('Target.getTargetInfo');await c.detach();if(r.targetInfo.targetId===TARGET)page=p;}
 if(!page)throw Error('Owned target missing');
 let retryAfter=0;
 page.on('response',response=>{if(response.status()===429){const value=response.headers()['retry-after'];const seconds=Number(value);retryAfter=Math.max(retryAfter,Number.isFinite(seconds)?Date.now()+Math.max(30,seconds)*1000:(Date.parse(value)||Date.now()+30000));}});
 const pageCDP=await ctx.newCDPSession(page);
 async function officialDownload(trigger,destination){
  await browserCDP.send('Browser.setDownloadBehavior',{behavior:'allowAndName',downloadPath:downloadCache,eventsEnabled:true});
  const {frameTree}=await pageCDP.send('Page.getFrameTree');
  let begin,progress,timer,guid,suggested;
  const completion=new Promise((resolve,reject)=>{
   begin=e=>{if(e.frameId===frameTree.frame.id){guid=e.guid;suggested=e.suggestedFilename;}};
   progress=e=>{if(e.guid!==guid)return;if(e.state==='completed')resolve({guid,suggested});if(e.state==='canceled')reject(Error('Official download canceled'));};
   browserCDP.on('Browser.downloadWillBegin',begin);browserCDP.on('Browser.downloadProgress',progress);
   timer=setTimeout(()=>reject(Error('Official download timed out')),120000);
  });
  try{await trigger();const result=await completion;const name=result.suggested.replace(/[\\/:*?"<>|\x00-\x1f]/g,'_');const dest=typeof destination==='function'?destination(name):destination;
   preservePayload(path.join(downloadCache,result.guid),dest);return {file:dest,suggested:name};
  }finally{clearTimeout(timer);browserCDP.off('Browser.downloadWillBegin',begin);browserCDP.off('Browser.downloadProgress',progress);}
 }
 const day=process.argv[2]||'2026-07-31';
 if(!/^\d{4}-\d{2}-\d{2}$/.test(day))throw Error('Invalid date');
 const [y,m,d]=day.split('-').map(Number);const query=`${y}年${m}月${d}日`;
 const dir=path.join(OUT,day);fs.mkdirSync(dir,{recursive:true});
 const partitionArg=process.argv.find(a=>a.startsWith('--partition='));
 const partition=partitionArg?partitionArg.slice(12).split('/').map(Number):null;
 if(partition&&(!(partition[0]>=0)||!(partition[1]>partition[0])||partition.length!==2))throw Error('Invalid partition');
 const masterFile=path.join(dir,'manifest.json');
 const manifestFile=partition?path.join(dir,`manifest-part-${partition[0]}-of-${partition[1]}.json`):masterFile;
 let manifest=fs.existsSync(manifestFile)?JSON.parse(fs.readFileSync(manifestFile)):(partition&&fs.existsSync(masterFile)?JSON.parse(fs.readFileSync(masterFile)):null);
 if(partition&&!manifest)throw Error('Collect complete master before partitioned downloads');
 const persist=()=>{manifest.partition=partition;atomicJson(manifestFile,manifest);};
 checkpoint=signal=>{if(manifest){manifest.run_state='interrupted';manifest.interrupted_by=signal;manifest.interrupted_at=stamp();persist();log(`${day} checkpoint saved for ${signal}; no automatic restart`);}};
 if(!manifest){
  await page.goto('https://photos.google.com/',{waitUntil:'domcontentloaded',timeout:60000});
  const search=page.getByRole('combobox',{name:'搜索您的照片和相册'});await search.fill(query);await search.press('Enter');
  await page.waitForURL('**/search/**',{timeout:60000});
  await page.getByRole('heading',{name:new RegExp(`${m}月${d}日`)}).first().waitFor({state:'visible',timeout:60000});
  const map=new Map();let stable=0;let lastCount=-1;let rounds=0;
  while(rounds++<250){
   const state=await page.evaluate(()=>{
    const visible=e=>!!(e.getClientRects().length)&&getComputedStyle(e).visibility!=='hidden';
    const els=[...document.querySelectorAll('*')].filter(e=>visible(e)&&e.clientHeight>300&&e.scrollHeight>e.clientHeight+100).sort((a,b)=>(b.scrollHeight-b.clientHeight)-(a.scrollHeight-a.clientHeight));
    const s=els[0];const items=[...document.querySelectorAll('a[href*="/photo/"]')].filter(visible).map(a=>({id:a.href.split('/photo/')[1].split(/[?#]/)[0],label:a.getAttribute('aria-label')}));
    if(!s)return {items,end:true};
    const end=s.scrollTop+s.clientHeight>=s.scrollHeight-5;s.scrollTop=Math.min(s.scrollHeight-s.clientHeight,s.scrollTop+s.clientHeight*0.7);return {items,end};
   });
   for(const item of state.items)map.set(item.id,item);
   stable=state.end&&map.size===lastCount?stable+1:0;lastCount=map.size;
   if(stable>=4)break;await sleep(500);
  }
  manifest={date:day,query,collectedAt:stamp(),collectionReachedEnd:stable>=4,items:[...map.values()].map(x=>({...x,status:'pending'})),bulk:[]};persist();
  log(`${day} collected ${manifest.items.length} IDs; end=${manifest.collectionReachedEnd}`);
  if(!manifest.collectionReachedEnd)throw Error('Collection did not reach stable bottom');
 }
 status(`${day}: ${manifest.items.length} media IDs frozen, acquisition mode=${process.argv.includes('--collect-only')?'metadata-only':process.argv.includes('--individual')?'individual':'whole-date'}.`);
 if(process.argv.includes('--collect-only')){persist();log(`${day} metadata-only count=${manifest.items.length}`);return;}
 if(!manifest.items.length)throw Error('Empty exact-date result');
 if(!process.argv.includes('--individual')){
  if(!process.argv.includes('--selected')){
   await page.goto('https://photos.google.com/',{waitUntil:'domcontentloaded',timeout:60000});
   const search=page.getByRole('combobox',{name:'搜索您的照片和相册'});await search.fill(query);await search.press('Enter');
   const check=page.locator(`[role="checkbox"][aria-label^="选择以下日期的所有照片：${m}月${d}日"] svg`).first();await check.waitFor({state:'visible',timeout:60000});await check.click();
  }
  await sleep(700);
  const selectedText=await page.locator('body').innerText();manifest.selectionText=selectedText.slice(0,600);persist();
  const dest=path.join(dir,'whole-date.zip');
  if(fs.existsSync(dest))throw Error('Bulk payload already exists; inspect before retry');
  try{
   const dl=await officialDownload(()=>page.keyboard.press('Shift+KeyD'),dest);
   manifest.bulk.push({at:stamp(),status:'downloaded',file:dest,bytes:fs.statSync(dest).size,suggested:dl.suggested});persist();log(`${day} bulk saved ${fs.statSync(dest).size} bytes`);status(`${day}: official bulk payload saved; extraction/count verification pending.`);return;
  }catch(e){manifest.bulk.push({at:stamp(),status:'failed',error:e.name||'Error'});persist();log(`${day} bulk failed (${e.name}); individual official Download available on resume.`);throw e;}
 }
 let consecutive=0;const start=Date.now();let completedThisRun=0;
 const secondsArg=process.argv.find(a=>a.startsWith('--max-seconds='));
 const deadline=start+(secondsArg?Number(secondsArg.split('=')[1]):1200)*1000;
 manifest.run_state='running';persist();
 const limitArg=process.argv.find(a=>a.startsWith('--limit='));const limit=limitArg?Number(limitArg.split('=')[1]):Infinity;
 for(const [itemIndex,item] of manifest.items.entries()){
  if(partition&&itemIndex%partition[1]!==partition[0])continue;
  if(completedThisRun>=limit)break;
  if(item.status==='done'&&fs.existsSync(item.file)&&fs.statSync(item.file).size===item.bytes)continue;
  if(Date.now()>=deadline)break;
  manifest.current_item_index=itemIndex;item.status='in_progress';persist();
  try{
   const dl=await retryOperation(async()=>{
   if(Date.now()>=deadline)throw Error('RUN_DEADLINE: checkpoint and resume later');
   if(page.url().startsWith('https://accounts.google.com/'))throw Error('AUTH_REQUIRED: existing login unavailable');
   if(retryAfter>Date.now())await sleep(retryAfter-Date.now());
   if(process.argv.includes('--fast-navigation') && !partition && page.url().includes('/photo/')){
    try{await page.getByRole('button',{name:'查看下一张照片',exact:true}).last().click({timeout:1500});await page.waitForURL(url=>url.pathname.endsWith(`/photo/${item.id}`),{timeout:2000});}
    catch{await page.goto(`https://photos.google.com/photo/${item.id}`,{waitUntil:'domcontentloaded',timeout:60000});}
   }else{await page.goto(`https://photos.google.com/photo/${item.id}`,{waitUntil:'domcontentloaded',timeout:60000});}
   const more=page.getByRole('button',{name:/更多选项|More options/}).last();
   await more.waitFor({state:'visible',timeout:30000});await more.click();
   const downloadItem=page.getByRole('menuitem',{name:/^下载|^Download/}).last();
   await downloadItem.waitFor({state:'visible',timeout:10000});
   return await officialDownload(()=>downloadItem.click(),name=>{item.pending_file=path.join(dir,`${item.id.slice(-10)}-${name}`);persist();return item.pending_file;});
   },{onAttempt:attempt=>{(item.attempts??=[]).push({at:stamp(),...attempt});persist();if(attempt.status==='failed')log(`${day} item=${itemIndex} attempt=${attempt.attempt} ${attempt.error}`);}});
   item.status='done';item.file=dl.file;item.bytes=fs.statSync(dl.file).size;consecutive=0;completedThisRun++;
  }catch(e){item.status='failed';item.error=safeError(e);consecutive++;persist();if(terminal(e)){manifest.run_state='paused_context_or_safety';persist();throw e;}}
  persist();const done=manifest.items.filter(x=>x.status==='done').length;
  if(done%10===0||consecutive) {log(`${day} done=${done}/${manifest.items.length} consecutive_failures=${consecutive}`);status(`${day} done=${done}/${manifest.items.length}, consecutive failures=${consecutive}.`);}
  if(consecutive>=10){manifest.run_state='paused_service_failures';break;}
  await sleep(process.argv.includes('--fast-navigation')?500:1200);
 }
 manifest.partition=partition;
 manifest.complete=manifest.items.every((x,i)=>(partition&&i%partition[1]!==partition[0])||(x.status==='done'&&fs.existsSync(x.file)&&fs.statSync(x.file).size===x.bytes));
 manifest.run_state=manifest.complete?'complete':'incomplete_resumable';persist();status(`${day}: done=${manifest.items.filter(x=>x.status==='done').length}/${manifest.items.length}; complete=${manifest.complete}`);
 return manifest.complete;
}
main().then(complete=>process.exit(complete===false?2:0)).catch(e=>{log(`STOP ${safeError(e)}`);process.exit(1);});
