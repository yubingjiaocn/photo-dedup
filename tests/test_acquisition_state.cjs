'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {atomicJson,retryOperation,preservePayload}=require('../scripts/research/acquisition_state.cjs');
test('transient failures retry this item and then succeed',async()=>{
 let calls=0;const events=[];const waits=[];
 const value=await retryOperation(async()=>{if(++calls<3)throw Error('temporary');return 7;},
 {onAttempt:e=>events.push(e),wait:async ms=>waits.push(ms)});
 assert.equal(value,7);assert.deepEqual(waits,[2000,4000]);assert.equal(events.length,3);
});
test('exhausted item preserves full failure history',async()=>{
 const events=[];
 await assert.rejects(retryOperation(async()=>{throw Error('temporary with detail');},
 {onAttempt:e=>events.push(e),wait:async()=>{}}));
 assert.equal(events.length,3);assert.match(events[2].error,/temporary with detail/);
});
test('closed owner context stops without repeated browser actions',async()=>{
 let calls=0;
 await assert.rejects(retryOperation(async()=>{calls++;throw Error('Target page, context or browser has been closed');},
 {wait:async()=>{}}));assert.equal(calls,1);
});
test('atomic checkpoint and existing payload never overwrite conflicting bytes',()=>{
 const directory=fs.mkdtempSync(path.join(__dirname,'../.venv/acquisition-test-'));
 try{
  const checkpoint=path.join(directory,'state.json');atomicJson(checkpoint,{state:'running'});atomicJson(checkpoint,{state:'interrupted'});
  assert.equal(JSON.parse(fs.readFileSync(checkpoint)).state,'interrupted');
  const source=path.join(directory,'source'),dest=path.join(directory,'payload');
  fs.writeFileSync(source,'original');preservePayload(source,dest);preservePayload(source,dest);
  fs.writeFileSync(source,'different');assert.throws(()=>preservePayload(source,dest),/PAYLOAD_CONFLICT/);
  assert.equal(fs.readFileSync(dest,'utf8'),'original');
 }finally{fs.rmSync(directory,{recursive:true});}
});
