// Run 172 — cancellation ownership and broken-pipe retry discipline.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed=0, failed=0;
function ok(cond,name){ if(cond) passed++; else {failed++; console.error('FAIL '+name);} }
function extract(name) {
  const start=src.indexOf('function '+name+'('); if(start<0) throw new Error('missing '+name);
  let depth=0,began=false,quote='',esc=false,line=false,block=false;
  for(let i=start;i<src.length;i++){const c=src[i],n=src[i+1];
    if(line){if(c==='\n')line=false;continue;} if(block){if(c==='*'&&n==='/'){block=false;i++;}continue;}
    if(quote){if(esc){esc=false;continue;}if(c==='\\'){esc=true;continue;}if(c===quote)quote='';continue;}
    if(c==='/'&&n==='/'){line=true;i++;continue;} if(c==='/'&&n==='*'){block=true;i++;continue;}
    if(c==='"'||c==="'"||c==='`'){quote=c;continue;} if(c==='{'){depth++;began=true;} else if(c==='}'&&--depth===0&&began)return src.slice(start,i+1);
  } throw new Error('unterminated '+name);
}
const abortSrc=extract('_panelTurnAbortError');
const delaySrc=extract('_panelTurnDelay');
const fallbackSrc=extract('_fetchWithReasoningFallback').replace('function _fetchWithReasoningFallback','async function _fetchWithReasoningFallback');
const streamSrc=extract('_panelApiCallStreaming');
const stopSrc=extract('_stopActivePanelResponse');
const submitSrc=extract('handleAIPanelSubmit');

ok(streamSrc.includes("reader.cancel('AI_REQUEST_CANCELLED')")||stopSrc.includes("reader.cancel('AI_REQUEST_CANCELLED')"),'Stop has direct stream-reader cancellation path');
ok(streamSrc.includes("if (streamBubble && streamBubble.parentNode) streamBubble.parentNode.removeChild(streamBubble)"),'cancel/fallback path removes provisional stream bubble');
ok(streamSrc.includes("throw new Error('AI_STREAM_BROKEN_PIPE')"),'empty unexpected stream close has stable broken-pipe error');
ok(streamSrc.includes('The visible partial answer was preserved; no automatic retry was attempted.'),'partial broken pipe preserves visible output without replay');
ok(submitSrc.includes('_panelActiveRequestToken')&&submitSrc.includes('_panelActiveRequestToken.cancelled = true')&&submitSrc.includes("reader.cancel('AI_REQUEST_SUPERSEDED')"),'new submit invalidates predecessor token and pending stream reader');
ok(src.includes('_panelUnlockComposerAfterCancel'),'cancel can release composer before transport shutdown completes');
ok(submitSrc.includes('stillOwnsPanelRequest')&&submitSrc.includes('if (stillOwnsPanelRequest)'),'superseded predecessor finally cannot unlock/focus newer turn UI');

function buildDelayRuntime(){
  const factory=new Function(`
    var _panelActiveRequestToken=null;
    ${abortSrc}
    ${delaySrc}
    return {
      run:function(ms,token){_panelActiveRequestToken=token;return _panelTurnDelay(ms,null,null,token);},
      cancel:function(token){token.cancelled=true;}
    };
  `);
  return factory();
}
const d=buildDelayRuntime();
const token={id:1,cancelled:false};
const started=Date.now();
const pending=d.run(1000,token).then(()=>({ok:true}),e=>({ok:false,name:e&&e.name}));
setTimeout(()=>d.cancel(token),20);
const delayed=await pending;
ok(!delayed.ok&&delayed.name==='AbortError','local delayed work rejects as AbortError when token is cancelled');
ok(Date.now()-started<500,'local delayed cancellation does not wait for original delay');

function buildFallbackRuntime(fetchImpl){
  const factory=new Function('fetchImpl',`
    var _fetch=fetchImpl;
    var opened=0;
    function _openReasoningCircuit(){opened++;}
    ${abortSrc}
    ${fallbackSrc}
    return {run:_fetchWithReasoningFallback,opened:function(){return opened;}};
  `);
  return factory(fetchImpl);
}

let calls=0, active=true;
let rt=buildFallbackRuntime(async()=>{calls++;active=false;throw new TypeError('pipe');});
let cancelledPipe=await rt.run('/x',{body:'with-reasoning'},'base',{},()=>active).then(()=>null,e=>e);
ok(cancelledPipe&&cancelledPipe.name==='AbortError','cancelled broken-pipe primary becomes AbortError');
ok(calls===1,'cancelled broken-pipe primary does not launch fallback request');

calls=0; active=true;
rt=buildFallbackRuntime(async()=>{calls++;active=false;return {ok:false,status:400};});
let cancelled400=await rt.run('/x',{body:'with-reasoning'},'base',{},()=>active).then(()=>null,e=>e);
ok(cancelled400&&cancelled400.name==='AbortError','cancelled 400 response cannot launch schema fallback');
ok(calls===1,'cancelled schema rejection performs no second request');

// Cancellation that happens while the one permitted fallback request is in
// flight must remain AbortError; inner fallback catches may not downgrade it
// back to the original network/HTTP failure.
calls=0; active=true;
rt=buildFallbackRuntime(async()=>{calls++; if(calls===1) throw new TypeError('pipe'); active=false; return {ok:true,status:200};});
let cancelledDuringPipeRetry=await rt.run('/x',{body:'with-reasoning'},'base',{},()=>active).then(()=>null,e=>e);
ok(cancelledDuringPipeRetry&&cancelledDuringPipeRetry.name==='AbortError'&&calls===2,'cancellation during pipe fallback is preserved as AbortError');
ok(rt.opened()===0,'cancelled fallback never opens reasoning circuit');

calls=0; active=true;
rt=buildFallbackRuntime(async()=>{calls++; if(calls===1) return {ok:false,status:400}; active=false; return {ok:true,status:200};});
let cancelledDuringHttpRetry=await rt.run('/x',{body:'with-reasoning'},'base',{},()=>active).then(()=>null,e=>e);
ok(cancelledDuringHttpRetry&&cancelledDuringHttpRetry.name==='AbortError'&&calls===2,'cancellation during HTTP fallback is preserved as AbortError');

calls=0; active=true;
rt=buildFallbackRuntime(async(_url,opt)=>{calls++; if(calls===1) throw new TypeError('pipe'); return {ok:true,status:200,body:opt.body};});
const recovered=await rt.run('/x',{body:'with-reasoning'},'base',{},()=>active);
ok(recovered&&recovered.ok&&calls===2,'active pre-response broken pipe retries exactly once with provider defaults');
ok(rt.opened()===1,'successful fallback opens the in-memory reasoning circuit once');

console.log(`${passed} passed, ${failed} failed`); if(failed) process.exit(1);
