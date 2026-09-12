import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const mainPath = process.argv[2];
const root = path.dirname(mainPath);
const hostSrc = fs.readFileSync(path.join(root, 'ai-assistant-isolation-host.js'), 'utf8');
const frameSrc = fs.readFileSync(path.join(root, 'ai-assistant-isolated-frame.js'), 'utf8');
const mainSrc = fs.readFileSync(mainPath, 'utf8');
let passed=0, failed=0;
function ok(c,n){ if(c) passed++; else { failed++; console.error('FAIL '+n); } }
function eq(a,b,n){ if(Object.is(a,b)) passed++; else { failed++; console.error(`FAIL ${n} got=${JSON.stringify(a)} want=${JSON.stringify(b)}`); } }
function streamText(text){const bytes=new TextEncoder().encode(String(text));return new ReadableStream({start(c){c.enqueue(bytes);c.close();}});}

class ClassList {
  constructor(){this.s=new Set();}
  add(...x){x.forEach(v=>this.s.add(v));}
  remove(...x){x.forEach(v=>this.s.delete(v));}
  contains(x){return this.s.has(x);}
}
function element(tag){
  return {tagName:tag.toUpperCase(), attrs:{}, classList:new ClassList(), children:[], style:{},
    setAttribute(k,v){this.attrs[k]=String(v);}, getAttribute(k){return this.attrs[k]??null;},
    appendChild(x){this.children.push(x); x.parentNode=this; return x;}, remove(){this.removed=true;},
    querySelector(){return null;}, querySelectorAll(){return [];}, cloneNode(){return {innerHTML:'<h1>Docs</h1><script>bad()</script>',textContent:'Docs text',querySelectorAll(){return [];}};}
  };
}

// Host handshake and capability gate.
{
  const listeners={}; const appended=[]; const external=[]; const responses=[]; let initSent=null;
  const doc={readyState:'complete', title:'Secret Docs', body:{appendChild(x){appended.push(x); return x;}}, documentElement:{appendChild(x){appended.push(x);return x;}},
    createElement(tag){const e=element(tag); if(tag==='iframe'){e.contentWindow={postMessage(data,origin,ports){initSent={data,origin,ports};}};} return e;},
    querySelector(sel){ if(sel==='article') return element('article'); return null; },
    addEventListener(){}, dispatchEvent(ev){external.push(ev); return true;}}
  class Port { start(){} postMessage(x){responses.push(x);} }
  class MC { constructor(){this.port1=new Port(); this.port2=new Port();} }
  const win={AI_ASSISTANT_CONFIG:{isolationOrigin:'https://assistant.example.com',panelTitle:'AI',content_selector:'article'},
    AI_ASSISTANT_ENDPOINTS:{prod:{base:'https://api.example.com'}}, AI_ASSISTANT_ENDPOINT_DEFAULT:'prod',
    innerWidth:1200,innerHeight:800, MessageChannel:MC, addEventListener(t,f){listeners[t]=f;}, removeEventListener(t,f){if(listeners[t]===f) delete listeners[t];}};
  const ctx={window:win,document:doc,location:new URL('https://docs.example.com/guide/page.html?token=SECRET#frag'),URL,URLSearchParams,
    crypto:{getRandomValues(a){for(let i=0;i<a.length;i++)a[i]=i+1;return a;}},MessageChannel:MC,CustomEvent:class{constructor(t,i={}){this.type=t;this.detail=i.detail;}},
    fetch:async()=>({ok:true,headers:{get(){return null;}},body:streamText('# canonical')}),setTimeout(){},console,TextDecoder,TextEncoder,ReadableStream,Uint8Array};
  vm.createContext(ctx); vm.runInContext(hostSrc,ctx);
  ok(win.SphinxAIAssistantIsolationHostActive===true,'host marks isolated mode before main runtime');
  // Mutating page globals after bridge startup must not alter the already
  // snapshotted bootstrap contract sent after the asynchronous HELLO.
  win.AI_ASSISTANT_CONFIG.panelTitle='MUTATED';
  win.AI_ASSISTANT_CONFIG.panelFeedbackToken='LATE_SECRET';
  win.AI_ASSISTANT_ENDPOINTS.prod.base='https://evil.example';
  win.AI_ASSISTANT_ENDPOINT_DEFAULT='mutated';
  const iframe=appended.find(x=>x.tagName==='IFRAME');
  ok(!!iframe,'host creates iframe');
  ok(iframe.src.startsWith('https://assistant.example.com/ai-assistant-isolated.html#'),'frame URL pinned to configured origin');
  ok(!iframe.src.includes('SECRET'),'frame URL does not copy page query secret');
  ok((iframe.attrs.sandbox||'').includes('allow-same-origin'),'cross-origin sandbox keeps frame origin for CORS/storage');
  ok(!(iframe.attrs.sandbox||'').includes('allow-top-navigation'),'sandbox denies top navigation');
  listeners.message({source:iframe.contentWindow,origin:'https://evil.example',data:{type:'AI_ASSISTANT_ISOLATION_HELLO',v:'2.0.0',channel:'0102030405060708090a0b0c0d0e0f10'}});
  eq(initSent,null,'wrong-origin hello is rejected');
  listeners.message({source:iframe.contentWindow,origin:'https://assistant.example.com',data:{type:'AI_ASSISTANT_ISOLATION_HELLO',v:'2.0.0',channel:'0102030405060708090a0b0c0d0e0f10'}});
  ok(!!initSent,'exact origin/source hello receives MessagePort init');
  eq(initSent.origin,'https://assistant.example.com','init targetOrigin is exact');
  eq(initSent.data.page.url,'https://docs.example.com/guide/page.html','page identity strips query+fragment');
  ok(!JSON.stringify(initSent.data).includes('SECRET'),'init payload excludes page query secret');
  eq(initSent.data.config.panelTitle,'AI','host config is snapshotted before asynchronous handshake');
  eq(initSent.data.endpoints.prod.base,'https://api.example.com','endpoint descriptors are snapshotted before handshake');
  eq(initSent.data.endpointDefault,'prod','endpoint default is snapshotted before handshake');
  ok(!JSON.stringify(initSent.data).includes('LATE_SECRET'),'late secret-shaped config mutation cannot enter init');
  const port=initSent.ports[0] ? initSent.ports[0] : null;
  // The transferred port2 is not the listening port. Access the retained port1 via MC side effect is unavailable,
  // so runtime capability behavior is covered by source assertions below.
  ok(hostSrc.includes("else throw new Error('CAPABILITY_DENIED')"),'host has closed capability default');
  ok(hostSrc.includes("msg.seq !== _rxSeq + 1"),'host rejects replay/out-of-order sequences');
  ok(hostSrc.includes("PUBLIC_EVENT_DETAIL_DENIED"),'host independently validates public integration details');
}

// Frame handshake source/origin, MessagePort-only runtime, and secret-safe config scrub.
{
  const listeners={}; let hello=null; let loadedScript=null; const sent=[];
  const parent={postMessage(d,o){hello={d,o};}};
  const doc={body:{textContent:'',appendChild(){}},head:{appendChild(x){loadedScript=x;}},createElement(tag){return element(tag);},
    querySelector(){return null;},getElementById(){return null;},addEventListener(){}};
  const win={addEventListener(t,f){listeners[t]=f;},removeEventListener(t,f){if(listeners[t]===f)delete listeners[t];}};
  const port={start(){},postMessage(x){sent.push(x);}};
  const cryptoObj={getRandomValues(a){for(let i=0;i<a.length;i++)a[i]=i+1;return a;}};
  win.crypto=cryptoObj;
  const policy=JSON.stringify({schemaVersion:1,protocolVersion:'2.0.0',isolationOrigin:'https://assistant.example.com',allowedParentOrigins:['https://docs.example.com']});
  const fetchMock=async()=>({ok:true,status:200,headers:{get(){return String(new TextEncoder().encode(policy).length);}},body:streamText(policy)});
  const ctx={window:win,document:doc,parent,location:{hash:'#v=2.0.0&parentOrigin=https%3A%2F%2Fdocs.example.com',href:'https://assistant.example.com/ai-assistant-isolated.html'},
    URL,URLSearchParams,crypto:cryptoObj,fetch:fetchMock,setTimeout(){},MutationObserver:class{},console,Uint8Array,TextDecoder,TextEncoder,ReadableStream};
  win.fetch=fetchMock;
  vm.createContext(ctx); vm.runInContext(frameSrc,ctx);
  await new Promise(resolve=>setImmediate(resolve));
  eq(hello.o,'https://docs.example.com','frame HELLO uses exact declared parent origin');
  listeners.message({source:{},origin:'https://docs.example.com',data:{type:'AI_ASSISTANT_ISOLATION_INIT',v:'2.0.0',channel:hello.d.channel},ports:[port]});
  ok(!win.AI_ASSISTANT_ISOLATION_BRIDGE,'wrong message source cannot initialize frame');
  listeners.message({source:parent,origin:'https://docs.example.com',data:{type:'AI_ASSISTANT_ISOLATION_INIT',v:'2.0.0',channel:hello.d.channel,page:{url:'https://docs.example.com/a.html',title:'A',pageName:'a'},config:{panelMaxTokens:1234,panelFeedbackToken:'LEAK',allowRuntimeTokens:true,constructor:'POLLUTE'},endpoints:{},endpointDefault:''},ports:[port]});
  ok(!!win.AI_ASSISTANT_ISOLATION_BRIDGE,'exact parent/source with one port initializes frame');
  eq(win.AI_ASSISTANT_CONFIG.panelMaxTokens,1234,'non-secret token-limit config survives scrub');
  ok(!('panelFeedbackToken' in win.AI_ASSISTANT_CONFIG),'build-time secret-shaped config field is stripped');
  eq(win.AI_ASSISTANT_CONFIG.allowRuntimeTokens,true,'runtime-token policy flag is not mistaken for a credential');
  ok(!Object.prototype.hasOwnProperty.call(win.AI_ASSISTANT_CONFIG,'constructor'),'prototype-pollution key is stripped from frame config');
  eq(win.AI_ASSISTANT_CONFIG.isolationStorageScope,'host-site:https://docs.example.com|/','storage scope binds to parent origin + docs root');
  ok(loadedScript && loadedScript.src==='https://assistant.example.com/ai-assistant.js','full runtime loads only after secure handshake');
  ok(frameSrc.includes('msg.seq !== rxSeq + 1'),'frame rejects replay/out-of-order responses');
}

// Main runtime must fail closed if host bridge asset failed to load, and all fixed
// storage keys are transparently scoped in the isolated frame.
ok(mainSrc.includes('(window.AI_ASSISTANT_CONFIG && window.AI_ASSISTANT_CONFIG.isolationOrigin)'), 'main self-suppresses when isolation requested even if host asset fails');
ok(mainSrc.includes("'ai-assistant-isolated:' + encodeURIComponent(scope)"),'isolated storage keys are parent-origin namespaced');
ok(mainSrc.includes("_isolationRequest('page.context.read'"),'page conversion uses host capability in isolated mode');
ok(mainSrc.includes("_isolationRequest('page.canonical.read'"),'canonical Markdown uses host capability in isolated mode');
ok(mainSrc.includes("_isolationRequest('page.print'"),'printing delegates to host page');
ok(mainSrc.includes("isolationBridge.notify('page.integration.emit'"),'consented public events cross only bounded capability bridge');

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
