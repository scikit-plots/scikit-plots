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

class ClassList { constructor(){this.s=new Set();} add(...x){x.forEach(v=>this.s.add(v));} remove(...x){x.forEach(v=>this.s.delete(v));} contains(x){return this.s.has(x);} }
function element(tag){ return {tagName:tag.toUpperCase(),attrs:{},classList:new ClassList(),style:{},children:[],
  setAttribute(k,v){this.attrs[k]=String(v);},getAttribute(k){return this.attrs[k]??null;},hasAttribute(k){return Object.hasOwn(this.attrs,k);},
  appendChild(x){this.children.push(x);return x;},remove(){this.removed=true;},querySelector(){return null;},querySelectorAll(){return [];},
  cloneNode(){return {innerHTML:'<h1>Docs</h1>',textContent:'Docs',querySelectorAll(){return [];}};}}; }

// Host: listener is installed before attachment, iframe URL carries no nonce,
// exact HELLO is consumed and gets one MessagePort.
{
  const listeners={}; const order=[]; let initSent=null; let stopped=false;
  class Port { start(){} postMessage(){} }
  class MC { constructor(){this.port1=new Port();this.port2=new Port();} }
  const frameWin={postMessage(data,origin,ports){initSent={data,origin,ports};}};
  const doc={readyState:'complete',title:'Docs',currentScript:null,
    body:{appendChild(x){order.push('append');return x;}},documentElement:{appendChild(x){order.push('append');return x;}},
    createElement(tag){const e=element(tag);if(tag==='iframe')e.contentWindow=frameWin;return e;},
    querySelector(sel){return sel==='article'?element('article'):null;},dispatchEvent(){return true;},addEventListener(){}};
  const win={AI_ASSISTANT_CONFIG:{isolationOrigin:'https://assistant.example.com',panelTitle:'AI',panelSpeakBanner:true,isolationAllowMicrophone:false},
    AI_ASSISTANT_ENDPOINTS:{},AI_ASSISTANT_ENDPOINT_DEFAULT:'',MessageChannel:MC,URL,URLSearchParams,
    addEventListener(t,f,capture){order.push(`listen:${capture===true}`);listeners[t]=f;},removeEventListener(){},setTimeout(){}};
  const ctx={window:win,document:doc,location:new URL('https://docs.example.com/guide/?q=SECRET#x'),URL,URLSearchParams,MessageChannel:MC,
    CustomEvent:class{},setTimeout(){},console,fetch:async()=>{},TextDecoder,Uint8Array};
  vm.createContext(ctx); vm.runInContext(hostSrc,ctx);
  const iframe=doc.body.lastChild || null;
  const created = order.includes('append');
  ok(created,'host attaches isolated iframe');
  ok(order.indexOf('listen:true') >= 0 && order.indexOf('listen:true') < order.indexOf('append'),'capture listener installed before iframe attachment');
  const hostFrame = (()=>{ // retrieve via closure is impossible; createElement product can be recovered by recording below through source property assertion
    return null;
  })();
  ok(hostSrc.includes("parentOrigin: location.origin") && !hostSrc.slice(hostSrc.indexOf('var fragment'),hostSrc.indexOf('frameUrl.hash')).includes('channel:'),'iframe fragment omits capability nonce');
  ok(hostSrc.includes("allow-scripts allow-same-origin allow-downloads allow-popups") && !hostSrc.includes('allow-popups-to-escape-sandbox'),'sandbox removes popup escape');
  ok(hostSrc.includes("cfg.isolationAllowMicrophone === true"),'microphone delegation requires separate opt-in');
  // We need event.source equality; host keeps frameWin as the iframe contentWindow.
  listeners.message({source:frameWin,origin:'https://assistant.example.com',data:{type:'AI_ASSISTANT_ISOLATION_HELLO',v:'2.0.0',channel:'00112233445566778899aabbccddeeff'},stopImmediatePropagation(){stopped=true;}});
  ok(!!initSent,'valid frame-generated HELLO receives INIT');
  eq(initSent.data.channel,'00112233445566778899aabbccddeeff','host binds transferred port to frame nonce');
  eq(initSent.origin,'https://assistant.example.com','host postMessage target origin is exact');
  ok(stopped,'host consumes valid HELLO before later parent listeners');
  ok(!JSON.stringify(initSent.data).includes('SECRET'),'bootstrap strips parent query/fragment material');
  eq(initSent.data.config.panelSpeakBanner,false,'voice UI hidden when microphone delegation is off');
}

function makeFrameContext({allowed=true, cryptoEnabled=true}={}) {
  const listeners={}; let hello=null; let loadedScript=null; let opened=null; let clickHandler=null;
  const parent={postMessage(d,o){hello={d,o};}};
  const policy=JSON.stringify({schemaVersion:1,protocolVersion:'2.0.0',isolationOrigin:'https://assistant.example.com',allowedParentOrigins:allowed?['https://docs.example.com']:['https://other.example.com']});
  const fetchMock=async()=>({ok:true,status:200,headers:{get(){return String(new TextEncoder().encode(policy).length);}},body:streamText(policy)});
  const doc={body:{textContent:'',appendChild(){}},head:{appendChild(x){loadedScript=x;}},
    createElement(tag){return element(tag);},querySelector(){return null;},getElementById(){return null;},
    addEventListener(t,f,capture){if(t==='click'&&capture===true)clickHandler=f;}};
  const win={fetch:fetchMock,open(url,target,features){opened={url,target,features};return {opener:{}};},
    addEventListener(t,f){listeners[t]=f;},removeEventListener(t,f){if(listeners[t]===f)delete listeners[t];}};
  const cryptoObj=cryptoEnabled?{getRandomValues(a){for(let i=0;i<a.length;i++)a[i]=(i+1)&255;return a;}}:null;
  if(cryptoObj) win.crypto=cryptoObj;
  const ctx={window:win,document:doc,parent,location:{hash:'#v=2.0.0&parentOrigin=https%3A%2F%2Fdocs.example.com',href:'https://assistant.example.com/ai-assistant-isolated.html'},
    URL,URLSearchParams,fetch:fetchMock,setTimeout(){},clearTimeout(){},MutationObserver:class{},console,Uint8Array,TextDecoder,TextEncoder,ReadableStream};
  if(cryptoObj) ctx.crypto=cryptoObj;
  vm.createContext(ctx); vm.runInContext(frameSrc,ctx);
  return {ctx,win,doc,parent,listeners,getHello:()=>hello,getScript:()=>loadedScript,getOpened:()=>opened,getClick:()=>clickHandler};
}

// Frame: policy allowlist + cryptographic nonce + path-scoped storage + navigation guard.
{
  const h=makeFrameContext({allowed:true,cryptoEnabled:true});
  await new Promise(resolve=>setImmediate(resolve));
  const hello=h.getHello();
  ok(!!hello,'allow-listed parent receives HELLO');
  eq(hello.o,'https://docs.example.com','HELLO target origin exact');
  ok(/^[a-f0-9]{32}$/.test(hello.d.channel),'frame creates 128-bit hex nonce');
  const port={start(){},postMessage(){}};
  h.listeners.message({source:h.parent,origin:'https://docs.example.com',data:{type:'AI_ASSISTANT_ISOLATION_INIT',v:'2.0.0',channel:hello.d.channel,
    page:{url:'https://docs.example.com/project/page.html',title:'Page',pageName:'project/page',docsRootUrl:'https://docs.example.com/project/'},
    config:{panelMaxTokens:4096,allowRuntimeTokens:true},endpoints:{},endpointDefault:''},ports:[port]});
  ok(!!h.win.AI_ASSISTANT_ISOLATION_BRIDGE,'valid INIT establishes isolated runtime');
  eq(h.win.AI_ASSISTANT_CONFIG.isolationStorageScope,'host-site:https://docs.example.com|/project/','storage namespace includes docs project root');
  ok(h.getScript() && h.getScript().src==='https://assistant.example.com/ai-assistant.js','main runtime loads only after policy + handshake');
  const click=h.getClick();
  let prevented=false,stopped=false;
  const anchor={hasAttribute(){return false;},getAttribute(k){return k==='href'?'/other/page.html':null;}};
  click({target:{closest(){return anchor;}},preventDefault(){prevented=true;},stopImmediatePropagation(){stopped=true;}});
  const opened=h.getOpened();
  ok(prevented && stopped,'HTTP navigation is intercepted before frame-self navigation');
  eq(opened.url,'https://docs.example.com/other/page.html','relative link resolves against docs page origin');
  eq(opened.target,'_blank','HTTP destination opens outside isolated browsing context');
  eq(opened.features,'noopener,noreferrer','opened navigation severs opener/referrer');
}

// Parent not in generated policy: fail closed before HELLO.
{
  const h=makeFrameContext({allowed:false,cryptoEnabled:true});
  await new Promise(resolve=>setImmediate(resolve));
  eq(h.getHello(),null,'non-allowlisted parent never receives HELLO');
  eq(h.doc.body.textContent,'AI isolation policy/handshake failed closed.','policy denial is visible fail-closed state');
}

// No WebCrypto: fail closed, never downgrade to Math.random.
{
  const h=makeFrameContext({allowed:true,cryptoEnabled:false});
  await new Promise(resolve=>setImmediate(resolve));
  eq(h.getHello(),null,'missing WebCrypto never emits a weak HELLO');
  eq(h.doc.body.textContent,'AI isolation policy/handshake failed closed.','missing secure RNG is fail closed');
}


// Central assistant-service fetch boundary: ambient credentials are omitted,
// caller attempts to request `include` are downgraded, explicit omit is sticky.
{
  const begin=mainSrc.indexOf('    var _fetchTransport =');
  const end=mainSrc.indexOf('    /**\n     * Stable radio-group name', begin);
  ok(begin>=0 && end>begin,'service fetch wrapper source segment found');
  const calls=[];
  const win={AI_ASSISTANT_CONFIG:{},AI_COMPAT:null};
  const ctx={window:win,fetch:async(url,opts)=>{calls.push({url,opts});return {ok:true};},Promise,Object};
  vm.createContext(ctx);
  vm.runInContext("function _cfg(){return window.AI_ASSISTANT_CONFIG||{};}\n"+mainSrc.slice(begin,end),ctx);
  await vm.runInContext("_fetch('https://api.example/a',{method:'POST',credentials:'include'})",ctx);
  eq(calls[0].opts.credentials,'omit','default service fetch strips caller include/ambient credentials');
  win.AI_ASSISTANT_CONFIG.allowCredentialedFetch=true;
  await vm.runInContext("_fetch('https://api.example/b',{method:'GET'})",ctx);
  eq(calls[1].opts.credentials,'same-origin','site-owner compatibility opt-in permits only same-origin credentials');
  await vm.runInContext("_fetch('https://api.example/c',{credentials:'omit'})",ctx);
  eq(calls[2].opts.credentials,'omit','explicit omit remains omit under compatibility opt-in');
  await vm.runInContext("_fetch('https://api.example/d',{credentials:'include'})",ctx);
  eq(calls[3].opts.credentials,'same-origin','include can never escape central wrapper even after opt-in');
}

ok(!hostSrc.includes('Math.random'), 'host has no weak random fallback');
ok(!frameSrc.includes('Math.random'), 'frame has no weak random fallback');
ok(frameSrc.includes("credentials:'omit'"), 'policy fetch carries no ambient credentials');
ok(frameSrc.includes("msg.seq !== rxSeq + 1"), 'frame requires contiguous response sequence');
ok(hostSrc.includes("msg.seq !== _rxSeq + 1"), 'host requires contiguous request sequence');

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
