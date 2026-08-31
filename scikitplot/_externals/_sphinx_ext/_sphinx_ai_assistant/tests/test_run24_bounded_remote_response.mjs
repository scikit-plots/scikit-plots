import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { importCfWorkerForNode } from './_import_cf_worker_for_node.mjs';

const mainPath = process.argv[2];
const root = path.dirname(mainPath);
const mainSrc = fs.readFileSync(mainPath, 'utf8');
const testsDir = path.dirname(fileURLToPath(import.meta.url));
const workerPath = path.join(path.dirname(testsDir), '_cf_worker', 'index.js');
let passed=0, failed=0;
function ok(c,n){if(c)passed++;else{failed++;console.error('FAIL '+n);}}
function eq(a,b,n){if(Object.is(a,b))passed++;else{failed++;console.error(`FAIL ${n} got=${JSON.stringify(a)} want=${JSON.stringify(b)}`);}}

function streamOf(parts){
  return new ReadableStream({start(controller){for(const p of parts)controller.enqueue(new Uint8Array(p));controller.close();}});
}

// Execute the real browser response helpers, not a rewritten test copy.
{
  const begin=mainSrc.indexOf('    // B43: remote response ceilings');
  const end=mainSrc.indexOf('    /**\n     * Stable radio-group name', begin);
  ok(begin>=0&&end>begin,'bounded browser helper source found');
  const ctx={TextDecoder,Uint8Array,JSON,Number,Math,Error};
  vm.createContext(ctx);vm.runInContext(mainSrc.slice(begin,end),ctx);

  const good={headers:{get(){return '7';}},body:streamOf([[123,34,97,34],[58,49,125]])};
  const doc=await vm.runInContext('_readResponseJsonBounded',ctx)(good,1024);
  eq(doc.a,1,'bounded JSON reader parses a valid streamed body');

  const declared={headers:{get(){return '2049';}},body:streamOf([[123,125]])};
  let declaredErr='';try{await vm.runInContext('_readResponseTextBounded',ctx)(declared,2048);}catch(e){declaredErr=e.message;}
  eq(declaredErr,'REMOTE_RESPONSE_TOO_LARGE','declared oversize fails before body read');

  const malformed={headers:{get(){return '12x';}},body:streamOf([[123,125]])};
  let malformedErr='';try{await vm.runInContext('_readResponseTextBounded',ctx)(malformed,2048);}catch(e){malformedErr=e.message;}
  eq(malformedErr,'REMOTE_RESPONSE_INVALID_LENGTH','malformed content-length fails closed');

  const chunked={headers:{get(){return null;}},body:streamOf([new Array(1024).fill(97),new Array(1025).fill(98)])};
  let chunkErr='';try{await vm.runInContext('_readResponseTextBounded',ctx)(chunked,2048);}catch(e){chunkErr=e.message;}
  eq(chunkErr,'REMOTE_RESPONSE_TOO_LARGE','unknown-length body is stopped at streamed byte ceiling');

  // Multibyte payload: encoded bytes exceed the limit while UTF-16 text length stays below it.
  // This independently proves that the byte counter, not only the secondary text-length cap, is active.
  const multibyteBytes=new TextEncoder().encode('😀'.repeat(600)); // 2400 UTF-8 bytes, 1200 UTF-16 code units.
  const multibyte={headers:{get(){return null;}},body:streamOf([multibyteBytes])};
  let multibyteErr='';try{await vm.runInContext('_readResponseTextBounded',ctx)(multibyte,2048);}catch(e){multibyteErr=e.message;}
  eq(multibyteErr,'REMOTE_RESPONSE_TOO_LARGE','streamed byte ceiling is independent of decoded text length');

  const unbounded={headers:{get(){return null;}},body:null,text:async()=>'{"a":1}'};
  let streamErr='';try{await vm.runInContext('_readResponseTextBounded',ctx)(unbounded,2048);}catch(e){streamErr=e.message;}
  eq(streamErr,'REMOTE_RESPONSE_STREAM_UNAVAILABLE','missing stream never falls back to whole-body text');
}

// Execute Worker chat forwarding against declared and chunked oversize bodies.
{
  const mod=await importCfWorkerForNode(workerPath,'run24-b43');
  const worker=mod.default;
  const env={HF_TOKEN:'hf-secret',ALLOWED_MODELS:'Qwen/safe-model',MAX_RESPONSE_BYTES:'1024',SHARE_KV:{async list(){return {keys:[]};},async put(){}}};
  const req=()=>new Request('https://worker.example/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({contract:'scikitplot-chat-v1',model:'Qwen/safe-model',user_message:'q',context:{},stream:false})});
  const realFetch=globalThis.fetch;
  try{
    const health=await worker.fetch(new Request('https://worker.example/health'),env);
    const healthDoc=await health.json();
    eq(healthDoc.limits.max_upstream_response_bytes,1024,'Worker health exposes only the effective response ceiling');
    ok(!JSON.stringify(healthDoc).includes('hf-secret'),'Worker health never exposes provider bearer material');

    globalThis.fetch=async()=>new Response(streamOf([[123,125]]),{status:200,headers:{'content-type':'application/json','content-length':'2048'}});
    const declared=await worker.fetch(req(),env);
    eq(declared.status,502,'Worker rejects declared oversized upstream response before forwarding');

    globalThis.fetch=async()=>new Response(streamOf([new Array(700).fill(97),new Array(700).fill(98)]),{status:200,headers:{'content-type':'application/json'}});
    const chunked=await worker.fetch(req(),env);
    eq(chunked.status,200,'unknown-length Worker response retains upstream status until stream crosses cap');
    let readFailed=false;try{await chunked.arrayBuffer();}catch(_e){readFailed=true;}
    ok(readFailed,'Worker bounded stream fails once unknown-length upstream crosses byte cap');

    globalThis.fetch=async()=>new Response(streamOf([[123,125]]),{status:200,headers:{'content-type':'application/json'}});
    const good=await worker.fetch(req(),env);
    eq(await good.text(),'{}','Worker passes a bounded unknown-length response');
  }finally{globalThis.fetch=realFetch;}
}

ok(mainSrc.includes('return _readResponseTextBounded(response, _CANONICAL_RESPONSE_MAX_BYTES);'),'static Markdown uses bounded stream reader');
ok(mainSrc.includes('resp.ok ? _readResponseJsonBounded(resp, _CONTROL_RESPONSE_MAX_BYTES)'),'dataset discovery uses bounded JSON reader');
ok(!mainSrc.includes('return response.text();'),'main runtime contains no direct response.text fallback');

console.log(`${passed} passed, ${failed} failed`);if(failed)process.exit(1);
