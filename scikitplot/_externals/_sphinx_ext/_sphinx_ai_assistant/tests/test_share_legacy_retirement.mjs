import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { importCfWorkerForNode } from './_import_cf_worker_for_node.mjs';

const testsDir = path.dirname(fileURLToPath(import.meta.url));
const workerPath = path.join(path.dirname(testsDir), '_cf_worker', 'index.js');
const worker = (await importCfWorkerForNode(workerPath, 'run14')).default;
let passed=0, failed=0;
function ok(v,n){ if(v){passed++;console.log('PASS',n)}else{failed++;console.log('FAIL',n)} }

const store = new Map();
const env = {
  ALLOWED_ORIGINS: 'https://docs.example.test',
  SHARE_WRITE_TOKEN: 'create-secret',
  SHARE_KV: {
    async get(k){ return store.has(k) ? store.get(k).value : null; },
    async put(k,v,opts={}){ store.set(k,{value:String(v),opts}); },
    async delete(k){ store.delete(k); },
    async list({prefix='' }={}){
      return { keys:[...store.entries()].filter(([k])=>k.startsWith(prefix)).map(([name,row])=>({name,metadata:row.opts?.metadata||{}})), list_complete:true };
    },
  },
};
const snapshot={schema_version:'2.0',session:{id:'s14',page_url:'https://docs.example.test/page',page_title:'Run14',assistant_name:'AI Assistant',exported_at:1,exported_at_iso:'2026-08-29T00:00:00Z'},records:[{turn_index:0,message_index:0,role:'user',text:'q',ts:1},{turn_index:0,message_index:1,role:'assistant',text:'a',ts:2}]};
const post=(p,body,headers={})=>new Request('https://worker.example'+p,{method:'POST',headers:{'content-type':'application/json',...headers},body:JSON.stringify(body)});
const legacyReq=(method,id,body,headers={})=>new Request('https://worker.example/v1/share/'+id,{method,headers:{...(body?{'content-type':'application/json'}:{}),...headers},...(body?{body:JSON.stringify(body)}:{})});

const created=await worker.fetch(post('/v1/share',{snapshot,format:'html',ttlDays:7},{Authorization:'Bearer create-secret'}),env); const c=await created.json();
const newRaw=JSON.parse((await env.SHARE_KV.get('sh:'+c.uuid)));
ok(newRaw.transportVersion===2,'new Worker share is transport generation 2');
ok((await worker.fetch(legacyReq('HEAD',c.uuid),env)).status===404,'generation2 HEAD legacy path is rejected');
ok((await worker.fetch(legacyReq('GET',c.uuid),env)).status===404,'generation2 GET legacy path is rejected');
ok((await worker.fetch(legacyReq('DELETE',c.uuid,null,{'X-Share-Edit-Token':c.editToken}),env)).status===404,'generation2 DELETE legacy path is rejected');

// Model a still-live pre-Run-14 KV object by removing the generation marker.
delete newRaw.transportVersion;
await env.SHARE_KV.put('sh:'+c.uuid,JSON.stringify(newRaw),{metadata:{bytes:newRaw.bytes,format:newRaw.format}});
const oldHead=await worker.fetch(legacyReq('HEAD',c.uuid),env);
ok(oldHead.status===200 && oldHead.headers.get('Deprecation')==='@1787961600','legacy HEAD remains bounded-compatible and deprecated');
ok((oldHead.headers.get('Link')||'').includes('successor-version'),'legacy HEAD advertises fixed successor');
ok(!!oldHead.headers.get('Sunset'),'legacy HEAD advertises object expiry sunset');
const oldGet=await worker.fetch(legacyReq('GET',c.uuid),env);
ok(oldGet.status===200 && oldGet.headers.get('Deprecation')==='@1787961600','legacy GET remains bounded-compatible and deprecated');

const retiredPatch=await worker.fetch(legacyReq('PATCH',c.uuid,{snapshot,format:'txt',ttlDays:365},{'X-Share-Edit-Token':c.editToken}),env); const rp=await retiredPatch.json();
ok(retiredPatch.status===410 && retiredPatch.headers.get('Deprecation')==='@1787961600','legacy PATCH is retired instead of extending path-capability lifetime');
ok(String(rp.error||'').includes('POST /v1/share/update'),'legacy PATCH directs clients to fixed update path');
ok((await worker.fetch(legacyReq('GET',c.uuid),env)).status===200,'retired PATCH leaves old object unchanged and readable until TTL');

const migratedSnap={...snapshot,records:[snapshot.records[0],{...snapshot.records[1],text:'migrated'}]};
const migrated=await worker.fetch(post('/v1/share/update',{shareId:c.uuid,snapshot:migratedSnap,format:'txt',ttlDays:5},{'X-Share-Edit-Token':c.editToken}),env); const m=await migrated.json();
ok(migrated.status===200 && new URL(m.url).hash==='#share='+c.uuid,'fixed update migrates old object to fragment successor');
const migratedRaw=JSON.parse(await env.SHARE_KV.get('sh:'+c.uuid));
ok(migratedRaw.transportVersion===2,'fixed update is one-way migration to generation2');
ok((await worker.fetch(legacyReq('GET',c.uuid),env)).status===404,'migrated object cannot fall back to legacy GET');
const fixedRead=await worker.fetch(post('/v1/share/read',{shareId:c.uuid}),env); const fr=await fixedRead.json();
ok(fixedRead.status===200 && String(fr.content||'').includes('migrated'),'migrated object remains readable through fixed path');

// A separate pre-generation object remains revocable during the compatibility window.
const c2r=await worker.fetch(post('/v1/share',{snapshot,format:'json',ttlDays:7},{Authorization:'Bearer create-secret'}),env); const c2=await c2r.json();
const old2=JSON.parse(await env.SHARE_KV.get('sh:'+c2.uuid)); delete old2.transportVersion; await env.SHARE_KV.put('sh:'+c2.uuid,JSON.stringify(old2),{metadata:{bytes:old2.bytes,format:old2.format}});
const revoked=await worker.fetch(legacyReq('DELETE',c2.uuid,null,{'X-Share-Edit-Token':c2.editToken}),env);
ok(revoked.status===200 && revoked.headers.get('Deprecation')==='@1787961600','pre-generation object can still be revoked via deprecated path');
ok((await env.SHARE_KV.get('sh:'+c2.uuid))===null,'legacy revoke removes old KV object');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
