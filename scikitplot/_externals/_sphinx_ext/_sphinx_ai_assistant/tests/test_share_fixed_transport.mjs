import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { importCfWorkerForNode } from './_import_cf_worker_for_node.mjs';

const testsDir = path.dirname(fileURLToPath(import.meta.url));
const workerPath = path.join(path.dirname(testsDir), '_cf_worker', 'index.js');
const worker = (await importCfWorkerForNode(workerPath, 'run13')).default;
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
const snapshot={schema_version:'2.0',session:{id:'s',page_url:'https://docs.example.test/page?x=1#frag',page_title:'Run13',assistant_name:'AI Assistant',exported_at:1,exported_at_iso:'2026-08-29T00:00:00Z'},records:[{turn_index:0,message_index:0,role:'user',text:'q',ts:1},{turn_index:0,message_index:1,role:'assistant',text:'a',ts:2}]};
const req=(path, body, headers={})=>new Request('https://worker.example'+path,{method:'POST',headers:{'content-type':'application/json',...headers},body:JSON.stringify(body)});

const created=await worker.fetch(req('/v1/share',{snapshot,format:'html',ttlDays:7},{Authorization:'Bearer create-secret'}),env); const c=await created.json();
ok(created.status===200,'worker creates share');
ok(new URL(c.url).pathname==='/v1/share','generated Global link uses fixed viewer path');
ok(new URL(c.url).hash==='#share='+c.uuid,'generated Global link carries read capability in fragment');
ok(!new URL(c.url).pathname.includes(c.uuid),'generated request path omits read capability');

const viewer=await worker.fetch(new Request('https://worker.example/v1/share'),env); const viewerText=await viewer.text();
ok(viewer.status===200 && viewerText.includes('location.hash'),'worker fixed viewer loads fragment client-side');
ok(viewerText.includes("fetch('/v1/share/read'"),'viewer resolves through fixed read path');
ok(viewerText.includes('textContent'),'viewer renders untrusted conversation values without HTML injection');

const status=await worker.fetch(req('/v1/share/status',{shareId:c.uuid}),env);
ok(status.status===200,'fixed status finds active share');
const read=await worker.fetch(req('/v1/share/read',{shareId:c.uuid}),env); const r=await read.json();
ok(read.status===200 && r.format==='html' && r.snapshot.records[1].text==='a','fixed read returns canonical viewer data');

const denied=await worker.fetch(req('/v1/share/update',{shareId:c.uuid,snapshot:{...snapshot,records:[...snapshot.records.slice(0,1),{...snapshot.records[1],text:'b'}]},format:'txt'},{'X-Share-Edit-Token':'wrong'}),env);
ok(denied.status===403,'fixed update requires private edit capability');
const update=await worker.fetch(req('/v1/share/update',{shareId:c.uuid,snapshot:{...snapshot,records:[...snapshot.records.slice(0,1),{...snapshot.records[1],text:'b'}]},format:'txt'},{'X-Share-Edit-Token':c.editToken}),env); const u=await update.json();
ok(update.status===200 && new URL(u.url).hash==='#share='+c.uuid,'fixed update preserves fragment-backed public URL');

const revoke=await worker.fetch(req('/v1/share/revoke',{shareId:c.uuid},{'X-Share-Edit-Token':c.editToken}),env);
ok(revoke.status===200,'fixed revoke deletes share');
const after=await worker.fetch(req('/v1/share/status',{shareId:c.uuid}),env);
ok(after.status===404,'revoked share is unavailable through fixed status');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
