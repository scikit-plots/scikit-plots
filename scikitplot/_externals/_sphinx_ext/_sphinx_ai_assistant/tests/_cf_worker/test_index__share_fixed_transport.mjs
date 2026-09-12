import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { importCfWorkerForNode } from './_import_cf_worker_for_node.mjs';

const runtimeRoot = process.argv[4] || path.dirname(path.dirname(process.argv[2]));
const workerPath = path.join(runtimeRoot, '_cf_worker', 'index.js');
const worker = (await importCfWorkerForNode(workerPath, 'run13')).default;
let passed=0, failed=0;
function ok(v,n){ if(v){passed++;console.log('PASS',n)}else{failed++;console.log('FAIL',n)} }
function yamlJsonScalar(text,key){
  const prefix=`${JSON.stringify(key)}: `;
  for(const rawLine of String(text||'').split('\n')){
    const line=rawLine.trimStart();
    if(!line.startsWith(prefix))continue;
    try{return JSON.parse(line.slice(prefix.length))}catch(_e){return undefined}
  }
  return undefined;
}

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
const snapshot={schema_version:'2.1',session:{id:'s',page_url:'https://docs.example.test/page?x=1#frag',page_title:'Run13',assistant_name:'AI Assistant',exported_at:1,exported_at_iso:'2026-08-29T00:00:00Z'},records:[{turn_index:0,message_index:0,role:'user',text:'q',ts:1,resources:{version:2,totalCount:1,includedCount:1,localOnlyCount:0,contextCount:1,rawCount:0,notSentCount:0,pageCount:1,replayCount:0,totalBytes:42,itemCount:1,omittedCount:0,complete:true,items:[{name:'Docs',badge:'PAGE',kind:'page',size:42,lineCount:2,included:true,localOnly:false,delivery:'context',modality:'text',intent:'context',replay:false,boundedExcerpt:false,status:'Current page',type:'',relativePath:'',sourceKind:'',archiveName:'',contextRole:'current',sourceUrl:'https://user:pass@docs.example.test/resource/?token=SECRET#frag'}]}},{turn_index:0,message_index:1,role:'assistant',text:'a',ts:2,feedback_rating_value:1,feedback_rating_label:'helpful',feedback_message:'note'}]};
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
ok(viewerText.includes('/v1/share/download')&&viewerText.includes('Download '+""),'viewer exposes capability-safe artifact download control');
ok(viewerText.includes('ai-conversation-global-share-'),'viewer labels downloaded artifacts with Global Share provenance');
ok(viewerText.includes('appendResources')&&viewerText.includes('feedback_rating_label'),'viewer renders resource and feedback metadata from canonical snapshot');
ok(!viewerText.includes('.innerHTML'),'viewer never uses innerHTML for Share content');

const status=await worker.fetch(req('/v1/share/status',{shareId:c.uuid}),env);
ok(status.status===200,'fixed status finds active share');
const read=await worker.fetch(req('/v1/share/read',{shareId:c.uuid}),env); const r=await read.json();
ok(read.status===200 && r.format==='html' && r.snapshot.records[1].text==='a','fixed read returns canonical viewer data');
ok(r.filename==='ai-conversation-global-share-html.html','fixed read names HTML artifact by Global Share provenance');
ok(r.snapshot.schema_version==='2.1','fixed read canonicalizes current schema 2.1');
ok(r.snapshot.records[0].resources.items[0].sourceUrl==='https://docs.example.test/resource/','fixed read retains sanitized resource provenance');
const canonicalResourceUrl=r.snapshot.records[0].resources.items[0].sourceUrl;
ok(r.snapshot.turns[0].user.resources.items[0].name==='Docs','fixed read rebuilds turn resource provenance server-side');

const denied=await worker.fetch(req('/v1/share/update',{shareId:c.uuid,snapshot:{...snapshot,records:[...snapshot.records.slice(0,1),{...snapshot.records[1],text:'b'}]},format:'txt'},{'X-Share-Edit-Token':'wrong'}),env);
ok(denied.status===403,'fixed update requires private edit capability');
const update=await worker.fetch(req('/v1/share/update',{shareId:c.uuid,snapshot:{...snapshot,records:[...snapshot.records.slice(0,1),{...snapshot.records[1],text:'b'}]},format:'txt'},{'X-Share-Edit-Token':c.editToken}),env); const u=await update.json();
ok(update.status===200 && new URL(u.url).hash==='#share='+c.uuid,'fixed update preserves fragment-backed public URL');

const revoke=await worker.fetch(req('/v1/share/revoke',{shareId:c.uuid},{'X-Share-Edit-Token':c.editToken}),env);
ok(revoke.status===200,'fixed revoke deletes share');
const after=await worker.fetch(req('/v1/share/status',{shareId:c.uuid}),env);
ok(after.status===404,'revoked share is unavailable through fixed status');

for(const fmt of ['json','html','txt','yaml','toml']){
  const matrixCreate=await worker.fetch(req('/v1/share',{snapshot,format:fmt,ttlDays:7},{Authorization:'Bearer create-secret'}),env); const mc=await matrixCreate.json();
  ok(matrixCreate.status===200,'worker creates Global '+fmt+' share');
  const matrixRead=await worker.fetch(req('/v1/share/read',{shareId:mc.uuid}),env); const mr=await matrixRead.json();
  ok(matrixRead.status===200&&mr.format===fmt,'worker reads Global '+fmt+' share');
  const expectedName=`ai-conversation-global-share-${fmt}.${fmt}`;
  ok(mr.filename===expectedName,'Global '+fmt+' read exposes provenance filename');
  const matrixDownload=await worker.fetch(req('/v1/share/download',{shareId:mc.uuid}),env); const downloadText=await matrixDownload.text();
  ok(matrixDownload.status===200,'worker downloads Global '+fmt+' share');
  ok(matrixDownload.headers.get('content-disposition')===`attachment; filename="${expectedName}"`,'Global '+fmt+' download uses provenance filename');
  ok(!String(matrixDownload.headers.get('content-disposition')||'').includes(mc.uuid),'Global '+fmt+' filename omits Share capability');
  ok(matrixDownload.headers.get('x-content-type-options')==='nosniff','Global '+fmt+' download is nosniff');
  if(fmt==='html'){
    ok(mr.snapshot.schema_version==='2.1'&&mr.snapshot.records[0].resources.items[0].sourceUrl==='https://docs.example.test/resource/','Global html viewer snapshot preserves canonical 2.1 resources');
    ok(downloadText.includes('<!doctype html>')||downloadText.includes('<!DOCTYPE html>'),'Global html download is trusted rendered HTML');
    ok(String(matrixDownload.headers.get('content-security-policy')||'').startsWith('sandbox;'),'Global html download keeps sandbox CSP header');
  }else{
    ok(!String(mr.content||'').includes('user:pass')&&!String(mr.content||'').includes('token=SECRET'),'Global '+fmt+' output strips private source URL components');
    ok(downloadText===String(mr.content||''),'Global '+fmt+' download bytes match canonical read representation');
    if(fmt==='json')ok(JSON.parse(mr.content).schema_version==='2.1','Global json output is canonical 2.1');
    if(fmt==='txt')ok(mr.content.includes('Schema: 2.1')&&mr.content.includes('Rating: helpful'),'Global txt output includes schema and rating metadata');
    if(fmt==='yaml')ok(yamlJsonScalar(mr.content,'schema_version')==='2.1'&&yamlJsonScalar(mr.content,'sourceUrl')===canonicalResourceUrl,'Global yaml output preserves canonical resource metadata');
    if(fmt==='toml')ok(mr.content.includes('schema_version = \"2.1\"')&&mr.content.includes('[[records.resources.items]]'),'Global toml output preserves canonical resource metadata');
  }
}

const legacy={...snapshot,schema_version:'2.0',records:snapshot.records.map((row,i)=>i===0?{...row,resources:undefined}:{...row})};
const legacyCreate=await worker.fetch(req('/v1/share',{snapshot:legacy,format:'json',ttlDays:7},{Authorization:'Bearer create-secret'}),env); const lc=await legacyCreate.json();
ok(legacyCreate.status===200,'worker accepts schema 2.0 migration input');
const legacyRead=await worker.fetch(req('/v1/share/read',{shareId:lc.uuid}),env); const lr=await legacyRead.json();
ok(legacyRead.status===200,'worker reads schema 2.0 migration input');
const migrated=JSON.parse(lr.content);
ok(migrated.schema_version==='2.1','worker migration output is canonical schema 2.1');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
