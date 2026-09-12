// Run 111 — complete live resource manifests, compact persistence, large-batch manager, all-export provenance.
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');

function extract(name) {
  const i = src.indexOf('function ' + name + '('); if (i < 0) throw new Error('missing '+name);
  let d=0, started=false, q=null, esc=false, line=false, block=false;
  for (let j=i;j<src.length;j++) {
    const c=src[j], n=src[j+1]||'';
    if (line) { if (c==='\n') line=false; continue; }
    if (block) { if (c==='*'&&n==='/') { block=false; j++; } continue; }
    if (q) { if (esc) esc=false; else if (c==='\\') esc=true; else if (c===q) q=null; continue; }
    if (c==='/'&&n==='/') { line=true; j++; continue; }
    if (c==='/'&&n==='*') { block=true; j++; continue; }
    if (c==='"'||c==="'"||c==='`') { q=c; continue; }
    if (c==='{') { d++; started=true; }
    else if (c==='}') { d--; if (started&&d===0) return src.slice(i,j+1); }
  }
  throw new Error('unbalanced '+name);
}
let pass=0, fail=0;
function ok(cond, name) { if (cond) pass++; else { fail++; console.error('FAIL '+name); } }
function eq(got,want,name) { ok(JSON.stringify(got)===JSON.stringify(want), name + ` (got ${JSON.stringify(got)}, want ${JSON.stringify(want)})`); }

ok(src.includes('var _ATTACHMENT_MAX_FILES = 256'), 'file intake remains 256');
ok(src.includes('var _TURN_RESOURCE_LIVE_MAX_ITEMS = 512'), 'live resource ceiling is independent of file intake');
ok(src.includes('var _TURN_RESOURCE_PERSIST_MAX_ITEMS = 24'), 'persistence row ceiling is 24');
ok(src.includes('var _TURN_RESOURCE_VISIBLE_MAX_ITEMS = 32'), 'visual shelf ceiling is 32');
ok(extract('_saveTranscript').includes('_compactTurnResourceManifest(resourceSource)'), 'session persistence compacts manifests before stringify');
ok(extract('_loadTranscript').includes('_TURN_RESOURCE_PERSIST_MAX_ITEMS'), 'restore revalidates compact manifests');
ok(extract('_renderComposerAttachments').includes('hiddenComposerCount'), 'large composer renders a bounded visible subset');
ok(extract('_renderComposerAttachments').includes("moreName.textContent = '+' + hiddenComposerCount + ' more'"), 'large composer exposes +N more affordance');
ok(extract('_openAttachmentManager').includes("st.mode = manifest ? 'turn' : 'composer'"), 'one manager supports composer and historical-turn manifests');
ok(extract('_attachmentManagerPageRows').includes('item.relativePath') && extract('_attachmentManagerPageRows').includes('item.type'), 'manager search covers path/type metadata');
ok(extract('_ensureAttachmentManagerLayer').includes("['pdf', 'PDF']"), 'manager already has first-class PDF filter slot');
ok(!extract('_renderAttachmentManagerList').includes('_readAttachmentText'), 'opening/filtering manager performs no attachment text read');
ok(css.includes('.ai-assistant-panel-attachment-manager-layer'), 'manager viewport layer styled');
ok(css.includes('.ai-assistant-panel-attachment-batch-summary'), 'large batch summary styled');

// Runtime manifest authority.
globalThis._TURN_RESOURCE_LIVE_MAX_ITEMS = 512;
globalThis._TURN_RESOURCE_PERSIST_MAX_ITEMS = 24;
globalThis._attachmentSafeName = (v) => String(v||'file').replace(/[\u0000-\u001f\u007f]/g,' ').trim().slice(0,240)||'file';
globalThis._attachmentExtension = (v) => { const n=_attachmentSafeName(v); const d=n.lastIndexOf('.'); return d>0?n.slice(d+1).toUpperCase().slice(0,10):'FILE'; };
globalThis._attachmentItemBadge = (item) => item && item.badge ? item.badge : (item && item.kind==='page' ? 'PAGE' : _attachmentExtension(item&&item.name));
globalThis._normalizeContextPageUrl = (v) => String(v||'').slice(0,2048);
for (const name of ['_sanitizeTurnAttachmentSummaries','_safeTurnResourceTotal','_turnResourceAggregates','_buildTurnResourceManifest','_sanitizeTurnResourceManifest','_compactTurnResourceManifest']) {
  globalThis[name]=(0,eval)('('+extract(name)+')');
}
const resources=[];
for (let i=0;i<91;i++) resources.push({name:`doc-${i}.txt`,badge:'TXT',kind:'text',size:100+i,included:true,localOnly:false,status:'Included once'});
for (let i=0;i<9;i++) resources.push({name:`image-${i}.png`,badge:'PNG',kind:'image',size:1000+i,included:false,localOnly:true,status:'Local only · not sent'});
const live=_buildTurnResourceManifest(resources,_TURN_RESOURCE_LIVE_MAX_ITEMS);
eq([live.totalCount,live.includedCount,live.localOnlyCount,live.itemCount,live.omittedCount,live.complete],[100,91,9,100,0,true],'live 100-resource manifest is complete and exact');
const compact=_compactTurnResourceManifest(live);
eq([compact.totalCount,compact.includedCount,compact.localOnlyCount,compact.itemCount,compact.omittedCount,compact.complete],[100,91,9,24,76,false],'compact manifest preserves exact aggregates and admits omitted rows');
ok(compact.totalBytes===live.totalBytes,'compact manifest preserves exact total bytes');
ok(compact.items.every(x=>!('file' in x)&&!('previewText' in x)),'manifest rows are content-free');
const hostile=_sanitizeTurnResourceManifest({totalCount:10,includedCount:10,localOnlyCount:10,items:[],complete:true},24);
ok(hostile.includedCount + hostile.localOnlyCount <= hostile.totalCount,'tampered aggregate counts cannot exceed total resource count');

// Canonical record/turn path carries the same manifest.
globalThis._transcript=[{role:'user',text:'Question',ts:1,resources:live},{role:'assistant',text:'Answer',ts:2,model:null}];
globalThis._feedbackStore={}; globalThis._sessionId='s';
globalThis._buildExportRecords=(0,eval)('('+extract('_buildExportRecords')+')');
globalThis._buildTurnsFromExportRecords=(0,eval)('('+extract('_buildTurnsFromExportRecords')+')');
const records=_buildExportRecords('https://example.test/','s');
ok(records[0].resources && records[0].resources.totalCount===100,'flat JSON record carries resource manifest');
ok(_buildTurnsFromExportRecords(records)[0].user.resources.totalCount===100,'nested turns view carries same resource manifest');

// TXT/YAML/TOML share the canonical manifest rather than reconstructing provenance.
globalThis._resourceManifestSummaryText=(0,eval)('('+extract('_resourceManifestSummaryText')+')');
globalThis._resourceManifestTextLines=(0,eval)('('+extract('_resourceManifestTextLines')+')');
globalThis._buildConversationSnapshot=()=>null;
globalThis._yamlScalar=(0,eval)('('+extract('_yamlScalar')+')');
globalThis._yamlKey=(0,eval)('('+extract('_yamlKey')+')');
globalThis._serializeYamlValue=(0,eval)('('+extract('_serializeYamlValue')+')');
globalThis._buildConvYamlString=(0,eval)('('+extract('_buildConvYamlString')+')');
globalThis._tomlString=(0,eval)('('+extract('_tomlString')+')');
globalThis._tomlScalar=(0,eval)('('+extract('_tomlScalar')+')');
globalThis._tomlWriteFields=(0,eval)('('+extract('_tomlWriteFields')+')');
globalThis._tomlWriteResourceManifest=(0,eval)('('+extract('_tomlWriteResourceManifest')+')');
globalThis._buildConvTomlString=(0,eval)('('+extract('_buildConvTomlString')+')');
globalThis._buildConvTxtString=(0,eval)('('+extract('_buildConvTxtString')+')');
const snap={schema_version:'2.1',session:{assistant_name:'AI',page_url:'https://example.test/',exported_at_iso:'2026-01-01T00:00:00.000Z'},turns:_buildTurnsFromExportRecords(records),records};
const txt=_buildConvTxtString(snap);
ok(txt.includes('[Resources used for this question: 100 · context 91 · raw 0 · not-sent 9]'),'TXT exposes explicit context/raw/not-sent aggregate provenance');
ok(txt.includes('- [TXT] doc-0.txt — Included once'),'TXT exposes per-resource provenance');
const yaml=_buildConvYamlString(snap);
ok(yaml.includes('"resources":') && yaml.includes('"totalCount": 100'),'YAML contains structured resource manifest');
const toml=_buildConvTomlString(snap);
ok(toml.includes('[records.resources]') && toml.includes('[[records.resources.items]]'),'TOML uses real nested resource tables');
ok(toml.includes('[turns.user.resources]') && toml.includes('[[turns.user.resources.items]]'),'TOML nested turns also preserve resources');
const py=spawnSync('python',['-c','import sys,tomllib; tomllib.loads(sys.stdin.read()); print("ok")'],{input:toml,encoding:'utf8'});
ok(py.status===0 && py.stdout.trim()==='ok','generated TOML resource manifest parses with Python tomllib');

// HTML path must render escaped resource cards and embed the same JSON payload.
ok(extract('_buildConvHtmlString').includes('_resourceManifestHtml(r.resources)'), 'HTML serializer consumes canonical record resources');
ok(extract('_resourceManifestHtml').includes('_escapeHtml(_attachmentSafeName(item.name))'), 'HTML resource names are escaped as untrusted data');
ok(extract('_resourceManifestHtml').includes('m.omittedCount'), 'HTML honestly renders restored metadata omission');

console.log(`${pass} passed, ${fail} failed`); if (fail) process.exit(1);
