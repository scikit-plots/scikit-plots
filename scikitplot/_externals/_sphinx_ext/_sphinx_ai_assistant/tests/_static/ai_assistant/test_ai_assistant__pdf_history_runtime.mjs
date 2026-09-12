// Run 114 — historical previews are memory-only and bounded; PDF never embeds a native iframe.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3],'utf8');
let pass=0,fail=0; function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
 for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}

const render=extract('_renderAttachmentPreviewBody');
const open=extract('_openVerifiedLocalPdf');
const save=extract('_saveTranscript');
const record=extract('_recordMessage');
const replay=extract('_replayTranscript');
const clear=extract('clearConversation');
const manager=extract('_attachmentManagerSourceManifest');

ok(!render.includes("createElement('iframe')"),'PDF preview contains no embedded native iframe');
ok(!src.includes("frame.className = 'ai-assistant-panel-attachment-preview-pdf'"),'old PDF iframe implementation removed globally');
ok(render.includes('Open the verified local file in a separate browser tab, or download it.'),'PDF preview exposes top-level-viewer fallback');
ok(open.includes("a.target = '_blank'") && open.includes("a.rel = 'noopener noreferrer'"),'verified PDF uses one isolated top-level anchor navigation');
ok(open.includes('_ATTACHMENT_PDF_OPEN_URL_TTL_MS'),'new-tab blob URL revocation is delayed/bounded');
ok(src.includes('navigator.pdfViewerEnabled === false'),'disabled native viewer can be explained to user');
ok(css.includes('.ai-assistant-panel-attachment-preview-pdf-safe'),'safe PDF action panel has CSS');
ok(!css.includes('.ai-assistant-panel-attachment-preview-pdf {'),'obsolete iframe sizing CSS removed');

ok(src.includes('var _TURN_RESOURCE_RUNTIME_MAX_BINARY_ITEMS = 12'),'historical binary preview count is bounded');
ok(src.includes('var _TURN_RESOURCE_RUNTIME_MAX_BINARY_BYTES = 96 * 1024 * 1024'),'historical binary preview bytes are bounded');
ok(src.includes('var _TURN_RESOURCE_RUNTIME_MAX_FILE_BYTES = 64 * 1024 * 1024'),'single historical binary retention is bounded');
ok(record.includes('entry.resourceRuntime = _buildTurnResourceRuntime'),'user turn records memory-only runtime capability');
ok(save.includes('delete row.resourceRuntime'),'session persistence strips runtime File/Blob authority');
ok(replay.includes('resourceRuntime: m.resourceRuntime || null'),'same-page transcript replay can reuse runtime previews');
ok(clear.includes('_releaseAllTurnResourceRuntime()'),'new chat releases historical runtime files');
ok(src.includes('_releaseTurnResourceRuntime(oldEntry.resourceRuntime)'),'transcript trimming releases old runtime files');
ok(manager.includes('_turnResourceRenderManifest(st.manifest, st.runtime)'),'historical resource manager overlays runtime preview capability');
ok(src.includes('Local file bytes are intentionally never persisted across reloads.'),'restored history explains metadata-only boundary');
ok(src.includes('local binary preview was released to keep attachment memory bounded'),'runtime eviction is explained rather than silently broken');
ok(!src.includes('var idx = value.lastIndexOf(_ATTACHMENT_CANONICAL_PREFIX);\n        var idx = value.lastIndexOf(_ATTACHMENT_CANONICAL_PREFIX);'),'duplicate canonical attachment index declaration removed');

// Execute the retention queue with fake 10 MiB binary files. The resulting
// authority must obey both count and byte budgets and mark evicted entries.
const names=['_turnResourceRuntimeKey','_refreshOpenAttachmentPreviewForRuntime','_releaseTurnResourceRuntimeItem','_releaseTurnResourceRuntimeText','_trimTurnResourceRuntime','_cacheTurnResourceRuntimeText','_releaseTurnResourceRuntime','_buildTurnResourceRuntime'];
const parts=names.map(extract).join('\n');
const factory=new Function(`
 var _TURN_RESOURCE_LIVE_MAX_ITEMS=512;
 var _TURN_RESOURCE_RUNTIME_MAX_BINARY_ITEMS=12;
 var _TURN_RESOURCE_RUNTIME_MAX_BINARY_BYTES=96*1024*1024;
 var _TURN_RESOURCE_RUNTIME_MAX_FILE_BYTES=64*1024*1024;
 var _TURN_RESOURCE_RUNTIME_MAX_TEXT_CHARS=2*1024*1024;
 var _ATTACHMENT_MAX_PREVIEW_BYTES=512*1024;
 var _turnResourceRuntimeQueue=[];
 var _turnResourceRuntimeBytes=0;
 var _turnResourceRuntimeTextQueue=[];
 var _turnResourceRuntimeTextChars=0;
 function _attachmentRevokeObjectUrl(item){item.objectUrl='';}
 var _attachmentPreviewState=null; function _attachmentPreviewMeta(){return '';} function _renderAttachmentPreviewBody(){}
 function _turnResourceLiveView(item){return item?Object.assign({},item,{turnScoped:true}):null;}
 ${parts}
 return {
   build:_buildTurnResourceRuntime,
   stats:()=>({count:_turnResourceRuntimeQueue.length,bytes:_turnResourceRuntimeBytes,queue:_turnResourceRuntimeQueue}),
   release:_releaseTurnResourceRuntime
 };
`);
const rt=factory();
const ten=10*1024*1024;
const source=Array.from({length:20},(_,i)=>({kind:'pdf',name:'p'+i+'.pdf',size:ten,file:{id:i},type:'application/pdf'}));
const manifest=rt.build(source);
const stats=rt.stats();
ok(manifest.items.length===20,'runtime records metadata for all 20 resources');
ok(stats.count<=12,'runtime queue never exceeds binary item cap');
ok(stats.bytes<=96*1024*1024,'runtime queue never exceeds binary byte cap');
ok(manifest.items.filter(x=>x.file).length===stats.count,'only retained runtime rows keep File authority');
ok(manifest.items.some(x=>x.runtimeEvicted===true),'oldest runtime rows are explicitly marked evicted');
const huge=rt.build([{kind:'pdf',name:'huge.pdf',size:65*1024*1024,file:{id:'huge'},type:'application/pdf'}]);
ok(!huge.items[0].file && huge.items[0].runtimeEvicted===true,'single >64 MiB file is not pinned in historical preview memory');
const before=rt.stats().bytes;
rt.release(manifest);
ok(rt.stats().bytes<before,'releasing a turn reduces retained binary bytes');

// Opening an already verified PDF must call window.open synchronously in the
// same user-gesture stack and schedule delayed URL cleanup.
const openFactory=new Function(`
 var created=0,opened=0,scheduled=0;
 var _ATTACHMENT_PDF_OPEN_URL_TTL_MS=60000;
 var URL={createObjectURL(){created++;return 'blob:test'},revokeObjectURL(){}};
 var anchor={href:'',target:'',rel:'',style:{},click(){opened++;},remove(){}};
 var document={body:{appendChild(){}},createElement(){return anchor;}};
 function setTimeout(fn,ms){scheduled=ms;}
 ${open}
 return {run:_openVerifiedLocalPdf,stats:()=>({created,opened,scheduled})};
`);
const opener=openFactory();
const openResult=opener.run({kind:'pdf',pdfVerified:true,file:{},name:'ok.pdf'});
const os=opener.stats();
ok(openResult===true && os.created===1 && os.opened===1,'verified PDF opens synchronously through top-level viewer');
ok(os.scheduled===60000,'open blob URL cleanup uses configured TTL');
ok(opener.run({kind:'pdf',pdfVerified:false,file:{},name:'bad.pdf'})===false,'unverified PDF cannot enter top-level opener');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
