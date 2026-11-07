// Run 116 — resource lifetime, historical preview and hostile metadata doctor.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0; function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
 for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}

const safeName=extract('_attachmentSafeName');
const safeNameFn=new Function(`${safeName}; return _attachmentSafeName;`)();
ok(safeNameFn('normal-file.pdf')==='normal-file.pdf','ordinary filename unchanged');
ok(safeNameFn('مرحبا-מסמך.pdf').includes('مرحبا-מסמך'),'real RTL-script letters remain intact');
ok(!safeNameFn('invoice\u202egnp.exe').includes('\u202e'),'bidi override control removed from direct filename');
ok(!safeNameFn('x\u2066evil\u2069.txt').includes('\u2066'),'bidi isolate controls removed from direct filename');
ok(!safeNameFn('a\u0000b.txt').includes('\u0000'),'NUL removed from direct filename');

// Execute bounded text reader with a fake FileReader to verify one sparse NUL
// is enough to classify mislabeled binary content, and Clear can abort a read.
const track=extract('_attachmentTrackReader');
const untrack=extract('_attachmentUntrackReader');
const abortReaders=extract('_attachmentAbortActiveReaders');
const readText=extract('_readAttachmentText');
const readerFactory=new Function(`
 var _ATTACHMENT_MAX_READ_BYTES=256*1024;
 var _attachmentActiveReaders=new Set();
 var nextText=''; var hold=false; var lastReader=null;
 class FileReader {
   constructor(){this.result='';this.readyState=0;lastReader=this;}
   readAsText(source){this.readyState=1;if(hold)return;this.result=nextText;this.readyState=2;if(this.onload)this.onload();}
   abort(){this.readyState=2;if(this.onabort)this.onabort();}
 }
 ${track}\n${untrack}\n${abortReaders}\n${readText}
 return {
   read:_readAttachmentText, abort:_attachmentAbortActiveReaders,
   setText(v){nextText=v;hold=false;}, hold(){hold=true;},
   count(){return _attachmentActiveReaders.size;}
 };
`);
const rr=readerFactory();
const fakeFile={size:5000,slice(){return this;}};
rr.setText('A'.repeat(3000)+'\u0000'+'B'.repeat(1000));
let sparseRejected=false; try{await rr.read(fakeFile,4096)}catch(e){sparseRejected=e.message==='ATTACHMENT_BINARY_CONTENT'}
ok(sparseRejected,'single sparse NUL rejects mislabeled binary text');
rr.setText('plain UTF-8 text');
ok(await rr.read(fakeFile,4096)==='plain UTF-8 text','ordinary text still previews');
rr.hold(); const pending=rr.read(fakeFile,4096).then(()=>false,e=>e.message==='ATTACHMENT_READ_ABORTED');
ok(rr.count()===1,'in-flight FileReader is tracked');
rr.abort(); ok(await pending,'clear-time abort rejects pending FileReader with bounded code');
ok(rr.count()===0,'aborted FileReader is removed from active registry');

// Global same-page historical preview-text budget: many 512 KiB previews must
// not accumulate unboundedly across turns.
const runtimeNames=['_turnResourceRuntimeKey','_refreshOpenAttachmentPreviewForRuntime','_releaseTurnResourceRuntimeItem','_releaseTurnResourceRuntimeText','_trimTurnResourceRuntime','_cacheTurnResourceRuntimeText','_releaseTurnResourceRuntime','_releaseAllTurnResourceRuntime','_buildTurnResourceRuntime'];
const runtimeParts=runtimeNames.map(extract).join('\n');
const runtimeFactory=new Function(`
 var _TURN_RESOURCE_LIVE_MAX_ITEMS=512;
 var _TURN_RESOURCE_RUNTIME_MAX_BINARY_ITEMS=12;
 var _TURN_RESOURCE_RUNTIME_MAX_BINARY_BYTES=96*1024*1024;
 var _TURN_RESOURCE_RUNTIME_MAX_FILE_BYTES=64*1024*1024;
 var _TURN_RESOURCE_RUNTIME_MAX_TEXT_CHARS=2*1024*1024;
 var _ATTACHMENT_MAX_PREVIEW_BYTES=512*1024;
 var _turnResourceRuntimeQueue=[]; var _turnResourceRuntimeBytes=0;
 var _turnResourceRuntimeTextQueue=[]; var _turnResourceRuntimeTextChars=0;
 var _transcript=[];
 function _attachmentRevokeObjectUrl(){}
 var _attachmentPreviewState=null; function _attachmentPreviewMeta(){return '';} function _renderAttachmentPreviewBody(){}
 function _turnResourceLiveView(x){return x?Object.assign({},x,{turnScoped:true}):null;}
 ${runtimeParts}
 return {build:_buildTurnResourceRuntime,release:_releaseTurnResourceRuntime,
 stats:()=>({chars:_turnResourceRuntimeTextChars,textCount:_turnResourceRuntimeTextQueue.length})};
`);
const rt=runtimeFactory(); const runtimes=[];
for(let i=0;i<10;i++) runtimes.push(rt.build([{kind:'text',name:'t'+i+'.txt',size:500000,previewText:'x'.repeat(400000)}]));
const rts=rt.stats();
ok(rts.chars<=2*1024*1024,'historical preview strings obey global 2 MiB character budget');
ok(runtimes.some(r=>r.items[0].runtimeTextEvicted===true),'oldest historical text preview is explicitly evicted');
ok(runtimes.filter(r=>typeof r.items[0].previewText==='string').length<=5,'only bounded number of 400k previews remain resident');

const render=extract('_renderAttachmentPreviewBody');
ok(render.includes("item.kind === 'pdf' && item.file && item.pdfVerified !== true"),'metadata-only historical PDF never enters async signature rerender loop');
ok(render.includes("item.kind === 'text' && item.file && typeof item.previewText !== 'string'"),'metadata-only historical text never enters async preview rerender loop');
ok(render.includes('bounded text preview was released'),'text-preview eviction is explained to reader');
ok(render.includes('Local file bytes are intentionally never persisted across reloads.'),'reload metadata-only boundary remains explicit');

const clear=extract('_clearComposerAttachments');
ok(clear.includes('_attachmentAbortActiveReaders()'),'Clear/New chat actively aborts FileReader work');
const exactStream=extract('_zipReadExactStream');
const inflate=extract('_zipInflateBounded');
const extractZip=extract('_extractZipEntry');
const commit=extract('_queueAttachmentImportCommit');
ok(exactStream.includes("reader.cancel('ZIP_CANCELLED')"),'ZIP stream reader is actively cancelled when generation changes');
ok(exactStream.includes("throw new Error('ZIP_CANCELLED')") && inflate.includes("throw new Error('ZIP_CANCELLED')"),'ZIP cancellation has bounded internal code');
ok(extractZip.includes('function assertLive()'),'ZIP extraction checks liveness around async boundaries');
ok(extractZip.includes('_zipInflateBounded(compressedBlob, entry.size, shouldCancel)'),'DEFLATE extraction receives cancellation predicate');
ok(commit.includes('return generation !== _attachmentStageGeneration'),'import commit ties ZIP cancellation to composer generation');

// Minimal dynamic cancellation check before decompression starts.
const inflateFactory=new Function(`
 var _ATTACHMENT_ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES=64*1024*1024;
 class DecompressionStream { constructor(){} }
 ${exactStream}
 ${inflate}
 return _zipInflateBounded;
`);
const inf=inflateFactory();
let zipCancelled=false; try{await inf({stream(){throw new Error('must not stream');}},10,()=>true)}catch(e){zipCancelled=e.message==='ZIP_CANCELLED'}
ok(zipCancelled,'already-invalid ZIP generation exits before touching compressed stream');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
