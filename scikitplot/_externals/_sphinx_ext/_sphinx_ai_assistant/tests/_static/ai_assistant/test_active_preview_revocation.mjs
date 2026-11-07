// Run 123 — an already-open historical preview must reflect runtime eviction immediately.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0;function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}
const keySrc=extract('_turnResourceRuntimeKey');
const refreshSrc=extract('_refreshOpenAttachmentPreviewForRuntime');
const releaseItemSrc=extract('_releaseTurnResourceRuntimeItem');
const releaseTextSrc=extract('_releaseTurnResourceRuntimeText');
const trimSrc=extract('_trimTurnResourceRuntime');
const cacheSrc=extract('_cacheTurnResourceRuntimeText');
const bindSrc=extract('_turnResourceBindRuntime');
const factory=new Function(`
 var _ATTACHMENT_MAX_PREVIEW_BYTES=512*1024;
 var _TURN_RESOURCE_RUNTIME_MAX_BINARY_ITEMS=12;
 var _TURN_RESOURCE_RUNTIME_MAX_BINARY_BYTES=96*1024*1024;
 var _TURN_RESOURCE_RUNTIME_MAX_TEXT_CHARS=2*1024*1024;
 var _turnResourceRuntimeQueue=[];var _turnResourceRuntimeBytes=0;
 var _turnResourceRuntimeTextQueue=[];var _turnResourceRuntimeTextChars=0;
 var renders=0,last={};
 var _attachmentPreviewState={item:null,layer:{hidden:false},meta:{textContent:''}};
 function _attachmentPreviewMeta(item){return item.file?'bytes present':'metadata only';}
 function _renderAttachmentPreviewBody(item){renders++;last={file:!!item.file,text:typeof item.previewText==='string',evicted:item.runtimeEvicted===true,textEvicted:item.runtimeTextEvicted===true};}
 function _attachmentRevokeObjectUrl(item){if(item)item.objectUrl='';}
 ${keySrc}
 ${refreshSrc}
 ${releaseItemSrc}
 ${releaseTextSrc}
 ${trimSrc}
 ${cacheSrc}
 ${bindSrc}
 return {bind:_turnResourceBindRuntime,release:_releaseTurnResourceRuntimeItem,releaseText:_releaseTurnResourceRuntimeText,st:_attachmentPreviewState,renders:()=>renders,last:()=>last};
`);
const h=factory();
const runtime={key:'pdf\nreport.pdf\n100',file:{name:'report.pdf'},_runtimeBytes:100,objectUrl:'blob:x',pdfVerified:true};
const row={kind:'pdf',name:'report.pdf',size:100};h.bind(row,runtime);h.st.item=row;
h.release(runtime);
ok(h.renders()===1,'binary eviction rerenders an already-open matching preview');
ok(h.last().file===false && h.last().evicted===true,'rerender sees revoked PDF/file authority');
ok(h.st.meta.textContent==='metadata only','open preview metadata updates after eviction');
ok(row.objectUrl==='' && row.file===undefined,'open row observes object URL and File revocation');
ok(!Object.keys(row).includes('_runtimeAuthority'),'runtime authority link is non-enumerable');

const tr={key:'text\nnotes.txt\n10',previewText:'hello',_runtimeTextChars:5};
const textRow={kind:'text',name:'notes.txt',size:10};h.bind(textRow,tr);h.st.item=textRow;
const before=h.renders();h.releaseText(tr);
ok(h.renders()===before+1,'text eviction rerenders an already-open matching preview');
ok(h.last().text===false && h.last().textEvicted===true,'rendered bounded text is released when FIFO evicts it');

const other={key:'pdf\nother.pdf\n1',file:{},_runtimeBytes:1};
const beforeOther=h.renders();h.release(other);
ok(h.renders()===beforeOther,'evicting unrelated runtime does not disturb open preview');

h.st.layer.hidden=true;
const again={key:'pdf\nreport2.pdf\n2',file:{},_runtimeBytes:2};const row2={kind:'pdf',name:'report2.pdf',size:2};h.bind(row2,again);h.st.item=row2;const beforeHidden=h.renders();h.release(again);
ok(h.renders()===beforeHidden,'hidden preview layer is not needlessly rerendered');

ok(refreshSrc.includes('st.item._runtimeAuthority !== runtime'),'refresh is identity-scoped to exact runtime capability');
ok(releaseItemSrc.includes('_refreshOpenAttachmentPreviewForRuntime(item)'),'binary release actively refreshes open preview');
ok(releaseTextSrc.includes('_refreshOpenAttachmentPreviewForRuntime(item)'),'text release actively refreshes open preview');
console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
