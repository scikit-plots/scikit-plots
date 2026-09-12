// Run 122 — Resource Manager pagination + revocable historical preview authority.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0; function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
 for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)
}
const runtimeKeySrc=extract('_turnResourceRuntimeKey');
const refreshSrc=extract('_refreshOpenAttachmentPreviewForRuntime');
const releaseItemSrc=extract('_releaseTurnResourceRuntimeItem');
const releaseTextSrc=extract('_releaseTurnResourceRuntimeText');
const trimSrc=extract('_trimTurnResourceRuntime');
const cacheTextSrc=extract('_cacheTurnResourceRuntimeText');
const bindSrc=extract('_turnResourceBindRuntime');
const filterSrc=extract('_attachmentManagerFilterMatch');
const pageSrc=extract('_attachmentManagerPageRows');
const closeSrc=extract('_closeAttachmentManager');

const runtimeFactory=new Function(`
 var _ATTACHMENT_MAX_PREVIEW_BYTES=512*1024;
 var _TURN_RESOURCE_RUNTIME_MAX_BINARY_ITEMS=12;
 var _TURN_RESOURCE_RUNTIME_MAX_BINARY_BYTES=96*1024*1024;
 var _TURN_RESOURCE_RUNTIME_MAX_TEXT_CHARS=2*1024*1024;
 var _turnResourceRuntimeQueue=[];
 var _turnResourceRuntimeBytes=0;
 var _turnResourceRuntimeTextQueue=[];
 var _turnResourceRuntimeTextChars=0;
 function _attachmentRevokeObjectUrl(item){ if(item) item.objectUrl=''; }
 var _attachmentPreviewState=null; function _attachmentPreviewMeta(){return '';} function _renderAttachmentPreviewBody(){}
 ${runtimeKeySrc}
 ${refreshSrc}
 ${releaseItemSrc}
 ${releaseTextSrc}
 ${trimSrc}
 ${cacheTextSrc}
 ${bindSrc}
 return {
  bind:_turnResourceBindRuntime, release:_releaseTurnResourceRuntimeItem,
  releaseText:_releaseTurnResourceRuntimeText,
  state(){return {binary:_turnResourceRuntimeBytes,text:_turnResourceRuntimeTextChars,queue:_turnResourceRuntimeTextQueue.slice()}}
 };
`);
let h=runtimeFactory();
const file={name:'report.pdf'};
const runtime={key:'pdf\nreport.pdf\n100',file,pdfVerified:true,rasterPreview:false,objectUrl:'blob:old'};
const row={kind:'pdf',name:'report.pdf',size:100,type:'application/pdf'};
h.bind(row,runtime);
ok(row.file===file && row.pdfVerified===true,'render row reads current runtime file/PDF authority');
ok(!Object.keys(row).includes('file') && !Object.keys(row).includes('previewText'),'runtime capabilities are non-enumerable and cannot enter export/persistence by enumeration');
runtime.file=undefined; runtime.runtimeEvicted=true; runtime.objectUrl='';
ok(row.file===undefined && row.runtimeEvicted===true,'already-rendered row observes later runtime eviction instead of retaining copied File');
ok(row.objectUrl==='', 'already-rendered row observes object-URL revocation');

const textRuntime={key:'text\nnotes.txt\n10'};
const textRow={kind:'text',name:'notes.txt',size:10,type:'text/plain'};
h.bind(textRow,textRuntime);
textRow.previewText='late historical preview';
ok(textRuntime.previewText==='late historical preview','late historical preview writes into runtime authority');
ok(h.state().text===23 && h.state().queue.includes(textRuntime),'late historical preview joins global text budget/FIFO');
h.releaseText(textRuntime);
ok(textRow.previewText===undefined && textRow.runtimeTextEvicted===true && h.state().text===0,'text eviction is immediately visible through bound render row');

const managerFactory=new Function(`
 var _attachmentManagerState={page:0,pageSize:100,search:{value:''},filter:{value:'all'}};
 ${filterSrc}
 ${pageSrc}
 return {state:_attachmentManagerState,page:_attachmentManagerPageRows};
`);
let m=managerFactory();
const items=Array.from({length:512},(_,i)=>({name:'file-'+i+'.txt',kind:'text',type:'text/plain',localOnly:false,status:'Included once'}));
let p=m.page({items});
ok(p.rows.length===100 && p.pageCount===6 && p.page===0,'512 resources render as 100-row page, not 512 DOM rows');
m.state.page=5; p=m.page({items});
ok(p.rows.length===12 && p.page===5,'last resource-manager page contains bounded remainder');
m.state.page=999; p=m.page({items});
ok(p.page===5 && p.rows.length===12,'out-of-range manager page clamps safely');
m.state.search.value='file-50'; m.state.page=5; p=m.page({items});
ok(p.matches.length===11 && p.page===0,'search resets/clamps paging against filtered result set');
m.state.search.value=''; m.state.filter.value='pdf'; p=m.page({items});
ok(p.matches.length===0 && p.rows.length===0,'filter supports empty result without page underflow');

function makeList(n){const nodes=Array.from({length:n},(_,i)=>({i}));return{nodes,get firstChild(){return this.nodes[0]||null},removeChild(node){if(node!==this.nodes[0])throw new Error('wrong node');this.nodes.shift()}}}
const closeFactory=new Function(`
 var focused=0;
 var _attachmentManagerState={layer:{hidden:false,setAttribute(){}},trigger:{focus(){focused++}},mode:'turn',manifest:{},runtime:{},page:4,list:null,count:{textContent:'100 shown'},omitted:{hidden:false,textContent:'omitted'}};
 ${closeSrc}
 return {state:_attachmentManagerState,close:_closeAttachmentManager,focused:()=>focused};
`);
const c=closeFactory(); c.state.list=makeList(100); c.close(true);
ok(c.state.list.nodes.length===0,'closing manager destroys rendered row/listener DOM immediately');
ok(c.state.runtime===null && c.state.manifest===null && c.state.page===0,'closing manager releases runtime/manifest references and paging state');
ok(c.focused()===1,'manager close restores focus to opener');

ok(src.includes("pageSize: 100"),'resource manager declares bounded page size');
ok(src.includes("_attachmentManagerPageRows(manifest)"),'resource manager renderer uses paging helper');
ok(src.includes("Object.defineProperty(row, 'file'")===false,'file authority is not defined as an independent copied value');
ok(bindSrc.includes("live('file', undefined)"),'file capability is a live runtime accessor');
ok(bindSrc.includes("live('objectUrl', '')"),'blob URL authority is also bound to evictable runtime state');
ok(closeSrc.includes('while (st.list.firstChild)'),'manager teardown removes DOM rather than only hiding layer');
console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
