// Run 121 — closing/clearing import inventories releases rendered row capabilities.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0; function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
 for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}

const clearSrc=extract('_clearAttachmentImportRenderedRows');
const advanceSrc=extract('_attachmentImportAdvance');
const clearJobsSrc=extract('_clearAttachmentImportJobs');

function makeList(n){
 const nodes=Array.from({length:n},(_,i)=>({i}));
 return {
  nodes,
  get firstChild(){return this.nodes[0]||null;},
  removeChild(node){if(this.nodes[0]!==node)throw new Error('wrong node');this.nodes.shift();}
 };
}
const factory=new Function(`
 var focused=0;
 var _attachmentImportState={
  active:null,queue:[],reservations:Object.create(null),queueFullNotified:true,busy:false,trigger:null,page:4,
  list:null,count:null,selected:null,search:null,filter:null,layer:null
 };
 var document={documentElement:{contains(){return false;}}};
 function requestAnimationFrame(fn){fn();}
 function _renderAttachmentImportList(){}
 ${clearSrc}
 ${advanceSrc}
 ${clearJobsSrc}
 return {state:_attachmentImportState,clearRows:_clearAttachmentImportRenderedRows,advance:_attachmentImportAdvance,clearJobs:_clearAttachmentImportJobs};
`);
let h=factory();
h.state.list=makeList(200); h.state.count={textContent:'200 shown'}; h.state.selected={textContent:'3 selected'};
h.clearRows();
ok(h.state.list.nodes.length===0,'explicit row teardown removes all 200 rendered import rows');
ok(h.state.count.textContent==='' && h.state.selected.textContent==='','row teardown clears stale hidden summary text');

h=factory();
h.state.active={entries:[{descriptor:{file:{name:'secret.txt'}}}]};
h.state.list=makeList(200); h.state.count={textContent:'x'}; h.state.selected={textContent:'y'};
h.advance(true);
ok(h.state.active===null && h.state.list.nodes.length===0,'discarding last active inventory clears hidden row DOM immediately');

h=factory();
h.state.active={entries:[]}; h.state.queue=[{entries:[]}]; h.state.list=makeList(200);
h.advance(true);
ok(h.state.active!==null,'discarding with queued work advances to next inventory');
ok(h.state.list.nodes.length===200,'advance does not blank rows before next inventory renderer owns them');

h=factory();
h.state.active={entries:[]}; h.state.queue=[{entries:[]}]; h.state.reservations={'1:zip':true}; h.state.list=makeList(200); h.state.count={textContent:'x'};h.state.selected={textContent:'y'};
h.clearJobs();
ok(h.state.active===null && h.state.queue.length===0 && h.state.list.nodes.length===0,'clear/new chat releases active, queued, and rendered inventory state');
ok(Object.keys(h.state.reservations).length===0 && h.state.queueFullNotified===false,'clear/new chat also invalidates reservations/backpressure notification state');

ok(advanceSrc.includes('_clearAttachmentImportRenderedRows()'),'no-active close branch explicitly tears down row DOM');
ok(clearJobsSrc.includes('_clearAttachmentImportRenderedRows()'),'global clear path explicitly tears down row DOM');
ok(clearSrc.includes('while (st.list.firstChild)'),'teardown removes nodes/listeners rather than merely hiding container');
console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
