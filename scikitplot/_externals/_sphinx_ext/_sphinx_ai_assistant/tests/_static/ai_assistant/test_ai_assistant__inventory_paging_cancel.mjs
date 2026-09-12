// Run 119 — huge import inventories stay DOM-bounded and stop after invalidation.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0; function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
 for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}

const visibleSrc=extract('_attachmentImportVisibleEntries');
const pageSrc=extract('_attachmentImportPageRows');
globalThis._attachmentImportState={
 active:{entries:Array.from({length:4096},(_,i)=>({name:`f${i}.txt`,relativePath:`root/f${i}.txt`,type:'text/plain',kind:'text',reason:'',selectable:true,rejected:false}))},
 search:null,filter:null,page:0,pageSize:200
};
const visible=new Function(`${visibleSrc}; return _attachmentImportVisibleEntries;`)();
globalThis._attachmentImportVisibleEntries=visible;
const pageRows=new Function(`${pageSrc}; return _attachmentImportPageRows;`)();
let p0=pageRows();
ok(p0.matches.length===4096,'all inventory metadata remains searchable');
ok(p0.rows.length===200,'first page renders only fixed 200 rows');
ok(p0.pageCount===21,'4096 entries produce bounded 21-page inventory');
ok(p0.rows[199].name==='f199.txt','first page boundary is deterministic');
_attachmentImportState.page=1; let p1=pageRows();
ok(p1.rows[0].name==='f200.txt' && p1.rows[199].name==='f399.txt','next page advances by exact page size');
_attachmentImportState.page=20; let plast=pageRows();
ok(plast.rows.length===96 && plast.rows[95].name==='f4095.txt','last page contains bounded remainder');
_attachmentImportState.page=999; let pclamp=pageRows();
ok(pclamp.page===20,'out-of-range page is clamped safely');
_attachmentImportState.search={value:'f4095'}; _attachmentImportState.page=0; let searched=pageRows();
ok(searched.matches.length===1 && searched.rows[0].name==='f4095.txt','search still reaches entries outside first page');

const dirFactory=new Function(`
 var _ATTACHMENT_IMPORT_MAX_ENTRIES=4096,_ATTACHMENT_IMPORT_MAX_PATH_CHARS=1024,_ATTACHMENT_DIRECTORY_MAX_DEPTH=24;
 ${extract('_attachmentSafeName')}
 ${extract('_attachmentSafeRelativePath')}
 ${extract('_directoryEntryFile')}
 ${extract('_directoryReadBatch')}
 ${extract('_inventoryDirectoryEntries')}
 return _inventoryDirectoryEntries;
`);
const inventoryDir=dirFactory();
let reads=0, fileCalls=0, checks=0;
const child={isFile:true,isDirectory:false,name:'x.txt',file(resolve){fileCalls++;resolve({name:'x.txt',size:1,type:'text/plain'})}};
const root={isDirectory:true,name:'root',createReader(){return {readEntries(resolve){reads++;resolve([child])}}}};
let dirCancelled=false;
try{await inventoryDir([root],()=>++checks>=4)}catch(e){dirCancelled=e.message==='DIRECTORY_CANCELLED'}
ok(dirCancelled,'directory generation cancellation propagates as bounded code');
ok(reads===1,'directory cancellation stops after current readEntries batch');
ok(fileCalls===0,'directory cancellation before batch processing avoids File handle fan-out');

const inspectSrc=extract('_inspectZipArchive');
const inspect=new Function(`${inspectSrc}; return _inspectZipArchive;`)();
let slices=0, zipCancelled=false;
try{await inspect({name:'x.zip',type:'application/zip',size:100,slice(){slices++;throw new Error('must not read')}},()=>true)}catch(e){zipCancelled=e.message==='ZIP_CANCELLED'}
ok(zipCancelled,'ZIP inventory cancellation exits before parser work');
ok(slices===0,'already-invalid ZIP inventory reads zero bytes');

const render=extract('_renderAttachmentImportList');
const ensure=extract('_ensureAttachmentImportLayer');
const queueZip=extract('_queueZipImport');
const queueDir=extract('_queueDirectoryEntryImport');
ok(render.includes('var pageInfo = _attachmentImportPageRows()'),'import renderer consumes paged rows');
ok(render.includes('var rows = pageInfo.rows'),'DOM iteration is limited to current page');
ok(ensure.includes("_attachmentImportPageRows().rows.forEach"),'Select shown eligible is page-scoped, not 4096-entry bulk select');
ok(ensure.includes('Previous import inventory page') && ensure.includes('Next import inventory page'),'pagination controls are keyboard/screen-reader named');
ok(queueZip.includes('_inspectZipArchive(file, function ()'),'ZIP queue passes generation cancellation into inventory');
ok(queueDir.includes('_inventoryDirectoryEntries(roots, function ()'),'directory queue passes generation cancellation into traversal');
ok(src.includes('pageSize: 200'),'default import DOM page size is explicitly bounded');

console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
