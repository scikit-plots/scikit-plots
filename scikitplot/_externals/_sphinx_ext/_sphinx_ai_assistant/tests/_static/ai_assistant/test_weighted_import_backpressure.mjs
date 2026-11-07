// Run 124 — weighted import backpressure for folder File-capability retention.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0;function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}
const reservationCount=extract('_attachmentImportReservationCount');
const outstanding=extract('_attachmentImportOutstandingCount');
const queueNotice=extract('_attachmentImportQueueFullNotice');
const kindCount=extract('_attachmentImportKindOutstandingCount');
const dirNotice=extract('_attachmentImportDirectoryFullNotice');
const reserve=extract('_reserveAttachmentImportInventory');
const release=extract('_releaseAttachmentImportReservation');
const enqueue=extract('_enqueueAttachmentImportJob');
const advance=extract('_attachmentImportAdvance');
const entryImport=extract('_queueDirectoryEntryImport');
const fileListImport=extract('_queueDirectoryFileListImport');

const factory=new Function(`
 var _ATTACHMENT_IMPORT_MAX_OUTSTANDING=8;
 var _ATTACHMENT_IMPORT_MAX_DIRECTORY_OUTSTANDING=2;
 var _attachmentImportReservationSeq=0;
 var notices=[];
 var _attachmentImportState={active:null,queue:[],reservations:Object.create(null),queueFullNotified:false,directoryFullNotified:false,queuedNoticeShown:false,trigger:null,page:0,search:null,filter:null,layer:null,list:null,count:null,selected:null};
 var document={activeElement:null,documentElement:{contains(){return false;}}};
 function requestAnimationFrame(fn){fn();}
 function showNotification(msg){notices.push(msg);}
 function _ensureAttachmentImportLayer(){return _attachmentImportState;}
 function _renderAttachmentImportList(){}
 function _clearAttachmentImportRenderedRows(){}
 ${reservationCount}
 ${outstanding}
 ${queueNotice}
 ${kindCount}
 ${dirNotice}
 ${reserve}
 ${release}
 ${enqueue}
 ${advance}
 return {st:_attachmentImportState,reserve:_reserveAttachmentImportInventory,release:_releaseAttachmentImportReservation,enqueue:_enqueueAttachmentImportJob,advance:_attachmentImportAdvance,kind:_attachmentImportKindOutstandingCount,out:_attachmentImportOutstandingCount,notices};
`);
let h=factory();
const d1=h.reserve('directory'),d2=h.reserve('directory'),d3=h.reserve('directory');
ok(!!d1&&!!d2&&d3===null,'third folder reservation is rejected before inventory work');
ok(h.kind('directory')===2,'directory outstanding count includes in-flight reservations');
ok(h.notices.filter(x=>x.includes('Two folder inventories')).length===1,'directory saturation warning is coalesced');
ok(h.reserve('directory')===null && h.notices.filter(x=>x.includes('Two folder inventories')).length===1,'repeated saturated folder drops do not toast-storm');
h.release(d1);const d4=h.reserve('directory');
ok(!!d4&&h.kind('directory')===2,'folder capacity rearms after a reservation is genuinely released');
let zips=[];for(let i=0;i<6;i++)zips.push(h.reserve('zip'));
ok(zips.every(Boolean)&&h.out()===8,'two folder reservations plus six ZIP reservations fill total eight-job ceiling');
ok(h.reserve('zip')===null&&h.reserve('zip')===null,'global ceiling rejects additional ZIP inventories');
ok(h.notices.filter(x=>x.includes('queue is full')).length===1,'global queue-full warning remains coalesced');

h=factory();h.st.active={kind:'directory',entries:[]};h.st.queue=[{kind:'directory',entries:[]}];
ok(h.enqueue({kind:'directory',entries:[]},null,null)===false,'direct enqueue cannot bypass two-folder retention ceiling');

h=factory();h.st.active={kind:'zip',entries:[]};
ok(h.enqueue({kind:'zip',entries:[]},null,null)===true,'first queued ZIP accepted behind active inventory');
ok(h.enqueue({kind:'zip',entries:[]},null,null)===true,'second queued ZIP accepted');
ok(h.notices.filter(x=>x.includes('queued behind')).length===1,'queued-behind notification is coalesced across batch');
h.advance(true);h.advance(true);
ok(h.st.queue.length===0&&h.st.queuedNoticeShown===false,'queued notice rearms once backlog drains');
ok(h.enqueue({kind:'zip',entries:[]},null,null)===true&&h.notices.filter(x=>x.includes('queued behind')).length===2,'new backlog episode may notify once again');

ok(src.includes('var _ATTACHMENT_IMPORT_MAX_DIRECTORY_OUTSTANDING = 2'),'directory retention ceiling is explicit');
ok(entryImport.indexOf("_reserveAttachmentImportInventory('directory')") < entryImport.indexOf('_inventoryDirectoryEntries('),'entry-based folder path reserves before recursive traversal');
ok(fileListImport.indexOf("_reserveAttachmentImportInventory('directory')") < fileListImport.indexOf('_inventoryDirectoryFileList('),'webkitdirectory FileList path reserves before building 4096 File descriptors');
ok((advance.match(/st\.page = 0;/g)||[]).length===1,'duplicate import page reset removed');
ok(enqueue.includes("job.kind === 'directory'")&&enqueue.includes('_ATTACHMENT_IMPORT_MAX_DIRECTORY_OUTSTANDING'),'direct enqueue path independently enforces folder ceiling');
console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
