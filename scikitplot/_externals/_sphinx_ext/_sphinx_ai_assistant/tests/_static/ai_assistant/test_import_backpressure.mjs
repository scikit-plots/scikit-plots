// Run 120 — import inventories apply backpressure before archive/folder reads.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0; function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
 for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}

const helpers=[
 '_attachmentImportReservationCount','_attachmentImportOutstandingCount','_attachmentImportQueueFullNotice',
 '_attachmentImportKindOutstandingCount','_attachmentImportDirectoryFullNotice','_reserveAttachmentImportInventory','_releaseAttachmentImportReservation','_enqueueAttachmentImportJob'
].map(extract).join('\n');

const helperFactory=new Function(`
 var _ATTACHMENT_IMPORT_MAX_OUTSTANDING=8;
 var _ATTACHMENT_IMPORT_MAX_DIRECTORY_OUTSTANDING=2;
 var _attachmentImportReservationSeq=0;
 var notices=[];
 var _attachmentImportState={active:null,queue:[],reservations:Object.create(null),queueFullNotified:false,directoryFullNotified:false,queuedNoticeShown:false};
 var document={activeElement:null};
 function showNotification(msg,good){notices.push([msg,good]);}
 function _ensureAttachmentImportLayer(){return _attachmentImportState;}
 function _attachmentImportAdvance(){ /* test only: active already owns the slot */ }
 ${helpers}
 return {state:_attachmentImportState,notices,
  reserve:_reserveAttachmentImportInventory,release:_releaseAttachmentImportReservation,
  enqueue:_enqueueAttachmentImportJob,count:_attachmentImportOutstandingCount};
`);
let h=helperFactory();
let tokens=[]; for(let i=0;i<100;i++){const t=h.reserve('zip'); if(t)tokens.push(t);}
ok(tokens.length===8,'only eight outstanding inventories can reserve capacity');
ok(h.count()===8,'reservation count reaches but never exceeds hard ceiling');
ok(h.notices.length===1,'100-item saturation emits one backpressure notice, not a toast storm');
ok(h.reserve('zip')===null && h.notices.length===1,'repeated rejection while saturated stays notification-coalesced');
h.release(tokens[0]);
const reopened=h.reserve('zip');
ok(!!reopened && h.count()===8,'capacity can reopen after a reservation is released');
ok(h.reserve('zip')===null && h.notices.length===2,'a later independent saturation episode may notify once again');

// Late releases from a cleared generation cannot corrupt newly-reserved work.
h=helperFactory();
const oldA=h.reserve('zip'),oldB=h.reserve('directory');
h.state.reservations=Object.create(null);
const fresh=h.reserve('zip');
ok(h.count()===1,'clear-style reservation table replacement removes stale outstanding work');
ok(h.release(oldA)===false && h.release(oldB)===false,'late stale releases are harmless after table replacement');
ok(h.count()===1 && h.release(fresh)===true,'stale finally release cannot decrement a new reservation');

// Direct callers cannot bypass the ceiling even without a reservation token.
h=helperFactory();
h.state.active={entries:[]};
h.state.queue=Array.from({length:7},()=>({entries:[]}));
const before=h.state.queue.length;
ok(h.enqueue({entries:[]},null,null)===false,'direct enqueue is rejected when active+queue already equals ceiling');
ok(h.state.queue.length===before && h.count()===8,'direct rejection does not grow queue beyond ceiling');
ok(h.notices.length===1,'direct bypass rejection uses the same coalesced warning');

// Exercise the actual ZIP queue with 100 archives. Reservation happens synchronously
// before any stage-queue parser task can run, so rejected archives must never be read.
const queueZipSrc=extract('_queueZipImport');
const zipFactory=new Function(`
 var _ATTACHMENT_IMPORT_MAX_OUTSTANDING=8;
 var _ATTACHMENT_IMPORT_MAX_DIRECTORY_OUTSTANDING=2;
 var _attachmentImportReservationSeq=0;
 var _attachmentImportState={active:null,queue:[],reservations:Object.create(null),queueFullNotified:false,directoryFullNotified:false,queuedNoticeShown:false};
 var _attachmentStageGeneration=7,_attachmentStagePending=0,_attachmentStageQueue=Promise.resolve();
 var inspectCalls=0,stageFallbackCalls=0,notices=[];
 var document={activeElement:null};
 function showNotification(msg,good){notices.push([msg,good]);}
 function _updateAttachmentStageUi(){}
 function _ensureAttachmentImportLayer(){return _attachmentImportState;}
 function _attachmentImportAdvance(){}
 async function _inspectZipArchive(file){inspectCalls++; file.reads=(file.reads||0)+1; return {name:file.name,file,entries:[],rejectedCount:0,truncated:false,centralDirectoryBytes:0,sourceBytes:file.size||1};}
 function _attachmentZipImportJob(info){return {kind:'zip',name:info.name,file:info.file,entries:[],rejectedCount:0};}
 async function _stageComposerFiles(){stageFallbackCalls++;}
 function _attachmentSafeName(v){return String(v||'');}
 ${helpers}
 ${queueZipSrc}
 return {queue:_queueZipImport,state:_attachmentImportState,notices,
   inspect:()=>inspectCalls,fallback:()=>stageFallbackCalls,pending:()=>_attachmentStagePending,outstanding:_attachmentImportOutstandingCount};
`);
const z=zipFactory();
const archives=Array.from({length:100},(_,i)=>({name:`a${i}.zip`,size:100,reads:0}));
const promises=archives.map(f=>z.queue(f,7));
// All reservation decisions happen before promise microtasks execute.
ok(z.outstanding()===8,'100 ZIP calls synchronously reserve only eight parser slots');
ok(archives.reduce((n,f)=>n+f.reads,0)===0,'reservation phase performs zero ZIP reads');
await Promise.all(promises);
ok(z.inspect()===8,'only admitted ZIPs reach container inspection');
ok(archives.slice(0,8).every(f=>f.reads===1),'each admitted ZIP is inspected exactly once');
ok(archives.slice(8).every(f=>f.reads===0),'ZIPs rejected by backpressure consume zero archive reads');
ok(z.state.active!==null && z.state.queue.length===7,'admitted ZIPs become one active plus seven queued inventories');
ok(z.outstanding()===8,'completed inventories preserve the same hard outstanding ceiling');
ok(z.pending()===0,'stage pending counter drains after admitted inventory jobs finish');
ok(z.fallback()===0,'queue pressure never converts rejected ZIPs into local-only staging fallbacks');
ok(z.notices.filter(x=>String(x[0]).includes('queue is full')).length===1,'ZIP storm produces one queue-full warning');

const queueZip=queueZipSrc;
const queueDir=extract('_queueDirectoryEntryImport');
const queueFileList=extract('_queueDirectoryFileListImport');
ok(queueZip.indexOf("_reserveAttachmentImportInventory('zip')") < queueZip.indexOf('_inspectZipArchive('),'ZIP capacity is reserved before any parser call');
ok(queueDir.indexOf("_reserveAttachmentImportInventory('directory')") < queueDir.indexOf('_inventoryDirectoryEntries('),'directory-entry capacity is reserved before traversal');
ok(queueFileList.indexOf("_reserveAttachmentImportInventory('directory')") < queueFileList.indexOf('_inventoryDirectoryFileList('),'directory FileList capacity is reserved before inventory construction');
ok(src.includes('var _ATTACHMENT_IMPORT_MAX_OUTSTANDING = 8;'),'outstanding import ceiling is explicit and reviewable');
ok(extract('_clearAttachmentImportJobs').includes('st.reservations = Object.create(null)'),'clear/new chat invalidates every pending inventory reservation');
ok(extract('_clearAttachmentImportJobs').includes('st.queueFullNotified = false'),'clear/new chat resets notification saturation state');

console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
