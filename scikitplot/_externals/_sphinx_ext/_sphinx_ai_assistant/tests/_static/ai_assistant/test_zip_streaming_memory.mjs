// Run 118 — ZIP extraction is single-buffer bounded and actively cancellable.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0; function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
 for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}

const exactSrc=extract('_zipReadExactStream');
const inflateSrc=extract('_zipInflateBounded');
const storedSrc=extract('_zipReadStoredBounded');
const extractSrc=extract('_extractZipEntry');
ok(exactSrc.includes('new Uint8Array(expected)'),'stream extraction preallocates one exact bounded output');
ok(!exactSrc.includes('chunks.push'),'stream extraction no longer retains a second chunk list');
ok(!inflateSrc.includes('chunks'),'DEFLATE wrapper no longer accumulates decompressed chunks');
ok(storedSrc.includes("typeof blob.stream === 'function'"),'stored ZIP entries prefer cancellable Blob streams');
ok(extractSrc.includes('_zipVerifyStoredBlob(compressedBlob, entry.size, entry.crc32, shouldCancel)'),'stored extraction uses streaming size/CRC verification without retaining a duplicate buffer');
ok(!extractSrc.includes('compressedBlob.arrayBuffer()'),'stored extraction no longer unconditionally performs uncancellable full arrayBuffer read');

globalThis._ATTACHMENT_ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES=64*1024*1024;
const exact=new Function(`${exactSrc}; return _zipReadExactStream;`)();
function streamOf(chunks,onCancel){
 return new ReadableStream({
   start(controller){for(const c of chunks)controller.enqueue(new Uint8Array(c));controller.close()},
   cancel(reason){if(onCancel)onCancel(reason)}
 });
}
const bytes=await exact(streamOf([[1,2],[3,4]]),4,()=>false);
ok(Array.from(bytes).join(',')==='1,2,3,4','exact stream preserves bytes across chunks');
ok(bytes.byteLength===4,'exact stream returns declared bounded length');

let short=false;try{await exact(streamOf([[1,2]]),3,()=>false)}catch(e){short=e.message==='ZIP_UNCOMPRESSED_SIZE_MISMATCH'}
ok(short,'short stream fails exact-size verification');
let overflow=false;try{await exact(streamOf([[1,2,3]]),2,()=>false)}catch(e){overflow=e.message==='ZIP_OUTPUT_LIMIT'}
ok(overflow,'overflowing stream is blocked before writing beyond declared size');
let limit=false;try{await exact(streamOf([]),64*1024*1024+1,()=>false)}catch(e){limit=e.message==='ZIP_OUTPUT_LIMIT'}
ok(limit,'declared output above hard per-entry bound fails before reader allocation');

let cancelled=false; let checks=0;
try{await exact(streamOf([[1],[2],[3]]),3,()=>++checks>=3)}catch(e){cancelled=e.message==='ZIP_CANCELLED'}
ok(cancelled,'generation cancellation interrupts stream between chunks');

const storedFactory=new Function(`
 var _ATTACHMENT_ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES=64*1024*1024;
 ${exactSrc}
 ${storedSrc}
 return _zipReadStoredBounded;
`);
const stored=storedFactory();
const blob=new Blob([new Uint8Array([9,8,7,6])]);
const storedBytes=await stored(blob,4,()=>false);
ok(Array.from(storedBytes).join(',')==='9,8,7,6','stored entry stream returns exact bytes');
let storedCancel=false;try{await stored(blob,4,()=>true)}catch(e){storedCancel=e.message==='ZIP_CANCELLED'}
ok(storedCancel,'stored entry checks cancellation before reading bytes');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
