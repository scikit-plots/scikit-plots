// Run 125 — stored ZIP entries verify in-stream and retain source-backed Blob slices.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0;function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}
const ensure=extract('_zipEnsureCrcTable'),update=extract('_zipCrc32Update'),crc=extract('_zipCrc32'),verify=extract('_zipVerifyStoredBlob'),extractEntry=extract('_extractZipEntry');
const factory=new Function(`
 var _ATTACHMENT_ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES=64*1024*1024;
 var _zipCrcTable=null;
 ${ensure}
 ${update}
 ${crc}
 ${verify}
 return {crc:_zipCrc32,verify:_zipVerifyStoredBlob};
`);
const h=factory();
const enc=new TextEncoder();const bytes=enc.encode('hello');const expected=h.crc(bytes);
let ab=0,cancelled=0;
function streamBlob(chunks,size=bytes.length){return {size,arrayBuffer(){ab++;return Promise.resolve(bytes.buffer)},stream(){let i=0;return{getReader(){return{async read(){return i<chunks.length?{done:false,value:chunks[i++]}:{done:true}},async cancel(){cancelled++},releaseLock(){}}}}}}}
const blob=streamBlob([bytes.slice(0,2),bytes.slice(2)]);
const got=await h.verify(blob,5,expected,()=>false);
ok(got===blob,'stored verifier returns original source-backed Blob');
ok(ab===0,'modern stored-entry verification performs zero arrayBuffer/full-copy reads');
ok(cancelled===0,'successful stored verification does not cancel reader');
let bad=false;try{await h.verify(streamBlob([bytes]),5,(expected+1)>>>0,()=>false)}catch(e){bad=e.message==='ZIP_CRC_MISMATCH'}
ok(bad,'stored stream still enforces CRC32');
let checks=0,wasCancelled=false;try{await h.verify(streamBlob([bytes.slice(0,2),bytes.slice(2)]),5,expected,()=>++checks>=3)}catch(e){wasCancelled=e.message==='ZIP_CANCELLED'}
ok(wasCancelled,'stored CRC stream remains generation-cancellable');

let legacyReads=0;const legacy={size:5,arrayBuffer(){legacyReads++;return Promise.resolve(bytes.slice().buffer)}};
ok(await h.verify(legacy,5,expected,()=>false)===legacy&&legacyReads===1,'legacy engine keeps one bounded arrayBuffer fallback');
let sizeBad=false;try{await h.verify({size:6,arrayBuffer(){throw new Error('must not read')}},5,expected,()=>false)}catch(e){sizeBad=e.message==='ZIP_UNCOMPRESSED_SIZE_MISMATCH'}
ok(sizeBad,'claimed stored size mismatch rejects before byte read');
const empty={size:0,stream(){return{getReader(){return{async read(){return{done:true}},releaseLock(){}}}}}};
ok(await h.verify(empty,0,0,()=>false)===empty,'empty stored entry verifies correctly');

ok(extractEntry.includes('_zipVerifyStoredBlob(compressedBlob, entry.size, entry.crc32, shouldCancel)'),'method 0 extraction uses streaming CRC verifier');
ok(!extractEntry.includes('bytes = await _zipReadStoredBounded'),'stored extraction no longer allocates full output Uint8Array');
ok(extractEntry.includes("entry.compressionMethod === 0 && payload && typeof payload.slice === 'function'"),'stored payload remains source-backed Blob slice');
ok(extractEntry.includes('_zipInflateBounded(compressedBlob, entry.size, shouldCancel)'),'DEFLATE path retains bounded materialization');
ok(extractEntry.includes('_zipCrc32(bytes)'),'DEFLATE output still receives CRC verification');
ok(verify.includes("reader.cancel('ZIP_CANCELLED')"),'stored streaming verifier actively cancels reader');
console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
