// Run 140 — reader-owned ZIP authorization, strict text proposals, and verified artifact save.
import fs from 'node:fs';
import crypto from 'node:crypto';
const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3],'utf8');
let pass=0,fail=0;
function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
  for(const pre of ['async function ','function ']){
    const i=src.indexOf(pre+name+'('); if(i<0)continue;
    let d=0,st=false,q=null,esc=false,line=false,block=false;
    for(let j=i;j<src.length;j++){
      const c=src[j],n=src[j+1]||'';
      if(line){if(c==='\n')line=false;continue}
      if(block){if(c==='*'&&n==='/'){block=false;j++}continue}
      if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}
      if(c==='/'&&n==='/'){line=true;j++;continue}
      if(c==='/'&&n==='*'){block=true;j++;continue}
      if(c==='"'||c==="'"||c==='`'){q=c;continue}
      if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)
    }
  }
  throw new Error('missing '+name);
}
function sliceBetween(start,end){const i=src.indexOf(start),j=src.indexOf(end,i);if(i<0||j<0)throw new Error('missing slice '+start);return src.slice(i,j)}

// Static authority/UI contract.
ok(src.includes("var _ZIP_EDIT_SERVER_CONTRACT = 'scikitplot-zip-edit-v1'"),'client pins Run139 server artifact contract');
ok(src.includes("var _ZIP_EDIT_TEXT_PROPOSAL_CONTRACT = 'scikitplot-zip-text-proposal-v1'"),'model proposal has separate content-only contract');
ok(src.includes('var _ZIP_EDIT_MAX_AI_PATHS = 8'),'model proposal path fanout is independently bounded');
ok(src.includes('var _ZIP_EDIT_MAX_AI_FILE_BYTES = 32 * 1024'),'AI source file bytes are independently bounded');
ok(src.includes('var _ZIP_EDIT_MAX_AI_CONTEXT_CHARS = 96 * 1024'),'AI source context is independently bounded');
ok(src.includes('var _ZIP_EDIT_MAX_PROPOSAL_TOTAL_BYTES = 512 * 1024'),'model replacement text has bounded aggregate client budget');
ok(src.includes('var _ZIP_EDIT_DIFF_PREVIEW_CHARS = 128 * 1024'),'complete review diff has an explicit bounded surface');
ok(/function _zipEditStrictProposal[\s\S]*?reviewDiff\.truncated[\s\S]*?ZIP_EDIT_PROPOSAL_DIFF_TOO_LARGE/.test(src),'proposal is rejected when its complete diff cannot fit the review surface');
ok(/async function _zipEditSaveArtifactResponse[\s\S]*?fallbackReader = response\.body\.getReader\(\)[\s\S]*?ZIP_EDIT_FALLBACK_DOWNLOAD_MAX_BYTES/.test(src),'fallback artifact download streams through a hard byte ceiling');
ok(/function _zipEditStrictProposal[\s\S]*?sourceById\[row\.proposalId\][\s\S]*?ZIP_EDIT_PROPOSAL_NOT_AUTHORIZED/.test(src),'proposal IDs must be exact members of browser-assigned reader selection');
ok(/async function _zipEditApply[\s\S]*?authorization: \{ paths: authorized\.slice\(\) \}/.test(src),'artifact authorization.paths is built from current UI selection');
ok(/async function _zipEditApply[\s\S]*?id: 'r' \+ \(i \+ 1\)/.test(src),'multipart replacement ids are client-generated rather than model-controlled');
ok(/async function _zipEditAskModel[\s\S]*?row\.proposalId = 'f' \+ \(i \+ 1\)/.test(src),'model file IDs are opaque browser assignments');
ok(/function _zipEditBuildUserMessage[\s\S]*?\"id\":\"f1\"/.test(src),'model response schema contains opaque id + content, not archive path');
ok(/async function _zipEditAskModel[\s\S]*?context: \{ page_text: context/.test(src),'selected file bodies use server-fenced untrusted page_text context');
ok(/async function _zipEditAskModel[\s\S]*?_privacyPreflightReview/.test(src),'selected ZIP text receives local sensitive-data preflight before model egress');
ok(/async function _zipEditReadAuthorizedText[\s\S]*?ZIP_EDIT_TEXT_MIXED_NEWLINES/.test(src),'mixed-newline source files fail closed instead of receiving invisible normalization');
ok(/function _zipEditBuildUserMessage[\s\S]*?never emit an archive path[\s\S]*?never add, delete, rename, authorize/i.test(src),'model instruction explicitly denies path and structural output authority');
ok(src.includes("editArchive.textContent = 'Edit selected files'"),'archive preview exposes explicit edit workflow action');
ok(src.includes('Reader authorization is the boundary.'),'workflow explains authority boundary in UI');
ok(src.includes('I reviewed the accepted file diffs'),'application requires explicit post-diff review');
ok(/function _removeComposerResourceItem[\s\S]*?_zipEditState\.sourceItem === item[\s\S]*?_closeZipEditLayer\(false\)/.test(src),'removing source archive invalidates active ZIP-edit state');
ok(/function _clearComposerAttachments[\s\S]*?_closeZipEditLayer\(false\)/.test(src),'clearing composer attachments invalidates ZIP-edit state');
ok(src.includes("flags & (~0x080e & 0xffff)"),'client ZIP inventory rejects unsupported flags with explicit 16-bit mask');
ok(css.includes('.ai-assistant-panel-zip-edit-notice'),'reader-authority notice is styled');
ok(css.includes('.ai-assistant-panel-zip-edit-diff'),'bounded per-file diff is styled');
ok(css.includes('.ai-assistant-panel-zip-edit-review'),'explicit review control is styled');

// Pure-JS incremental SHA-256 must agree with Node's trusted implementation.
const hashPrelude=sliceBetween('var _ZIP_EDIT_SHA256_K =','async function _zipEditSha256Blob');
const hashFactory=new Function(`${hashPrelude}\nreturn {state:_zipEditSha256State,update:_zipEditSha256Update,digest:_zipEditSha256Digest};`);
const hashApi=hashFactory();
function ours(bytes,chunks){const st=hashApi.state();let p=0;for(const n of chunks){hashApi.update(st,bytes.subarray(p,Math.min(bytes.length,p+n)));p+=n}if(p<bytes.length)hashApi.update(st,bytes.subarray(p));return hashApi.digest(st)}
function expected(bytes){return crypto.createHash('sha256').update(bytes).digest('hex')}
const vectors=[
  new Uint8Array(),
  new TextEncoder().encode('abc'),
  new TextEncoder().encode('The quick brown fox jumps over the lazy dog'),
  Uint8Array.from({length:63},(_,i)=>i),
  Uint8Array.from({length:64},(_,i)=>255-i),
  Uint8Array.from({length:65},(_,i)=>(i*17)&255),
  Uint8Array.from({length:1024*1024+137},(_,i)=>(i*31+7)&255)
];
for(const [i,v] of vectors.entries()) ok(ours(v,[1,2,7,31,64,511,65537])===expected(v),'incremental SHA-256 vector '+i+' matches Node crypto');
let finished=false;try{const st=hashApi.state();hashApi.digest(st);hashApi.update(st,new Uint8Array([1]))}catch(e){finished=e.message==='ZIP_EDIT_HASH_STATE'}
ok(finished,'SHA-256 state rejects update after final digest');

// Capability parser is separate and fail closed.
const capFactory=new Function(`
 var _ZIP_EDIT_SERVER_CONTRACT='scikitplot-zip-edit-v1';
 var _ZIP_EDIT_RECEIPT_CONTRACT='scikitplot-zip-edit-receipt-v1';
 ${extract('_zipEditCapabilityParse')}
 return _zipEditCapabilityParse;
`);
const parseCaps=capFactory();
const capDoc={capabilities:{zip_edit_artifact:{version:1,contract:'scikitplot-zip-edit-v1',receipt_contract:'scikitplot-zip-edit-receipt-v1',endpoint:'/v1/artifacts/zip-edit',multipart:true,tree_authority:'source-archive',max_request_bytes:805306368,max_source_bytes:536870912,max_entry_bytes:67108864,max_replacement_total_bytes:268435456,max_authorized_paths:4096,max_replacements:256}}};
const caps=parseCaps(capDoc,'https://proxy.example');
ok(caps && caps.endpoint==='https://proxy.example/v1/artifacts/zip-edit','valid capability resolves same-origin artifact endpoint');
ok(caps && caps.maxSourceBytes===536870912 && caps.maxEntryBytes===67108864,'client consumes server-advertised byte ceilings');
ok(parseCaps({capabilities:{zip_edit_artifact:{...capDoc.capabilities.zip_edit_artifact,contract:'other'}}},'https://proxy.example')===null,'wrong artifact contract fails closed');
ok(parseCaps({capabilities:{zip_edit_artifact:{...capDoc.capabilities.zip_edit_artifact,endpoint:'https://evil.example/edit'}}},'https://proxy.example')===null,'absolute/cross-origin capability endpoint fails closed');
ok(parseCaps({capabilities:{zip_edit_artifact:{...capDoc.capabilities.zip_edit_artifact,max_source_bytes:536870913}}},'https://proxy.example')===null,'capability values above supported client ceiling fail closed');

// Strict proposal parser cannot escape exact source-row authority and preserves text conventions.
const proposalFactory=new Function(`
 var _ZIP_EDIT_TEXT_PROPOSAL_CONTRACT='scikitplot-zip-text-proposal-v1';
 var _ZIP_EDIT_MAX_PROPOSAL_ENTRY_BYTES=256*1024,_ZIP_EDIT_MAX_PROPOSAL_TOTAL_BYTES=512*1024,_ZIP_EDIT_MAX_AI_PATHS=8,_ZIP_EDIT_DIFF_PREVIEW_CHARS=128*1024,_CHAT_RESPONSE_MAX_BYTES=8*1024*1024;
 ${extract('_zipEditNormalizeReplacementText')}
 ${extract('_zipEditReplacementBlob')}
 ${extract('_zipEditBoundedDiff')}
 ${extract('_zipEditStrictProposal')}
 return _zipEditStrictProposal;
`);
const strictProposal=proposalFactory();
const sourceRows=[{proposalId:'f1',path:'src/a.py',text:'a\r\nb\r\n',hadBom:true,newline:'\r\n'},{proposalId:'f2',path:'docs/readme.md',text:'old\n',hadBom:false,newline:'\n'}];
const parsed=strictProposal(JSON.stringify({contract:'scikitplot-zip-text-proposal-v1',replacements:[{id:'f1',content:'a\nc\n'}]}),sourceRows);
ok(parsed.length===1 && parsed[0].path==='src/a.py','strict proposal accepts one exact authorized changed path');
ok(parsed[0].newText==='a\r\nc\r\n','replacement text preserves source newline convention');
const parsedBytes=new Uint8Array(await parsed[0].blob.arrayBuffer());
ok(parsedBytes[0]===0xef&&parsedBytes[1]===0xbb&&parsedBytes[2]===0xbf,'replacement blob preserves source UTF-8 BOM');
let outside=false;try{strictProposal(JSON.stringify({contract:'scikitplot-zip-text-proposal-v1',replacements:[{id:'f3',content:'x'}]}),sourceRows)}catch(e){outside=e.message==='ZIP_EDIT_PROPOSAL_NOT_AUTHORIZED'}
ok(outside,'proposal ID outside browser-assigned selection is rejected');
let extra=false;try{strictProposal(JSON.stringify({contract:'scikitplot-zip-text-proposal-v1',replacements:[{id:'f1',content:'x',mode:'rename'}]}),sourceRows)}catch(e){extra=e.message==='ZIP_EDIT_PROPOSAL_INVALID'}
ok(extra,'proposal entries reject structural/extra fields');
let nul=false;try{strictProposal(JSON.stringify({contract:'scikitplot-zip-text-proposal-v1',replacements:[{id:'f1',content:'x\u0000y'}]}),sourceRows)}catch(e){nul=e.message==='ZIP_EDIT_PROPOSAL_BINARY'}
ok(nul,'replacement content containing binary NUL is rejected');
let surrogate=false;try{strictProposal('{"contract":"scikitplot-zip-text-proposal-v1","replacements":[{"id":"f1","content":"\\ud800"}]}',sourceRows)}catch(e){surrogate=e.message==='ZIP_EDIT_PROPOSAL_INVALID_UTF16'}
ok(surrogate,'replacement content containing an unpaired surrogate is rejected');
let duplicate=false;try{strictProposal(JSON.stringify({contract:'scikitplot-zip-text-proposal-v1',replacements:[{id:'f1',content:'x'},{id:'f1',content:'y'}]}),sourceRows)}catch(e){duplicate=e.message==='ZIP_EDIT_PROPOSAL_NOT_AUTHORIZED'}
ok(duplicate,'duplicate proposal ID is rejected');
const unchanged=strictProposal(JSON.stringify({contract:'scikitplot-zip-text-proposal-v1',replacements:[{id:'f2',content:'old\n'}]}),sourceRows);
ok(unchanged.length===0,'byte-equivalent text proposal is omitted rather than creating a fake change');
let oversizedDiff=false;try{strictProposal(JSON.stringify({contract:'scikitplot-zip-text-proposal-v1',replacements:[{id:'f1',content:Array.from({length:70000},(_,i)=>'x'+i).join('\n')}]}),sourceRows)}catch(e){oversizedDiff=e.message==='ZIP_EDIT_PROPOSAL_DIFF_TOO_LARGE'||e.message==='ZIP_EDIT_PROPOSAL_ENTRY_TOO_LARGE'}
ok(oversizedDiff,'unreviewably large replacement/diff fails closed before acceptance');

// Receipt validation binds source/output hashes, authorization count, tree arithmetic, and preservation booleans.
const receiptFactory=new Function(`
 var _ZIP_EDIT_RECEIPT_CONTRACT='scikitplot-zip-edit-receipt-v1';
 ${extract('_zipEditReceipt')}
 return _zipEditReceipt;
`);
const verifyReceipt=receiptFactory();
const sourceSha='1'.repeat(64), outputSha='2'.repeat(64);
function responseFor(doc){return {headers:new Headers({'x-ai-artifact-contract':'scikitplot-zip-edit-receipt-v1','x-ai-artifact-sha256':outputSha,'x-ai-artifact-receipt':JSON.stringify(doc)})}}
const good={contract:'scikitplot-zip-edit-receipt-v1',source_sha256:sourceSha,output_sha256:outputSha,entry_count:205,authorized_count:4,applied_count:2,unchanged_count:203,tree_preserved:true,unchanged_content_preserved:true,metadata_preserved:true};
ok(verifyReceipt(responseFor(good),sourceSha,outputSha,2,4).entry_count===205,'valid receipt binds authorization and complete-tree counts');
let badAuth=false;try{verifyReceipt(responseFor({...good,authorized_count:5}),sourceSha,outputSha,2,4)}catch(e){badAuth=e.message==='ZIP_EDIT_RECEIPT_INVALID'}
ok(badAuth,'receipt with wrong authorization count is rejected');
let badMath=false;try{verifyReceipt(responseFor({...good,unchanged_count:202}),sourceSha,outputSha,2,4)}catch(e){badMath=e.message==='ZIP_EDIT_RECEIPT_INVALID'}
ok(badMath,'receipt with inconsistent complete-tree arithmetic is rejected');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
