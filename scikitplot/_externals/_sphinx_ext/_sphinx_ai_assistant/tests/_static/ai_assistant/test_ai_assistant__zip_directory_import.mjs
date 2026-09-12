// Run 113 — bounded folder/ZIP inventory, explicit selection, and verified extraction.
import fs from 'node:fs';
import zlib from 'node:zlib';
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

// Static architecture / UI contracts.
ok(src.includes('var _ATTACHMENT_IMPORT_MAX_ENTRIES = 4096'),'inventory has independent 4096-entry metadata ceiling');
ok(src.includes('var _ATTACHMENT_DIRECTORY_MAX_DEPTH = 24'),'directory recursion depth is bounded');
ok(src.includes('var _ATTACHMENT_ZIP_MAX_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024'),'ZIP central directory read is bounded');
ok(src.includes('var _ATTACHMENT_ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES = 64 * 1024 * 1024'),'per-entry ZIP output is bounded');
ok(src.includes('var _ATTACHMENT_ZIP_MAX_SELECTED_UNCOMPRESSED_BYTES = 256 * 1024 * 1024'),'selected ZIP output has aggregate bound');
ok(src.includes('var _ATTACHMENT_ZIP_MAX_COMPRESSION_RATIO = 500'),'ZIP compression ratio is bounded');
ok(src.includes("folderInput.setAttribute('webkitdirectory', '')"),'folder picker requests directory selection');
ok(src.includes('<strong>Add folder</strong>'),'attach menu exposes folder inventory action');
ok(src.includes('Drop files, folders, or ZIPs'),'drop overlay explains expanded intake');
ok(/function _attachmentFilesFromDrop[\s\S]*?directoryEntries\.push\(entry\)/.test(src),'drop snapshot retains directory entry handles');
ok(/panel\.addEventListener\('drop'[\s\S]*?_queueDirectoryEntryImport\(dropped\.directoryEntries\)/.test(src),'dropped directories route to bounded inventory');
ok(/function _queueComposerFiles[\s\S]*?Archives are[\s\S]*?first-class staged resources/.test(src) && !/function _queueComposerFiles[\s\S]*?_queueZipImport/.test(src),'top-level ZIP preserves original archive identity by default');
ok(src.includes("kind: 'archive', modality: 'archive'") && src.includes("rawEligible: true") && src.includes('Inspect contents'),'archives are raw-resource eligible while secure inventory remains explicit');
ok(src.includes('Inventory is metadata-only. ZIP payload bytes are extracted only for selected entries'),'import UI states metadata/extraction trust boundary');
ok(src.includes("add.textContent = 'Add selected'"),'import requires explicit commit action');
ok(src.includes("if (e.key !== 'Tab' || layer.hidden) return") && src.includes("dialog.querySelectorAll('button:not([disabled])"),'import dialog traps Tab focus');
ok(src.includes('var trigger = st.trigger') && src.includes('trigger.focus()'),'import dialog restores trigger focus when queue closes');
ok(src.includes("e.dataTransfer.dropEffect = 'copy'") && src.includes('Composer full · ZIP/folder inventory still available'),'full composer still permits metadata-only folder/ZIP drops');
ok(src.includes('ZIP_DUPLICATE_LOCAL_HEADER') && src.includes('localOffsets[localOffsetKey]'),'ZIP inventory blocks duplicate local-header targets');
ok(src.includes('var remaining = Math.max(0, room - selectedNow)'),'bulk inventory selection respects remaining composer slots');
ok(/function _queueAttachmentImportCommit[\s\S]*?entry\.selected[\s\S]*?_extractZipEntry/.test(src),'ZIP extraction is downstream of explicit selected entries');
ok(/function _inspectZipArchive[\s\S]*?Math\.min\(size, 22\)[\s\S]*?_ATTACHMENT_ZIP_EOCD_TAIL_BYTES/.test(src),'ZIP parser uses 22-byte EOCD fast probe before bounded comment-tail expansion');
ok(src.includes("throw new Error('ZIP64_UNSUPPORTED')"),'ZIP64 has explicit unsupported boundary');
ok(src.includes("'ZIP_ENCRYPTED_ENTRY'"),'encrypted ZIP entries are blocked');
ok(src.includes("'ZIP_SYMLINK_BLOCKED'"),'ZIP symlinks are blocked');
ok(src.includes("'ZIP_SPECIAL_FILE_BLOCKED'"),'ZIP special files are blocked');
ok(src.includes("'ZIP_DUPLICATE_PATH_ALIAS'"),'duplicate normalized path aliases are blocked');
ok(src.includes("new DecompressionStream('deflate-raw')"),'deflate ZIP extraction uses raw DEFLATE capability');
ok(src.includes("throw new Error('ZIP_OUTPUT_LIMIT')"),'decompressed stream has hard actual-byte stop');
ok(src.includes("throw new Error('ZIP_CRC_MISMATCH')"),'ZIP extraction verifies CRC32');
ok(src.includes("throw new Error('ZIP_LOCAL_NAME_MISMATCH')"),'local header filename must agree with central directory');
ok(src.includes("relativePath: item.relativePath ? _attachmentSafeRelativePath(item.relativePath) : ''"),'canonical turn manifest preserves safe relative path');
ok(src.includes("type: typeof item.type === 'string' ? item.type.slice(0, 120) : ''"),'canonical turn manifest preserves bounded MIME type');
ok(src.includes("sourceKind: typeof item.sourceKind === 'string'"),'canonical manifest preserves import origin');
ok(src.includes("['archive', 'ZIP/archive']"),'resource manager can filter archives');
ok(css.includes('.ai-assistant-panel-attachment-import-row'),'import inventory rows are styled');
ok(css.includes('.ai-assistant-panel-attachment-import-add'),'explicit Add selected action is styled');

// Path normalization security.
const pathFactory=new Function(`
 var _ATTACHMENT_IMPORT_MAX_PATH_CHARS=1024;
 ${extract('_attachmentSafeRelativePath')}
 ${extract('_attachmentPathAlias')}
 return {safe:_attachmentSafeRelativePath,alias:_attachmentPathAlias};
`);
const pathApi=pathFactory();
ok(pathApi.safe('docs/a.txt')==='docs/a.txt','safe relative path accepted');
ok(pathApi.safe('../evil.txt')==='','parent traversal rejected');
ok(pathApi.safe('docs/../evil.txt')==='','nested parent traversal rejected');
ok(pathApi.safe('/etc/passwd')==='','absolute POSIX path rejected');
ok(pathApi.safe('C:/secret.txt')==='','Windows drive-root path rejected');
ok(pathApi.safe('docs\\..\\evil.txt')==='','backslash traversal rejected');
ok(pathApi.safe('safe/\u202Eevil.txt')==='','bidi-format path rejected');
ok(pathApi.alias('Folder/File.txt')===pathApi.alias('folder/file.txt'),'alias comparison is case-insensitive');
ok(pathApi.alias('folder/file.txt.')===pathApi.alias('folder/file.txt'),'alias comparison catches trailing-dot collision');

// Chromium-style directory readers can return only batches; verify repeated draining.
const dirFactory=new Function(`
 var _ATTACHMENT_IMPORT_MAX_ENTRIES=4096,_ATTACHMENT_IMPORT_MAX_PATH_CHARS=1024,_ATTACHMENT_DIRECTORY_MAX_DEPTH=24;
 ${extract('_attachmentSafeName')}
 ${extract('_attachmentSafeRelativePath')}
 ${extract('_directoryEntryFile')}
 ${extract('_directoryReadBatch')}
 ${extract('_inventoryDirectoryEntries')}
 return _inventoryDirectoryEntries;
`);
const inventoryDirectory=dirFactory();
let readCalls=0;
const fakeFiles=Array.from({length:205},(_,i)=>({
  isFile:true,isDirectory:false,name:`f${i}.txt`,
  file(resolve){resolve({name:`f${i}.txt`,type:'text/plain',size:i+1})}
}));
const batches=[fakeFiles.slice(0,100),fakeFiles.slice(100,200),fakeFiles.slice(200),[]];
const root={isDirectory:true,name:'root',createReader(){return {readEntries(resolve){readCalls++;resolve(batches.shift()||[])}}}};
const dirInventory=await inventoryDirectory([root]);
ok(dirInventory.entries.length===205,'directory inventory drains all 205 files');
ok(readCalls===4,'directory reader repeats readEntries until empty batch');
ok(dirInventory.entries[204].relativePath==='root/f204.txt','directory inventory preserves bounded relative path');
ok(dirInventory.truncated===false,'ordinary 205-file directory does not report truncation');

// ZIP parser/extractor runtime with a real classic ZIP built in-memory.
const zipFactory=new Function(`
 var _ATTACHMENT_IMPORT_MAX_ENTRIES=4096,_ATTACHMENT_IMPORT_MAX_PATH_CHARS=1024;
 var _ATTACHMENT_ZIP_EOCD_TAIL_BYTES=22+65535,_ATTACHMENT_ZIP_MAX_CENTRAL_DIRECTORY_BYTES=8*1024*1024;
 var _ATTACHMENT_ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES=64*1024*1024,_ATTACHMENT_ZIP_MAX_SELECTED_UNCOMPRESSED_BYTES=256*1024*1024;
 var _ATTACHMENT_ZIP_MAX_COMPRESSION_RATIO=500,_ATTACHMENT_ZIP_EXT_RE=/\\.zip$/i;
 var _ATTACHMENT_TEXT_EXT_RE=/\\.(?:txt|md|json)$/i;
 ${extract('_attachmentSafeName')}
 ${extract('_attachmentIsZipCandidate')}
 ${extract('_attachmentSafeRelativePath')}
 ${extract('_attachmentPathAlias')}
 ${extract('_attachmentGuessMimeFromName')}
 ${extract('_readAttachmentBlobSlice')}
 ${extract('_zipU16')}
 ${extract('_zipU32')}
 ${extract('_zipDecodeName')}
 ${extract('_zipEntryReject')}
 ${extract('_inspectZipArchive')}
 var _zipCrcTable=null;
 ${extract('_zipEnsureCrcTable')}
 ${extract('_zipCrc32Update')}
 ${extract('_zipCrc32')}
 ${extract('_zipVerifyStoredBlob')}
 ${extract('_zipReadExactStream')}
 ${extract('_zipInflateBounded')}
 ${extract('_zipReadStoredBounded')}
 ${extract('_extractZipEntry')}
 return {inspect:_inspectZipArchive,extract:_extractZipEntry,crc:_zipCrc32};
`);
const zipApi=zipFactory();
function u16(view,o,v){view.setUint16(o,v,true)} function u32(view,o,v){view.setUint32(o,v>>>0,true)}
function concat(...arrays){const n=arrays.reduce((x,a)=>x+a.length,0),out=new Uint8Array(n);let p=0;for(const a of arrays){out.set(a,p);p+=a.length}return out}
function buildZip({name='notes.txt',plain=new TextEncoder().encode('hello\n'),method=0,flags=0x0800,crcOverride=null,uncompressedOverride=null,compressedOverride=null}={}){
  const nameBytes=new TextEncoder().encode(name);
  const compressed=method===8?new Uint8Array(zlib.deflateRawSync(plain)):plain;
  const crc=crcOverride==null?zipApi.crc(plain):crcOverride>>>0;
  const usize=uncompressedOverride==null?plain.length:uncompressedOverride>>>0;
  const csize=compressedOverride==null?compressed.length:compressedOverride>>>0;
  const local=new Uint8Array(30+nameBytes.length+compressed.length); const lv=new DataView(local.buffer);
  u32(lv,0,0x04034b50);u16(lv,4,20);u16(lv,6,flags);u16(lv,8,method);u32(lv,14,crc);u32(lv,18,csize);u32(lv,22,usize);u16(lv,26,nameBytes.length);u16(lv,28,0);
  local.set(nameBytes,30); local.set(compressed,30+nameBytes.length);
  const central=new Uint8Array(46+nameBytes.length); const cv=new DataView(central.buffer);
  u32(cv,0,0x02014b50);u16(cv,4,0x0314);u16(cv,6,20);u16(cv,8,flags);u16(cv,10,method);u32(cv,16,crc);u32(cv,20,csize);u32(cv,24,usize);
  u16(cv,28,nameBytes.length);u16(cv,30,0);u16(cv,32,0);u16(cv,34,0);u32(cv,38,(0x81a4<<16)>>>0);u32(cv,42,0);central.set(nameBytes,46);
  const eocd=new Uint8Array(22); const ev=new DataView(eocd.buffer);u32(ev,0,0x06054b50);u16(ev,8,1);u16(ev,10,1);u32(ev,12,central.length);u32(ev,16,local.length);u16(ev,20,0);
  return {bytes:concat(local,central,eocd),centralOffset:local.length,centralSize:central.length,eocdOffset:local.length+central.length,plain};
}
function instrumentZip(built,type='application/zip'){
  const blob=new Blob([built.bytes],{type}); const ranges=[];
  return {file:{name:'sample.zip',type,size:blob.size,slice(a,b){ranges.push([a,b]);return blob.slice(a,b)}},ranges};
}
const built=buildZip(); const observed=instrumentZip(built);
const info=await zipApi.inspect(observed.file);
ok(info.entries.length===1 && info.entries[0].selectable===true,'classic stored ZIP inventories one eligible file');
ok(info.entries[0].relativePath==='notes.txt','ZIP inventory preserves safe entry path');
ok(observed.ranges[0][0]===built.bytes.length-22 && observed.ranges[0][1]===built.bytes.length,'EOCD fast path reads only final 22 bytes first');
ok(observed.ranges.length===2 && observed.ranges[1][0]===built.centralOffset && observed.ranges[1][1]===built.centralOffset+built.centralSize,'inventory reads central directory separately and no payload bytes');
const extracted=await zipApi.extract({file:observed.file,name:'sample.zip'},info.entries[0]);
ok((await extracted.file.text())==='hello\n','stored ZIP extraction returns exact verified bytes');
ok(extracted.relativePath==='notes.txt' && extracted.sourceKind==='zip','extracted file carries ZIP provenance');

const unsafe=instrumentZip(buildZip({name:'../evil.txt'})); const unsafeInfo=await zipApi.inspect(unsafe.file);
ok(unsafeInfo.entries[0].rejected===true && unsafeInfo.entries[0].reason==='ZIP_PATH_UNSAFE','ZIP traversal entry blocked during inventory');
const encrypted=instrumentZip(buildZip({flags:0x0801})); const encryptedInfo=await zipApi.inspect(encrypted.file);
ok(encryptedInfo.entries[0].reason==='ZIP_ENCRYPTED_ENTRY','encrypted entry blocked before extraction');
const unsupported=instrumentZip(buildZip({method:12})); const unsupportedInfo=await zipApi.inspect(unsupported.file);
ok(unsupportedInfo.entries[0].reason==='ZIP_UNSUPPORTED_COMPRESSION','unsupported compression method blocked');
const ratio=instrumentZip(buildZip({method:8,plain:new Uint8Array([65]),uncompressedOverride:5000})); const ratioInfo=await zipApi.inspect(ratio.file);
ok(ratioInfo.entries[0].reason==='ZIP_COMPRESSION_RATIO','suspicious declared compression ratio blocked');

const crcBuilt=buildZip(); const crcObs=instrumentZip(crcBuilt); const crcInfo=await zipApi.inspect(crcObs.file); crcInfo.entries[0].crc32^=1;
let crcBlocked=false;try{await zipApi.extract({file:crcObs.file,name:'sample.zip'},crcInfo.entries[0])}catch(e){crcBlocked=e.message==='ZIP_CRC_MISMATCH'}
ok(crcBlocked,'CRC mismatch blocks conversion into staged File');

const deflated=instrumentZip(buildZip({method:8,plain:new TextEncoder().encode('deflate me '.repeat(40))})); const defInfo=await zipApi.inspect(deflated.file);
let deflateOk=false;
try{const d=await zipApi.extract({file:deflated.file,name:'sample.zip'},defInfo.entries[0]);deflateOk=(await d.file.text())==='deflate me '.repeat(40)}catch(e){deflateOk=e.message==='ZIP_DEFLATE_UNAVAILABLE'}
ok(deflateOk,'deflate entry either verifies correctly or fails closed on missing capability');

const dupBase=buildZip();
const dupLocal=dupBase.bytes.slice(0,dupBase.centralOffset);
const dupCentral1=dupBase.bytes.slice(dupBase.centralOffset,dupBase.centralOffset+dupBase.centralSize);
const dupCentral2=dupCentral1.slice(); new TextEncoder().encode('other.txt').forEach((b,i)=>{dupCentral2[46+i]=b});
const dupEocd=new Uint8Array(22); const dupEv=new DataView(dupEocd.buffer); u32(dupEv,0,0x06054b50);u16(dupEv,8,2);u16(dupEv,10,2);u32(dupEv,12,dupCentral1.length+dupCentral2.length);u32(dupEv,16,dupLocal.length);u16(dupEv,20,0);
const dupBytes=concat(dupLocal,dupCentral1,dupCentral2,dupEocd); const dupBlob=new Blob([dupBytes],{type:'application/zip'});
const dupFile={name:'duplicate-target.zip',type:'application/zip',size:dupBlob.size,slice:(a,b)=>dupBlob.slice(a,b)};
const dupInfo=await zipApi.inspect(dupFile);
ok(dupInfo.entries[0].rejected===false && dupInfo.entries[1].reason==='ZIP_DUPLICATE_LOCAL_HEADER','two distinct names cannot alias the same ZIP local-header target');

const zip64=buildZip(); const zbytes=zip64.bytes.slice(); const zev=new DataView(zbytes.buffer); u16(zev,zip64.eocdOffset+8,0xffff);u16(zev,zip64.eocdOffset+10,0xffff);
const zblob=new Blob([zbytes],{type:'application/zip'}); const zfile={name:'zip64.zip',type:'application/zip',size:zblob.size,slice:(a,b)=>zblob.slice(a,b)};
let zip64Blocked=false;try{await zipApi.inspect(zfile)}catch(e){zip64Blocked=e.message==='ZIP64_UNSUPPORTED'}
ok(zip64Blocked,'ZIP64 sentinel routes to explicit unsupported capability path');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
