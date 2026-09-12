// Run 117 — historical actions must reflect actual runtime preview/byte authority.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let pass=0, fail=0;
function ok(cond,name){ if(cond) pass++; else { fail++; console.error('FAIL '+name); } }
function extract(name){
  for(const pre of ['async function ','function ']){
    const i=src.indexOf(pre+name+'('); if(i<0) continue;
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

globalThis._ATTACHMENT_IMAGE_PREVIEW_MAX_BYTES = 12 * 1024 * 1024;
const safeName = new Function(`${extract('_attachmentSafeName')}; return _attachmentSafeName;`)();
globalThis._attachmentSafeName = safeName;
const hasPreview = new Function(`${extract('_attachmentHasPreviewAuthority')}; return _attachmentHasPreviewAuthority;`)();
globalThis._attachmentHasPreviewAuthority = hasPreview;
const label = new Function(`${extract('_attachmentInteractionLabel')}; return _attachmentInteractionLabel;`)();
globalThis._attachmentInteractionLabel = label;
const aria = new Function(`${extract('_attachmentInteractionAria')}; return _attachmentInteractionAria;`)();

ok(safeName('normal.pdf')==='normal.pdf','ordinary basename preserved');
ok(!safeName('../../secret.pdf').includes('/'),'forward path separators removed from direct filename');
ok(!safeName('..\\..\\secret.pdf').includes('\\'),'backslash separators removed from direct filename');
ok(!safeName('invoice\u202egnp.exe').includes('\u202e'),'bidi override remains neutralised');
ok(safeName('مرحبا-מסמך.pdf').includes('مرحبا-מסמך'),'real RTL-script filename remains readable');

ok(hasPreview({kind:'pdf',file:{}})===true,'live PDF bytes grant preview/open authority');
ok(hasPreview({kind:'pdf',pdfVerified:true,runtimeEvicted:true})===false,'evicted verified PDF has no preview authority');
ok(label({kind:'pdf',pdfVerified:true,runtimeEvicted:true})==='Details','evicted PDF action becomes Details');
ok(label({kind:'pdf',file:{}})==='Preview','live PDF action remains Preview');
ok(aria({kind:'pdf',pdfVerified:true,runtimeEvicted:true,name:'report.pdf'})==='Details report.pdf','historical PDF aria copy tells capability truth');

ok(hasPreview({kind:'text',previewText:'abc'})===true,'retained text prefix is previewable without File');
ok(hasPreview({kind:'text',file:{}})===true,'live text File can be lazily previewed');
ok(hasPreview({kind:'text',runtimeTextEvicted:true})===false,'evicted text preview has details only');
ok(hasPreview({kind:'page',previewText:'# doc'})===true,'page snapshot remains previewable');
ok(hasPreview({kind:'page'})===false,'page metadata without text is details only');
ok(hasPreview({kind:'image',file:{},rasterPreview:true,size:100})===true,'small retained raster image is previewable');
ok(hasPreview({kind:'image',file:{},rasterPreview:true,size:20*1024*1024})===false,'oversized image uses details/download path');
ok(hasPreview({kind:'image',rasterPreview:true,size:100})===false,'historical image metadata alone is not previewable');
ok(hasPreview({kind:'archive',file:{}})===false,'archive action is details rather than fake inline preview');
ok(hasPreview({kind:'file',file:{}})===false,'unknown binary action is details rather than fake inline preview');

const appendPdf=extract('_appendLocalPdfActions');
const render=extract('_renderAttachmentPreviewBody');
const meta=extract('_attachmentPreviewMeta');
ok(appendPdf.includes('!item.file'),'PDF actions require retained local bytes');
ok(appendPdf.includes('item.pdfVerified !== true'),'PDF actions require verified signature');
ok(render.includes("item.pdfVerified === true && item.file"),'verified-PDF renderer requires actual File authority');
ok(render.includes("item.kind === 'pdf' && !item.file && item.runtimeEvicted === true"),'evicted PDF gets explicit bounded-cache explanation');
ok(render.includes("item.kind === 'pdf' && !item.file"),'metadata-only PDF has dedicated historical explanation');
ok(meta.includes('Local PDF bytes released'),'preview metadata exposes binary eviction state');
ok(meta.includes('PDF metadata only'),'preview metadata exposes reload/no-byte state');
ok(meta.includes('Bounded text preview released'),'preview metadata exposes text eviction state');

ok(src.includes("preview.textContent = _attachmentInteractionLabel(item);"),'resource manager derives action label from capability');
ok((src.match(/preview\.setAttribute\('aria-label', _attachmentInteractionAria\(item\)\);/g)||[]).length>=2,'composer and historical cards derive aria action from capability');
ok(!render.includes("createElement('iframe')"),'capability truth does not regress to embedded PDF iframe');

console.log(`${pass} passed, ${fail} failed`);
if(fail) process.exit(1);
