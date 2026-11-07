// Run 110 — metadata-first staging, fixed shared send budget, lazy previews,
// thumbnail bounds, and mutation-safe async request preparation.
import fs from 'node:fs';
const js = fs.readFileSync(process.argv[2], 'utf8');
let pass=0, fail=0;
function t(name, cond){ if(cond) pass++; else { fail++; console.error('FAIL '+name); } }
function extract(name){
  for (const prefix of ['async function ','function ']) {
    const i=js.indexOf(prefix+name+'(');
    if(i<0) continue;
    let d=0, started=false, str=null, esc=false, line=false, block=false;
    for(let j=i;j<js.length;j++){
      const c=js[j], n=js[j+1]||'';
      if(line){ if(c==='\n') line=false; continue; }
      if(block){ if(c==='*'&&n==='/'){block=false;j++;} continue; }
      if(str){ if(esc) esc=false; else if(c==='\\') esc=true; else if(c===str) str=null; continue; }
      if(c==='/'&&n==='/'){line=true;j++;continue;} if(c==='/'&&n==='*'){block=true;j++;continue;}
      if(c==='"'||c==="'"||c==='`'){str=c;continue;} if(c==='{'){d++;started=true;} else if(c==='}'&&started&&--d===0) return js.slice(i,j+1);
    }
  }
  throw new Error('missing '+name);
}

t('file limit is 256', js.includes('var _ATTACHMENT_MAX_FILES = 256'));
t('one fixed 48k context cap remains authoritative', js.includes('var _ATTACHMENT_MAX_TEXT_CHARS = 48000'));
t('per-file context share capped near 12k', js.includes('var _ATTACHMENT_MAX_FILE_CONTEXT_CHARS = 12000'));
t('staging is metadata-first', !extract('_stageComposerFiles').includes('_readAttachmentText'));
t('preview reads only on demand', extract('_ensureAttachmentTextPreview').includes('_readAttachmentText(item.file, _ATTACHMENT_MAX_PREVIEW_BYTES)'));
t('send plan reads lazily', extract('_prepareComposerAttachmentPlan').includes('await _readAttachmentText(item.file, readBytes)'));
t('send plan removes whole tail parts instead of raw slicing', extract('_prepareComposerAttachmentPlan').includes('parts.pop()') && !extract('_prepareComposerAttachmentPlan').includes('text.slice(0, _ATTACHMENT_MAX_TEXT_CHARS)'));
t('normal thumbnail budget 24', js.includes('var _ATTACHMENT_THUMBNAIL_NORMAL_MAX = 24'));
t('large batch thumbnail budget 8', js.includes('var _ATTACHMENT_THUMBNAIL_LARGE_MAX = 8'));
t('out-of-budget thumbnails revoke blob URLs', /thumbnailsUsed < thumbnailBudget[\s\S]*_attachmentRevokeObjectUrl\(item\)/.test(js));
t('send snapshots mutation revision', js.includes('var attachmentRevision = _attachmentMutationRevision'));
t('send aborts on attachment mutation', js.includes('Attachments changed while preparing this request. Review the visible batch and send again.'));
t('duplicate stage pending declaration removed', (js.match(/var _attachmentStagePending = 0/g)||[]).length===1);
t('duplicate pending send guard removed', (js.match(/Attachments are still being prepared\. Send when preparation finishes\./g)||[]).length===1);

// Execute metadata staging with 120 synthetic files. No read helper is even
// present in the stage factory, so any accidental eager read throws.
const stage=extract('_stageComposerFiles');
const stageFactory=new Function(`
 var _attachmentStageGeneration=0,_attachmentMutationRevision=0,_ATTACHMENT_MAX_FILES=256;
 var _composerAttachments=[]; function _attachmentSafeName(v){return String(v||'file').slice(0,240)}
 function _attachmentIsPdfCandidate(){return false} function _attachmentIsImage(){return false} function _attachmentIsRasterPreview(){return false}
 function _attachmentIsText(){return true} function _attachmentClassifyMetadata(){return {kind:'text',modality:'text',localOnly:false,sendEligible:true,rawEligible:true}} function _renderComposerAttachments(){} function showNotification(){}
 function _readAttachmentText(){throw new Error('EAGER_READ')}
 ${stage}
 return async function(files){await _stageComposerFiles(files,0);return {count:_composerAttachments.length,rev:_attachmentMutationRevision};};
`);
const staged=await stageFactory()(Array.from({length:120},(_,i)=>({name:`f${i}.txt`,type:'text/plain',size:4*1024**3})));
t('120-file staging succeeds', staged.count===120);
t('120-file staging increments revision for every item', staged.rev===120);

const safeName=extract('_attachmentSafeName'), header=extract('_attachmentHeader'), select=extract('_attachmentSelectContextCandidates'), planFn=extract('_prepareComposerAttachmentPlan');
const planFactory=new Function(`
 var _ATTACHMENT_MAX_TEXT_CHARS=48000,_ATTACHMENT_MAX_FILE_CONTEXT_CHARS=12000,_ATTACHMENT_MAX_CONTEXT_READ_BYTES=64*1024;
 ${safeName}\n${header}\n${select}
 async function _readAttachmentText(file,maxBytes){return String(file.body||'').slice(0,maxBytes)}
 ${planFn}
 return _prepareComposerAttachmentPlan;
`);
const buildPlan=planFactory();
const hundred=Array.from({length:100},(_,i)=>({name:`notes-${i}.txt`,type:'text/plain',kind:'text',sendEligible:true,size:4*1024**3,file:{body:'x'.repeat(100000)}}));
const plan=await buildPlan(hundred);
t('100-file outbound plan stays <=48k', plan.text.length<=48000);
t('100-file plan can include all reasonable headers', plan.included.length===100);
t('no file receives more than 12k chars', plan.included.every(r=>r.bodyChars<=12000));
t('huge text is treated as bounded excerpt, not rejected by file size', plan.included.every(r=>r.boundedExcerpt===true));

const hostile=Array.from({length:256},(_,i)=>({name:('x'.repeat(235)+String(i)+'.txt'),type:'text/plain',kind:'text',sendEligible:true,size:1,file:{body:'z'}}));
const hostilePlan=await buildPlan(hostile);
t('pathological long-name batch safely subsets', hostilePlan.included.length<256 && hostilePlan.included.length>0);
t('pathological plan stays <=48k', hostilePlan.text.length<=48000);
t('context starts at complete attachment header', hostilePlan.text.startsWith('Attachment: '));
t('every serialized part has a complete header boundary', hostilePlan.text.split('\n\n').every(part=>part.startsWith('Attachment: ')));

console.log(`${pass} passed, ${fail} failed`); if(fail) process.exit(1);
