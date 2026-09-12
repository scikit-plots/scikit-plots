// Run 112 contract, upgraded by Runs 114/126 — PDF remains signature-gated and never enters text context; raw delivery is a separate resource-plane capability.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3],'utf8');
let pass=0,fail=0; function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
 for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}

const candidate=extract('_attachmentIsPdfCandidate');
const stage=extract('_stageComposerFiles');
const classStart=src.indexOf('function _attachmentClassifyMetadata(');
const classEnd=src.indexOf('function _readAttachmentBlobSlice(',classStart);
const classify=src.slice(classStart,classEnd).trim();
const prefix=extract('_readAttachmentPrefixBytes');
const verify=extract('_ensurePdfSignature');
const render=extract('_renderAttachmentPreviewBody');
ok(src.includes('var _ATTACHMENT_PDF_SIGNATURE_BYTES = 8'),'signature probe is only 8 bytes');
ok(!src.includes('_ATTACHMENT_PDF_INLINE_MAX_BYTES'),'native inline PDF decode cap removed with iframe renderer');
ok(candidate.includes("type === 'application/pdf'") && candidate.includes('/\\.pdf$/i'),'MIME or filename only classify a PDF candidate');
ok(classify.indexOf('_attachmentIsPdfCandidate(proxy)') < classify.indexOf('_attachmentIsText(proxy)'),'PDF candidate classification precedes text MIME classification');
ok(classify.includes("kind: 'pdf', modality: 'document'") && classify.includes('sendEligible: false') && classify.includes('rawEligible: true'),'PDF cannot enter text context but is eligible for first-class raw resource routing');
ok(!stage.includes('_readAttachmentPrefixBytes') && !stage.includes('_ensurePdfSignature'),'metadata staging performs zero PDF byte reads');
ok(prefix.includes('file.slice(0, cap)'),'signature reader slices only bounded prefix');
ok(verify.includes("bytes[0] === 0x25") && verify.includes("bytes[4] === 0x2d"),'signature verifier requires %PDF- magic bytes');
ok(!render.includes('createElement(\'iframe\')') && !render.includes('frame.src = pdfUrl'),'PDF preview no longer embeds a native PDF iframe');
ok(render.includes('Inline native PDF plug-ins are intentionally not embedded'),'UI explains the cross-browser/freeze safety choice');
ok(render.includes('navigator.pdfViewerEnabled === false'),'browser viewer capability hint is surfaced when available');
ok(render.includes('This preview does not extract PDF text.') && render.includes('governed separately by the resource router'),'UI separates local preview from model resource delivery');
ok(render.includes('Open the verified local file in a separate browser tab, or download it.'),'verified PDFs use explicit top-level viewer or download');
ok(src.includes('var _ATTACHMENT_PDF_OPEN_URL_TTL_MS = 60 * 1000'),'new-tab blob URL has bounded lifetime');
ok(render.includes("item.previewError === 'ATTACHMENT_PDF_SIGNATURE'"),'spoofed PDF gets dedicated blocked state');
ok(extract('_openLocalPdf').includes('_openVerifiedLocalPdf(item)') && extract('_openLocalPdf').includes('_ensurePdfSignature(item)'),'explicit open stays signature-gated and synchronous after verification');
ok(css.includes('.ai-assistant-panel-attachment-preview-pdf-safe'),'non-iframe PDF safety panel styled');
ok(css.includes('.ai-assistant-panel-attachment-preview-pdf-actions'),'Open/Download action group styled');

// Runtime signature checks use Blob slices and never read beyond eight bytes.
globalThis._ATTACHMENT_PDF_SIGNATURE_BYTES=8;
globalThis._readAttachmentPrefixBytes=(0,eval)('('+prefix+')');
globalThis._ensurePdfSignature=(0,eval)('('+verify+')');
let maxSlice=0;
const validBytes=new Uint8Array([0x25,0x50,0x44,0x46,0x2d,0x31,0x2e,0x37,0x0a,0x58,0x58]);
const validBlob=new Blob([validBytes],{type:'application/pdf'});
const observed={size:validBlob.size,slice(a,b){maxSlice=Math.max(maxSlice,b||0);return validBlob.slice(a,b)}};
const validItem={kind:'pdf',file:observed,pdfVerified:false,previewError:''};
const isValid=await _ensurePdfSignature(validItem);
ok(isValid===true && validItem.pdfVerified===true,'valid %PDF- file passes local signature gate');
ok(maxSlice===8,'valid PDF verifier reads exactly the 8-byte prefix cap');
const fakeBlob=new Blob([new TextEncoder().encode('NOTPDF plain text')],{type:'application/pdf'});
const fakeItem={kind:'pdf',file:fakeBlob,pdfVerified:false,previewError:''};
const fakeValid=await _ensurePdfSignature(fakeItem);
ok(fakeValid===false && fakeItem.previewError==='ATTACHMENT_PDF_SIGNATURE','fake application/pdf is rejected by magic bytes');

// Runtime staging: .pdf with misleading text MIME is still never treated as sendable text.
const stageFactory=new Function(`
 var _attachmentStageGeneration=0,_attachmentMutationRevision=0,_ATTACHMENT_MAX_FILES=256,_composerAttachments=[];
 function _attachmentSafeName(v){return String(v||'file').slice(0,240)}
 ${candidate}
 function _attachmentIsImage(){return false} function _attachmentIsRasterPreview(){return false}
 function _attachmentIsText(){return true} function _attachmentClassifyMetadata(file){return _attachmentIsPdfCandidate(file)?{kind:'pdf',modality:'document',localOnly:false,sendEligible:false,rawEligible:true}:{kind:'text',modality:'text',localOnly:false,sendEligible:true,rawEligible:true}} function _renderComposerAttachments(){} function showNotification(){}
 ${stage}
 return async function(file){await _stageComposerFiles([file],0);return _composerAttachments[0]};
`);
const staged=await stageFactory()({name:'report.pdf',type:'text/plain',size:1024});
ok(staged.kind==='pdf' && staged.localOnly===false && staged.sendEligible===false && staged.rawEligible===true,'misleading text MIME + .pdf cannot leak into text context but remains raw-resource eligible');
ok(staged.pdfVerified===false,'staging does not pretend PDF has already been verified');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
