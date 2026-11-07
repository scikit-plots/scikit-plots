// Run 126 — first-class browser resource classification and multipart transport.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3] || process.argv[2].replace(/ai-assistant\.js$/, 'ai-assistant.css'),'utf8');
let pass=0,fail=0;function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}
const names=['_attachmentSafeName','_attachmentIsRasterPreview','_attachmentIsImage','_attachmentIsPdfCandidate','_attachmentIsText','_attachmentIsZipCandidate','_attachmentGuessMimeFromName','_prepareComposerRawResources','_resourceDescriptorForWire'];
const f=Object.fromEntries(names.map(n=>[n,extract(n)]));
const classStart=src.indexOf('function _attachmentClassifyMetadata(');
const classEnd=src.indexOf('function _readAttachmentBlobSlice(', classStart);
if(classStart<0||classEnd<0)throw new Error('missing classifier boundaries');
f._attachmentClassifyMetadata=src.slice(classStart,classEnd).trim();
const factory=new Function(`
 var _ATTACHMENT_MAX_FILES=256;
 var _ATTACHMENT_TEXT_EXT_RE=/\\.(?:txt|md|markdown|rst|py|pyi|js|mjs|cjs|ts|tsx|jsx|json|jsonl|ipynb|yaml|yml|toml|csv|tsv|xml|html|htm|css|scss|less|ini|cfg|conf|log|sql|sh|bash|zsh|fish|ps1|bat|cmd|c|cc|cpp|cxx|h|hpp|java|kt|kts|go|rs|rb|php|swift|scala|r|jl)$/i;
 var _ATTACHMENT_ZIP_EXT_RE=/\\.zip$/i;
 var _ATTACHMENT_AUDIO_EXT_RE=/\\.(?:mp3|wav|wave|m4a|aac|ogg|oga|flac|opus|aiff|aif|weba)$/i;
 var _ATTACHMENT_VIDEO_EXT_RE=/\\.(?:mp4|m4v|mov|webm|mpeg|mpg|avi|wmv|flv|3gp|3gpp)$/i;
 var _ATTACHMENT_DATA_EXT_RE=/\\.(?:xls|xlsx|ods|parquet|arrow|feather)$/i;
 ${f._attachmentSafeName}\n${f._attachmentIsRasterPreview}\n${f._attachmentIsImage}\n${f._attachmentIsPdfCandidate}\n${f._attachmentIsText}\n${f._attachmentIsZipCandidate}\n${f._attachmentGuessMimeFromName}\n${f._attachmentClassifyMetadata}\n${f._prepareComposerRawResources}\n${f._resourceDescriptorForWire}
 return {classify:_attachmentClassifyMetadata, raw:_prepareComposerRawResources, wire:_resourceDescriptorForWire};
`);
const h=factory();
function cls(name,type=''){return h.classify({name,type},name)}
const matrix=[
 ['photo.jpg','image/jpeg','image','image'],
 ['photo.png','image/png','image','image'],
 ['anim.gif','image/gif','image','animated_image'],
 ['diagram.svg','image/svg+xml','vector_image','vector_image'],
 ['speech.mp3','audio/mpeg','audio','audio'],
 ['movie.mp4','video/mp4','video','video'],
 ['clip.webm','video/webm','video','video'],
 ['paper.pdf','application/pdf','pdf','document'],
 ['project.zip','application/zip','archive','archive'],
 ['sheet.xlsx','','data','data'],
 ['data.parquet','','data','data'],
 ['notes.md','text/markdown','text','text'],
 ['mystery.bin','','file','binary'],
];
for(const [name,type,kind,modality] of matrix){const c=cls(name,type);ok(c.kind===kind && c.modality===modality,`${name} => ${kind}/${modality}`);}
ok(cls('diagram.svg','image/svg+xml').rasterPreview===false,'SVG is not treated as safe raster preview');
ok(matrix.filter(x=>!['notes.md'].includes(x[0])).every(([n,t])=>cls(n,t).rawEligible===true),'non-text modalities are raw-resource eligible');
const items=[
 {name:'notes.md',type:'text/markdown',size:10,kind:'text',modality:'text',rawEligible:true,transportIntent:'context',file:{}},
 {name:'paper.pdf',type:'application/pdf',size:20,kind:'pdf',modality:'document',rawEligible:true,transportIntent:'auto',file:{}},
 {name:'movie.mp4',type:'video/mp4',size:30,kind:'video',modality:'video',rawEligible:true,transportIntent:'raw',file:{}},
 {name:'inside.png',type:'image/png',size:40,kind:'image',modality:'image',rawEligible:true,transportIntent:'auto',relativePath:'assets/inside.png',archiveName:'project.zip',file:{}},
];
const raw=h.raw(items);
ok(raw.length===3,'default text context is not duplicated into raw resource plane');
ok(raw.map(x=>x.id).join(',')==='r0,r1,r2','raw resources get deterministic request-local ids');
ok(raw[0].modality==='document'&&raw[1].modality==='video'&&raw[2].relative_path==='assets/inside.png','resource modality/path metadata survives planner');
ok(raw[2].archive_name==='project.zip','archive provenance survives raw planner');
ok(h.wire(raw[1]).intent==='raw','explicit raw transport intent survives wire descriptor');
const q=extract('_queueComposerFiles');
ok(!q.includes('_queueZipImport'),'top-level ZIP keeps original identity instead of forced extraction');
const body=extract('_buildResourceFormData');
ok(body.includes("form.append('request', JSON.stringify(bodyObj))") && body.includes("form.append('resource:' + row.id"),'multipart contains one JSON control part plus raw resource parts');
const api=extract('_panelApiCall');
ok(api.includes('var resourceForm = _buildResourceFormData') && api.includes('body: resourceForm'),'resource requests use FormData body');
ok(!/Content-Type[^\n]*multipart\/form-data/i.test(api),'browser never authors multipart boundary header manually');
ok(api.includes('if (requestResources.length && !useStructuredProxy)'),'raw resources require trusted bundled proxy contract');
ok(api.includes('if (!requestResources.length && streamingEnabled'),'resource upload avoids unsupported SSE upload path in Run126');
ok(src.includes("media.preload = 'metadata'") && src.includes("item.kind === 'audio' || item.kind === 'video'"),'audio/video local previews are metadata-only until playback');
ok(!src.includes("iframe.className = 'ai-assistant-panel-attachment-preview-pdf'"),'PDF iframe remains retired');
ok(src.includes('Raw SVG is not embedded in the assistant preview'),'SVG preview remains non-executable/local-safe');
ok(/\.ai-assistant-panel-attachment-preview-media\s*\{[\s\S]*max-height:/.test(css),'media preview dimensions are bounded');
ok(src.includes("['text','image','vector_image','audio','video','data','file','replay','page','pdf','archive']"),'turn manifest preserves every Run126 resource kind');
ok(src.includes("delivery: rawIncluded ? 'raw' : (textIncluded ? 'context' : 'not_sent')"),'turn manifest records context/raw/not-sent delivery authority');
ok(src.includes('contextCount: aggregates.contextCount') && src.includes('rawCount: aggregates.rawCount') && src.includes('notSentCount: aggregates.notSentCount'),'manifest aggregates separate context, raw, and not-sent resources');
ok(src.includes("' · context ' + m.contextCount") && src.includes("' · raw ' + m.rawCount") && src.includes("' · not-sent ' + m.notSentCount"),'human exports expose delivery classes instead of ambiguous included/local-only counts');
ok(src.includes("['media', 'Media']") && src.includes("['data', 'Data']"),'Resource Manager can filter new media/data modalities');
ok(src.includes('Raw delivery decided by Resource Router'),'PDF preview distinguishes local preview from raw provider delivery');
console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
