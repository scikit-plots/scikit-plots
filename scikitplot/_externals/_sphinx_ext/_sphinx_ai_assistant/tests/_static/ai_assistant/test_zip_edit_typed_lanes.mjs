// Run 141 — typed ZIP edit lanes: passive SVG editing + capability-gated read-only media references.
import fs from 'node:fs';
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

function sliceFunction(name,nextName){
  const i=src.indexOf('    function '+name+'(');
  const j=src.indexOf('    '+nextName,i);
  if(i<0||j<0)throw new Error('missing slice '+name);
  return src.slice(i,j);
}
const svgFn=sliceFunction('_zipEditValidateSvgText','async function _zipEditReadAuthorizedText');

ok(src.includes('var _ZIP_EDIT_MAX_SVG_FILE_BYTES = 64 * 1024'),'SVG edit source has independent 64 KiB input ceiling');
ok(src.includes('var _ZIP_EDIT_MAX_IMAGE_REFERENCE_BYTES = 8 * 1024 * 1024'),'image reference lane has independent client byte ceiling');
ok(src.includes('var _ZIP_EDIT_MAX_AUDIO_REFERENCE_BYTES = 16 * 1024 * 1024'),'audio reference lane has independent client byte ceiling');
ok(src.includes('var _ZIP_EDIT_MAX_VIDEO_REFERENCE_BYTES = 32 * 1024 * 1024'),'video reference lane has independent client byte ceiling');
ok(/async function _openZipEditWorkflow[\s\S]*?_resourceTransportDiscover\(st\.target\.endpoint, st\.target\.model\)/.test(src),'typed media lanes are capability-discovered for the exact active model');
ok(/async function _zipEditAskModel[\s\S]*?resources: referenceRows\.map\(_resourceDescriptorForWire\)/.test(src),'raw media references use the existing first-class resource descriptor contract');
ok(/async function _zipEditAskModel[\s\S]*?_buildResourceFormData\(bodyObj, referenceRows\)/.test(src),'media references use bounded multipart resource transport rather than JSON/base64');
ok(/async function _zipEditApply[\s\S]*?selectedStats\.editable\.map/.test(src),'server authorization is reconstructed from editable selections only');
ok(/function _zipEditBuildUserMessage[\s\S]*?use only the fN editable ids[\s\S]*?mN media ids are reference-only/i.test(src),'model instructions separate read-only mN media from editable fN ids');
ok(src.includes('Media mN ids never enter server authorization'),'UI explains that media visibility is not mutation authority');
ok(src.includes('Media bytes are read-only references and are not text-redacted by the local privacy preflight.'),'consent accurately distinguishes binary media egress from text privacy redaction');
ok(css.includes('.ai-assistant-panel-zip-edit-row[data-lane="svg"]'),'SVG lane receives explicit UI treatment');
ok(css.includes('.ai-assistant-panel-zip-edit-row[data-lane="media-reference"]'),'media-reference lane receives explicit UI treatment');

// Pure lane classification: text/SVG are editable, raw media requires executable exact-model capability.
const laneFactory=new Function(`
 var _ZIP_EDIT_MAX_AI_FILE_BYTES=32*1024,_ZIP_EDIT_MAX_SVG_FILE_BYTES=64*1024,
     _ZIP_EDIT_MAX_IMAGE_REFERENCE_BYTES=8*1024*1024,_ZIP_EDIT_MAX_AUDIO_REFERENCE_BYTES=16*1024*1024,
     _ZIP_EDIT_MAX_VIDEO_REFERENCE_BYTES=32*1024*1024;
 function _formatByteSize(n){return String(n)}
 function _attachmentImportReasonLabel(x){return String(x||'blocked')}
 function _attachmentGuessMimeFromName(name){name=String(name||'').toLowerCase();if(name.endsWith('.png'))return 'image/png';if(name.endsWith('.gif'))return 'image/gif';if(name.endsWith('.mp4'))return 'video/mp4';if(name.endsWith('.wav'))return 'audio/wav';if(name.endsWith('.svg'))return 'image/svg+xml';return ''}
 function _attachmentSafeName(x){return String(x||'')}
 function DOMParser(){}
 function _resourceExecutionAvailable(c,m){return !!(c&&c.execution==='enabled'&&c.models&&c.models[m]&&c.models[m].execution==='enabled')}
 function _resourceRouteCheck(c,m,d){const r=c&&c.models&&c.models[m]&&c.models[m].routes&&c.models[m].routes[d.modality];if(!Array.isArray(r)||!r.includes('native'))return {supported:false,route:'unsupported'};return {supported:true,route:'native',maxFiles:2}}
 ${extract('_zipEditMediaReferenceLimit')}
 ${extract('_zipEditReferenceDescriptor')}
 ${extract('_zipEditEntryEligibility')}
 return _zipEditEntryEligibility;
`);
const eligible=laneFactory();
const caps={execution:'enabled',maxFileBytes:64*1024*1024,models:{vision:{execution:'enabled',routes:{image:['native'],animated_image:['native'],audio:['native'],video:['native']}}}};
let e=eligible({kind:'text',size:100,selectable:true,rejected:false,name:'a.py',relativePath:'a.py'},null,'vision');
ok(e.ok&&e.editable&&!e.reference&&e.lane==='text','UTF-8 text remains editable without media capability dependency');
e=eligible({kind:'vector_image',size:1000,selectable:true,rejected:false,name:'logo.svg',relativePath:'img/logo.svg'},null,'vision');
ok(e.ok&&e.editable&&e.lane==='svg','SVG receives independent editable source lane');
e=eligible({kind:'image',type:'image/png',size:1000,selectable:true,rejected:false,name:'plot.png',relativePath:'img/plot.png'},null,'vision');
ok(!e.ok&&!e.editable&&!e.reference,'raster media fails closed when resource capability is absent');
e=eligible({kind:'image',type:'image/png',size:1000,selectable:true,rejected:false,name:'plot.png',relativePath:'img/plot.png'},caps,'vision');
ok(e.ok&&!e.editable&&e.reference&&e.modality==='image'&&e.route==='native','supported raster image is read-only model reference, never editable');
e=eligible({kind:'image',type:'image/gif',size:1000,selectable:true,rejected:false,name:'anim.gif',relativePath:'img/anim.gif'},caps,'vision');
ok(e.ok&&e.reference&&e.modality==='animated_image','GIF reference preserves animated-image modality');
e=eligible({kind:'video',type:'video/mp4',size:33*1024*1024,selectable:true,rejected:false,name:'clip.mp4',relativePath:'clip.mp4'},caps,'vision');
ok(!e.ok&&e.lane==='media-reference','video reference above client lane ceiling fails closed');

// SVG source/replacement policy is deliberately passive and text-reviewable.
const svgFactory=new Function(`function DOMParser(){this.parseFromString=function(){return {documentElement:{localName:'svg'},querySelector:function(){return null}}}} ${svgFn} return _zipEditValidateSvgText;`);
const svgPolicy=svgFactory();
ok(svgPolicy('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h1v1z"/></svg>').startsWith('<svg'),'passive SVG is accepted');
ok(svgPolicy('<svg/>')==='<svg/>','self-closing passive SVG is accepted');
ok(svgPolicy('<svg><defs><linearGradient id="g"/></defs><rect fill="url(#g)"/></svg>').startsWith('<svg'),'passive SVG permits local fragment references');
ok(svgPolicy('<svg><image href="data:image/png;base64,iVBORw0KGgo="/></svg>').startsWith('<svg'),'passive SVG permits embedded raster data only');
for(const [payload,label] of [
  ['<svg><script>alert(1)</script></svg>','script'],
  ['<svg onclick="x()"></svg>','event handler'],
  ['<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/passwd">]><svg/>','XML entity/doctype'],
  ['<svg><image href="https://evil.example/a.png"/></svg>','external network href'],
  ['<svg><foreignObject><div>html</div></foreignObject></svg>','foreign HTML'],
  ['<svg><animate attributeName="href" to="https://evil.example/x"/></svg>','SMIL mutation'],
  ['<?xml-stylesheet href="https://evil.example/a.css"?><svg/>','external XML stylesheet'],
  ['<svg><image href="icons/a.png"/></svg>','relative external href']
]){
  let blocked=false;try{svgPolicy(payload)}catch(_){blocked=true}
  ok(blocked,'passive SVG policy rejects '+label);
}

const proposalFactory=new Function(`
 var _ZIP_EDIT_TEXT_PROPOSAL_CONTRACT='scikitplot-zip-text-proposal-v1';
 function DOMParser(){this.parseFromString=function(){return {documentElement:{localName:'svg'},querySelector:function(){return null}}}}
 var _ZIP_EDIT_MAX_PROPOSAL_ENTRY_BYTES=256*1024,_ZIP_EDIT_MAX_PROPOSAL_TOTAL_BYTES=512*1024,_ZIP_EDIT_MAX_AI_PATHS=8,_ZIP_EDIT_DIFF_PREVIEW_CHARS=128*1024,_CHAT_RESPONSE_MAX_BYTES=8*1024*1024;
 ${svgFn}
 ${extract('_zipEditNormalizeReplacementText')}
 ${extract('_zipEditReplacementBlob')}
 ${extract('_zipEditBoundedDiff')}
 ${extract('_zipEditStrictProposal')}
 return _zipEditStrictProposal;
`);
const strict=proposalFactory();
const svgSource=[{proposalId:'f1',path:'img/logo.svg',lane:'svg',text:'<svg><path d="M0 0"/></svg>\n',hadBom:false,newline:'\n'}];
const good=strict(JSON.stringify({contract:'scikitplot-zip-text-proposal-v1',replacements:[{id:'f1',content:'<svg><path d="M1 1"/></svg>\n'}]}),svgSource);
ok(good.length===1&&good[0].lane==='svg'&&good[0].path==='img/logo.svg','strict proposal accepts passive SVG replacement under opaque editable id');
let activeBlocked=false;try{strict(JSON.stringify({contract:'scikitplot-zip-text-proposal-v1',replacements:[{id:'f1',content:'<svg onload="alert(1)"></svg>\n'}]}),svgSource)}catch(e){activeBlocked=e.message==='ZIP_EDIT_SVG_ACTIVE_CONTENT'}
ok(activeBlocked,'strict proposal blocks active SVG generated by model before review/application');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
