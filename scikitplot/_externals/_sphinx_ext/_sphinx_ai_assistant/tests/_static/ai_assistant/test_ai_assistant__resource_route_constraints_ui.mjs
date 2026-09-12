// Run 134 — route-specific provider limits/MIME constraints are browser preflight authority.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0;function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}
const names=['_resourceRouteConstraintsParse','_resourceTransportCapsParse','_resourceSelectedRoute','_resourceRouteCheck','_resourceRouteSupported','_resourceModelCapsParse'];
const parts=names.map(extract).join('\n');
const factory=new Function(`
 var _RESOURCE_MODALITIES=['text','image','animated_image','vector_image','audio','video','document','archive','data','binary'];
 var _RESOURCE_ROUTES=['native','tool','extract','context','unsupported'];
 var _RESOURCE_AUTO_ROUTE_ORDER=['native','tool','extract','context'];
 var _RESOURCE_MIME_RE=/^[A-Za-z0-9!#$&^_.+*-]{1,80}\\/[A-Za-z0-9!#$&^_.+*-]{1,80}$/;
 function _safeInt(value,min,max,fallback){var n=Number(value);if(!Number.isFinite(n))return fallback;n=Math.floor(n);return Math.max(min,Math.min(max,n));}
 ${parts}
 return {parse:_resourceTransportCapsParse,model:_resourceModelCapsParse,selected:_resourceSelectedRoute,check:_resourceRouteCheck,supported:_resourceRouteSupported};
`);
const h=factory();
const MiB=1024*1024;
const doc={version:3,multipart:true,execution:'enabled',max_files:256,max_file_bytes:512*MiB,max_total_bytes:1024*MiB,models:{
 'gemini-3.8-flash':{adapter:'gemini',execution:'enabled',routes:{
   text:['tool','context'],video:['native'],document:['native'],archive:['tool'],data:['tool']
 },route_constraints:{
   text:{tool:{max_file_bytes:100*MiB}},
   document:{native:{max_file_bytes:50*MiB,mime_types:['application/pdf']}},
   archive:{tool:{max_file_bytes:100*MiB,mime_types:['application/zip']}},
   data:{tool:{max_file_bytes:100*MiB,mime_types:['application/vnd.ms-excel','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']}},
   video:{native:{max_file_bytes:'not-a-number',mime_types:['bad mime','video/mp4']}}
 }}
}};
const caps=h.parse(doc), model=caps.models['gemini-3.8-flash'];
ok(!!caps && caps.version===3,'v3 resource capability document parses');
ok(model.routeConstraints.archive.tool.maxFileBytes===100*MiB,'archive tool per-route size limit preserved');
ok(model.routeConstraints.archive.tool.mimeTypes[0]==='application/zip','archive MIME allowlist preserved');
ok(model.routeConstraints.video.native.mimeTypes.length===1 && model.routeConstraints.video.native.mimeTypes[0]==='video/mp4','malformed MIME/size constraint members are discarded independently');
ok(h.selected(caps,'gemini-3.8-flash',{modality:'archive',intent:'auto'})==='tool','auto route selection mirrors server native/tool/extract/context order');
ok(h.selected(caps,'gemini-3.8-flash',{modality:'text',intent:'context'})==='context','explicit context remains explicit');
ok(h.check(caps,'gemini-3.8-flash',{modality:'archive',intent:'raw',size:99*MiB,mime_type:'application/zip'}).supported,'99 MiB ZIP accepted before upload');
let c=h.check(caps,'gemini-3.8-flash',{modality:'archive',intent:'raw',size:101*MiB,mime_type:'application/zip'});
ok(!c.supported && c.reason==='size' && c.maxFileBytes===100*MiB,'101 MiB ZIP rejected by File Search route limit');
c=h.check(caps,'gemini-3.8-flash',{modality:'data',intent:'auto',size:1,mime_type:'application/vnd.apache.parquet'});
ok(!c.supported && c.reason==='mime','Parquet is rejected although generic data modality has a tool route');
ok(h.supported(caps,'gemini-3.8-flash',{modality:'data',intent:'auto',size:1,mime_type:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}),'XLSX accepted by exact File Search MIME allowlist');
ok(h.supported(caps,'gemini-3.8-flash',{modality:'video',intent:'auto',size:150*MiB,mime_type:'video/mp4'}),'150 MiB MP4 is not incorrectly capped by File Search 100 MiB limit');
ok(!h.supported(caps,'gemini-3.8-flash',{modality:'video',intent:'auto',size:1,mime_type:'video/quicktime'}),'route MIME constraint is fail-closed when provider vocabulary is explicit');
const onDemand=h.model(doc.models['gemini-3.8-flash']);
ok(onDemand.routeConstraints.data.tool.mimeTypes.length===2,'on-demand model capability parser preserves route constraints');
const bad=h.model({adapter:'gemini',execution:'enabled',routes:{archive:['tool']},route_constraints:{archive:{native:{max_file_bytes:1},tool:{mime_types:['javascript:evil','application/zip','application/zip']}}}});
ok(!bad.routeConstraints.archive.native && bad.routeConstraints.archive.tool.mimeTypes.length===1,'constraints cannot introduce unavailable routes and MIME list is sanitized/deduplicated');
const api=extract('_panelApiCall');
const preflight=api.indexOf('var routeCheck = _resourceRouteCheck(resourceCaps, modelName, row)');
const form=api.indexOf('var resourceForm = _buildResourceFormData');
ok(preflight>=0 && form>preflight,'route size/MIME preflight occurs before FormData construction');
ok(api.includes("new Error('AI_RESOURCE_ROUTE_SIZE_LIMIT')"),'route-specific size failure has distinct bounded diagnostic');
ok(api.includes("new Error('AI_RESOURCE_ROUTE_MIME_UNSUPPORTED')"),'route-specific MIME failure has distinct bounded diagnostic');
console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
