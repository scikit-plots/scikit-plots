// Run 127 — browser trusts proxy resource capability discovery, never provider/model-name guesses.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0;function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}
const constraintSrc=extract('_resourceRouteConstraintsParse');
const parseSrc=extract('_resourceTransportCapsParse');
const selectedSrc=extract('_resourceSelectedRoute');
const checkSrc=extract('_resourceRouteCheck');
const routeSrc=extract('_resourceRouteSupported');
const execSrc=extract('_resourceExecutionAvailable');
const factory=new Function(`
 var _RESOURCE_MODALITIES=['text','image','animated_image','vector_image','audio','video','document','archive','data','binary'];
 var _RESOURCE_ROUTES=['native','tool','extract','context','unsupported'];
 var _RESOURCE_AUTO_ROUTE_ORDER=['native','tool','extract','context'];
 var _RESOURCE_MIME_RE=/^[A-Za-z0-9!#$&^_.+*-]{1,80}\\/[A-Za-z0-9!#$&^_.+*-]{1,80}$/;
 function _safeInt(value,min,max,fallback){var n=Number(value);if(!Number.isFinite(n))return fallback;n=Math.floor(n);return Math.max(min,Math.min(max,n));}
 ${constraintSrc}\n${parseSrc}\n${selectedSrc}\n${checkSrc}\n${routeSrc}\n${execSrc}
 return {parse:_resourceTransportCapsParse,supported:_resourceRouteSupported,exec:_resourceExecutionAvailable};
`);
const h=factory();
const doc={version:2,multipart:true,execution:'enabled',model_capability_endpoint:'/v1/resource-capabilities',max_files:80,max_file_bytes:123456,max_total_bytes:987654,models:{
 'openai/gpt-oss-120b':{adapter:'huggingface',execution:'plan-only',routes:{text:['context'],image:['unsupported'],video:['unsupported'],archive:['unsupported']}},
 'gemini-video-model':{adapter:'gemini',execution:'plan-only',routes:{text:['context'],image:['native'],audio:['native'],video:['native'],document:['native'],archive:['tool','extract']}},
 'custom-model':{adapter:'custom<script>',execution:'enabled',routes:{binary:['bogus','tool','tool','native','extract','context','unsupported']}}
}};
const caps=h.parse(doc);
ok(!!caps && caps.version===2 && caps.multipart===true,'v2 multipart resource capability document parses');
ok(caps.modelCapabilityEndpoint==='/v1/resource-capabilities','model capability endpoint is sanitized and preserved');
ok(caps.maxFiles===80 && caps.maxFileBytes===123456 && caps.maxTotalBytes===987654,'server-advertised resource limits survive bounded parser');
ok(caps.models['openai/gpt-oss-120b'].adapter==='huggingface','provider authority comes from health document, not openai model namespace');
ok(caps.models['openai/gpt-oss-120b'].routes.video[0]==='unsupported','HF-routed OpenAI-named model remains unsupported for video unless server opts in');
ok(caps.models['gemini-video-model'].routes.video[0]==='native','server can explicitly advertise native video for a selected model');
ok(caps.models['custom-model'].adapter==='customscript','adapter label is sanitized before UI storage');
ok(caps.models['custom-model'].routes.binary.length===4 && !caps.models['custom-model'].routes.binary.includes('bogus'),'route lists accept only known routes and stay bounded');
ok(h.exec(caps,'gemini-video-model')===false,'plan-only model routes are not executable');
ok(h.exec(caps,'custom-model')===true,'enabled model under enabled transport is executable');
ok(h.supported(caps,'gemini-video-model',{modality:'video',intent:'auto'})===true,'auto accepts any non-unsupported server route');
ok(h.supported(caps,'gemini-video-model',{modality:'archive',intent:'raw'})===true,'explicit raw accepts native/tool route');
ok(h.supported(caps,'gemini-video-model',{modality:'archive',intent:'extract'})===true,'explicit extract requires extract route');
ok(h.supported(caps,'gemini-video-model',{modality:'archive',intent:'context'})===false,'explicit context does not silently become tool/extract');
ok(h.supported(caps,'openai/gpt-oss-120b',{modality:'image',intent:'auto'})===false,'unsupported health route blocks upload before provider call');
ok(h.parse({version:2,multipart:true,execution:'bogus',models:{}})===null,'unknown execution state fails closed');
ok(h.parse({version:1,multipart:true,execution:'enabled',models:{}})===null,'legacy/ambiguous resource capability versions fail closed');
ok(h.parse({version:2,multipart:false,models:{}})===null,'resource transport requires explicit multipart support');
ok(h.parse({version:2,multipart:true,models:null})===null,'malformed model capability map fails closed');
const discover=extract('_resourceTransportDiscover');
ok(discover.includes("origin + '/health'") && discover.includes("credentials: 'omit'") && discover.includes("cache: 'no-store'"),'resource discovery uses credential-free no-store /health request');
ok(discover.includes('_readResponseTextBounded') && discover.includes('_CAPS_MAX_BYTES'),'resource discovery bounds health response bytes');
ok(discover.includes('_CHAT_RESOURCE_KEY_PREFIX') && discover.includes('_CAPS_TTL_MS'),'resource discovery is session-cached with bounded TTL');
ok(discover.includes('modelCapabilityEndpoint') && discover.includes('encodeURIComponent'),'models omitted from bounded health map are discovered on demand without URL injection');
const chat=extract('_chatContractDiscover');
ok(chat.includes('_resourceTransportCapsParse') && chat.includes('_CHAT_RESOURCE_KEY_PREFIX'),'chat-contract discovery caches resource capability document from same health response');
const api=extract('_panelApiCall');
const discoverAt=api.indexOf('await _resourceTransportDiscover(endpoint, modelName)');
const formAt=api.indexOf('var resourceForm = _buildResourceFormData');
ok(discoverAt>=0 && formAt>discoverAt,'capability/limit preflight happens before FormData resource upload');
ok(api.includes("new Error('AI_RESOURCE_EXECUTOR_PENDING')"),'plan-only route is rejected before uploading resource bytes');
ok(api.includes('requestResources.length > resourceCaps.maxFiles'),'panel enforces server-advertised file-count limit');
ok(api.includes('> resourceCaps.maxFileBytes'),'panel enforces server-advertised per-file byte limit');
ok(api.includes('rawTotal > resourceCaps.maxTotalBytes'),'panel enforces server-advertised aggregate byte limit');
ok(api.includes('_resourceRouteCheck(resourceCaps, modelName, row)'),'panel checks exact selected-model modality/intent route and route constraints');
ok(api.includes("new Error('AI_RESOURCE_MODEL_UNSUPPORTED')"),'unsupported model/resource combination fails closed before upload');
ok(!api.includes("provider === 'openai'") || discoverAt>=0,'provider labels never replace resource health discovery');
ok(src.includes("var _RESOURCE_MODALITIES = ['text','image','animated_image','vector_image','audio','video','document','archive','data','binary']"),'browser capability parser covers all Run126 modalities');
console.log(`${pass} passed, ${fail} failed`);if(fail)process.exit(1);
