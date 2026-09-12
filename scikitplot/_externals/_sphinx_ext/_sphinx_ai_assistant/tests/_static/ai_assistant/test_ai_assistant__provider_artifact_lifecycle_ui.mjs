// Run 144 — provider-generated artifact lifecycle UX/security boundary.
import fs from 'node:fs';

const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3],'utf8');
let pass=0,fail=0;
function ok(cond,msg){if(cond){pass++;console.log('ok',pass,'-',msg)}else{fail++;console.error('not ok -',msg)}}
function extract(name){
  const needle='function '+name+'(';
  const i=src.indexOf(needle); if(i<0)throw new Error('missing '+name);
  const b=src.indexOf('{',i); let d=0,q=null,esc=false,line=false,block=false;
  for(let j=b;j<src.length;j++){
    const c=src[j],n=src[j+1];
    if(line){if(c==='\n')line=false;continue}
    if(block){if(c==='*'&&n==='/'){block=false;j++}continue}
    if(q){if(esc){esc=false;continue}if(c==='\\'){esc=true;continue}if(c===q)q=null;continue}
    if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}
    if(c==='"'||c==="'"||c==='`'){q=c;continue}
    if(c==='{')d++; else if(c==='}'&&--d===0)return src.slice(i,j+1);
  }
  throw new Error('unterminated '+name);
}

ok(src.includes("var _PROVIDER_ARTIFACT_LIFECYCLE_CONTRACT = 'scikitplot-provider-artifact-lifecycle-v1'"),'lifecycle contract is explicit');
ok(src.includes("var _PROVIDER_ARTIFACT_CANCEL_CONTRACT = 'scikitplot-provider-artifact-cancel-v1'"),'cancel contract is explicit');
ok(/function _zipEditProviderArtifactCapabilityParse[\s\S]*?raw\.version !== 1 && raw\.version !== 2/.test(src),'legacy v1 and lifecycle v2 are deliberately distinguished');
ok(/function _zipEditProviderArtifactCapabilityParse[\s\S]*?raw\.lifecycle_contract !== _PROVIDER_ARTIFACT_LIFECYCLE_CONTRACT[\s\S]*?raw\.cancel_contract !== _PROVIDER_ARTIFACT_CANCEL_CONTRACT/.test(src),'v2 discovery requires exact lifecycle and cancel contracts');
ok(/function _zipEditProviderArtifactCapabilityParse[\s\S]*?cancel_endpoint[\s\S]*?\/\^\\\/[A-Za-z0-9\/_-]/.test(src),'cancel endpoint is restricted to a relative path');
ok(/function _zipEditProviderArtifactCapabilityParse[\s\S]*?candidate_ttl_seconds[\s\S]*?duplicate_window_seconds/.test(src),'candidate TTL and duplicate window are capability-bound');
ok(/function _zipEditProviderArtifactCapabilityParse[\s\S]*?24 \* 60 \* 60/.test(src),'client caps server-advertised lifecycle TTL at one day');
ok(/function _zipEditRandomHex[\s\S]*?crypto\.getRandomValues/.test(src),'browser lifecycle ids use cryptographic randomness');
ok(/function _zipEditRandomHex[\s\S]*?ZIP_EDIT_PROVIDER_LIFECYCLE_UNAVAILABLE/.test(src),'missing cryptographic RNG fails closed');
ok(/async function _zipEditStageProviderReplacement[\s\S]*?cancel_token=cancelToken[\s\S]*?dedupe_key=entry\.providerArtifactDedupeKey/.test(src),'generation request carries cancel capability plus per-target opaque dedupe key');
ok(/async function _zipEditStageProviderReplacement[\s\S]*?regenerate_of=regenerateOf/.test(src),'explicit regeneration binds the predecessor lifecycle');
ok(!extract('_zipEditStageProviderReplacement').includes('provider_artifact_id'),'generation request still cannot grant ZIP provenance/write authority itself');
ok(/async function _zipEditStageProviderReplacement[\s\S]*?if \(st\.providerGeneration\) throw new Error\('ZIP_EDIT_PROVIDER_DUPLICATE'\)/.test(src),'browser suppresses concurrent duplicate generation locally');
ok(/async function _zipEditCancelProviderGeneration[\s\S]*?_PROVIDER_ARTIFACT_CANCEL_CONTRACT[\s\S]*?cancel_token:current\.cancelToken/.test(src),'cancel action uses separate ephemeral cancellation contract');
ok(/async function _zipEditCancelProviderGeneration[\s\S]*?st\.abortController\.abort/.test(src),'cancel action also aborts local generation transport');
ok(/function _closeZipEditLayer[\s\S]*?_zipEditCancelProviderGeneration\(true\)/.test(src),'closing ZIP workflow cancels in-flight provider generation');
ok(src.includes('Cancel generation'),'generation has explicit visible cancellation action');
ok(/function _zipEditProviderCandidateExpired[\s\S]*?Date\.now\(\)\/1000/.test(src),'browser candidate expiry uses receipt deadline');
ok(/function _zipEditExpireProviderCandidates[\s\S]*?row\.blob=null[\s\S]*?row\.originalBlob=null/.test(src),'expiry releases browser-resident generated and preview-source bytes');
ok(/function _zipEditScheduleProviderExpiry[\s\S]*?setTimeout/.test(src),'candidate expiry is actively scheduled rather than checked only at apply');
ok(/function _zipEditProviderArtifactReceipt[\s\S]*?doc\.state!=='ready'[\s\S]*?expires-created>Number\(caps\.candidateTtlSeconds\)\+5/.test(src),'receipt requires ready state and bounded server TTL');
ok(/function _zipEditProviderArtifactReceipt[\s\S]*?regeneration!==expected/.test(src),'receipt regeneration lineage must match the browser request');
ok(/source==='provider-binary'\) return row\.readerAccepted===true[\s\S]*?row\.lifecycleState!=='applied'/.test(extract('_zipEditAcceptedProposals')),'applied provider candidate cannot re-enter accepted write set');
ok(/function _zipEditProviderLifecycleEvent[\s\S]*?lifecycleState==='applied'[\s\S]*?observedAfterApply:true/.test(src),'applied lifecycle is terminal even if later UI events occur');
ok(/proposal\.source==='provider-binary'[\s\S]*?proposal\.lifecycleState==='applied'/.test(extract('_renderZipEditProposal')),'applied provider checkbox is disabled');
ok(/Download candidate[\s\S]*?proposal\.lifecycleState==='applied'/.test(extract('_renderZipEditProposal')),'applied candidate cannot be downloaded in a way that revives lifecycle state');
ok(/Regenerate[\s\S]*?_zipEditStageProviderReplacement\(entry,true\)/.test(extract('_renderZipEditProposal')),'regeneration is an explicit action creating a new lifecycle');
ok(/provider_artifact_id/.test(extract('_zipEditApply')),'ZIP manifest carries lifecycle id only at final reviewed apply');
ok(/providerArtifactIds\.push\(providerArtifactId\)/.test(extract('_zipEditApply')),'final apply collects provider lifecycle ids independently from archive paths');
ok(/function _zipEditReceipt[\s\S]*?provider_artifact_ids/.test(src),'browser validates lifecycle correlation returned by ZIP receipt');
ok(/accepted\.forEach[\s\S]*?readerAccepted=false[\s\S]*?'applied'/.test(extract('_zipEditApply')),'successful ZIP application locally makes exact provider candidate non-replayable');
ok(src.includes('provider lifecycle receipt'),'success status surfaces provenance correlation without exposing prompts or bytes');
ok(/lifecycleEvents\.length>8/.test(extract('_zipEditProviderLifecycleEvent')),'browser lifecycle event history is bounded');
ok(!src.includes('localStorage.setItem("providerArtifact'),'lifecycle state is not persisted to localStorage');
ok(css.includes('.ai-assistant-panel-zip-edit-foot'),'lifecycle controls remain inside bounded ZIP editor footer');

const capFactory=new Function(`
 var _PROVIDER_ARTIFACT_SERVER_CONTRACT='scikitplot-provider-artifact-output-v1';
 var _PROVIDER_ARTIFACT_RECEIPT_CONTRACT='scikitplot-provider-artifact-output-receipt-v1';
 var _PROVIDER_ARTIFACT_LIFECYCLE_CONTRACT='scikitplot-provider-artifact-lifecycle-v1';
 var _PROVIDER_ARTIFACT_CANCEL_CONTRACT='scikitplot-provider-artifact-cancel-v1';
 ${extract('_zipEditProviderArtifactCapabilityParse')}
 return _zipEditProviderArtifactCapabilityParse;
`);
const parseCaps=capFactory();
function health(overrides={}){
 const raw={version:2,contract:'scikitplot-provider-artifact-output-v1',receipt_contract:'scikitplot-provider-artifact-output-receipt-v1',lifecycle_contract:'scikitplot-provider-artifact-lifecycle-v1',cancel_contract:'scikitplot-provider-artifact-cancel-v1',endpoint:'/v1/artifacts/provider-output',cancel_endpoint:'/v1/artifacts/provider-output/cancel',candidate_ttl_seconds:900,duplicate_window_seconds:90,max_request_bytes:65536,max_prompt_chars:12000,max_output_bytes:67108864,chat_text_is_output_authority:false,resource_input_is_output_authority:false,generators:[{id:'openai/gpt-image-2',provider:'openai',model:'gpt-image-2',kind:'image',mime_types:['image/png'],max_output_bytes:16777216,diagnostic:false}]};
 Object.assign(raw,overrides); return {capabilities:{provider_artifact_output:raw}};
}
let c=parseCaps(health(),'https://proxy.example');
ok(c&&c.lifecycleEnabled===true,'valid v2 capability enables lifecycle client path');
ok(c&&c.cancelEndpoint==='https://proxy.example/v1/artifacts/provider-output/cancel','cancel endpoint is resolved under discovered health origin only');
ok(c&&c.candidateTtlSeconds===900&&c.duplicateWindowSeconds===90,'bounded lifecycle timing survives normalization');
ok(parseCaps(health({lifecycle_contract:'evil'}),'https://proxy.example')===null,'wrong lifecycle contract fails discovery');
ok(parseCaps(health({cancel_contract:'evil'}),'https://proxy.example')===null,'wrong cancel contract fails discovery');
ok(parseCaps(health({cancel_endpoint:'https://evil.example/cancel'}),'https://proxy.example')===null,'absolute/cross-origin cancel endpoint fails closed');
ok(parseCaps(health({candidate_ttl_seconds:0}),'https://proxy.example')===null,'zero TTL fails lifecycle discovery');
ok(parseCaps(health({duplicate_window_seconds:901}),'https://proxy.example')===null,'duplicate window cannot exceed candidate TTL');
let legacy=parseCaps(health({version:1,lifecycle_contract:undefined,cancel_contract:undefined,cancel_endpoint:undefined,candidate_ttl_seconds:undefined,duplicate_window_seconds:undefined}),'https://proxy.example');
ok(legacy&&legacy.lifecycleEnabled===false&&legacy.cancelEndpoint==='','legacy v1 output remains readable without inventing lifecycle authority');

const acceptedFactory=new Function(`
 var _zipEditState={proposals:[],localReplacements:[]};
 function _zipEditProviderCandidateExpired(row){return !!row.expired;}
 ${extract('_zipEditAcceptedProposals')}
 return {state:_zipEditState,accepted:_zipEditAcceptedProposals};
`);
const a=acceptedFactory();
a.state.localReplacements=[{source:'provider-binary',readerAccepted:true,lifecycleState:'accepted',expired:false},{source:'provider-binary',readerAccepted:true,lifecycleState:'applied',expired:false},{source:'provider-binary',readerAccepted:true,lifecycleState:'accepted',expired:true}];
ok(a.accepted().length===1,'runtime accepted-set filter excludes expired and already-applied provider candidates');

const eventFactory=new Function(`
 ${extract('_zipEditProviderLifecycleEvent')}
 return _zipEditProviderLifecycleEvent;
`);
const event=eventFactory();
let row={source:'provider-binary',lifecycleState:'accepted',lifecycleEvents:[]}; event(row,'applied'); event(row,'downloaded');
ok(row.lifecycleState==='applied','runtime lifecycle event cannot revive terminal applied state');
ok(row.lifecycleEvents.at(-1).observedAfterApply===true,'post-apply UI observation is logged without state mutation');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
