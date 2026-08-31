// Run 8 execution harness for unified Share sheet + artifact lifecycle.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
process.on('uncaughtException', (err) => {
  console.log(`  FAIL runtime construction/execution\n       ${err?.stack || err}`);
  console.log('\n0 passed, 1 failed'); process.exit(1);
});
function extract(name){
  const i=src.indexOf('function '+name+'('); if(i<0) throw new Error('not found '+name);
  let d=0,started=false,q=null,line=false,block=false;
  for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1];
    if(line){if(c==='\n')line=false;continue} if(block){if(c==='*'&&n==='/'){block=false;j++;}continue}
    if(q){if(c==='\\'){j++;continue}if(c===q)q=null;continue}
    if(c==='/'&&n==='/'){line=true;j++;continue} if(c==='/'&&n==='*'){block=true;j++;continue}
    if(c==='"'||c==="'"||c==='`'){q=c;continue} if(c==='{'){d++;started=true}else if(c==='}'){d--;if(started&&d===0)return src.slice(i,j+1)}
  } throw new Error('unbalanced '+name);
}
function extractArray(name){const i=src.indexOf('var '+name+' = [');if(i<0)throw new Error('missing '+name);const st=src.indexOf('[',i);let d=0,q=null;for(let j=st;j<src.length;j++){const c=src[j];if(q){if(c==='\\'){j++;continue}if(c===q)q=null;continue}if(c==='"'||c==="'"){q=c;continue}if(c==='[')d++;else if(c===']'){d--;if(d===0)return src.slice(st,j+1)}}throw new Error('unbalanced array')}
function makeNode(tag){
  const n={tagName:String(tag).toUpperCase(),className:'',id:'',type:'',value:'',textContent:'',innerHTML:'',readOnly:false,disabled:false,checked:false,hidden:false,title:'',style:{},attrs:{},children:[],listeners:{},parentNode:null,
    setAttribute(k,v){this.attrs[k]=String(v)},getAttribute(k){return k in this.attrs?this.attrs[k]:null},removeAttribute(k){delete this.attrs[k]},
    appendChild(c){if(!c||typeof c!=='object')throw new TypeError('append non-node');c.parentNode=this;this.children.push(c);return c},
    removeChild(c){const i=this.children.indexOf(c);if(i>=0){this.children.splice(i,1);c.parentNode=null;}return c},
    addEventListener(e,fn){(this.listeners[e]=this.listeners[e]||[]).push(fn)},focus(){document.activeElement=this},select(){},
    async click(){for(const fn of (this.listeners.click||[]))await fn({type:'click',target:this,preventDefault(){},stopPropagation(){}})},
    dispatch(e,extra={}){const ev=Object.assign({type:e,target:this,preventDefault(){},stopPropagation(){}},extra);for(const fn of(this.listeners[e]||[]))fn(ev)},
    querySelector(sel){return findAll(this,sel)[0]||null},querySelectorAll(sel){return findAll(this,sel)},
  }; Object.defineProperty(n,'firstChild',{get(){return this.children[0]||null}}); return n;
}
function matches(n,s){if(!n?.tagName)return false;if(s.startsWith('.'))return n.className.split(/\s+/).includes(s.slice(1));if(s.startsWith('#'))return n.id===s.slice(1);const a=s.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/);if(a){const v=n.getAttribute(a[1]);return a[2]===undefined?v!==null:v===a[2]}return n.tagName===s.toUpperCase()}
function findAll(r,s){const out=[];for(const c of r.children||[]){if(matches(c,s))out.push(c);out.push(...findAll(c,s))}return out}
function byText(r,t){if(r.textContent===t)return r;for(const c of r.children||[]){const f=byText(c,t);if(f)return f}return null}
const root=makeNode('div'), docListeners={};
globalThis.document={body:root,activeElement:null,createElement:makeNode,createTextNode(t){const n=makeNode('#text');n.textContent=String(t);return n},querySelector(s){return findAll(root,s)[0]||null},querySelectorAll(s){return findAll(root,s)},addEventListener(e,fn){(docListeners[e]=docListeners[e]||[]).push(fn)},dispatchEvent(ev){for(const fn of(docListeners[ev.type]||[]))fn(ev);return true}};
globalThis.CustomEvent=function(type,init){return{type,detail:init?.detail||null}};
let currentConversationId='conv-a', ss={}, blobN=0, revoked=[], opens=[], notices=[], copied=[], downloads=[], legacyClears=0, globalPosts=[], globalPatches=[], globalDeletes=[], globalStatus=200;
globalThis.location={href:'https://docs.example.test/page?secret=x#frag'};
globalThis.window={sessionStorage:{getItem:k=>k in ss?ss[k]:null,setItem:(k,v)=>{ss[k]=String(v)},removeItem:k=>{delete ss[k]}},open:(url)=>{opens.push(url);return{opener:null}}};
globalThis.Blob=function(parts,opt){this.parts=parts;this.type=opt?.type};
const NativeURL=globalThis.URL; NativeURL.createObjectURL=()=> 'blob:fake-'+(++blobN); NativeURL.revokeObjectURL=(u)=>revoked.push(u); globalThis.URL=NativeURL;
globalThis.setTimeout=(fn)=>{fn();return 1};
globalThis.ICONS={exportJson:'<svg/>',exportHtml:'<svg/>',exportTxt:'<svg/>',close:'<svg/>',menu:'<svg/>'};
globalThis._cfg=()=>({panelTitle:'AI Assistant',panelGlobalShareEndpoint:'https://share.example.test/v1/share',panelGlobalShareTtlDays:30,panelTrainingEndpoint:''});
globalThis._getConversationId=()=>currentConversationId;
globalThis._ssGet=k=>k in ss?ss[k]:null; globalThis._ssSet=(k,v)=>{ss[k]=String(v)}; globalThis._ssDel=k=>{delete ss[k]};
globalThis._transcript=[{role:'user',text:'Hello',ts:1},{role:'assistant',text:'World',ts:2}]; globalThis._feedbackStore={};
globalThis._EP={hasProfiles:()=>false,resolveEndpoint:()=>'',resolve:()=>'',resolveToken:()=>''};
globalThis._resolveFlatFeatureEndpoint=(raw)=>raw||'';
globalThis._sanitizePage=()=> 'https://docs.example.test/page';
globalThis._managedConversationArtifacts=[]; globalThis._managedConversationArtifactSeq=0; globalThis._managedConversationArtifactRenderHook=null;
globalThis._registerManagedConversationArtifact=(0,eval)('('+extract('_registerManagedConversationArtifact')+')');
globalThis._forgetManagedConversationArtifact=(0,eval)('('+extract('_forgetManagedConversationArtifact')+')');
globalThis._conversationContentPreset=(0,eval)('('+extract('_conversationContentPreset')+')');
globalThis._normalizeConversationContentOptions=(0,eval)('('+extract('_normalizeConversationContentOptions')+')');
globalThis._buildConversationSnapshot=(options)=>{const o=_normalizeConversationContentOptions(options);return{schema_version:'2.0',session:{page_url:o.includeSafeSourcePage?'https://docs.example.test/page':null,page_title:o.includePageTitle?'Page':null,id:o.includeSessionId?'sid':null},turns:[{turn_index:0,user:{text:'Hello'},assistant:{text:'World'}}],records:[{turn_index:0,message_index:0,role:'user',text:'Hello',ts:o.includeTimestamps?1:null,model_id:null,feedback_rating_value:null,session_id:o.includeSessionId?'sid':null,page_url:o.includeSafeSourcePage?'https://docs.example.test/page':null},{turn_index:0,message_index:1,role:'assistant',text:'World',ts:o.includeTimestamps?2:null,model_id:o.includeModel?'m':null,feedback_rating_value:o.includeRatings?1:null,session_id:o.includeSessionId?'sid':null,page_url:o.includeSafeSourcePage?'https://docs.example.test/page':null}]}};
globalThis._buildConvJsonString=s=>JSON.stringify(s);globalThis._buildConvHtmlString=s=>'<!doctype html>'+JSON.stringify(s);globalThis._buildConvTxtString=s=>s.records.map(r=>r.text).join('\n');globalThis._buildConvYamlString=s=>'yaml:'+JSON.stringify(s);globalThis._buildConvTomlString=s=>'toml:'+JSON.stringify(s);
globalThis._privacyPreflightScan=()=>({flagged:false,control_findings:[]}); let privacyHook=async v=>({action:'continue',value:v,scan:{flagged:false}}); globalThis._privacyPreflightReview=(v,o)=>privacyHook(v,o);
globalThis._utf8ByteLength=(0,eval)('('+extract('_utf8ByteLength')+')'); globalThis._formatByteSize=(0,eval)('('+extract('_formatByteSize')+')');
globalThis._resolveGlobalPublicReadUrl=undefined;
globalThis._strHash=()=> 'hash'; globalThis._isoFileStamp=()=> '20260829';
globalThis._buildPortableSelfContainedDataUrl=(snap,fmt)=>({url:`data:text/html;charset=utf-8;base64,PORTABLE-${fmt}-${snap.records.length}`,content:'<!doctype html><p>portable</p>',mime:'text/html;charset=utf-8',bytes:128,urlChars:96}); globalThis._buildDataUri=(content,mime)=>`data:${mime},${content.length}`;
globalThis._downloadBlob=(content,mime,name)=>downloads.push({content,mime,name}); globalThis._idbClearShares=cb=>{legacyClears++;cb(true,null)};
globalThis._newOperationEnvelope=()=>({operationId:'a'.repeat(32),resourceId:'c'.repeat(32),managementToken:'d'.repeat(43),managementTokenHash:'e'.repeat(64),operationCreatedAt:Date.now()});globalThis._postGlobalShare=(base,token,payload,envelope,ok)=>{const n=globalPosts.length+1,id=n.toString(16).padStart(32,'0');globalPosts.push({base,token,payload,envelope,id});ok({uuid:id,editToken:'edit-secret-'+id,expiresAt:'2026-09-28T00:00:00Z',storage:{durable:true,shared:true}})};
globalThis._patchGlobalShare=(base,id,edit,payload,ok)=>{globalPatches.push({base,id,edit,payload});ok({uuid:id,url:'https://share.example.test/v1/share#share='+id,editToken:edit,expiresAt:'2026-09-28T00:00:00Z'})};
globalThis._deleteGlobalShare=(base,id,edit,ok)=>{globalDeletes.push({base,id,edit});ok({ok:true})};
globalThis._probeGlobalShareStatus=(url,cb)=>cb({status:globalStatus,ok:globalStatus===200});
globalThis._buildModelInfo=()=>({}); globalThis._postTrainingContribution=()=>{}; globalThis._remotePost=()=>{};
globalThis._downloadConversationFormat=(0,eval)('('+extract('_downloadConversationFormat')+')');
globalThis.showNotification=(msg,err)=>notices.push({msg,err}); globalThis.copyToClipboard=v=>copied.push(v);
globalThis._createIconBtn=(suffix,label,icon)=>{const b=makeNode('button');b.id='x-'+suffix;b.setAttribute('aria-label',label);b.innerHTML=icon;return b};globalThis._buildSheetHamburgerBtn=()=>null;
const _EXPORT_FORMATS=(0,eval)('('+extractArray('_EXPORT_FORMATS')+')');globalThis._EXPORT_FORMATS=_EXPORT_FORMATS;globalThis._getExportFormat=(0,eval)('('+extract('_getExportFormat')+')');globalThis._buildFmtSharePanel=(0,eval)('('+extract('_buildFmtSharePanel')+')');globalThis._buildConversationShareSheet=(0,eval)('('+extract('_buildConversationShareSheet')+')');
let pass=0,fail=0;function t(n,g,w=true){if(g===w)pass++;else{fail++;console.log(`  FAIL ${n}\n       got  ${JSON.stringify(g)}\n       want ${JSON.stringify(w)}`)}}
const sheet=_buildConversationShareSheet('json');root.appendChild(sheet);
t('sheet constructs',!!sheet);t('five format tabs',sheet.querySelectorAll('.ai-assistant-conv-share-format-btn').length,5);t('initial format json',sheet._getSelectedExportFormat(),'json');t('one destination section',sheet.querySelectorAll('.ai-assistant-conv-share-destinations').length,1);t('three destinations',sheet.querySelectorAll('.ai-assistant-conv-share-destination').length,3);t('format panel present',sheet.querySelectorAll('.ai-assistant-conv-share-format-panel').length,1);
t('switch YAML',sheet._selectExportFormat('yaml'));t('selected YAML',sheet._getSelectedExportFormat(),'yaml');t('switch does not duplicate destination section',sheet.querySelectorAll('.ai-assistant-conv-share-destinations').length,1);t('still one descriptor panel',sheet.querySelectorAll('.ai-assistant-conv-share-format-panel').length,1);
// Content & privacy presets always expose their concrete checkbox meaning.
const privacyGrid=sheet.querySelector('.ai-assistant-conv-share-custom-grid');
const presetControls=sheet.querySelectorAll('.ai-assistant-conv-share-preset');const standardPreset=presetControls.find(b=>b.textContent==='Standard');const minimalPreset=presetControls.find(b=>b.textContent==='Minimal');const customizePreset=presetControls.find(b=>b.textContent==='Customize');
t('privacy option checklist is visible for Standard',privacyGrid?.style?.display,'');
t('privacy hint explains automatic Customize transition',!!byText(sheet,'Preset selections are shown below. Change any option to switch to Customize.'));
t('Standard remains active while merely viewing options',standardPreset?.getAttribute('aria-pressed'),'true');
t('Customize is not selected merely because options are visible',customizePreset?.getAttribute('aria-pressed'),'false');
const privacyChecks=privacyGrid?.querySelectorAll('input')||[];
t('Standard checklist reflects preset selections',privacyChecks.length===7 && privacyChecks[0].checked===false && privacyChecks[1].checked===false && privacyChecks[2].checked===false && privacyChecks[3].checked===false && privacyChecks[4].checked===true && privacyChecks[5].checked===false && privacyChecks[6].checked===false);
await minimalPreset.click();
t('privacy option checklist stays visible for Minimal',privacyGrid?.style?.display,'');
t('Minimal remains active after selecting the preset',minimalPreset?.getAttribute('aria-pressed'),'true');
t('Minimal checklist reflects preset selections',privacyChecks.every(c=>c.checked===false));
privacyChecks[0].checked=true;privacyChecks[0].dispatch('change');
t('editing one option automatically selects Customize',customizePreset?.getAttribute('aria-pressed'),'true');
t('editing one option clears Minimal preset state',minimalPreset?.getAttribute('aria-pressed'),'false');
t('option checklist stays visible in Customize',privacyGrid?.style?.display,'');
await standardPreset.click();
t('choosing Standard restores Standard preset after customization',standardPreset?.getAttribute('aria-pressed'),'true');
t('Standard re-applies its predefined checkbox selections',privacyChecks[0].checked===false && privacyChecks[4].checked===true && privacyChecks[6].checked===false);
// Local preview -> copy/open/inspect current-browser Blob URL, then real local revoke.
const localDest=sheet.querySelector('[data-key="local"]');await localDest.click();const primary=sheet.querySelector('.ai-assistant-conv-share-generate-btn');await primary.click();t('local artifact created',sheet._getManagedShareArtifactCount(),1);t('preview opens blob',opens.some(x=>String(x).startsWith('blob:fake-')));let resultWrap=sheet.querySelector('.ai-assistant-conv-share-result');let resultInput=sheet.querySelector('.ai-assistant-conv-share-link-input');const localUrl=String(resultInput?.value||'');t('local result holds browser-local Blob URL',localUrl.startsWith('blob:fake-'));let localCopyResult=byText(resultWrap,'Copy link');let localInspect=byText(resultWrap,'Inspect');let localRemoveResult=byText(resultWrap,'Remove from browser');t('local result exposes Copy link',!!localCopyResult);t('local result exposes Inspect',!!localInspect);t('local result uses Remove from browser',!!localRemoveResult);await localCopyResult.click();t('local Copy link copies exact Blob URL',copied.at(-1),localUrl);t('local URL hidden until Inspect',resultInput?.style?.display,'none');await localInspect.click();t('local Inspect reveals Blob URL',resultInput?.style?.display,'');let artifactRows=sheet.querySelectorAll('.ai-assistant-conv-share-artifact');let localRowCopy=artifactRows[0].children.find(c=>c.textContent==='Copy link');let remove=artifactRows[0].children.find(c=>c.textContent==='Remove');t('managed local row exposes Copy link',!!localRowCopy);t('managed local row keeps standard Remove',!!remove);await localRowCopy.click();t('managed local Copy link copies exact Blob URL',copied.at(-1),localUrl);await remove.click();t('local artifact removed',sheet._getManagedShareArtifactCount(),0);t('local Blob revoked',revoked.some(x=>String(x).startsWith('blob:fake-')));
// Self-contained -> forget only, never server revoke.
const selfDest=sheet.querySelector('[data-key="self_contained"]');await selfDest.click();await primary.click();t('self-contained artifact created',sheet._getManagedShareArtifactCount(),1);resultInput=sheet.querySelector('.ai-assistant-conv-share-link-input');let copyResult=byText(sheet,'Copy data link');t('self-contained current transport is data html',String(resultInput?.value||'').startsWith('data:text/html;charset=utf-8;base64,'));t('self-contained exposes Copy data link',!!copyResult);const selfDataUrl=String(resultInput?.value||'');const openResult=byText(sheet,'Open');const opensBeforeSelf=opens.length;await openResult.click();t('self-contained Open uses exact generated data URL',opens.at(-1),selfDataUrl);t('self-contained Open creates no extra Blob URL',blobN,1);t('self-contained Open performed one direct navigation attempt',opens.length,opensBeforeSelf+1);artifactRows=sheet.querySelectorAll('.ai-assistant-conv-share-artifact');let selfCopy=artifactRows[0].children.find(c=>c.textContent==='Copy link');t('managed self-contained row keeps Copy link',!!selfCopy);remove=artifactRows[0].children.find(c=>c.textContent==='Remove');const delBefore=globalDeletes.length;await remove.click();t('self-contained removed locally',sheet._getManagedShareArtifactCount(),0);t('self-contained never calls server DELETE',globalDeletes.length,delBefore);t('truthful non-revocable notice',notices.some(n=>n.msg.includes('cannot be revoked')));
// Global -> tracked lifecycle, explicit status check, true server revocation.
const globalDest=sheet.querySelector('[data-key="global"]');await globalDest.click();await primary.click();
t('Global POST happened',globalPosts.length,1);t('Global artifact created',sheet._getManagedShareArtifactCount(),1);t('Global ledger tracks provided link',sheet._getTrackedGlobalArtifactCount(),1);resultInput=sheet.querySelector('.ai-assistant-conv-share-link-input');t('Global UUID-only response synthesizes public read URL',resultInput?.value,'https://share.example.test/v1/share#share='+(1).toString(16).padStart(32,'0'));t('Global result visibly exposes share URL',resultInput?.style?.display,'');
let ledgerRaw=ss['ai-assistant-global-share-ledger:v1']||'';t('ledger never persists edit token',!ledgerRaw.includes('edit-secret'));t('ledger never persists snapshot',!ledgerRaw.includes('Hello'));
artifactRows=sheet.querySelectorAll('.ai-assistant-conv-share-artifact');let globalCopy=artifactRows[0].children.find(c=>c.textContent==='Copy link');t('managed Global row exposes Copy link',!!globalCopy);await globalCopy.click();t('Global Copy link copies public read URL',copied.at(-1),'https://share.example.test/v1/share#share='+(1).toString(16).padStart(32,'0'));let statusBtn=artifactRows[0].children.find(c=>c.textContent==='Check status');t('Global Check status exists',!!statusBtn);await statusBtn.click();t('status check reports active',notices.some(n=>n.msg==='Global link is active'));
// A 404 is reason-unknown and therefore non-terminal. Keep the bounded public
// capability and live page-memory edit token so the user can recheck or revoke.
globalStatus=404;await statusBtn.click();artifactRows=sheet.querySelectorAll('.ai-assistant-conv-share-artifact');statusBtn=artifactRows[0].children.find(c=>c.textContent==='Check status');let revoke=artifactRows[0].children.find(c=>c.textContent==='Revoke');let forgetUnavailable=artifactRows[0].children.find(c=>c.textContent==='Forget');ledgerRaw=ss['ai-assistant-global-share-ledger:v1']||'';
t('404 remains explicitly recheckable',!!statusBtn);t('404 keeps live Revoke capability in page memory',!!revoke);t('404 also exposes truthful local Forget',!!forgetUnavailable);t('404 ledger retains public URL for later recheck',ledgerRaw.includes((1).toString(16).padStart(32,'0')));t('404 reports unavailable without claiming revoked',notices.some(n=>n.msg==='Global link is no longer available on the server'));
globalStatus=200;await statusBtn.click();artifactRows=sheet.querySelectorAll('.ai-assistant-conv-share-artifact');revoke=artifactRows[0].children.find(c=>c.textContent==='Revoke');t('200 can recover a previously unavailable Global link',!!revoke);
t('Global Revoke exists',!!revoke);await revoke.click();t('server DELETE called',globalDeletes.length,1);t('DELETE uses private edit capability',globalDeletes[0]?.edit,'edit-secret-'+(1).toString(16).padStart(32,'0'));
t('revoked Global remains as lifecycle history',sheet._getManagedShareArtifactCount(),1);artifactRows=sheet.querySelectorAll('.ai-assistant-conv-share-artifact');let forgetGlobal=artifactRows[0].children.find(c=>c.textContent==='Forget');t('revoked Global changes to Forget',!!forgetGlobal);t('revoked Global Forget is enabled after revoke completes',!!forgetGlobal && !forgetGlobal.disabled);t('revoked row says revoked',!!byText(artifactRows[0],'Global link'));
ledgerRaw=ss['ai-assistant-global-share-ledger:v1']||'';t('revoked ledger tombstone clears public URL',!ledgerRaw.includes('/g1'));await forgetGlobal.click();t('revoked Global history can be forgotten',sheet._getManagedShareArtifactCount(),0);t('forgotten Global removed from ledger',sheet._getTrackedGlobalArtifactCount(),0);
// Multiple links handed to the user survive new-chat and reload as a bounded read-only ledger.
await primary.click();t('second Global created',globalPosts.length,2);currentConversationId='conv-b';document.dispatchEvent(new CustomEvent('ai-assistant-conversation-reset',{detail:{conversationId:'conv-b'}}));t('new chat preserves prior provided Global artifact',sheet._getManagedShareArtifactCount(),1);
await globalDest.click();await primary.click();t('third Global created for new chat',globalPosts.length,3);t('ledger tracks both active provided links',sheet._getTrackedGlobalArtifactCount(),2);
// Delayed privacy review cannot publish previous conversation.
await selfDest.click();let resolvePrivacy;privacyHook=v=>new Promise(r=>{resolvePrivacy=()=>r({action:'continue',value:v,scan:{flagged:true}})});const before=sheet._getManagedShareArtifactCount();const delayed=primary.click();await Promise.resolve();currentConversationId='conv-c';document.dispatchEvent(new CustomEvent('ai-assistant-conversation-reset',{detail:{conversationId:'conv-c'}}));resolvePrivacy();await delayed;t('stale privacy decision creates no artifact',sheet._getManagedShareArtifactCount(),before);privacyHook=async v=>({action:'continue',value:v,scan:{flagged:false}});
// Direct toolbar download also enters the same page-memory artifact lifecycle registry.
const directBefore=sheet._getManagedShareArtifactCount();const downloadsBefore=downloads.length;_downloadConversationFormat('yaml');
t('direct toolbar download invoked',downloads.length,downloadsBefore+1);t('direct toolbar download tracked',sheet._getManagedShareArtifactCount(),directBefore+1);
artifactRows=sheet.querySelectorAll('.ai-assistant-conv-share-artifact');const directDownloadRow=artifactRows.find(r=>byText(r,'Downloaded artifact'));const directForget=directDownloadRow?.children.find(c=>c.textContent==='Forget');t('direct toolbar download has Forget',!!directForget);await directForget.click();t('direct toolbar download can be forgotten',sheet._getManagedShareArtifactCount(),directBefore);
// Download lifecycle is truthful: forget record only, external file remains user-controlled.
const advancedToggle=byText(sheet,'Advanced')?.parentNode || byText(sheet,'Advanced');if(advancedToggle?.click)await advancedToggle.click();const downloadBtn=byText(sheet,'Download current snapshot');t('download action exists',!!downloadBtn);await downloadBtn.click();t('download invoked',downloads.length,downloadsBefore+2);artifactRows=sheet.querySelectorAll('.ai-assistant-conv-share-artifact');const downloadRow=artifactRows.find(r=>byText(r,'Downloaded artifact'));const forget=downloadRow?.children.find(c=>c.textContent==='Forget');t('download Forget exists',!!forget);await forget.click();t('download record forget notice is truthful',notices.some(n=>n.msg.includes('Delete the file itself from your device')));
// Simulated page reload: memory-only edit capabilities disappear, sessionStorage public ledger remains.
_managedConversationArtifacts.length=0;_managedConversationArtifactRenderHook=null;currentConversationId='conv-b';const reloaded=_buildConversationShareSheet('html');root.appendChild(reloaded);
t('reload restores all tracked Global links',reloaded._getManagedShareArtifactCount(),2);t('reload ledger count remains two',reloaded._getTrackedGlobalArtifactCount(),2);
let restoredRows=reloaded.querySelectorAll('.ai-assistant-conv-share-artifact');t('restored Global links have no Revoke',restoredRows.every(r=>!r.children.find(c=>c.textContent==='Revoke')));t('restored Global links expose Forget',restoredRows.every(r=>!!r.children.find(c=>c.textContent==='Forget')));
// Explicit status check can transition a restored public link to expired lifecycle history.
globalStatus=410;statusBtn=restoredRows[0].children.find(c=>c.textContent==='Check status');await statusBtn.click();restoredRows=reloaded.querySelectorAll('.ai-assistant-conv-share-artifact');t('expired status keeps lifecycle history',reloaded._getManagedShareArtifactCount(),2);t('expired status removes Open action',!restoredRows[0].children.find(c=>c.textContent==='Open'));
globalStatus=200;
// Fail-closed migration: a legacy/tampered v2 recovery record containing an
// edit token or conversation-derived fingerprint must be scrubbed before use.
_managedConversationArtifacts.length=0;_managedConversationArtifactRenderHook=null;ss={};currentConversationId='conv-legacy';
const legacyId='f'.repeat(32);ss['ai-assistant-global-share:v2']=JSON.stringify({uuid:legacyId,url:'https://share.example.test/v1/share/'+legacyId,conversationId:'conv-legacy',format:'json',expiresAt:'2026-09-28T00:00:00Z',editToken:'legacy-edit-secret',contentHash:'derived-private-fingerprint',snapshot:{secret:'conversation'}});
const legacyPostsBefore=globalPosts.length,legacyPatchesBefore=globalPatches.length;const legacySheet=_buildConversationShareSheet('json');root.appendChild(legacySheet);const scrubbedRaw=ss['ai-assistant-global-share:v2']||'';const legacyRows=legacySheet.querySelectorAll('.ai-assistant-conv-share-artifact');
t('legacy recovery record is destructively scrubbed',!scrubbedRaw.includes('legacy-edit-secret')&&!scrubbedRaw.includes('contentHash')&&!scrubbedRaw.includes('snapshot'));
t('legacy recovery never restores Revoke authority',legacyRows.every(r=>!r.children.find(c=>c.textContent==='Revoke')));
const legacyGlobalDest=legacySheet.querySelector('[data-key="global"]');await legacyGlobalDest.click();const legacyPrimary=legacySheet.querySelector('.ai-assistant-conv-share-generate-btn');await legacyPrimary.click();
t('legacy edit token is never used for PATCH',globalPatches.length,legacyPatchesBefore);t('legacy restored link creates a fresh Global object',globalPosts.length,legacyPostsBefore+1);

// Run 16.2.2: Global creation must never look like a silent/stub control.
ss={}; currentConversationId='conv-error'; _managedConversationArtifacts.length=0;
globalThis._postGlobalShare=(base,token,payload,envelope,ok,fail)=>{globalPosts.push({base,token,payload,envelope,error:true});fail({status:502,message:'The service returned a non-JSON success response.'})};
const errorSheet=_buildConversationShareSheet('json');root.appendChild(errorSheet);await errorSheet.querySelector('[data-key="global"]').click();const errorPrimary=errorSheet.querySelector('.ai-assistant-conv-share-generate-btn');await errorPrimary.click();
t('Global failure renders inline result panel',errorSheet.querySelector('.ai-assistant-conv-share-result')?.style?.display,'');
t('ambiguous 5xx renders outcome-unknown title',!!byText(errorSheet,'Global link outcome unknown'));
t('ambiguous 5xx does not claim definite failure',!errorSheet.querySelector('.ai-assistant-conv-share-result-meta')?.textContent.includes('not created'),true);
t('ambiguous 5xx explains server may have created link',errorSheet.querySelector('.ai-assistant-conv-share-result-meta')?.textContent.includes('may have created'),true);
t('ambiguous 5xx exposes safe retry',!!byText(errorSheet,'Retry safely'));
t('Global failure re-enables primary button',errorPrimary.disabled,false);

ss={}; currentConversationId='conv-pending'; _managedConversationArtifacts.length=0;
globalThis._postGlobalShare=(base,token,payload,envelope,ok,fail)=>{globalPosts.push({base,token,payload,envelope,pending:true})};
const pendingSheet=_buildConversationShareSheet('json');root.appendChild(pendingSheet);await pendingSheet.querySelector('[data-key="global"]').click();const pendingPrimary=pendingSheet.querySelector('.ai-assistant-conv-share-generate-btn');await pendingPrimary.click();
t('Global request immediately renders pending panel',pendingSheet.querySelector('.ai-assistant-conv-share-result')?.style?.display,'');
t('Global request renders Creating title before callback',!!byText(pendingSheet,'Creating global link…'));
t('Global pending result is aria-busy',pendingSheet.querySelector('.ai-assistant-conv-share-result')?.getAttribute('aria-busy'),'true');

console.log(`\n${pass} passed, ${fail} failed`);process.exit(fail?1:0);
