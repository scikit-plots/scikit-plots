import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function t(name, got, want) {
  if (Object.is(got, want)) passed++;
  else { failed++; console.error(`FAIL ${name}: got=${JSON.stringify(got)} want=${JSON.stringify(want)}`); }
}
function ok(cond, name) { t(name, !!cond, true); }
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('missing ' + name);
  let depth=0, started=false, q=null, line=false, block=false;
  for (let j=i; j<src.length; j++) {
    const c=src[j], n=src[j+1];
    if (line) { if (c==='\n') line=false; continue; }
    if (block) { if (c==='*' && n==='/') { block=false; j++; } continue; }
    if (q) { if (c==='\\') { j++; continue; } if (c===q) q=null; continue; }
    if (c==='/' && n==='/') { line=true; j++; continue; }
    if (c==='/' && n==='*') { block=true; j++; continue; }
    if (c==='"' || c==="'" || c==='`') { q=c; continue; }
    if (c==='{') { depth++; started=true; }
    else if (c==='}') { depth--; if (started && depth===0) return src.slice(i,j+1); }
  }
  throw new Error('unbalanced ' + name);
}

// Executable public-DOM egress permission boundary.
function domRuntime(seed={}) {
  const store = new Map(Object.entries(seed));
  const localStorage = {
    getItem(k){ return store.has(k) ? store.get(k) : null; },
    setItem(k,v){ store.set(k,String(v)); },
    removeItem(k){ store.delete(k); },
  };
  const events=[];
  const nodes=new Map();
  const document = {
    dispatchEvent(ev){ events.push(ev); return true; },
    getElementById(id){ return nodes.get(id) || null; },
    querySelectorAll(){ return []; },
  };
  class CustomEvent { constructor(type, init={}) { this.type=type; this.detail=init.detail; } }
  const code = `
    var _FEEDBACK_TELEMETRY_CONSENT_VERSION='1.0.0';
    var _feedbackTelemetryGrantedAt=1700000000000;
    var _FEEDBACK_DOM_CONSENT_VERSION='2.0.0';
    var _FEEDBACK_DOM_PREF_KEY='ai-assistant-page-integration-consent';
    ${extract('_readFeedbackDomConsent')}
    var _feedbackDomIntegrationEnabled=_readFeedbackDomConsent();
    ${extract('_feedbackTelemetryPayload')}
    ${extract('_feedbackLocalEventPayload')}
    ${extract('_dispatchFeedbackIntegrationEvent')}
    ${extract('_feedbackDomStatusText')}
    ${extract('_setFeedbackDomIntegrationMode')}
    return {enabled:()=>_feedbackDomIntegrationEnabled, dispatch:_dispatchFeedbackIntegrationEvent,
      set:_setFeedbackDomIntegrationMode, status:_feedbackDomStatusText};
  `;
  const api = new Function('localStorage','document','CustomEvent',code)(localStorage,document,CustomEvent);
  return {api,store,events};
}

const detail={ratingValue:1,ratingLabel:'helpful',ratingTitle:'Helpful',ratingMode:'quick',answerIndex:2,
  query:'SECRET QUESTION',answer:'SECRET ANSWER',message:'SECRET NOTE',model:{id:'secret'},
  page:'https://private.example/x?token=secret',conversationId:'stable',sessionId:'event-id',ts:123};

let r=domRuntime();
t('page integration absent consent defaults off', r.api.enabled(), false);
t('off dispatch reports false', r.api.dispatch(detail), false);
t('off dispatch emits no public event', r.events.length, 0);
t('off status is explicit', r.api.status(), 'Page integration off — assistant lifecycle events stay on the private internal event bus.');

r=domRuntime({'ai-assistant-page-integration-consent':'not-json'});
t('malformed page integration consent fails closed', r.api.enabled(), false);
r=domRuntime({'ai-assistant-page-integration-consent':JSON.stringify({enabled:true,version:'1.0.0'})});
t('stale page integration version fails closed', r.api.enabled(), false);
r=domRuntime({'ai-assistant-page-integration-consent':JSON.stringify({enabled:true,version:'2.0.0',grantedAt:1})});
t('current page integration consent restores', r.api.enabled(), true);
t('allowed dispatch reports true', r.api.dispatch(detail), true);
t('one public event emitted', r.events.length, 1);
t('public event name', r.events[0].type, 'ai-assistant-feedback');
for (const key of ['query','answer','message','model','page','conversationId','telemetryConsent','telemetryConsentVersion','telemetryConsentAt']) {
  ok(!(key in r.events[0].detail), `public event omits ${key}`);
}
t('public event keeps bounded rating value', r.events[0].detail.ratingValue, 1);
t('public event keeps answer index', r.events[0].detail.answerIndex, 2);

r.api.set(false);
t('explicit page integration opt-out disables', r.api.enabled(), false);
ok(!r.store.has('ai-assistant-page-integration-consent'),'opt-out deletes integration consent record');
const beforeEvents=r.events.length;
t('dispatch blocked after opt-out',r.api.dispatch(detail),false);
t('no event after opt-out',r.events.length,beforeEvents);
r.api.set(true);
t('explicit page integration opt-in enables',r.api.enabled(),true);
const saved=JSON.parse(r.store.get('ai-assistant-page-integration-consent'));
t('integration consent version',saved.version,'2.0.0');
t('integration consent enabled marker',saved.enabled,true);
ok(Number.isFinite(saved.grantedAt)&&saved.grantedAt>0,'integration consent timestamp exists');

// Source-level capability and recovery invariants.
const headersFn=extract('_operationHeaders');
ok(headersFn.includes("'X-AI-Management-Token-Hash': envelope.managementTokenHash"),'CREATE sends capability digest header');
ok(!headersFn.includes("managementToken'"),'CREATE headers do not serialize raw management token');
ok(!headersFn.includes('envelope.managementToken,'),'CREATE headers do not serialize raw capability value');
const sharePost=extract('_postGlobalShare');
ok(sharePost.includes('headers: _operationHeaders(envelope)'),'Global Share CREATE uses operation envelope headers');
ok(sharePost.includes('res.editToken = envelope.managementToken'),'Share UI recovers its local token after server response');
ok(!sharePost.includes("'X-Share-Edit-Token': envelope.managementToken"),'Share CREATE does not transmit local edit token');
const safeLog=extract('_safeUrlForLog');
ok(safeLog.includes('parsed.protocol + \'//\' + parsed.host + parsed.pathname'),'safe endpoint log keeps scheme host path');
ok(!safeLog.includes('parsed.search'),'safe endpoint log omits query');
ok(!safeLog.includes('parsed.hash'),'safe endpoint log omits fragment');

const endpointSecret=extract('_endpointHasSecretQueryName');
for (const name of ['api[_-]?key','access[_-]?token','authorization','client[_-]?secret','signature']) {
  ok(endpointSecret.includes(name),`credential-query detector covers ${name}`);
}

const rememberFn=extract('_rememberConversationPermission');
ok(rememberFn.includes("stored === 'true'"),'remember preference honors explicit ON');
ok(rememberFn.includes("stored === 'false'"),'remember preference honors explicit OFF');
ok(rememberFn.includes('cfg.panelRememberConversation !== false'),'missing tab choice falls back to site default');
const rememberSetFn=extract('_setRememberConversationInTab');
ok(rememberSetFn.includes("allow ? 'true' : 'false'"),'explicit OFF is persisted instead of erased');
const persistFn=extract('_persistEnabled');
ok(persistFn.includes('_rememberConversationPermission()'),'transcript restore respects the configured default or explicit per-tab choice');
const loadFn=extract('_loadTranscript');
ok(loadFn.includes('_TRANSCRIPT_RESTORE_MAX_STORAGE_CHARS'),'transcript restore bounds raw storage');
ok(loadFn.includes('_TRANSCRIPT_RESTORE_MAX_ENTRIES'),'transcript restore bounds entry count');
ok(loadFn.includes('_TRANSCRIPT_RESTORE_MAX_TEXT_CHARS'),'transcript restore bounds message text');
ok(loadFn.includes('_ssDel(_TRANSCRIPT_KEY)'),'invalid/off transcript state is destructively cleared');

ok(src.includes("sessionStorage.getItem('ai-assistant-mic-device-id')"),'microphone device preference is session-scoped');
ok(!src.includes("localStorage.getItem('ai-assistant-mic-device-id')"),'microphone device ID is not read from persistent localStorage');
ok(src.includes('Save private receipt'),'explicit contribution receipt export is present');
ok(src.includes('Import private receipt'),'explicit contribution receipt import is present');
ok(src.includes('Copy private withdrawal code'),'portable private withdrawal capability is present');
ok(src.includes('Copy support reference'),'non-secret maintainer support fallback is present');
ok(src.includes("status: 'outcome_unknown'"),'contribution receipt models unknown outcome');
ok(src.includes("phase: outcomeUnknown ? 'outcome_unknown' : 'error'"),'Global Share models unknown create outcome');
ok(src.includes("recoveringGlobal ? _pendingGlobalCreate.payload"),'Global Share retry reuses exact prepared payload');
ok(src.includes('if (!outcomeUnknown) _pendingGlobalCreate = null;'),'unknown Share outcome preserves retry envelope');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
