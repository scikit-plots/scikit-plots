import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed=0, failed=0;
function t(name, got, want){ if(Object.is(got,want)){passed++;}else{failed++;console.error(`FAIL ${name}: got=${JSON.stringify(got)} want=${JSON.stringify(want)}`);} }
function ok(cond,name){t(name,!!cond,true);}
function slice(a,b){const i=src.indexOf(a);const j=src.indexOf(b,i+1);if(i<0||j<0)throw new Error(`missing slice ${a}`);return src.slice(i,j);}

const init = slice("    var _FEEDBACK_TELEMETRY_CONSENT_VERSION = '1.0.0';", '    /**\n     * Selected microphone device ID.');
const telemetry = slice('    function _feedbackTelemetryPayload(detail) {', '    /**\n     * POST a content-free telemetry supersession marker.');
const retract = slice('    function _postFeedbackRetract(url, token, prevSessionId, answerIndex, conversationId) {', '    /**\n     * Render the post-submission thank-you state');
const setter = slice('    function _setFeedbackPersistMode(enabled) {', '    // ══════════════════════════════════════════════════════════════════════════\n    // LEGACY LOCAL SHARE STORAGE');

function runtime(seed={}) {
  const store = new Map(Object.entries(seed));
  const localStorage = {
    getItem(k){return store.has(k)?store.get(k):null;},
    setItem(k,v){store.set(k,String(v));},
    removeItem(k){store.delete(k);},
  };
  const document = {getElementById(){return null;},querySelectorAll(){return [];}};
  const posts=[];
  function _remotePost(...args){posts.push(args);}
  const fn = new Function('localStorage','document','_remotePost', `${init}\n${telemetry}\n${retract}\n${setter}\nreturn {enabled:()=>_feedbackPersistEnabled, grantedAt:()=>_feedbackTelemetryGrantedAt, set:_setFeedbackPersistMode, post:_postFeedback, retract:_postFeedbackRetract, event:_feedbackLocalEventPayload, payload:_feedbackTelemetryPayload};`);
  return {api:fn(localStorage,document,_remotePost),store,posts};
}

let r=runtime();
t('absent consent defaults off',r.api.enabled(),false);
t('post blocked without consent',r.api.post('https://example/v1/feedback','',{ratingValue:1}),false);
t('no request without consent',r.posts.length,0);

r=runtime({'ai-assistant-feedback-telemetry':'true','ai-assistant-feedback-persist':'true'});
t('legacy booleans do not authorize telemetry',r.api.enabled(),false);

r=runtime({'ai-assistant-feedback-telemetry-consent':'not-json'});
t('malformed consent fails closed',r.api.enabled(),false);
r=runtime({'ai-assistant-feedback-telemetry-consent':JSON.stringify({version:'1.0.0',grantedAt:1700000000000})});
t('missing enabled flag fails closed',r.api.enabled(),false);
r=runtime({'ai-assistant-feedback-telemetry-consent':JSON.stringify({enabled:true,version:'0.9.0',grantedAt:1})});
t('stale consent version fails closed',r.api.enabled(),false);
r=runtime({'ai-assistant-feedback-telemetry-consent':JSON.stringify({enabled:true,version:'1.0.0',grantedAt:1700000000000})});
t('current structured consent restores telemetry',r.api.enabled(),true);
t('stored grant timestamp restored',r.api.grantedAt(),1700000000000);

const detail={ratingValue:1,ratingLabel:'helpful',ratingTitle:'Helpful',ratingMode:'quick',answerIndex:2,query:'SECRET QUESTION',answer:'SECRET ANSWER',message:'SECRET NOTE',model:{id:'secret'},page:'https://private',conversationId:'stable',sessionId:'event-id',ts:123};
const local=r.api.event(detail);
for(const k of ['query','answer','message','model','page','conversationId','telemetryConsent','telemetryConsentVersion','telemetryConsentAt']) ok(!(k in local),`public event omits ${k}`);
t('public event keeps rating',local.ratingValue,1);

r.api.post('https://example/v1/feedback','token',detail);
t('consented post occurs',r.posts.length,1);
const sent=r.posts[0][2];
t('network schema v4',sent.schemaVersion,4);
t('network consent marker',sent.telemetryConsent,true);
t('network consent version',sent.telemetryConsentVersion,'1.0.0');
t('network consent timestamp',sent.telemetryConsentAt,1700000000000);
for(const k of ['query','answer','message','model','page','conversationId']) ok(!(k in sent),`network payload omits ${k}`);

r.api.set(false);
t('toggle off disables telemetry',r.api.enabled(),false);
ok(!r.store.has('ai-assistant-feedback-telemetry-consent'),'toggle off removes consent record');
const before=r.posts.length;
t('post stays blocked after opt-out',r.api.post('https://example/v1/feedback','token',detail),false);
t('opt-out causes zero extra request',r.posts.length,before);
t('retract also blocked after opt-out',r.api.retract('https://example/v1/feedback','token','old',2,'stable'),false);
t('blocked retract causes zero extra request',r.posts.length,before);

r.api.set(true);
t('explicit toggle on enables telemetry',r.api.enabled(),true);
const stored=JSON.parse(r.store.get('ai-assistant-feedback-telemetry-consent'));
t('stored consent version',stored.version,'1.0.0');
t('stored consent enabled',stored.enabled,true);
ok(Number.isFinite(stored.grantedAt)&&stored.grantedAt>0,'stored consent has grant timestamp');
ok(!r.store.has('ai-assistant-feedback-telemetry'),'legacy telemetry key removed');
ok(!r.store.has('ai-assistant-feedback-persist'),'legacy persist key removed');

console.log(`${passed} passed, ${failed} failed`);
if(failed)process.exit(1);
