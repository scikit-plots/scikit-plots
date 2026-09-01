import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed=0, failed=0;
function ok(cond,name){if(cond){passed++;}else{failed++;console.error('FAIL '+name);}}
function eq(got,want,name){ok(Object.is(got,want),name);if(!Object.is(got,want))console.error(`  got=${JSON.stringify(got)} want=${JSON.stringify(want)}`);}
function slice(a,b){const i=src.indexOf(a),j=src.indexOf(b,i+1);if(i<0||j<0)throw new Error(`missing slice ${a}`);return src.slice(i,j);}

const prefs = slice("    var _FEEDBACK_TELEMETRY_CONSENT_VERSION = '1.0.0';", '    /**\n     * Selected microphone device ID.');

function runtime(cfg={}, seed={}) {
  const store=new Map(Object.entries(seed));
  const localStorage={
    getItem:k=>store.has(k)?store.get(k):null,
    setItem:(k,v)=>store.set(k,String(v)),
    removeItem:k=>store.delete(k),
  };
  const document={querySelectorAll(){return [];}};
  function _cfg(){return cfg;}
  function _dispatchAssistantEvent(){}
  function CustomEvent(type,init){this.type=type;this.detail=init&&init.detail;}
  const fn=new Function('localStorage','document','_cfg','_dispatchAssistantEvent','CustomEvent', `${prefs}\nreturn {telemetry:()=>_feedbackPersistEnabled,review:()=>_feedbackReviewEnabled,page:()=>_feedbackDomIntegrationEnabled,stream:()=>_streamingOn,effectiveStream:_effectiveStreamingEnabled,setReview:_setFeedbackReviewMode,setStream:_setStreamingMode};`);
  return {api:fn(localStorage,document,_cfg,_dispatchAssistantEvent,CustomEvent),store};
}

let r=runtime();
eq(r.api.telemetry(),false,'built-in telemetry default is false');
eq(r.api.review(),true,'built-in maintainer review default is true');
eq(r.api.page(),false,'built-in page integration default is false');
eq(r.api.stream(),true,'built-in streaming preference default is true');
eq(r.api.effectiveStream(),true,'streaming is effective when capability and preference are on');

r=runtime({panelFeedbackTelemetryDefault:true,panelFeedbackReviewDefault:false,panelPageIntegrationDefault:true,panelStreamingDefault:false,panelApiStreaming:true});
eq(r.api.telemetry(),true,'site can configure telemetry initial value');
eq(r.api.review(),false,'site can configure maintainer review initial value');
eq(r.api.page(),true,'site can configure page integration initial value');
eq(r.api.stream(),false,'site can configure streaming initial value');
eq(r.api.effectiveStream(),false,'streaming preference off disables runtime SSE');

r=runtime(
  {panelFeedbackTelemetryDefault:true,panelFeedbackReviewDefault:true,panelPageIntegrationDefault:true,panelStreamingDefault:true,panelApiStreaming:true},
  {
    'ai-assistant-feedback-telemetry-consent':JSON.stringify({enabled:false,version:'1.0.0',grantedAt:null}),
    'ai-assistant-feedback-review-consent':JSON.stringify({enabled:false,version:'2.0.0',grantedAt:null}),
    'ai-assistant-page-integration-consent':JSON.stringify({enabled:false,version:'2.0.0',changedAt:1}),
    'ai-assistant-streaming-on':'false',
  }
);
eq(r.api.telemetry(),false,'stored telemetry OFF beats site default ON');
eq(r.api.review(),false,'stored review OFF beats site default ON');
eq(r.api.page(),false,'stored page-integration OFF beats site default ON');
eq(r.api.stream(),false,'stored streaming OFF beats site default ON');

r=runtime(
  {panelFeedbackTelemetryDefault:false,panelFeedbackReviewDefault:false,panelPageIntegrationDefault:false,panelStreamingDefault:false,panelApiStreaming:true},
  {
    'ai-assistant-feedback-telemetry-consent':JSON.stringify({enabled:true,version:'1.0.0',grantedAt:10}),
    'ai-assistant-feedback-review-consent':JSON.stringify({enabled:true,version:'2.0.0',grantedAt:11}),
    'ai-assistant-page-integration-consent':JSON.stringify({enabled:true,version:'2.0.0',changedAt:12}),
    'ai-assistant-streaming-on':'true',
  }
);
eq(r.api.telemetry(),true,'stored telemetry ON beats site default OFF');
eq(r.api.review(),true,'stored review ON beats site default OFF');
eq(r.api.page(),true,'stored page-integration ON beats site default OFF');
eq(r.api.stream(),true,'stored streaming ON beats site default OFF');

r=runtime({panelStreamingDefault:true,panelApiStreaming:false},{'ai-assistant-streaming-on':'true'});
eq(r.api.stream(),true,'reader streaming preference remains on behind disabled capability');
eq(r.api.effectiveStream(),false,'site streaming capability is a hard ceiling');

r=runtime({panelFeedbackReviewDefault:true});
r.api.setReview(false);
let saved=JSON.parse(r.store.get('ai-assistant-feedback-review-consent'));
eq(saved.enabled,false,'explicit review OFF is persisted');
r.api.setStream(false);
eq(r.store.get('ai-assistant-streaming-on'),'false','explicit streaming OFF is persisted');

ok(src.includes("var streamingEnabled = _effectiveStreamingEnabled();"),'API request path consumes reader streaming preference');
ok(src.includes("data-streaming-toggle"),'streaming UI synchronizes from the centralized preference');

console.log(`${passed} passed, ${failed} failed`);
if(failed)process.exit(1);
