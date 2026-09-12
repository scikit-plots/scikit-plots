import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function eq(got, want, name) { if (Object.is(got, want)) passed++; else { failed++; console.error(`FAIL ${name}\n got=${JSON.stringify(got)}\nwant=${JSON.stringify(want)}`); } }
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('missing ' + name);
  let depth=0, started=false, quote=null, line=false, block=false;
  for (let j=i;j<src.length;j++) {
    const c=src[j], n=src[j+1];
    if (line) { if (c==='\n') line=false; continue; }
    if (block) { if (c==='*'&&n==='/') { block=false; j++; } continue; }
    if (quote) { if (c==='\\') { j++; continue; } if (c===quote) quote=null; continue; }
    if (c==='/'&&n==='/') { line=true; j++; continue; }
    if (c==='/'&&n==='*') { block=true; j++; continue; }
    if (c==='"'||c==="'"||c==='`') { quote=c; continue; }
    if (c==='{') { depth++; started=true; }
    else if (c==='}') { depth--; if (started&&depth===0) return src.slice(i,j+1); }
  }
  throw new Error('unbalanced '+name);
}

// Execute the public projection/dispatch contract with a private event target.
{
  const internal=[], external=[];
  const document = { dispatchEvent(ev){ external.push(ev); return true; } };
  class CustomEvent { constructor(type, init={}) { this.type=type; this.detail=init.detail; } }
  const code = `
    var _feedbackDomIntegrationEnabled=false;
    var _FEEDBACK_TELEMETRY_CONSENT_VERSION='1.0.0';
    var _feedbackTelemetryGrantedAt=123;
    var _assistantEvents={dispatchEvent(ev){ internal.push(ev); return true; }};
    ${extract('_feedbackTelemetryPayload')}
    ${extract('_feedbackLocalEventPayload')}
    ${extract('_publicAssistantEventDetail')}
    ${extract('_dispatchAssistantEvent')}
    return {emit:_dispatchAssistantEvent, set(v){_feedbackDomIntegrationEnabled=!!v;}};
  `;
  const api = new Function('internal','document','CustomEvent',code)(internal,document,CustomEvent);
  const sensitive={reason:'model-edited',id:'provider-model-id',model:{id:'secret'},endpoint:'https://x/?token=secret',token:'secret'};
  api.emit(new CustomEvent('ai-assistant-model-change',{detail:sensitive}));
  eq(internal.length,1,'private bus receives event while integration off');
  eq(external.length,0,'document receives nothing while integration off');
  eq(internal[0].detail.id,'provider-model-id','internal detail keeps model id');
  api.set(true);
  api.emit(new CustomEvent('ai-assistant-model-change',{detail:sensitive}));
  eq(external.length,1,'explicit page integration emits one public projection');
  eq(external[0].detail.reason,'model-edited','public model event keeps bounded reason');
  ok(!('id' in external[0].detail),'public model event strips model id');
  ok(!('model' in external[0].detail),'public model event strips model object');
  ok(!('endpoint' in external[0].detail),'public model event strips endpoint');
  ok(!('token' in external[0].detail),'public model event strips token');

  api.emit(new CustomEvent('ai-assistant:profile-changed',{detail:{activeKey:'prod-secret-key',activeLabel:'Production',isBuiltin:false}}));
  const profile = external.at(-1).detail;
  eq(profile.activeLabel,'Production','profile projection keeps display label');
  ok(!('activeKey' in profile),'profile projection strips stable/internal key');

  api.emit(new CustomEvent('ai-assistant-model-edit',{detail:{model:{id:'private-model',provider:'secret'},isCustom:true}}));
  const edit = external.at(-1).detail;
  eq(edit.isCustom,true,'model edit projection keeps only coarse custom flag');
  ok(!('model' in edit),'model edit projection strips full model object');
}

// Runtime bearer-token policy: OFF strips programmatic injection; explicit
// site-owner opt-in keeps the legacy page-memory-only compatibility path.
{
  const start = src.indexOf('var _EP = (function () {');
  const endMarker='\n    }());';
  const end=src.indexOf(endMarker,start);
  const block=src.slice(start,end+endMarker.length);
  function runtime(allow) {
    const storage=new Map();
    const context={URL,console:{warn(){},log(){},error(){}},CustomEvent:function(t,i){this.type=t;this.detail=i?.detail;},
      document:{dispatchEvent(){}},localStorage:{getItem(k){return storage.get(k)||null;},setItem(k,v){storage.set(k,String(v));},removeItem(k){storage.delete(k);}},
      window:{AI_ASSISTANT_ENDPOINT_DEFAULT:'',AI_ASSISTANT_ENDPOINTS:{},AI_ASSISTANT_CONFIG:{allowRuntimeTokens:allow}}};
    vm.createContext(context); vm.runInContext(block,context); return context._EP;
  }
  let ep=runtime(false);
  ok(ep.addProfile('locked',{label:'Locked',base:'https://example.com',shareToken:'SECRET',feedbackToken:'SECRET2'}).ok,'default-off profile still saves routing');
  ep.setActive('locked');
  eq(ep.resolveToken('shareToken'),'','default-off strips share token authority');
  eq(ep.resolveToken('feedbackToken'),'','default-off strips feedback token authority');
  eq(ep.getProfile('locked').shareToken,'','default-off getProfile cannot reveal injected token');

  ep=runtime(true);
  ok(ep.addProfile('compat',{label:'Compat',base:'https://example.com',shareToken:'SHORT',feedbackToken:'SHORT2'}).ok,'explicit opt-in accepts short-lived token fields');
  ep.setActive('compat');
  eq(ep.resolveToken('shareToken'),'SHORT','opt-in keeps share token in page memory');
  eq(ep.resolveToken('feedbackToken'),'SHORT2','opt-in keeps feedback token in page memory');
}

ok(src.includes("if (!_feedbackDomIntegrationEnabled) {\n                showNotification('Attachment integration is off."),'attachment integration is gated by page permission');
console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
