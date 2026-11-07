// Run 108: six visible stub modes + local fallback parity.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let p=0,f=0; const ok=(c,n)=>{if(c)p++;else{f++;console.error('FAIL '+n);}};

const start=src.indexOf('var _JS_STUB_MODELS_DEFAULT_ENABLED');
const end=src.indexOf('// B41 page environment abstraction', start);
const block=src.slice(start,end);
const rt=new Function('window', block+'\nreturn {cfg:_cfg,entries:_JS_STUB_MODEL_ENTRIES,ids:_JS_STUB_MODEL_IDS};');
let r=rt({AI_ASSISTANT_CONFIG:{}}); let cfg=r.cfg();
const expected=['stub-echo','stub-mirror','stub-error','stub-hostile','stub-qa','stub-slow'];
ok(JSON.stringify(cfg.panelApiModels.map(x=>x.id))===JSON.stringify(expected),'standalone JS exposes six modes in canonical order');
ok(r.entries.length===6,'JS fallback catalog has six entries');
ok(Object.keys(r.ids).length===6,'JS built-in id authority has six entries');
ok(cfg.panelApiModels.find(x=>x.id==='stub-error').model==='stub/error:503','error fixture defaults to HTTP 503');
ok(cfg.panelApiModels.find(x=>x.id==='stub-slow').model==='stub/slow:1500','slow fixture defaults to 1500 ms');
ok(cfg.panelApiModels.find(x=>x.id==='stub-mirror').label==='Stub · mirror request chain','mirror label describes client boundary');

r=rt({AI_ASSISTANT_CONFIG:{panelStubModels:false,panelApiModels:cfg.panelApiModels.concat([{id:'real',model:'x'}])}});
ok(r.cfg().panelApiModels.length===1 && r.cfg().panelApiModels[0].id==='real','explicit false removes all six stubs');

ok(src.includes("/^stub\\/(?:echo|mirror|error(?::[0-9]+)?|hostile|qa|slow(?::[0-9]+)?)$/i"),'local built-in recognizer accepts all six mode shapes');
ok(src.includes("if (mode === 'error')"),'browser-local fallback has deterministic error mode');
ok(src.includes("if (mode === 'slow')"),'browser-local fallback has deterministic slow mode');
ok(src.includes('browser-local request-chain inspector'),'browser-local Mirror remains available without a proxy');
ok(src.includes('No proxy endpoint is configured, so **no HTTP request exists to mirror**.'),'local Mirror does not falsely claim a wire request');
ok(src.includes("'`echo`, `mirror`, `error`, `hostile`, `qa`, `slow`'"),'local Mirror advertises all six modes');
ok(src.includes('_splitQuestionWithAttachments(question)'),'local Mirror separates user input from one-turn files');
ok(src.includes('_scanInjection(split.attachmentContext)'),'local Mirror scans uploaded file text for advisory injection indicators');
ok(src.includes('_localMirrorRedact'),'local Mirror uses a display-only secret redaction layer');

console.log(`${p} passed, ${f} failed`); if(f)process.exit(1);
