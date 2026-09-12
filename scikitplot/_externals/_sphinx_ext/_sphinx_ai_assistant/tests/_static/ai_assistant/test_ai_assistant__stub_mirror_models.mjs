// Run 107 regression — stub catalog parity + JS fallback + local mirror routing.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let passed=0,failed=0; function ok(c,n){if(c)passed++;else{failed++;console.error('FAIL '+n);}}

const start=src.indexOf('var _JS_STUB_MODELS_DEFAULT_ENABLED');
const end=src.indexOf('// B41 page environment abstraction', start);
ok(start>=0 && end>start,'JS contains built-in stub catalog fallback block');
const block=src.slice(start,end);
const runtime=new Function('window', block+`\nreturn {cfg:_cfg,sync:_syncJsStubModels,enabled:_stubModelsEnabled,entries:_JS_STUB_MODEL_ENTRIES};`);

let win={AI_ASSISTANT_CONFIG:{}};
let r=runtime(win);
let cfg=r.cfg();
ok(cfg.panelApiModels.length===6,'missing Python config defaults to six JS stub models');
ok(cfg.panelApiModels.map(x=>x.id).join(',')==='stub-echo,stub-mirror,stub-error,stub-hostile,stub-qa,stub-slow','JS fallback order matches six canonical modes');
ok(cfg.panelApiModels.every(x=>x._localStubFallback===true),'no endpoint marks JS fallback stubs local');
ok(r.cfg().panelApiModels.length===6,'repeated config reads are idempotent and do not duplicate stubs');

win={AI_ASSISTANT_CONFIG:{panelStubModels:false,panelApiModels:[{id:'stub-echo',model:'stub/echo'},{id:'real',model:'real/model'}]}};
r=runtime(win); cfg=r.cfg();
ok(cfg.panelApiModels.length===1 && cfg.panelApiModels[0].id==='real','explicit panelStubModels=false strips built-in stub ids');
ok(r.enabled(cfg)===false,'explicit false is authoritative over JS default');

win={AI_ASSISTANT_CONFIG:{panelStubModels:true,panelApiModels:[{id:'real',model:'real/model',endpoint:'https://proxy.test/v1/chat/completions'}]}};
r=runtime(win); cfg=r.cfg();
ok(cfg.panelApiModels.length===7,'real model plus six stubs are available when enabled');
ok(cfg.panelApiModels.slice(1).every(x=>x.endpoint==='https://proxy.test/v1/chat/completions'),'JS stubs inherit the real proxy endpoint');
ok(cfg.panelApiModels.slice(1).every(x=>!x._localStubFallback),'endpoint-backed stubs are not marked local fallback');

const pythonLike=[
 {id:'real',model:'real/model',endpoint:'https://proxy.test/v1/chat/completions'},
 {id:'stub-echo',model:'stub/echo',endpoint:'https://proxy.test/v1/chat/completions'},
 {id:'stub-mirror',model:'stub/mirror',endpoint:'https://proxy.test/v1/chat/completions'},
 {id:'stub-error',model:'stub/error:503',endpoint:'https://proxy.test/v1/chat/completions'},
 {id:'stub-hostile',model:'stub/hostile',endpoint:'https://proxy.test/v1/chat/completions'},
 {id:'stub-qa',model:'stub/qa',endpoint:'https://proxy.test/v1/chat/completions'},
 {id:'stub-slow',model:'stub/slow:1500',endpoint:'https://proxy.test/v1/chat/completions'}
];
win={AI_ASSISTANT_CONFIG:{panelStubModels:true,panelApiModels:pythonLike}};
r=runtime(win); cfg=r.cfg();
ok(cfg.panelApiModels.length===7,'JS does not duplicate Python-injected stub models');

ok(src.includes('"panelStubModels": _cfg_bool')===false,'JS file does not contain Python config injection text');
ok(src.includes("id: 'stub-mirror', model: 'stub/mirror'"),'JS catalog includes mirror model');
ok(src.includes("label: 'Stub · mirror request chain'"),'mirror has an explicit client-boundary diagnostic label');
ok(src.includes('function _panelLocalStubReply('),'standalone JS has deterministic local stub execution');
ok(src.includes("mode === 'mirror'"),'local responder has a dedicated mirror branch');
ok(src.includes('browser-local request-chain inspector'),'local mirror is truthful about no HTTP request existing');
ok(src.includes("if (!endpoint && _isBuiltInStubModel(activeModel))"),'API-enabled/no-endpoint stub falls back locally instead of failing endpoint guard');
ok(src.includes('var localStubNoNetwork = _stubUsesLocalFallback'),'submit detects no-network local stub before privacy transport preflight');
ok(src.includes('if (cfg.panelApiEnabled && !localStubNoNetwork)'),'local stubs do not show a misleading network preflight');
ok(src.includes('var localActiveModel = _getActiveModel(cfg);'),'API-disabled submit still honors selected built-in stub model');
ok(src.includes("await _panelLocalStubReply(\n                        requestQuestion"),'API-disabled selected stub executes locally');

// Python-injected config contract is visible to JS by name; absence means fallback true.
ok(src.includes('return cfg.panelStubModels !== false;'),'JS fallback default is true unless explicit injected false');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
