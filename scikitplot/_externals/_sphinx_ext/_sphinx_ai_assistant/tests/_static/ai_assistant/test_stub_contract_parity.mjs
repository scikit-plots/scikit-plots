// Run 115 — client/proxy stub identity is a fail-closed diagnostic contract.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let pass=0,fail=0; function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
 for(const pre of ['async function ','function ']){const i=src.indexOf(pre+name+'(');if(i<0)continue;let d=0,st=false,q=null,esc=false,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1]||'';if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)}}throw new Error('missing '+name)}

const isStub=extract('_isBuiltInStubModel');
const modeName=extract('_stubModeName');
const compat=extract('_stubReplyCompatibility');
const requireCompat=extract('_requireStubReplyCompatibility');
const replyExtract=extract('_extractPanelReply');
const factory=new Function(`
 var _JS_STUB_MODEL_IDS={};
 ${isStub}\n${modeName}\n${compat}\n${requireCompat}\n${replyExtract}
 return {compat:_stubReplyCompatibility,required:_requireStubReplyCompatibility,extract:_extractPanelReply};
`);
const h=factory();
const mirror={id:'stub-mirror',model:'stub/mirror'};
const echo={id:'stub-echo',model:'stub/echo'};
const normal={id:'normal',model:'openai/gpt'};
const stale='**Stub echo** — no model was called and no credential was read.\n\n- model: `stub/mirror`\n- available modes: `echo`, `error`, `hostile`, `qa`, `slow`';
const goodMirror='**Stub mirror · request-chain security inspector**\n\n- model: `stub/mirror`';
const goodEcho='**Stub echo** — no model was called.\n\n- available modes: `echo`, `mirror`, `error`, `hostile`, `qa`, `slow`';

ok(h.compat(goodMirror,{model:'stub/mirror',stub_report:{mode:'mirror',model:'stub/mirror'}},mirror).ok,'correct mirror identity accepted');
ok(!h.compat(stale,{model:'stub/mirror',stub_report:{mode:'mirror',model:'stub/mirror'}},mirror).ok,'textual echo cannot masquerade as mirror even when metadata reflects requested model');
ok(!h.compat(goodMirror,{stub_report:{mode:'echo',model:'stub/echo'}},mirror).ok,'structured mode mismatch fails closed');
let threw=false;try{h.required(stale,{model:'stub/mirror'},mirror)}catch(e){threw=e.message==='AI_STUB_MODE_MISMATCH'&&e.code==='AI_STUB_MODE_MISMATCH'}
ok(threw,'required compatibility raises bounded AI_STUB_MODE_MISMATCH');

const legacyEcho=h.compat(stale.replace('`stub/mirror`','`stub/echo`'),{model:'stub/echo'},echo);
ok(legacyEcho.ok,'legacy echo itself remains usable');
ok(legacyEcho.text.includes('Compatibility warning:'),'legacy five-mode echo receives explicit compatibility warning');
ok(legacyEcho.text.includes('does not include `mirror`'),'warning names missing mirror capability');
const currentEcho=h.compat(goodEcho,{model:'stub/echo',stub_report:{mode:'echo'}},echo);
ok(currentEcho.ok && !currentEcho.text.includes('Compatibility warning:'),'current six-mode echo has no false warning');
ok(h.compat('anything',{model:'openai/gpt'},normal).ok,'ordinary provider replies are not stub-validated');

ok(h.extract({choices:[{message:{content:' openai '}}]},false)==='openai','shared extractor reads OpenAI shape');
ok(h.extract({content:[{type:'text',text:'anthropic'}]},true)==='anthropic','shared extractor reads Anthropic shape');
ok(h.extract({reply:'flat'},false)==='flat','shared extractor reads flat shim shape');
ok(h.extract({answer:'answer'},false)==='answer','shared extractor reads flat answer shape');
ok(h.extract({text:'text'},false)==='text','shared extractor reads flat text shape');

const nonstream=extract('_panelApiCall');
const streaming=extract('_panelApiCallStreaming');
ok(nonstream.includes('_extractPanelReply(data, isAnthropic)'),'non-streaming path uses shared response extractor');
ok(nonstream.includes('_requireStubReplyCompatibility(reply, data, activeModel)'),'non-streaming path validates stub identity before display');
ok(streaming.includes("_extractPanelReply(data2, provider === 'anthropic')"),'JSON streaming fallback uses shared extractor');
ok(streaming.includes('_requireStubReplyCompatibility(reply2, data2, activeModel)'),'JSON streaming fallback validates stub identity');
ok(streaming.includes("_extractPanelReply(fbData, provider === 'anthropic')"),'getReader compatibility fallback parses OpenAI/Anthropic/flat consistently');
ok(streaming.includes('_requireStubReplyCompatibility(fbReply, fbData, activeModel)'),'getReader fallback validates stub identity');
ok(streaming.includes('var streamStubMeta = null'),'SSE path captures structured stub report');
ok(streaming.includes('var validatingStub = _isBuiltInStubModel(activeModel)'),'SSE recognizes diagnostic stubs');
ok(streaming.includes('if (!validatingStub)'),'SSE diagnostic text is buffered before validation');
ok(streaming.includes('_requireStubReplyCompatibility(accumulated, streamStubMeta || {}, activeModel)'),'SSE validates final identity before rendering/recording');
ok((streaming.match(/var sseBuf = '';/g)||[]).length===1,'SSE buffer has one declaration');
ok(src.includes("message === 'AI_STUB_MODE_MISMATCH'"),'safe diagnostic explicitly owns stub mismatch code');
ok(src.includes("d.code === 'AI_STUB_MODE_MISMATCH'"),'reader-facing error has dedicated stale-proxy explanation');
ok(src.includes('The proxy appears stale or incompatible'),'mismatch error explains remediation without echoing provider body');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
