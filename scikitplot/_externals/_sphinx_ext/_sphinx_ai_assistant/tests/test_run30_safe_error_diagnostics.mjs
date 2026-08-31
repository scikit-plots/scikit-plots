import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(process.argv[2] || new URL('../_static/ai-assistant.js', import.meta.url), 'utf8');
const start = src.indexOf('    var _SENSITIVE_KEY = ');
const end = src.indexOf('    /**\n     * Gated, scrubbing logger', start);
if (start < 0 || end < 0) throw new Error('logger helper block not found');
const block = src.slice(start, end);
const helpers = new Function(block + '\nreturn {_sanitizeDiagnosticText,_safeErrorDiagnostic,_requestFailureDisplayText,_scrubArg};')();
const { _sanitizeDiagnosticText, _safeErrorDiagnostic, _requestFailureDisplayText, _scrubArg } = helpers;

let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function eq(got, want, name) { ok(JSON.stringify(got) === JSON.stringify(want), `${name}: got=${JSON.stringify(got)} want=${JSON.stringify(want)}`); }

const http = _scrubArg(new Error('AI request failed (HTTP 502).'), 0);
eq(http, {name:'Error', status:502, code:'AI_HTTP_ERROR', message:'AI request failed (HTTP 502).'}, 'HTTP failure keeps safe status/message');


const modelErr = new Error('AI request failed (HTTP 400).');
modelErr.status = 400; modelErr.code = 'PROXY_MODEL_NOT_ALLOWED';
const modelDiag = _scrubArg(modelErr, 0);
ok(modelDiag.code === 'PROXY_MODEL_NOT_ALLOWED' && modelDiag.status === 400, 'proxy-owned model rejection code retained');
ok(_requestFailureDisplayText(modelErr).includes('ALLOWED_MODELS'), 'model rejection points to server allow-list');

const upstream403 = new Error('AI request failed (HTTP 403).');
upstream403.status = 403; upstream403.code = 'UPSTREAM_AUTH_OR_ACCESS_REJECTED';
ok(_requestFailureDisplayText(upstream403).includes('HF_TOKEN'), 'upstream 403 points to server-side provider credential');

const direct = new Error('AI_DIRECT_PROVIDER_ENDPOINT');
ok(_requestFailureDisplayText(direct).includes('server-side proxy'), 'direct provider endpoint is explained as unsafe browser configuration');

const remote = _scrubArg(new Error('REMOTE_RESPONSE_TOO_LARGE'), 0);
ok(remote.code === 'REMOTE_RESPONSE_TOO_LARGE' && remote.message.includes('byte limit'), 'bounded-read code retained');
ok(!('stack' in remote), 'raw stack excluded');

ok(_requestFailureDisplayText(new Error('AI request failed (HTTP 404).')).includes('Endpoint Configuration'), '404 gives actionable endpoint hint');
ok(_requestFailureDisplayText(new Error('AI request failed (HTTP 429).')).includes('rate-limited'), '429 gives rate-limit hint');
ok(_requestFailureDisplayText(vm.runInNewContext("new TypeError('Failed to fetch')")).includes('CORS'), 'network error gives network/CORS hint');
ok(_requestFailureDisplayText(new SyntaxError('private response fragment')).includes('valid JSON'), 'parse error never echoes response fragment');

const secret = 'hf_abcdefghijklmnopqrstuvwxyz123456';
const dirty = new Error(`Bearer ${secret} at https://alice:pw@example.test/private?token=${secret} person@example.test`);
const clean = _scrubArg(dirty, 0);
ok(!JSON.stringify(clean).includes(secret), 'error credential redacted');
ok(!JSON.stringify(clean).includes('example.test'), 'error URL redacted');
ok(!JSON.stringify(clean).includes('person@example.test'), 'error email redacted');
ok(clean.code === 'AI_REQUEST_ERROR' && clean.message === 'AI request failed.', 'arbitrary error text is not logged');

const crossRealm = vm.runInNewContext("new Error('Failed to fetch')");
const cross = _safeErrorDiagnostic(crossRealm);
ok(cross && cross.name === 'Error' && cross.code === 'AI_NETWORK_ERROR' && cross.message.includes('could not be reached'), 'cross-realm network Error recognized');

const obj = _scrubArg({ authorization:'Bearer nope', endpoint:'https://example.test/x?token=abc', note:`sk-abcdefghijklmnopqrstuvwx` }, 0);
ok(obj.authorization === '[redacted]', 'sensitive key redacted');
ok(obj.endpoint === '<url-redacted>', 'ordinary URL string redacted');
ok(!obj.note.includes('sk-abcdefghijklmnopqrstuvwx'), 'credential-shaped ordinary string redacted');
ok(_sanitizeDiagnosticText('AKIAABCDEFGHIJKLMNOP', 320) === '<credential-redacted>', 'AWS access-key shape redacted');

const hostile = {};
Object.defineProperty(hostile, 'boom', { enumerable:true, get(){ throw new Error('getter executed'); } });
const hostileOut = _scrubArg(hostile, 0);
ok(hostileOut.boom === '<unreadable>', 'throwing getter cannot break logger');

const many = Array.from({length:40}, (_,i)=>i);
const manyOut = _scrubArg(many, 0);
ok(manyOut.length === 33 && manyOut[32] === '[+8 more]', 'arrays are bounded');

const forged = _sanitizeDiagnosticText('safe\nFAKE\rnext\u0000tail', 320);
ok(!forged.includes('\n') && !forged.includes('\r') && !forged.includes('\u0000'), 'literal control chars removed');
ok(forged.includes('\\n') && forged.includes('\\r') && forged.includes('<nul>'), 'control chars escaped');

const long = _sanitizeDiagnosticText('x'.repeat(2000), 64);
ok(long.startsWith('x'.repeat(64)) && long.endsWith('…<truncated>'), 'diagnostic text bounded');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
