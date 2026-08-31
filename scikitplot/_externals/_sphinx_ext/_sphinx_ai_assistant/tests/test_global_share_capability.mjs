import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const target = process.argv[2];
if (!target) throw new Error('usage: node test_global_share_capability.mjs <ai-assistant.js>');
const src = fs.readFileSync(target, 'utf8');
const here = path.dirname(fileURLToPath(import.meta.url));
const workerPath = path.join(here, '..', '_cf_worker', 'index.js');
const worker = fs.readFileSync(workerPath, 'utf8');

let passed = 0, failed = 0;
function ok(cond, name) {
  if (cond) { console.log('PASS', name); passed++; }
  else { console.log('FAIL', name); failed++; }
}
function has(text, re, name) { ok(re.test(text), name); }
function lacks(text, re, name) { ok(!re.test(text), name); }

const remote = src.slice(src.indexOf('function _remotePost'), src.indexOf('function _postFeedback'));
has(remote, /opts\.headers/, 'remote POST accepts explicit safe custom headers');
has(remote, /headers\[key\]\s*=\s*String\(opts\.headers\[key\]\)/, 'custom headers are copied as strings');

const patch = src.slice(src.indexOf('function _patchGlobalShare'), src.indexOf('function _postTrainingContribution'));
has(patch, /X-Share-Edit-Token/, 'fixed update sends per-share edit capability header');
lacks(patch, /Authorization[^\n]*editToken/, 'fixed update does not overload endpoint Authorization with edit capability');

const outerStart = src.indexOf('function _buildConversationShareSheet');
const outerEnd = src.indexOf('/**\n     * Build the "Project Links"', outerStart);
const outer = src.slice(outerStart, outerEnd > outerStart ? outerEnd : src.length);
has(outer, /var payload = recoveringGlobal \? _pendingGlobalCreate\.payload : \{ snapshot: snapshot, format: meta\.fmt, ttlDays: g\.ttlDays \}/, 'Global payload sends canonical snapshot');
has(outer, /format:\s*meta\.fmt/, 'Global payload sends allowlisted format id');
lacks(outer, /payload\s*=\s*\{[^}]*content\s*:/, 'Global payload no longer sends rendered content');
lacks(outer, /payload\s*=\s*\{[^}]*mimeType\s*:/, 'Global payload no longer sends MIME authority');
lacks(outer, /payload\s*=\s*\{[^}]*ext\s*:/, 'Global payload no longer sends extension authority');
has(outer, /editToken:\s*editToken/, 'POST result retains edit capability in live state');
has(outer, /_globalShareState && _globalShareState\.uuid && _globalShareState\.editToken/, 'fixed update requires live edit capability');
has(outer, /_patchGlobalShare\([\s\S]{0,220}_globalShareState\.editToken/, 'fixed update call passes the per-share edit capability');
has(outer, /read-only restored · status not checked/, 'restored public URL is explicitly read-only for mutation');
const loadSSStart = outer.indexOf('function _loadGlobalSS');
const loadSSEnd = outer.indexOf('function _saveGlobalSS', loadSSStart);
const loadSS = outer.slice(loadSSStart, loadSSEnd);
const saveSSStart = outer.indexOf('function _saveGlobalSS');
const saveSSEnd = outer.indexOf('_globalShareState = _loadGlobalSS();', saveSSStart);
const saveSS = outer.slice(saveSSStart, saveSSEnd);
lacks(loadSS, /return\s+state\s*;/, 'sessionStorage loader never trusts a parsed legacy object wholesale');
has(loadSS, /_saveGlobalSS\(safe\)/, 'sessionStorage loader destructively scrubs legacy forbidden fields');
lacks(loadSS, /editToken\s*:/, 'sessionStorage loader cannot rebuild edit capability');
lacks(saveSS, /editToken\s*:/, 'sessionStorage serializer excludes edit token');
lacks(saveSS, /contentHash\s*:/, 'sessionStorage serializer excludes conversation-derived content fingerprint');
has(saveSS, /uuid:\s*state\.uuid/, 'sessionStorage keeps public UUID only');
has(outer, /_deleteGlobalShare\([\s\S]{0,180}artifact\.editToken/, 'Global revoke uses per-share edit capability');
has(outer, /err\.status === 404 \|\| err\.status === 405 \|\| err\.status === 410/, 'fixed update fallback treats explicit expiry as a create-new boundary');
has(outer, /unavailable · reason unknown · recheck or forget/, 'HTTP 404 lifecycle copy remains reason-agnostic and recheckable');
has(outer, /if \(state === 'revoked' \|\| state === 'expired'\) \{[\s\S]{0,180}artifact\.editToken = ''/, 'only confirmed terminal lifecycle states erase live mutation capability');
has(outer, /artifact\.state === 'unavailable' && artifact\.editToken[\s\S]{0,420}_forgetGlobalArtifactRecord\(artifact\)/, 'unavailable with live capability exposes separate truthful Forget action');

const ledgerStart = outer.indexOf('function _saveGlobalLedger');
const ledgerEnd = outer.indexOf('function _findGlobalLedgerItem', ledgerStart);
const ledgerSave = outer.slice(ledgerStart, ledgerEnd);
has(outer, /ai-assistant-global-share-ledger:v1/, 'Global public-artifact ledger is session scoped and versioned');
has(outer, /_GLOBAL_LEDGER_MAX\s*=\s*25/, 'Global public-artifact ledger is bounded');
has(outer, /safeItems = _globalLedger\.map\([\s\S]{0,140}\.slice\(0, _GLOBAL_LEDGER_MAX\)/, 'Global ledger save enforces configured bound');
lacks(ledgerSave, /editToken\s*:/, 'Global public-artifact ledger never persists edit capability');
lacks(ledgerSave, /snapshot\s*:/, 'Global public-artifact ledger never persists conversation snapshot');
has(outer, /Check status/, 'Global artifact lifecycle exposes explicit status check');
has(src, /function _probeGlobalShareStatus[\s\S]{0,700}\/status[\s\S]{0,300}method:\s*'POST'/, 'Global status probe uses fixed POST path without capability in URL');
has(src, /function _probeGlobalShareStatus[\s\S]{0,900}body:\s*JSON\.stringify\(\{ shareId: loc\.id \}\)/, 'Global status probe carries public capability in request body');
has(src, /function _probeGlobalShareStatus[\s\S]{0,500}redirect:\s*'error'/, 'Global status probe does not follow redirects');
has(outer, /Global link revoked on the server/, 'Global revoke reports true remote deletion');
has(outer, /artifact\.busy = false;[\s\S]{0,220}_markGlobalArtifactState\(artifact, 'revoked'\)/, 'successful Global revoke clears busy before rendering Forget');

has(outer, /Already copied self-contained links cannot be revoked/, 'self-contained removal does not claim remote revocation');

has(src, /function _patchGlobalShare[\s\S]{0,500}\+ '\/update'/, 'Global update uses fixed request path');
has(src, /function _deleteGlobalShare[\s\S]{0,500}\+ '\/revoke'/, 'Global revoke uses fixed request path');
has(src, /function _globalShareLocator[\s\S]{0,500}#/, 'Global locator understands fragment-backed public links');

has(worker, /const SHARE_FORMATS = Object\.freeze/, 'Worker owns format allowlist');
has(worker, /yaml:\s*\{[^}]*application\/yaml/, 'Worker owns YAML MIME');
has(worker, /toml:\s*\{[^}]*application\/toml/, 'Worker owns TOML MIME');
has(worker, /_canonicalShareSnapshot\(shPayload\.snapshot\)/, 'Worker canonicalizes POST snapshot');
has(worker, /_shareFormat\(shPayload\.format\)/, 'Worker validates POST format enum');
lacks(worker.slice(worker.indexOf("request.method === 'POST' && url.pathname === '/v1/share'"), worker.indexOf("request.method === 'PATCH'", worker.indexOf("request.method === 'POST' && url.pathname === '/v1/share'"))), /shPayload\.mimeType|shPayload\.content|shPayload\.ext/, 'Worker POST ignores caller MIME/content/ext authority');
has(worker, /const editHash = operation \? operation\.editHash : await _sha256Hex\(editToken\)/, 'Worker stores only a management capability digest for envelope creates');
has(worker, /_verifyShareEditToken\(shToken, await _sha256Hex\(env\.SHARE_WRITE_TOKEN\)\)/, 'Worker create-token comparison is digest/constant-time based');
has(worker, /X-Share-Edit-Token/, 'Worker fixed update/revoke and legacy delete use edit capability header');
has(worker, /_verifyShareEditToken/, 'Worker verifies edit token before mutation');
has(worker, /request\.method === 'DELETE'/, 'Worker implements share revocation');
has(worker, /request\.method === 'HEAD'[\s\S]{0,220}v1\/share/, 'Worker implements content-free Share lifecycle probe');
has(worker, /request\.method === 'GET' && url\.pathname === '\/v1\/share'/, 'Worker serves fixed Share viewer path');
has(worker, /request\.method === 'POST' && url\.pathname === '\/v1\/share\/read'/, 'Worker fixed read endpoint keeps capability out of request path');
has(worker, /request\.method === 'POST' && url\.pathname === '\/v1\/share\/status'/, 'Worker fixed status endpoint keeps capability out of request path');
has(worker, /request\.method === 'POST' && url\.pathname === '\/v1\/share\/update'/, 'Worker fixed update endpoint keeps capability out of request path');
has(worker, /request\.method === 'POST' && url\.pathname === '\/v1\/share\/revoke'/, 'Worker fixed revoke endpoint keeps capability out of request path');
has(worker, /\/v1\/share#share=\$\{shareUuid\}/, 'Worker emits fragment-backed Global links');
has(worker, /request\.method === 'PATCH'[\s\S]{0,700}share\.legacy_update_retired[\s\S]{0,500}status:\s*410/, 'Worker legacy PATCH is retired and cannot extend capability-bearing transport');
has(worker, /request\.method === 'DELETE'[\s\S]{0,1800}current\.expiresAt[\s\S]{0,260}status:\s*410/, 'Worker DELETE reports explicit expiry instead of false revoke');
has(worker, /request\.method === 'GET'[\s\S]{0,1200}entry\.expiresAt[\s\S]{0,260}status:\s*410/, 'Worker GET rejects explicitly expired entries');
has(worker, /Access-Control-Allow-Methods[^\n]*HEAD/, 'Worker CORS allows explicit Share status HEAD');
has(worker, /Cache-Control': 'private, no-store'/, 'Worker share GET is no-store');
has(worker, /X-Robots-Tag': 'noindex, nofollow, noarchive'/, 'Worker share GET blocks indexing/archive');
has(worker, /Content-Security-Policy'[\s\S]{0,200}sandbox/, 'Worker HTML response CSP is sandboxed');
has(worker, /_shareUsage\(/, 'Worker enforces entry/aggregate capacity gate');
const shareRoutes = worker.slice(worker.indexOf("request.method === 'POST' && url.pathname === '/v1/share'"), worker.indexOf('// ── Method Guard'));
const shareLines = shareRoutes.split(/\r?\n/);
const writeLogLine = shareLines.find(line => line.includes("'share.write'")) || '';
const readLogLine = shareLines.find(line => line.includes("'share.read'")) || '';
const kvFailLine = shareLines.find(line => line.includes("'share.kv_fail'")) || '';
lacks(writeLogLine, /uuid\s*:/, 'Worker share.write log omits public capability');
lacks(readLogLine, /uuid\s*:/, 'Worker share.read log omits public capability');
lacks(kvFailLine, /uuid\s*:/, 'Worker KV failure log omits public capability');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
