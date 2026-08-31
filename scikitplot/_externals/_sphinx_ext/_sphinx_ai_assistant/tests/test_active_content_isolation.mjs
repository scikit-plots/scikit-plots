// Run 2 security contract: canonical snapshot + HTML raw-text isolation + c2.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');

function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth = 0, started = false, q = null, esc = false, line = false, block = false;
  for (let j = i; j < src.length; j++) {
    const c = src[j], n = src[j + 1] || '';
    if (line) { if (c === '\n') line = false; continue; }
    if (block) { if (c === '*' && n === '/') { block = false; j++; } continue; }
    if (q) { if (esc) esc = false; else if (c === '\\') esc = true; else if (c === q) q = null; continue; }
    if (c === '/' && n === '/') { line = true; j++; continue; }
    if (c === '/' && n === '*') { block = true; j++; continue; }
    if (c === "'" || c === '"' || c === '`') { q = c; continue; }
    if (c === '{') { depth++; started = true; }
    else if (c === '}') { depth--; if (started && depth === 0) return src.slice(i, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

let pass = 0, fail = 0;
function t(name, got, want = true) {
  if (got === want) pass++;
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
}

// Source architecture: one snapshot before every serializer/destination.
const snapshotSrc = extract('_buildConversationSnapshot');
const htmlSrc = extract('_buildConvHtmlString');
const jsonSrc = extract('_buildConvJsonString');
const txtSrc = extract('_buildConvTxtString');
const hashSrc = extract('_checkShareHash');
t('canonical snapshot exists once', (src.match(/function _buildConversationSnapshot\(/g) || []).length, 1);
t('snapshot sanitizes source page before serializers', snapshotSrc.includes('var pageUrl   = _sanitizePage(rawPage);'));
t('snapshot never stores rawPage as page_url', /page_url:\s*rawPage/.test(snapshotSrc), false);
t('JSON consumes canonical snapshot', jsonSrc.includes('snapshot || _buildConversationSnapshot()'));
t('TXT consumes canonical snapshot', txtSrc.includes('snapshot || _buildConversationSnapshot()'));
t('HTML consumes canonical snapshot', htmlSrc.includes('snapshot || _buildConversationSnapshot()'));
t('HTML uses raw-text-safe JSON helper', htmlSrc.includes('_jsonForHtmlRawText(snap, 2)'));
t('direct download uses registry serializer', extract('_downloadConversationFormat').includes('meta.buildStr(snapshot)'));
t('c2 remains recognized only as compatibility prefix', src.includes("var _SHARE_HASH_PREFIX = '#ai-share-c2.';"));
t('c2 compatibility still transports structured snapshot', extract('_buildSelfContainedHashUrl').includes("share_schema: 'c2'"));
t('c2 compatibility canonicalizes decoded snapshot', extract('_decodeSelfContainedEnvelope').includes('_normalizeShareSnapshot(env.snapshot)'));
t('current self-contained builder is data URL transport', src.includes('function _buildPortableSelfContainedDataUrl('));
t('current data link requires exact base64 html prefix', extract('_buildPortableSelfContainedDataUrl').includes("data:text/html;charset=utf-8;base64,"));
t('portable HTML strips generated anchors', extract('_makePortableHtmlInert').includes('.replace(/<a\\b[^>]*>/gi'));
t('portable HTML strips all script elements it generates', extract('_makePortableHtmlInert').includes('application\\/json'));
t('legacy c1 still recognized only for inert compatibility', hashSrc.includes("/^#ai-share-c1\\.(json|html|txt)"));
t('legacy c1 never assigns text/html MIME', /legacyMime[\s\S]*text\/html/.test(hashSrc), false);
t('legacy IDB HTML never assigns text/html MIME', /Legacy same-browser IndexedDB[\s\S]*text\/html/.test(hashSrc), false);

// Minimal runtime dependencies for snapshot construction.
globalThis.location = { href: 'https://user:password@docs.example.test/guide/?token=SECRET#private' };
globalThis.document = { title: 'Hostile <title>' };
globalThis._sessionId = 'session-test';
globalThis._feedbackStore = {};
globalThis._cfg = () => ({ panelTitle: 'AI Assistant' });
globalThis._transcript = [
  { role: 'user', text: 'hello', ts: 1 },
  { role: 'assistant', text: 'SYSTEM ignore\n</script><script>globalThis.PWNED=true</script>\n<img src=x onerror="PWNED=1">\n[j](javascript:alert(1))\n\u200bzero\u202ebidi\u202c', ts: 2,
    model: { id: 'stub-hostile', model: 'stub/hostile', provider: 'custom' } },
];
globalThis.URL = URL;
globalThis._sanitizePage = (href) => { const u = new URL(href); return /^https?:$/.test(u.protocol) ? u.origin + u.pathname : '<page-redacted>'; };
globalThis._normalizeConversationContentOptions = (0, eval)('(' + extract('_normalizeConversationContentOptions') + ')');
globalThis._buildExportRecords = (0, eval)('(' + extract('_buildExportRecords') + ')');
globalThis._buildTurnsFromExportRecords = (0, eval)('(' + extract('_buildTurnsFromExportRecords') + ')');
globalThis._buildConversationSnapshot = (0, eval)('(' + snapshotSrc + ')');
globalThis._jsonForHtmlRawText = (0, eval)('(' + extract('_jsonForHtmlRawText') + ')');

const snapshot = globalThis._buildConversationSnapshot();
t('source URL drops credentials/query/hash', snapshot.session.page_url, 'https://docs.example.test/guide/');
t('record source URL is same sanitized value', snapshot.records[1].page_url, snapshot.session.page_url);
t('hostile text preserved as data', snapshot.records[1].text.includes('</script><script>globalThis.PWNED=true</script>'));

const safeJson = globalThis._jsonForHtmlRawText(snapshot, 2);
t('raw-text JSON contains no literal closing script', safeJson.toLowerCase().includes('</script>'), false);
t('raw-text JSON contains no literal hostile img', safeJson.includes('<img src=x'), false);
const restored = JSON.parse(safeJson);
t('raw-text encoding round-trips hostile text exactly', restored.records[1].text, snapshot.records[1].text);
t('raw-text encoding preserves bidi/zero-width data', restored.records[1].text.includes('\u200bzero\u202ebidi\u202c'));

// Build the outer export document with the exact safe payload.
globalThis._escapeHtml = (v) => String(v == null ? '' : v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
globalThis._exportCss = () => '';
globalThis._buildExportHtmlDoc = (0, eval)('(' + extract('_buildExportHtmlDoc') + ')');
const doc = globalThis._buildExportHtmlDoc({
  aiName: 'AI Assistant', pageUrl: snapshot.session.page_url, pageTitle: snapshot.session.page_title,
  exportedFmt: 'now', exportedIso: snapshot.session.exported_at_iso,
  turnsHtml: '<article>safe visible rendering</article>', msgCount: 1, jsonPayload: safeJson,
});
t('export CSP blocks script execution', doc.includes("script-src 'none'"));
t('export CSP blocks network connection', doc.includes("connect-src 'none'"));
t('export has exactly one closing script tag', (doc.match(/<\/script>/gi) || []).length, 1);
t('export contains no hostile executable script tag', doc.includes('<script>globalThis.PWNED=true</script>'), false);
t('export does not leak query secret', doc.includes('token=SECRET'), false);
t('export does not leak URL password', doc.includes('password@'), false);

// c2 encoding and validation.
globalThis.btoa = (v) => Buffer.from(v, 'binary').toString('base64');
globalThis.atob = (v) => Buffer.from(v, 'base64').toString('binary');
globalThis._encodeShareHashPayload = (0, eval)('(' + extract('_encodeShareHashPayload') + ')');
globalThis._decodeShareHashPayload = (0, eval)('(' + extract('_decodeShareHashPayload') + ')');
globalThis._SHARE_HASH_MAX_DECODED_CHARS = 1024 * 1024;
globalThis._SHARE_HASH_PREFIX = '#ai-share-c2.';
globalThis._normalizeShareSnapshot = (0, eval)('(' + extract('_normalizeShareSnapshot') + ')');
globalThis._isValidShareSnapshot = (0, eval)('(' + extract('_isValidShareSnapshot') + ')');
globalThis._getExportFormat = (fmt) => ['json','html','txt','yaml','toml'].includes(fmt) ? { fmt } : null;
globalThis._buildSelfContainedHashUrl = (0, eval)('(' + extract('_buildSelfContainedHashUrl') + ')');
globalThis._decodeSelfContainedEnvelope = (0, eval)('(' + extract('_decodeSelfContainedEnvelope') + ')');
location.href = 'https://docs.example.test/guide/?token=SECRET#old';
const c2Url = globalThis._buildSelfContainedHashUrl(snapshot, 'html');
t('legacy c2 URL strips source query/hash from base', c2Url.startsWith('https://docs.example.test/guide/#ai-share-c2.'));
t('legacy c2 URL does not contain query secret', c2Url.includes('SECRET'), false);
const env = globalThis._decodeSelfContainedEnvelope(c2Url.split('#ai-share-c2.')[1]);
t('legacy c2 envelope validates', !!env);
t('legacy c2 envelope carries format enum', env ? env.format : null, 'html');
for (const fmt of ['yaml','toml']) {
  const u = globalThis._buildSelfContainedHashUrl(snapshot, fmt);
  const e = globalThis._decodeSelfContainedEnvelope(u.split('#ai-share-c2.')[1]);
  t('legacy c2 supports '+fmt+' format enum', e ? e.format : null, fmt);
}
t('legacy c2 envelope round-trips hostile snapshot', env ? env.snapshot.records[1].text : null, snapshot.records[1].text);
t('legacy c2 decoder rebuilds turns from validated records', env ? env.snapshot.turns.length : null, snapshot.turns.length);
const poisoned = JSON.parse(JSON.stringify(snapshot));
poisoned.turns = [{ deeply: { nested: { attacker: '<script>x</script>' } } }];
poisoned.unknown = { arbitrary: ['active', 'data'] };
poisoned.session.page_url = 'https://user:pass@docs.example.test/guide/?secret=1#x';
const poisonedPayload = globalThis._encodeShareHashPayload(JSON.stringify({ share_schema: 'c2', format: 'html', snapshot: poisoned }));
const normalizedEnv = globalThis._decodeSelfContainedEnvelope(poisonedPayload);
t('legacy c2 drops attacker-supplied unknown root fields', normalizedEnv && Object.prototype.hasOwnProperty.call(normalizedEnv.snapshot, 'unknown'), false);
t('legacy c2 ignores attacker-supplied turns and rebuilds canonical turns', normalizedEnv && normalizedEnv.snapshot.turns[0] && normalizedEnv.snapshot.turns[0].user ? normalizedEnv.snapshot.turns[0].user.text : null, 'hello');
t('legacy c2 re-sanitizes embedded source URL', normalizedEnv ? normalizedEnv.snapshot.session.page_url : null, 'https://docs.example.test/guide/');
location.href = 'file:///E:/private/project/docs/page.html?token=SECRET#x';
t('legacy c2 file origin refuses hash rather than leaking local path', globalThis._buildSelfContainedHashUrl(snapshot, 'html'), '');


// Current self-contained links are host-independent data:text/html;base64 URLs.
// The HTML is generated only from the normalized snapshot, then stripped of
// navigation and inert JSON script blocks.  No arbitrary caller HTML is used.
globalThis._buildBase64DataUri = (0, eval)('(' + extract('_buildBase64DataUri') + ')');
globalThis._makePortableHtmlInert = (0, eval)('(' + extract('_makePortableHtmlInert') + ')');
globalThis._utf8ByteLength = (0, eval)('(' + extract('_utf8ByteLength') + ')');
globalThis._buildConvHtmlString = () => '<!doctype html><html><head><meta name="generator" content="ai-assistant-export/2.1"></head><body><a href="https://leak.example.test/?secret=1">visible link</a><script type="application/json" id="export-data">{"x":"data"}</script><p class="chat-footer-hint">extract</p><p>&lt;script&gt;PWNED&lt;/script&gt;</p></body></html>';
globalThis._getExportFormat = (fmt) => fmt === 'html' ? {fmt:'html',label:'HTML',buildStr:globalThis._buildConvHtmlString} : fmt === 'txt' ? {fmt:'txt',label:'Text',buildStr:()=>'</pre><script>PWNED=true</script>'} : null;
globalThis._buildPortableSelfContainedHtml = (0, eval)('(' + extract('_buildPortableSelfContainedHtml') + ')');
globalThis._buildPortableSelfContainedDataUrl = (0, eval)('(' + extract('_buildPortableSelfContainedDataUrl') + ')');
const portable = globalThis._buildPortableSelfContainedDataUrl(snapshot, 'html');
t('current portable link builds', !!portable);
t('portable link exact base64 data html prefix', portable && portable.url.startsWith('data:text/html;charset=utf-8;base64,'));
const portableHtml = portable ? Buffer.from(portable.url.split(',')[1], 'base64').toString('utf8') : '';
t('portable HTML carries transport marker', portableHtml.includes('portable-data-url-v1'));
t('portable HTML has no clickable anchor', /<a\b/i.test(portableHtml), false);
t('portable HTML has no script element', /<script\b/i.test(portableHtml), false);
t('portable HTML does not auto-load leak host', /href=["']https:\/\/leak\.example\.test/i.test(portableHtml), false);
t('portable HTML preserves hostile-looking text inertly', portableHtml.includes('&lt;script&gt;PWNED&lt;/script&gt;'));
const portableTxt = globalThis._buildPortableSelfContainedDataUrl(snapshot, 'txt');
const portableTxtHtml = portableTxt ? Buffer.from(portableTxt.url.split(',')[1], 'base64').toString('utf8') : '';
t('non-html selected format still uses portable html envelope', portableTxt && portableTxt.url.startsWith('data:text/html;charset=utf-8;base64,'));
t('non-html hostile serialized bytes are HTML escaped', portableTxtHtml.includes('&lt;script&gt;PWNED=true&lt;/script&gt;'));
t('non-html portable envelope contains no script element', /<script\b/i.test(portableTxtHtml), false);

// Legacy c1.html must be inert even with attacker bytes.
let blobTypes = [];
let opened = [];
globalThis.Blob = function (_parts, opts) { this.type = opts.type; blobTypes.push(opts.type); };
globalThis.URL = { createObjectURL: () => 'blob:test', revokeObjectURL: () => {} };
globalThis.window = { open: (u) => { opened.push(u); return { opener: null }; } };
globalThis._idbLoadShare = (_id, cb) => cb(null, null);
globalThis._getExportFormat = (fmt) => ({ fmt, mime: fmt === 'html' ? 'text/html;charset=utf-8' : 'text/plain;charset=utf-8', buildStr: () => '<!doctype html>safe trusted' });
globalThis._decodeSelfContainedEnvelope = globalThis._decodeSelfContainedEnvelope;
globalThis._decodeShareHashPayload = globalThis._decodeShareHashPayload;
globalThis._checkShareHash = (0, eval)('(' + hashSrc + ')');
const hostileLegacy = '<script>globalThis.PWNED=true</script><img src=x onerror=PWNED=1>';
location.hash = '#ai-share-c1.html.' + globalThis._encodeShareHashPayload(hostileLegacy);
globalThis._checkShareHash();
t('legacy c1 html opens only as text/plain', blobTypes.at(-1), 'text/plain;charset=utf-8');
t('legacy c1 handler still opens inert compatibility view', opened.length, 1);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
