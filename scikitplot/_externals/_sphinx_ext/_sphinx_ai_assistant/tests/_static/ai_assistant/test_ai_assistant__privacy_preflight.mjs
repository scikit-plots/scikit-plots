import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) {
  if (cond) passed++;
  else { failed++; console.error('FAIL ' + name); }
}
function section(a, b) {
  const i = src.indexOf(a); const j = src.indexOf(b, i + 1);
  return i >= 0 && j > i ? src.slice(i, j) : '';
}

const start = src.indexOf('var _SECRET_PATTERNS = [');
const end = src.indexOf('var _HIDDEN_SELECTORS = [', start);
ok(start >= 0 && end > start, 'privacy helper source section found');
const helper = src.slice(start, end);
const ctx = { Promise, console };
vm.createContext(ctx);
vm.runInContext(helper + `\nthis.__api = {scan:_privacyPreflightScan, redact:_privacyRedactValue};`, ctx);
const { scan, redact } = ctx.__api;

const secret = 'sk-' + 'A'.repeat(32);
const hf = 'hf_' + 'b'.repeat(32);
const jwt = 'eyJabcdefghijk.abcdefghijk.abcdefghijk';
const email = 'alice@example.com';
const phone = '+90 555 123 45 67';
const ip = '203.0.113.42';
const card = '4111 1111 1111 1111';
const hostile = `contact ${email} ${phone} ${ip} card ${card} key ${secret} ${hf} ${jwt} \u202Ehidden\u202C zero\u200Bwidth`;
const findings = scan(hostile);
ok(findings.flagged === true, 'sensitive text is flagged');
ok(findings.secret_findings.length >= 3, 'multiple credential classes are counted');
ok(findings.personal_findings.some(x => x.pattern === 'email_address'), 'email category detected');
ok(findings.personal_findings.some(x => x.pattern === 'international_phone'), 'phone category detected');
ok(findings.personal_findings.some(x => x.pattern === 'ipv4_address'), 'IP category detected');
ok(findings.personal_findings.some(x => x.pattern === 'payment_card_number'), 'Luhn-valid card category detected');
ok(findings.control_findings.some(x => x.codepoint === 'U+202E'), 'RTL override surfaced by codepoint');
ok(findings.control_findings.some(x => x.codepoint === 'U+200B'), 'zero-width character surfaced by codepoint');

const serializedFindings = JSON.stringify(findings);
for (const value of [secret, hf, jwt, email, phone, ip, card]) {
  ok(!serializedFindings.includes(value), 'finding object never retains matched value');
}

const clean = scan('Ordinary documentation question about array shapes and model metrics.');
ok(clean.flagged === false, 'ordinary prose is not flagged');

const original = {records: [{text: hostile}], nested: {note: email}};
const redacted = redact(original);
ok(original.records[0].text === hostile, 'redaction does not mutate original object');
ok(!redacted.records[0].text.includes(secret), 'redaction removes OpenAI-shaped secret');
ok(!redacted.records[0].text.includes(hf), 'redaction removes HF-shaped secret');
ok(!redacted.records[0].text.includes(email), 'redaction removes email');
ok(!redacted.records[0].text.includes(phone), 'redaction removes phone-like number');
ok(!redacted.records[0].text.includes(ip), 'redaction removes IP');
ok(!redacted.records[0].text.includes(card), 'redaction removes payment-card-like number');
ok(!/[\u200B-\u200F\u202A-\u202E\u2060-\u2064\u2066-\u2069\uFEFF]/.test(redacted.records[0].text), 'explicit redaction removes invisible/bidi controls');
ok(redacted.records[0].text.includes('[redacted:email_address]'), 'redaction is type-labelled, not silent deletion');

const review = section('function _privacyPreflightReview(value, options)', 'var _HIDDEN_SELECTORS = [');
ok(review.includes("'Redact & continue'"), 'dialog offers explicit redaction action');
ok(review.includes("'Continue unchanged'"), 'dialog offers explicit unchanged action');
ok(review.includes("'Go back'"), 'dialog offers explicit cancel/edit action');
ok(review.includes('Detection is advisory and incomplete'), 'dialog does not claim complete detection');
ok(review.includes('No matching value is logged or sent by this warning'), 'dialog states warning itself does not exfiltrate matches');
ok(review.includes("e.key === 'Escape'"), 'Escape cancels preflight');
ok(review.includes("e.key === 'Tab'"), 'Tab focus is trapped inside preflight dialog');
ok(review.includes("return Promise.resolve({ action: 'cancel'"), 'flagged data fails closed when dialog cannot be built');

const submit = section('async function handleAIPanelSubmit()', '// ── API + stub');
ok(submit.includes('await _privacyPreflightReview(outboundCandidate'), 'inference uses privacy preflight');
ok(submit.indexOf('await _privacyPreflightReview(outboundCandidate') < submit.indexOf('_appendPanelMessage(questionText'), 'inference preflight runs before transcript mutation');
ok(submit.includes("page_context: preparedPageContext.text"), 'inference preflight includes page context');
ok(submit.includes("cancelLabel: 'Edit message'"), 'cancel preserves edit workflow');

const sharePanel = section('function _buildConversationShareSheet(initialFmt)', '/**\n     * Build the "Project Links"');
ok(sharePanel.includes('await _privacyPreflightReview(snapshot'), 'all Share destinations use shared privacy preflight');
ok(sharePanel.includes("destination: destinationLabel"), 'Share preflight is destination-aware');
ok(sharePanel.includes("var artifact = _addArtifact({ kind: 'local', url: url, snapshot: snapshot"), 'local preview retains exact reviewed snapshot');
ok(sharePanel.includes("var artifact = _addArtifact({ kind: 'self_contained', url: url, bytes: portable.bytes"), 'self-contained artifact stores canonical reviewed data URL without duplicating the reviewed snapshot');
ok(sharePanel.includes("snapshot: snapshot, bytes: bytes, format: meta.fmt"), 'Global managed artifact retains reviewed snapshot');
const contributionPanel = section('function _buildDatasetContributionSheet()', 'function _buildKeyboardShortcutsSheet()');
ok(contributionPanel.includes("var review = await _privacyPreflightReview(payload"), 'dataset contribution uses privacy preflight');
ok(sharePanel.includes('var content = meta.buildStr(snapshot)'), 'serializer consumes reviewed snapshot');
ok(sharePanel.includes("window.open(artifact.url, '_blank', 'noopener,noreferrer')"), 'artifact reopening uses the already-generated canonical URL and never rebuilds Self-contained content from the raw transcript');

ok(!review.includes('PII free'), 'user-facing review never claims PII-free status');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
