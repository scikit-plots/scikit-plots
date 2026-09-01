import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) { passed++; } else { failed++; console.error('FAIL ' + name); } }
function section(a, b) { const i=src.indexOf(a); const j=src.indexOf(b, i+1); return i>=0 && j>i ? src.slice(i,j) : ''; }

ok(src.includes("var _FEEDBACK_TELEMETRY_PREF_KEY = 'ai-assistant-feedback-telemetry-consent';"), 'rating telemetry uses a dedicated consent record');
ok(src.includes("saved.enabled !== true"), 'telemetry consent defaults fail closed');
ok(src.includes("saved.version !== _FEEDBACK_TELEMETRY_CONSENT_VERSION"), 'stale telemetry consent is rejected');
ok(src.includes("localStorage.removeItem('ai-assistant-feedback-telemetry')"), 'legacy boolean telemetry preference is revoked');
ok(src.includes('return false;'), 'feedback telemetry storage failure defaults off');
const telemetry = section('function _feedbackTelemetryPayload(detail)', 'function _postFeedback(url');
ok(telemetry.includes('schemaVersion: 4'), 'network feedback uses consent-aware schema v4');
ok(telemetry.includes('telemetryConsent: true'), 'network feedback carries explicit consent marker');
ok(telemetry.includes('telemetryConsentVersion: _FEEDBACK_TELEMETRY_CONSENT_VERSION'), 'network feedback carries versioned consent');
ok(telemetry.includes('telemetryConsentAt: _feedbackTelemetryGrantedAt'), 'network feedback carries consent timestamp');
for (const forbidden of ['query:', 'answer:', 'message:', 'model:', 'page:', 'conversationId:']) {
  ok(!telemetry.includes(forbidden), 'telemetry omits ' + forbidden);
}

const localEvent = section('function _feedbackLocalEventPayload(detail)', 'function _postFeedback(url');
ok(localEvent.includes('delete out.telemetryConsent'), 'public feedback event does not expose network consent state');
for (const forbidden of ['query:', 'answer:', 'message:', 'model:', 'page:', 'conversationId:']) {
  ok(!localEvent.includes(forbidden), 'public feedback event does not add ' + forbidden);
}
ok(src.includes("detail: _feedbackLocalEventPayload(detail || {})") || src.includes("return _feedbackLocalEventPayload(detail);"), 'public feedback event uses privacy-minimal detail');
ok(src.includes("if (!_feedbackPersistEnabled || !_feedbackTelemetryGrantedAt) { return false; }"), 'network helper self-gates on consent');
ok(src.includes("'Send anonymous rating telemetry'"), 'settings name is truthful');
ok(src.includes('Question text, answer text, written notes, model identity, page URL and conversation identifiers are excluded from telemetry'), 'settings explain excluded sensitive fields');
ok(src.includes("var _CONTRIBUTION_CONSENT_VERSION = '2.0.0';"), 'new contribution uses consent v2');
ok(src.includes('var _CONTRIBUTION_SCHEMA_VERSION = 4;'), 'new contribution uses schema v4');
const builder = section('function _buildDatasetContributionPayload(', 'function _datasetContributionPayloadBytes(');
ok(!builder.includes('sessionId:'), 'contribution does not send stable session id');
ok(!builder.includes('feedbackId:'), 'contribution does not cross-link feedback event id');
ok(!builder.includes('prevFeedbackId:'), 'contribution does not cross-link feedback edit chain');
const contribution = section('function _buildDatasetContributionSheet()', 'function _buildKeyboardShortcutsSheet()');
ok(contribution.includes("res && res.status === 'quarantined'"), 'UI recognizes quarantine state');
ok(contribution.includes("'Delete pending / withdraw training use'"), 'UI keeps receipt management meaningful after promotion');
ok(contribution.includes("'X-Contribution-Delete-Token'"), 'delete/withdraw uses separate receipt capability');
ok(contribution.includes("lifecycle.status === 'withdrawn'"), 'UI recognizes post-promotion withdrawal');
ok(contribution.includes('versioned provider history is not claimed physically erased'), 'UI does not overclaim provider-history erasure');
ok(contribution.includes('does not claim forensic deletion from database pages, backups, or infrastructure snapshots'), 'UI does not overclaim pending forensic erasure');
ok(contribution.includes('not training-eligible until an authorized review promotes it'), 'UI does not claim immediate training ingestion');
ok(src.includes("persistLabel.textContent = 'Anonymous rating telemetry'"), 'feedback popup labels telemetry without implying dataset save');
ok(src.includes("contributeLabel.textContent = 'Contribute this Q&A"), 'feedback popup has explicit Q&A contribution shortcut');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
