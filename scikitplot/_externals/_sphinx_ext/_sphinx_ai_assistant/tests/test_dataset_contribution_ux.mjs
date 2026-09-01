import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed=0, failed=0;
function t(name, got, want=true){ if(got===want){passed++;} else {failed++; console.error(`FAIL ${name}\n  got: ${JSON.stringify(got)}\n want: ${JSON.stringify(want)}`);} }
function section(a,b){const i=src.indexOf(a);const j=src.indexOf(b,i+1);return i>=0&&j>i?src.slice(i,j):'';}

const share = section('function _buildConversationShareSheet(', 'function _buildLinksSheet()');
const contribution = section('function _buildDatasetContributionSheet()', 'function _buildKeyboardShortcutsSheet()');
const payload = section('function _buildDatasetContributionPayload(', 'function _datasetContributionPayloadBytes(');
const conversationBuilder = section('function _buildWholeConversationContributionRecord(', '/** Build the exact schema-v4 payload');
const feedbackPopup = section('// ── Popup ─', '// ── Expand button wiring');
const endpoint = section('var extSection = _buildSheetSection', '// ══════════════════════════════════════════════════════════════════════\n        // MOUNT + SUBSCRIBE');
const usage = section('function _buildUsagePolicySheet()', 'function _buildDatasetContributionSheet()');

t('dedicated contribution sheet exists', contribution.includes("sheet.id = 'ai-assistant-panel-contribution-sheet'"));
t('shared workspace title is explicit', contribution.includes("hStrong.textContent = 'Feedback & contribution'"));
t('This Q&A scope exists', contribution.includes("_scopeButton('qa', 'This Q&A'"));
t('Rated answers scope exists', contribution.includes("_scopeButton('rated', 'Rated answers'"));
t('Whole conversation scope exists', contribution.includes("_scopeButton('conversation', 'Whole conversation'"));
t('exact JSON inspection exists', contribution.includes("inspectBtn.textContent = 'Inspect JSON'"));
t('contribution actions reuse Endpoint Configuration I/O primitives',
  contribution.includes("ai-assistant-panel-ep-io-row ai-assistant-panel-contribution-inspect-row") &&
  contribution.includes("ai-assistant-panel-ep-io-btn ai-assistant-panel-contribution-action-btn"));
t('payload tools include local copy and download without submission',
  contribution.includes('⎘ Copy JSON to clipboard') &&
  contribution.includes('↓ Download JSON file') &&
  contribution.includes('Nothing was submitted.'));
t('receipt actions are grouped by authority and lifecycle',
  contribution.includes("'Private recovery'") && contribution.includes("'Maintainer support'") &&
  contribution.includes("'Review lifecycle'") && contribution.includes("tone === 'danger'"));
t('recovery actions use shared endpoint I/O button style',
  contribution.includes("recoveryActions.className = 'ai-assistant-panel-ep-io-row ai-assistant-panel-contribution-recovery-actions'"));
t('contribution uses human-readable section framing',
  contribution.includes("'Choose content'") && contribution.includes("'Review details'") &&
  contribution.includes("'Inspect payload'") && contribution.includes("'Consent & manage'"));
t('JSON preview is keyboard focusable and context-sized',
  contribution.includes("preview.setAttribute('tabindex', '0')") &&
  contribution.includes("preview.dataset.size = lines <= 14 ? 'compact' : (lines <= 32 ? 'medium' : 'large')"));
t('optional reviewer note exposes live character count',
  contribution.includes("noteCounter.textContent = noteInput.value.length + ' / ' + _CONTRIBUTION_NOTE_MAX_CHARS"));
t('preview uses current payload object', contribution.includes('JSON.stringify(payload, null, 2)'));
t('inspect copy explains redaction boundary', contribution.includes('exact payload entering privacy review') && contribution.includes('only the reviewed/redacted copy is sent'));
t('privacy review receives payload', contribution.includes('_privacyPreflightReview(payload'));
t('submission uses reviewed value', contribution.includes('_sendReviewedContribution(endpoint, review.value, envelope)'));
t('quarantine copy exists', contribution.includes('not training-eligible until an authorized review promotes it'));
t('provider-native review renders human IN REVIEW state', contribution.includes('Status: IN REVIEW') && contribution.includes("res.reviewMode === 'provider-pr'"));
t('provider review status maps closed review to NOT ACCEPTED', contribution.includes('Status: NOT ACCEPTED') && contribution.includes("reviewState === 'closed'"));
t('receipt management capability remains page-memory action', contribution.includes("'X-Contribution-Delete-Token': res.deleteToken"));
t('pending deletion does not overclaim forensic erasure', contribution.includes('does not claim forensic deletion from database pages, backups, or infrastructure snapshots'));
t('withdrawal does not overclaim provider erasure', contribution.includes('versioned provider history is not claimed physically erased'));

t('Share has no contribution button', !share.includes('Contribute rated answers'));
t('Share has no contribution controller', !share.includes('_postTrainingContribution'));
t('Share has no contribution delete capability', !share.includes('X-Contribution-Delete-Token'));

t('feedback popup labels anonymous telemetry clearly', feedbackPopup.includes("persistLabel.textContent = 'Anonymous rating telemetry'"));
t('feedback popup exposes explicit Q&A contribution', feedbackPopup.includes('Contribute this Q&A'));
t('Q&A shortcut dispatches contribution event', feedbackPopup.includes("'ai-assistant-open-contribution'"));
t('telemetry toggle and contribution shortcut are separate rows', feedbackPopup.includes('popSepContribution'));

t('schema v4 constant exists', src.includes('var _CONTRIBUTION_SCHEMA_VERSION = 4;'));
t('consent v2 constant exists', src.includes("var _CONTRIBUTION_CONSENT_VERSION = '2.0.0';"));
t('payload has no session identity', !payload.includes('sessionId'));
t('payload has no feedback event id', !payload.includes('feedbackId'));
t('conversation scope emits one record', payload.includes('if (conversation) records.push(conversation)'));
t('conversation record type explicit', conversationBuilder.includes("recordType: 'conversation'"));
t('conversation messages are ordered from transcript', conversationBuilder.includes('_transcript.forEach'));
t('errors excluded from conversation content', conversationBuilder.includes("if (m.role === 'error') answerIndex++"));
t('assistant model is per message', conversationBuilder.includes('model: m.model ?'));
t('assistant feedback is per message', conversationBuilder.includes('feedback: fb ?'));
t('conversation note is not silently truncated before review', !conversationBuilder.includes('.slice(0, _CONTRIBUTION_NOTE_MAX_CHARS)'));

t('main subbar Contribute action exists', src.includes("contributeLbl.textContent = 'Contribute'"));
t('hamburger has Contribute item', src.includes("label: 'Contribute'"));
t('canonical sheet registry includes contribution', /key:\s*'contribution'[\s\S]{0,80}sheet:\s*contributionSheet/.test(src));
t('private event opens canonical contribution sheet', src.includes("addEventListener('ai-assistant-open-contribution'") && src.includes("_assistantEvents !== 'undefined'"));

t('Endpoint Configuration renamed Runtime & Data', endpoint.includes("_buildSheetSection('Runtime & Data')"));
t('Endpoint has Feedback telemetry block', endpoint.includes("_buildExtSub('Feedback telemetry')"));
t('Endpoint has Dataset contributions block', endpoint.includes("_buildExtSub('Dataset contributions')"));
t('Endpoint can open contribution sheet', endpoint.includes("openContribution.textContent = 'Open contribution sheet'"));
t('user-facing endpoint label is dataset contribution', src.includes("label: 'Dataset contribution endpoint'"));

t('Usage Policy distinguishes telemetry and contribution', usage.includes('Feedback telemetry and dataset contribution are different'));
t('Usage Policy describes quarantine review promotion', usage.includes('contribution &rarr; quarantine &rarr; review &rarr; authorized promotion'));

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
