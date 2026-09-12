// Run 63 post-upload audit: pending FileReader work must gate Send, and
// Retry/Edit must preserve canonical attachment context without flattening it
// into the textarea/question limit.
import fs from 'node:fs';
const js = fs.readFileSync(process.argv[2], 'utf8');
const cssPath = process.argv[3] || process.argv[2].replace(/ai-assistant\.js$/, 'ai-assistant.css');
const css = fs.readFileSync(cssPath, 'utf8');
let pass=0, fail=0;
function t(name, got, want=true) {
  if (got === want) pass++;
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
}

t('pending counter exists', js.includes('var _attachmentStagePending = 0'));
t('queue marks staging pending before async work', /function _queueComposerFiles[\s\S]*?_attachmentStagePending\+\+[\s\S]*?_updateAttachmentStageUi\(\)/.test(js));
t('pending count clears in finally for current generation', js.includes('_attachmentStagePending = Math.max(0, _attachmentStagePending - 1)'));
t('clear invalidates pending staging state', /function _clearComposerAttachments[\s\S]*?_attachmentStagePending = 0/.test(js));
t('send refuses to race pending attachment reads', /async function handleAIPanelSubmit[\s\S]*?if \(_attachmentStagePending > 0\)[\s\S]*?Attachments are still being prepared/.test(js));
t('composer exposes accessible staging status', js.includes("attachmentStageStatus.setAttribute('role', 'status')") && js.includes("'Preparing attachments\\u2026'"));
t('send button exposes pending state', js.includes("sendBtn.toggleAttribute('data-attachment-pending', pending)"));
t('request completion respects pending staging', js.includes('sendBtn.disabled = _attachmentStagePending > 0'));
t('pending status has styling', css.includes('.ai-assistant-panel-attachment-stage-status'));

t('canonical attachment split helper exists', js.includes('function _splitQuestionWithAttachments(canonicalText)'));
t('retry splits canonical context before resend', /userRetryBtn[\s\S]*?_splitQuestionWithAttachments\(canonicalQuestion\)[\s\S]*?_setComposerReplayAttachmentContext/.test(js));
t('edit splits canonical context before textarea fill', /editBtn[\s\S]*?_splitQuestionWithAttachments\(canonicalQuestion\)[\s\S]*?input\.value = replay\.question/.test(js));
t('assistant retry also splits canonical context', /retryMenuBtn[\s\S]*?_splitQuestionWithAttachments\(q\)[\s\S]*?_setComposerReplayAttachmentContext/.test(js));
t('submit awaits effective staged plus replay plan', js.includes('var attachmentPlan = await _prepareComposerEffectiveAttachmentPlan(attachmentSnapshot);'));
t('replay context is bounded', js.includes("_composerReplayAttachmentContext = String(value || '').slice(0, _ATTACHMENT_MAX_TEXT_CHARS)"));
t('replay context is visibly disclosed', js.includes('Reusing bounded attachment context from the prior turn'));
t('replay context can be removed', js.includes("attachmentReplayRemove.addEventListener('click', _clearComposerReplayAttachmentContext)"));
t('new staged files take priority under total cap', /function _prepareComposerEffectiveAttachmentPlan[\s\S]*?_prepareComposerAttachmentPlan[\s\S]*?_mergeAttachmentContexts\(plan\.text, _composerReplayAttachmentContext\)/.test(js));
t('clear removes hidden replay context', /function _clearComposerAttachments[\s\S]*?_composerReplayAttachmentContext = ''/.test(js));
t('replay disclosure styling exists', css.includes('.ai-assistant-panel-attachment-replay'));

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
