import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function section(a, b, from = 0) {
  const i = src.indexOf(a, from), j = src.indexOf(b, i + 1);
  if (i < 0 || j < 0) throw new Error(`missing section ${a}`);
  return src.slice(i, j);
}

const sheet = section('    function _buildDatasetContributionSheet() {', '    function _buildKeyboardShortcutsSheet() {');
const feedback = section('        function _refreshFeedbackWorkspace() {', '        function _activityCard(kind, item, slot) {', src.indexOf('    function _buildDatasetContributionSheet() {'));
const popup = section('    function _buildFbkFloat(answerIndex, answerText, questionText) {', '    function _buildFeedbackBlock(answerIndex, answerText, questionText) {');
const telemetrySetter = section('    function _setFeedbackPersistMode(enabled) {', '    // ══════════════════════════════════════════════════════════════════════════\n    // LEGACY LOCAL SHARE STORAGE');

ok(sheet.includes("_workspacePane('feedback')"), 'Feedback remains a first-class workspace pane');
ok(sheet.includes("_workspacePane('contribution')"), 'Contribution remains a sibling workspace pane');
ok(feedback.includes("ai-assistant-panel-contribution-body ai-assistant-panel-feedback-workspace-body"), 'Feedback reuses contribution body rhythm');

const order = [
  feedback.indexOf("'Current Q&A'"),
  feedback.indexOf("'Privacy channels'"),
  feedback.indexOf("'Why is this feedback useful?'"),
  feedback.indexOf("'Inspect payload'"),
  feedback.indexOf("'Review status'")
];
ok(order.every(i => i >= 0), 'all requested Feedback review stages exist');
ok(order.every((v, i) => i === 0 || order[i - 1] < v), 'Feedback stages follow contribution-compatible order');

ok(feedback.includes("telemetryTitle.textContent = 'Anonymous rating telemetry'"), 'anonymous telemetry control moved into Feedback workspace');
ok(feedback.includes("data-feedback-telemetry-toggle"), 'workspace telemetry switch has semantic state hook');
ok(feedback.includes("data-feedback-telemetry-status"), 'workspace telemetry status has semantic state hook');
ok(feedback.includes("telemetryIcon.textContent = '\\uD83D\\uDCBE'"), 'telemetry workspace preserves the save/privacy icon cue');
ok(!popup.includes("persistLabel.textContent = 'Anonymous rating telemetry'"), 'quick popup no longer duplicates anonymous telemetry control');
ok(popup.includes("label: 'Feedback center\\u2026'") && popup.includes("new CustomEvent('ai-assistant-open-feedback-center'"), 'quick popup still provides navigation to full Feedback workspace');

ok(feedback.includes("feedbackNoteInput.maxLength = _CONTRIBUTION_NOTE_MAX_CHARS"), 'feedback optional note uses bounded contribution-compatible limit');
ok(feedback.includes("current.message = feedbackNoteInput.value.slice"), 'workspace note synchronizes with the existing local feedback note');
ok(feedback.includes("reviewTitle.textContent = 'Maintainer feedback review'") && feedback.includes("reviewToggle.setAttribute('aria-label', 'Share feedback with maintainers')"), 'Privacy channels owns the single maintainer-sharing control');
ok(feedback.includes("lifecycle.className = 'ai-assistant-panel-contribution-action-group ai-assistant-panel-feedback-lifecycle'"), 'review lifecycle is folded into Review status');
ok(!feedback.includes("_contributionSection('Review lifecycle'"), 'review lifecycle is no longer a detached peer section');

ok(telemetrySetter.includes("querySelectorAll('[data-feedback-telemetry-toggle]')"), 'telemetry setter synchronizes semantic telemetry controls');
ok(telemetrySetter.includes("querySelectorAll('[data-feedback-telemetry-status]')"), 'telemetry setter synchronizes semantic telemetry status');
ok(!telemetrySetter.includes("querySelectorAll('.ai-assistant-fbk-popup-mini-pill')"), 'telemetry setter cannot overwrite independent popup review switches');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
