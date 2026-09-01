import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let pass=0, fail=0;
function ok(v,n){ if(v){pass++;}else{fail++;console.error('FAIL '+n);} }
function section(a,b){ const i=src.indexOf(a), j=src.indexOf(b,i+1); return i>=0&&j>i ? src.slice(i,j) : ''; }
const endpoint=section('var extSection = _buildSheetSection', '// ══════════════════════════════════════════════════════════════════════\n        // MOUNT + SUBSCRIBE');
const sheet=section('    function _buildDatasetContributionSheet() {','    function _buildKeyboardShortcutsSheet() {');
const feedback=section('        function _refreshFeedbackWorkspace() {','        function _activityCard(kind, item, slot) {');

ok(endpoint.includes("_buildExtSub('Page integration events')"), 'Endpoint Configuration keeps page integration controls');
ok(!endpoint.includes("_buildExtSub('Feedback telemetry')"), 'Endpoint Configuration no longer duplicates Feedback permissions');
ok(!endpoint.includes("'Send anonymous rating telemetry'"), 'Endpoint Configuration no longer renders telemetry permission');
ok(!endpoint.includes("'Share feedback for review & model improvement'"), 'Endpoint Configuration no longer renders maintainer-review permission');

ok(feedback.includes("telemetryTitle.textContent = 'Anonymous rating telemetry'"), 'Feedback workspace owns anonymous telemetry');
ok(feedback.includes("reviewTitle.textContent = 'Maintainer feedback review'"), 'Feedback workspace owns maintainer review');
ok(feedback.includes("reviewToggle.setAttribute('aria-label', 'Share feedback with maintainers')"), 'maintainer review has one concise sharing control');
ok(!feedback.includes("toggleStrong.textContent = 'Share with maintainers'"), 'old duplicate maintainer toggle removed');
ok(!feedback.includes("'Share current feedback'"), 'duplicate manual share button removed');
ok(!feedback.includes("'Update maintainer review'"), 'duplicate manual update button removed');
ok(feedback.includes("'Review status'"), 'final feedback section is lifecycle/status only');
ok(feedback.includes('the next quick/detailed feedback save will create or update') || feedback.includes('The next quick/detailed feedback save will create or update'), 'workspace explains automatic review update behavior');

for (const part of [sheet, feedback]) {
  ok(part.includes("'JSON'"), 'inspector has JSON format control');
  ok(part.includes("'JSONL'"), 'inspector has JSONL format control');
  ok(part.includes('ai-assistant-panel-payload-format-tabs'), 'inspector uses centralized format switcher');
  ok(part.includes('JSON.stringify('), 'JSON stays pretty-readable');
  ok(part.includes('_readableJsonl('), 'JSONL uses expanded readable local view');
}
ok(src.includes("join('\\n\\n')"), 'readable JSONL separates expanded records clearly');
ok(src.includes('Copy/Download still emits strict one-JSON-object-per-line NDJSON') || src.includes('copied/downloaded JSONL remains strict one-record-per-line NDJSON'), 'UI explains readable-view versus strict JSONL export');
ok(sheet.includes("feedbackInspectFormat === 'json'"), 'Feedback copy/download follows selected format');
ok(sheet.includes("contributionInspectFormat === 'json'"), 'Contribution copy/download follows selected format');
ok(feedback.includes("telemetrySummary.textContent = 'Anonymous telemetry JSONL · separate privacy-minimal row'"), 'telemetry JSONL is nested rather than another toolbar button');
ok(src.includes("application/x-ndjson"), 'JSONL download remains NDJSON MIME');
ok(src.includes("application/json"), 'JSON download uses JSON MIME');

console.log(`${pass} passed, ${fail} failed`);
if(fail) process.exit(1);
