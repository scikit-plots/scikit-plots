import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed=0, failed=0;
function ok(cond,name){if(cond){passed++;}else{failed++;console.error('FAIL '+name);}}
function section(a,b){const i=src.indexOf(a),j=src.indexOf(b,i+1);if(i<0||j<0)throw new Error('missing section '+a);return src.slice(i,j);}

const review = section('    var _FEEDBACK_REVIEW_SCHEMA_VERSION = 1;', '    var _CONTRIBUTION_SCHEMA_VERSION = 4;');
const sheet = section('    function _buildDatasetContributionSheet() {', '    function _buildKeyboardShortcutsSheet() {');
const popup = section('    function _buildFbkFloat(answerIndex, answerText, questionText) {', '    function _buildFeedbackBlock(answerIndex, answerText, questionText) {');

ok(review.includes('var qa = answerIndex != null ? _contributionQaAtIndex(answerIndex) : null;'), 'feedback review resolves the rated transcript Q&A');
ok(review.includes("? qa.model"), 'originating assistant-turn model wins attribution');
ok(review.includes("return 'Originating model attribution is unavailable"), 'missing model fails client preflight');
ok(review.includes('function _feedbackReviewModelAttribution(raw)') && review.includes('model: _feedbackReviewModelAttribution(model)'), 'feedback request minimizes model metadata before network submission');
ok(review.includes("page: _sanitizePage(String(detail.page || ''))"), 'feedback request sanitizes page source before network submission');
ok(review.includes('delete stable.consentAt;') && review.includes('return JSON.stringify(stable);'), 'saved feedback semantics participate in no-op fingerprint while volatile consentAt does not');
ok(sheet.includes("'Inspect payload'"), 'Feedback tab uses the shared inspect-section label');
ok(sheet.includes("_feedbackWorkspaceButton('JSONL'") && sheet.includes("feedbackPreviewStrong.textContent = 'Cloud projection JSONL'"), 'Feedback inspect section exposes canonical cloud-projection JSONL tab');
ok(sheet.includes("'⎘ Copy JSON'") && sheet.includes("'⎘ Copy JSONL'") && sheet.includes("feedbackInspectFormat === 'json'"), 'Feedback inspector has one format-aware copy action');
ok(sheet.includes("'↓ Download JSON'") && sheet.includes("'↓ Download JSONL'") && sheet.includes("application/x-ndjson"), 'Feedback inspector has one format-aware download action');
ok(sheet.includes("'Cloud-projection feedback-review JSONL copied locally. Nothing was submitted.'"), 'copy explicitly stays local and names the projection role');
ok(sheet.includes("_feedbackReviewArtifactFilename('request-json')") && sheet.includes("_feedbackReviewArtifactFilename('cloud-projection-jsonl')"), 'download filename identifies request JSON vs cloud-projection JSONL role');
ok(sheet.includes("'Originating model'"), 'Feedback tab visibly identifies originating model evidence');
ok(sheet.includes("'Request JSON'"), 'Feedback transport envelope remains separately inspectable');
ok(sheet.includes("'pre-save projection'") && sheet.includes('cloud-owned placeholders'), 'JSONL view makes pre-save cloud projection semantics explicit');
ok(sheet.includes("'Anonymous telemetry JSONL · separate privacy-minimal row'"), 'Feedback telemetry row is separately inspectable inside JSONL view');
ok(sheet.includes("idleText.textContent = reviewPayloadIssue") && !sheet.includes('share.disabled = !entry'), 'invalid review payload is surfaced without a duplicate share button');
ok(!popup.includes('data-feedback-review-toggle'), 'quick popup no longer duplicates maintainer-review permission');
ok(popup.includes('feedbackCenterIcon.innerHTML = ICONS.pulse'), 'feedback center uses pulse icon');
ok(src.includes("pulse: '<svg viewBox=\"0 0 16 16\""), 'pulse icon is shipped in shared ICONS map');

console.log(`${passed} passed, ${failed} failed`);
if(failed)process.exit(1);
