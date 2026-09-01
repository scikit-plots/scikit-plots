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
ok(review.includes('modelProvider: payload.model && payload.model.provider'), 'model identity participates in no-op fingerprint');
ok(sheet.includes("'Inspect feedback payload'"), 'Feedback tab has inspect section');
ok(sheet.includes("'Inspect JSON'"), 'Feedback inspect section can toggle JSON');
ok(sheet.includes("'⎘ Copy JSON to clipboard'"), 'Feedback inspect section can copy JSON');
ok(sheet.includes("'↓ Download JSON file'"), 'Feedback inspect section can download JSON');
ok(sheet.includes("'Feedback review JSON copied locally. Nothing was submitted.'"), 'copy explicitly stays local');
ok(sheet.includes("'ai-feedback-review-payload-' + _isoFileStamp() + '.json'"), 'download has feedback-specific filename');
ok(sheet.includes("'Originating model'"), 'Feedback tab visibly identifies originating model evidence');
ok(sheet.includes('share.disabled = !entry || !_feedbackReviewEnabled || !!reviewPayloadIssue;'), 'share is disabled when model/Q&A preflight fails');
ok(popup.includes('reviewShareIcon.innerHTML = ICONS.commentDiscussion'), 'review permission uses comment-discussion icon');
ok(popup.includes('feedbackCenterIcon.innerHTML = ICONS.pulse'), 'feedback center uses pulse icon');
ok(src.includes("pulse: '<svg viewBox=\"0 0 16 16\""), 'pulse icon is shipped in shared ICONS map');

console.log(`${passed} passed, ${failed} failed`);
if(failed)process.exit(1);
