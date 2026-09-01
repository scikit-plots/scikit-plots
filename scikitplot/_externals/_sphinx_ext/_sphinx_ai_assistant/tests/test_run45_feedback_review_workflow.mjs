import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed=0, failed=0;
function ok(cond,name){if(cond){passed++;}else{failed++;console.error('FAIL '+name);}}
function eq(got,want,name){ok(Object.is(got,want),name);if(!Object.is(got,want))console.error(`  got=${JSON.stringify(got)} want=${JSON.stringify(want)}`);}
function slice(a,b){const i=src.indexOf(a), j=src.indexOf(b,i+1);if(i<0||j<0)throw new Error(`missing slice ${a}`);return src.slice(i,j);}

const consent = slice("    var _FEEDBACK_REVIEW_CONSENT_VERSION = '2.0.0';", '    /**\n     * Selected microphone device ID.');
const review = slice('    var _FEEDBACK_REVIEW_SCHEMA_VERSION = 1;', '    var _CONTRIBUTION_SCHEMA_VERSION = 4;');
const helpers = slice('    function _syncFeedbackRatingControls(answerIndex) {', '    /**\n     * POST a share payload');
const quick = slice('    function _buildFbkFloat(answerIndex, answerText, questionText) {', '    function _buildFeedbackBlock(answerIndex, answerText, questionText) {');
const detailed = slice('    function _buildFeedbackBlock(answerIndex, answerText, questionText) {', '    // ── Sheet hamburger helper');
const sheet = slice('    function _buildDatasetContributionSheet() {', '    function _buildKeyboardShortcutsSheet() {');

ok(consent.includes("ai-assistant-feedback-review-consent"),'review sharing has separate versioned consent key');
ok(!consent.includes('ai-assistant-feedback-telemetry-consent'),'review consent cannot reuse telemetry consent key');
ok(consent.includes("var _FEEDBACK_TRAINING_CONSENT_VERSION = '1.0.0'"),'review consent carries independently versioned training authority');
ok(review.includes("replace(/\\/v1\\/contribute\\/?$/i, '/v1/feedback/review')"),'review endpoint is derived from contribution-capable service');
ok(review.includes("method: 'PUT'"),'changed feedback updates same review');
ok(review.includes("reason: 'unchanged'"),'identical feedback is client-side no-op');
ok(review.includes("'X-Feedback-Review-Token': active.deleteToken"),'feedback review uses separate management capability');
ok(review.includes("method: 'DELETE'"),'feedback review has participant withdrawal');
ok(review.includes('trainingConsentFlag: true'),'review payload requires explicit training-consent marker');
ok(review.includes('ratingScaleMin'),'review payload carries rating scale for server-side quality normalization');
ok(helpers.includes('delete _feedbackStore[answerIndex]'),'withdrawal clears local authoritative feedback state');
ok(helpers.includes('_feedbackGivenSet.delete(answerIndex)'),'withdrawal clears given-rating guard');
ok(helpers.includes('_syncFeedbackRatingControls(answerIndex)'),'withdrawal/update synchronizes all rating surfaces');
ok(quick.includes('_queueFeedbackReview(detail, answerIndex, answerText, questionText, cfg);'),'quick rating can update maintainer review after explicit permission');
ok(detailed.includes('_queueFeedbackReview(detail, answerIndex, answerText, questionText, cfg);'),'detailed feedback can update same maintainer review');
ok(quick.includes("data-feedback-mode', 'quick'"),'quick buttons expose synchronized state identity');
ok(detailed.includes("data-feedback-mode', 'panel'"),'detailed buttons expose synchronized state identity');
ok(sheet.includes("_workspaceButton('feedback', 'Feedback')"),'workspace has Feedback tab');
ok(sheet.includes("_workspaceButton('contribution', 'Dataset contribution')"),'workspace has Dataset contribution tab');
ok(sheet.includes("_workspaceButton('activity', 'Activity')"),'workspace has Activity tab');
ok(sheet.includes('Feedback is exactly one Q&A'),'feedback tab explains one-Q&A scope');
ok(sheet.includes('training-eligible only if a maintainer merges') || sheet.includes('merge required for training eligibility'),'workspace states merge-gated training invariant');
ok(src.includes("'Share feedback for review & model improvement'"),'settings expose explicit review/training permission');
ok(src.includes("'Send anonymous rating telemetry'"),'telemetry remains separately named and permissioned');

// Execute the consent primitive so this is not only a source-layout contract.
function consentRuntime(seed={}){
  const store=new Map(Object.entries(seed));
  const localStorage={getItem:k=>store.has(k)?store.get(k):null,setItem:(k,v)=>store.set(k,String(v)),removeItem:k=>store.delete(k)};
  const nodes=[];
  const document={querySelectorAll(){return nodes;}};
  function _dispatchAssistantEvent(){}
  function CustomEvent(type,init){this.type=type;this.detail=init&&init.detail;}
  const fn=new Function('localStorage','document','_dispatchAssistantEvent','CustomEvent', `${consent}\nreturn {enabled:()=>_feedbackReviewEnabled, granted:()=>_feedbackReviewGrantedAt, set:_setFeedbackReviewMode, status:_feedbackReviewStatusText};`);
  return {api:fn(localStorage,document,_dispatchAssistantEvent,CustomEvent),store};
}
let legacy=consentRuntime({'ai-assistant-feedback-review-consent':JSON.stringify({enabled:true,version:'1.0.0',grantedAt:123})});
eq(legacy.api.enabled(),false,'legacy review-only consent fails closed after training semantics change');
let r=consentRuntime();
eq(r.api.enabled(),false,'review sharing defaults off');
ok(r.api.status().startsWith('Local only'),'off state explains local-only content');
r.api.set(true);
eq(r.api.enabled(),true,'explicit review-sharing opt-in enables permission');
const saved=JSON.parse(r.store.get('ai-assistant-feedback-review-consent'));
eq(saved.version,'2.0.0','review consent stores current version');
eq(saved.enabled,true,'review consent stores enabled flag');
ok(!r.store.has('ai-assistant-feedback-telemetry-consent'),'review opt-in does not create telemetry consent');
r.api.set(false);
eq(r.api.enabled(),false,'review opt-out disables future repository sharing');
ok(!r.store.has('ai-assistant-feedback-review-consent'),'review opt-out removes its own consent record');

console.log(`${passed} passed, ${failed} failed`);
if(failed)process.exit(1);
