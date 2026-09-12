import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function extract(name){
  const i=src.indexOf('function '+name+'('); if(i<0) throw new Error('missing '+name);
  let d=0,started=false,q=null,line=false,block=false;
  for(let j=i;j<src.length;j++){
    const c=src[j],n=src[j+1];
    if(line){if(c==='\n')line=false;continue} if(block){if(c==='*'&&n==='/'){block=false;j++;}continue}
    if(q){if(c==='\\'){j++;continue}if(c===q)q=null;continue}
    if(c==='/'&&n==='/'){line=true;j++;continue} if(c==='/'&&n==='*'){block=true;j++;continue}
    if(c==='"'||c==="'"||c==='`'){q=c;continue} if(c==='{'){d++;started=true}else if(c==='}'){d--;if(started&&d===0)return src.slice(i,j+1)}
  }
  throw new Error('unbalanced '+name);
}
function section(a,b){const i=src.indexOf(a),j=src.indexOf(b,i+1);if(i<0||j<0)throw new Error('missing section');return src.slice(i,j)}

const activity=section('        function _armActivityForget(button, confirmText, onConfirm) {','        function _setWorkspaceTab(key) {');
ok(activity.includes("_feedbackWorkspaceButton('Forget', '', 'danger')"),'each activity card exposes Forget');
ok(activity.includes("'Confirm forget'"),'individual Forget is two-step confirmed');
ok(activity.includes('_forgetActiveContributionReviewSlot(slot)'),'dataset contribution Forget removes the exact tracked slot');
ok(activity.includes('_forgetActiveFeedbackReviewSlot(slot)'),'feedback Forget removes the exact tracked slot');
ok(activity.includes("_feedbackWorkspaceButton('Forget all', '', 'danger')"),'section-level Forget all exists');
ok(activity.includes("'Confirm all'"),'Forget all is two-step confirmed');
ok(activity.includes('Remote data and provider review state were not changed'),'individual Forget is explicitly local-only');
ok(activity.includes('No remote data was changed'),'Forget all is explicitly local-only');
ok(activity.includes('_activityNewestKeys'),'activity is ordered newest-first');
ok(activity.includes('no background status requests are made'),'activity explains no hidden polling');

const feedbackWorkspace=section('        function _refreshFeedbackWorkspace() {','        function _armActivityForget(button, confirmText, onConfirm) {');
ok(feedbackWorkspace.includes('_feedbackReviewStatus(context.answerIndex, function (res)'),'Feedback workspace status check uses answerIndex rather than a review object');

// Exercise the tab-local storage helpers directly.
const ss = Object.create(null);
globalThis._ACTIVE_FEEDBACK_REVIEW_KEY='ai-assistant-active-feedback-review-v1';
globalThis._ACTIVE_CONTRIBUTION_REVIEW_KEY='ai-assistant-active-contribution-review-v1';
globalThis._activeFeedbackReviewMemory={};
globalThis._activeContributionReviewMemory={};
globalThis._persistEnabled=()=>true;
globalThis._ssGet=(k)=>ss[k]||'';
globalThis._ssSet=(k,v)=>{ss[k]=String(v)};
let conv='conv-a';globalThis._getConversationId=()=>conv;
globalThis._safeUrlForLog=(v)=>String(v||'');
for (const n of ['_feedbackReviewSlot','_readActiveFeedbackReviews','_writeActiveFeedbackReviews','_rememberActiveFeedbackReview','_forgetActiveFeedbackReviewSlot','_forgetActiveFeedbackReview','_forgetAllActiveFeedbackReviews','_reviewTrackingTerminal','_contributionReviewSlot','_readActiveContributionReviews','_rememberActiveContributionReview','_forgetActiveContributionReviewSlot','_forgetActiveContributionReview','_forgetAllActiveContributionReviews']) {
  globalThis[n]=(0,eval)('('+extract(n)+')');
}
for(let i=0;i<30;i++) _rememberActiveFeedbackReview({receiptId:String(i).padStart(32,'a').slice(-32),deleteToken:'x'.repeat(43),reviewRevision:1},'https://svc/review',i,'fp'+i);
let fm=_readActiveFeedbackReviews();
ok(Object.keys(fm).length===24,'feedback tracking is bounded to 24 active receipts');
const fslot=Object.keys(fm)[0];
ok(_forgetActiveFeedbackReviewSlot(fslot)===true,'individual feedback slot can be forgotten');
ok(!_readActiveFeedbackReviews()[fslot],'forgotten feedback slot is removed from merged memory/storage view');
const fcount=_forgetAllActiveFeedbackReviews();
ok(fcount===23,'feedback Forget all reports removed count');
ok(Object.keys(_readActiveFeedbackReviews()).length===0,'feedback Forget all clears tab ledger');

_rememberActiveContributionReview({receiptId:'c'.repeat(32),deleteToken:'d'.repeat(43),reviewRevision:1,reviewProvider:'huggingface'},'https://svc/v1/contribute','conversation',null);
_rememberActiveContributionReview({receiptId:'e'.repeat(32),deleteToken:'f'.repeat(43),reviewRevision:1,reviewProvider:'huggingface'},'https://svc/v1/contribute','qa',2);
let cm=_readActiveContributionReviews();
ok(Object.keys(cm).length===2,'two contribution receipts are tracked');
const cslot=Object.keys(cm)[0];
ok(_forgetActiveContributionReviewSlot(cslot)===true,'individual contribution slot can be forgotten');
ok(Object.keys(_readActiveContributionReviews()).length===1,'individual contribution Forget removes only one receipt');
ok(_forgetAllActiveContributionReviews()===1,'contribution Forget all reports remaining count');
ok(Object.keys(_readActiveContributionReviews()).length===0,'contribution Forget all clears tab ledger');

ok(_reviewTrackingTerminal('eligible',''),'eligible/merged contribution is terminal');
ok(_reviewTrackingTerminal('reviewed',''),'reviewed feedback is terminal');
ok(_reviewTrackingTerminal('quarantined','closed'),'closed provider branch is terminal');
ok(_reviewTrackingTerminal('in_review','')===false,'open review remains tracked');

// Feedback status checks must prune terminal and missing remote reviews.
let forgot=0, remembered=0, mode='reviewed';
globalThis._getActiveFeedbackReview=()=>({receiptId:'a'.repeat(32),deleteToken:'b'.repeat(43),fingerprint:'fp'});
globalThis._resolveFeedbackReviewEndpoint=()=> 'https://svc/v1/feedback/review';
globalThis._forgetActiveFeedbackReview=()=>{forgot++};
globalThis._rememberActiveFeedbackReview=()=>{remembered++};
globalThis._remotePost=(url,tok,payload,opt)=>{ if(mode==='404') opt.onError({status:404}); else opt.onSuccess({status:mode}); };
globalThis._feedbackReviewStatus=(0,eval)('('+extract('_feedbackReviewStatus')+')');
_feedbackReviewStatus(0,()=>{});
ok(forgot===1 && remembered===0,'merged/reviewed feedback status is auto-pruned');
mode='404';_feedbackReviewStatus(0,()=>{});
ok(forgot===2,'missing feedback review (404) is auto-pruned');

const receiptFlow=section('            var check = _contributionActionButton(\'↻ Check status\'','        loadRecoveryCode.addEventListener');
ok(receiptFlow.includes('_reviewTrackingTerminal(state, reviewState)'),'contribution lifecycle uses shared terminal-state detector');
ok(receiptFlow.includes('status === 404 || status === 410'),'missing/expired contribution receipt is detected');
ok(receiptFlow.includes('_forgetActiveContributionReview(selectedScope, context.answerIndex)'),'terminal contribution status clears active tracking');
ok(receiptFlow.includes('no remote deletion was attempted'),'missing receipt cleanup is explicitly non-destructive');

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
