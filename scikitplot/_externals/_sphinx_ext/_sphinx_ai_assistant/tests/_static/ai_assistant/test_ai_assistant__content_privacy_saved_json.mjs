import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function extract(name) {
  const i = src.indexOf('function ' + name + '('); if (i < 0) throw new Error('missing ' + name);
  let d=0, started=false, q=null, line=false, block=false;
  for (let j=i;j<src.length;j++) { const c=src[j], n=src[j+1];
    if(line){if(c==='\n')line=false;continue} if(block){if(c==='*'&&n==='/'){block=false;j++}continue}
    if(q){if(c==='\\'){j++;continue}if(c===q)q=null;continue}
    if(c==='/'&&n==='/'){line=true;j++;continue} if(c==='/'&&n==='*'){block=true;j++;continue}
    if(c==='"'||c==="'"||c==='`'){q=c;continue} if(c==='{'){d++;started=true}else if(c==='}'){d--;if(started&&d===0)return src.slice(i,j+1)}
  } throw new Error('unbalanced ' + name);
}
function section(a,b){const i=src.indexOf(a),j=src.indexOf(b,i+1);if(i<0||j<0)throw new Error('missing section '+a);return src.slice(i,j)}

globalThis._reviewContentPreset=(0,eval)('('+extract('_reviewContentPreset')+')');
globalThis._applyContributionContentOptions=(0,eval)('('+extract('_applyContributionContentOptions')+')');
globalThis._applyFeedbackReviewContentOptions=(0,eval)('('+extract('_applyFeedbackReviewContentOptions')+')');
globalThis._feedbackReviewModelAttribution=(0,eval)('('+extract('_feedbackReviewModelAttribution')+')');
globalThis._contributionModelAttribution=(0,eval)('('+extract('_contributionModelAttribution')+')');
globalThis._contributionArtifactScopeSlug=(0,eval)('('+extract('_contributionArtifactScopeSlug')+')');
globalThis._storagePreviewModel=(0,eval)('('+extract('_storagePreviewModel')+')');
globalThis._feedbackSavedJsonStructure=(0,eval)('('+extract('_feedbackSavedJsonStructure')+')');
globalThis._contributionSavedJsonStructure=(0,eval)('('+extract('_contributionSavedJsonStructure')+')');


const rawModel={id:'m',provider:'p',model:'owner/m',label:'Private label',endpoint:'https://internal.example/?token=SECRET',info_url:'https://private.example/info',description:'private operator note',default:true};
const safeModel=_feedbackReviewModelAttribution(rawModel);
ok(JSON.stringify(Object.keys(safeModel))===JSON.stringify(['id','provider','model']),'feedback review sends minimum model attribution keys only');
ok(!('endpoint' in safeModel)&&!('info_url' in safeModel)&&!('description' in safeModel)&&!('label' in safeModel),'feedback review model attribution excludes transport and descriptive metadata');
const safeStored=_storagePreviewModel(safeModel);
ok(safeStored.endpoint===null&&safeStored.info_url===null&&safeStored.description===null&&safeStored.label===null,'feedback cloud projection expands stripped model metadata as explicit nulls');
const safeContributionModel=_contributionModelAttribution(rawModel);
ok(JSON.stringify(Object.keys(safeContributionModel))===JSON.stringify(['id','provider','model']),'contribution sends minimum model attribution keys only');
ok(!('endpoint' in safeContributionModel)&&!('info_url' in safeContributionModel)&&!('description' in safeContributionModel)&&!('label' in safeContributionModel),'contribution model attribution excludes transport and descriptive metadata');
ok(_feedbackReviewModelAttribution({id:{secret:'x'},provider:'p',model:'owner/m'}).id===null,'non-string model id is dropped rather than stringified');
ok(_feedbackReviewModelAttribution({id:'m',provider:{secret:'x'},model:'owner/m'})===null,'non-string provider fails closed');
ok(_feedbackReviewModelAttribution({id:'m',provider:'p\nprivate',model:'owner/m'})===null,'control characters in model attribution fail closed');

const feedbackPayload={schemaVersion:1,consentFlag:true,consentVersion:'1',consentAt:10,trainingConsentFlag:true,trainingConsentVersion:'1',feedbackId:'fb-1',prevFeedbackId:'fb-0',editCount:2,answerIndex:0,ratingValue:1,ratingLabel:'helpful',ratingTitle:'Helpful',ratingMode:'quick',ratingScaleMin:-1,ratingScaleMax:1,message:'useful note',query:'Q',answer:'A',model:{id:'m',provider:'p',model:'owner/m'},page:'https://docs.example/page',ts:20};
const fmin=_applyFeedbackReviewContentOptions(feedbackPayload,_reviewContentPreset('feedback','minimal'));
ok(!('consentAt' in fmin),'feedback minimal removes optional client consent timestamp');
ok(fmin.ts===null,'feedback minimal clears client event timestamp');
ok(fmin.page==='','feedback minimal clears safe source page');
ok(fmin.feedbackId===null&&fmin.prevFeedbackId===null&&fmin.editCount===0,'feedback minimal clears feedback/revision identifiers');
ok(fmin.message==='','feedback minimal clears written note');
ok(fmin.query==='Q'&&fmin.answer==='A','feedback required Q&A stays locked on');
ok(fmin.model?.provider==='p'&&fmin.ratingValue===1,'feedback required model/rating evidence stays locked on');
const fstandard=_applyFeedbackReviewContentOptions(feedbackPayload,_reviewContentPreset('feedback','standard'));
ok(fstandard.message==='useful note'&&fstandard.page==='', 'feedback Standard keeps note but omits source metadata');
const fcomplete=_applyFeedbackReviewContentOptions(feedbackPayload,_reviewContentPreset('feedback','complete'));
ok(fcomplete.page===feedbackPayload.page&&fcomplete.feedbackId==='fb-1'&&fcomplete.ts===20,'feedback Complete preserves prior payload behavior');

const contributionPayload={schemaVersion:4,consentFlag:true,consentVersion:'2.0.0',page:'https://docs.example/page',model:null,records:[{recordType:'conversation',message:'why useful',ts:30,messages:[{role:'user',content:'Q',ts:1},{role:'assistant',content:'A',ts:2,model:{id:'m',provider:'p',model:'owner/m',label:'Private label',endpoint:'https://internal.example/?token=SECRET',info_url:'https://private.example/info',description:'private operator note',default:true},feedback:{ratingValue:1,ratingLabel:'helpful',ratingTitle:'Helpful',ratingMode:'quick',note:'good'}}]}]};
const cmin=_applyContributionContentOptions(contributionPayload,_reviewContentPreset('contribution','minimal'));
ok(cmin.page==='','contribution minimal clears safe source page');
ok(cmin.records[0].ts===null&&cmin.records[0].messages.every(m=>m.ts===null),'contribution minimal clears timestamps');
ok(cmin.records[0].messages[1].model===null,'contribution minimal clears per-answer model metadata');
ok(cmin.records[0].messages[1].feedback===null,'contribution minimal clears rating/feedback metadata');
ok(cmin.records[0].message==='','contribution minimal clears reviewer note');
ok(cmin.records[0].messages.map(m=>m.content).join('|')==='Q|A','contribution required conversation content stays locked on');
const cstandard=_applyContributionContentOptions(contributionPayload,_reviewContentPreset('contribution','standard'));
ok(cstandard.page===''&&cstandard.records[0].message==='why useful','contribution Standard omits source metadata but keeps reviewer note');
ok(cstandard.records[0].messages[1].model?.provider==='p'&&cstandard.records[0].messages[1].feedback?.ratingValue===1,'contribution Standard keeps model and rating evidence');
ok(!('endpoint' in cstandard.records[0].messages[1].model),'contribution request keeps only minimized per-message model attribution');
const ccomplete=_applyContributionContentOptions(contributionPayload,_reviewContentPreset('contribution','complete'));
ok(ccomplete.page===contributionPayload.page&&ccomplete.records[0].messages[0].ts===1,'contribution Complete preserves prior payload behavior');


const canonicalKeys=['schemaVersion','_source','_ts','_dedup_key','conversationId','feedbackId','feedbackChainId','recordType','answerIndex','action','prevFeedbackId','prevFeedbackIds','editCount','status','trainingStatus','ratingValue','ratingSlug','ratingTitle','ratingMode','ratingScaleMin','ratingScaleMax','qualityScore','qualityPercent','message','query','answer','messages','model','modelEvidence','page','consentVersion','trainingConsentVersion','ts'];
const fsaved=_feedbackSavedJsonStructure(fcomplete)[0];
ok(canonicalKeys.every((k,i)=>Object.keys(fsaved)[i]===k),'feedback saved preview follows canonical JSONL key order');
ok(fsaved._source==='feedback'&&fsaved.recordType==='qa'&&fsaved.action==='review','feedback saved preview uses feedback review record family');
ok(fsaved._ts==='<server-assigned>'&&fsaved._dedup_key==='<receipt-id>:feedback','feedback saved preview clearly marks server-owned values');
ok(fsaved.model && Object.keys(fsaved.model).length===8,'feedback saved preview expands canonical 8-key model object');
const csaved=_contributionSavedJsonStructure(ccomplete)[0];
ok(canonicalKeys.every((k,i)=>Object.keys(csaved)[i]===k),'contribution saved preview follows canonical JSONL key order');
ok(csaved._source==='contribution'&&csaved.recordType==='conversation'&&csaved.trainingStatus==='eligible','contribution saved preview uses provider-review/future-main eligible bytes');
ok(csaved._dedup_key==='<receipt-id>:conversation','contribution saved preview marks receipt-scoped dedup key');
ok(csaved.messages[1].model && Object.keys(csaved.messages[1].model).length===8,'conversation saved preview expands per-message canonical model object');

const sheet=section('    function _buildDatasetContributionSheet() {','    function _buildKeyboardShortcutsSheet() {');
const feedback=section('        function _refreshFeedbackWorkspace() {','        function _activityCard(kind, item, slot) {');
const popup=section('    function _buildFbkFloat(answerIndex, answerText, questionText) {','    function _buildFeedbackBlock(answerIndex, answerText, questionText) {');
ok(sheet.includes("_buildReviewContentPrivacyControl(\n            'contribution'"),'Contribution renders shared Content & privacy control');
ok(feedback.includes("_buildReviewContentPrivacyControl(\n                'feedback'"),'Feedback renders shared Content & privacy control');
ok(src.includes("['standard', 'minimal', 'complete', 'custom']"),'shared control exposes Standard / Minimal / Complete / Customize presets');
ok(src.includes("chk.dataset.contentPrivacyOption = spec[0]"),'granular checkboxes have stable semantic hooks');
ok(sheet.includes('contributionContentOptions);'),'Contribution preview/submission builder receives selected content options');
ok(src.includes('return _applyFeedbackReviewContentOptions(payload, _feedbackReviewContentOptions);'),'Feedback request builder applies selected content options');
ok(src.includes('delete stable.consentAt;') && src.includes('return JSON.stringify(stable);'),'feedback no-op fingerprint tracks saved semantics while excluding volatile consentAt');
ok(!popup.includes('data-feedback-review-toggle'),'quick popup no longer owns maintainer-review permission');
ok(!popup.includes('Share feedback for review & model improvement'),'quick popup no longer duplicates review consent copy');
ok(feedback.includes("reviewTitle.textContent = 'Maintainer feedback review'") && feedback.includes("reviewToggle.setAttribute('aria-label', 'Share feedback with maintainers')"),'maintainer review permission has one owner in Feedback Privacy channels');
ok(src.includes('canonical contribution record written to provider review/future main'),'Contribution inspection identifies repository projection rather than ledger state');
ok(src.includes('canonical repository record projection'),'Feedback inspection identifies review repository projection');
ok(sheet.includes("inspectBtn = _contributionActionButton('JSONL'") && sheet.includes("previewStrong.textContent = 'Cloud projection JSONL'"),'Contribution Inspect has explicit cloud-projection JSONL view');
ok(feedback.includes("feedbackInspectBtn = _feedbackWorkspaceButton('JSONL'") && feedback.includes("feedbackPreviewStrong.textContent = 'Cloud projection JSONL'"),'Feedback Inspect has explicit cloud-projection JSONL view');
ok(src.includes('canonical contributions/*.jsonl repository record'),'Contribution cloud projection identifies contributions storage family');
ok(src.includes('canonical cloud feedback-review JSONL row'),'Feedback saved preview identifies feedback review storage family');


ok(src.includes('function _feedbackTelemetrySavedJsonStructure(detail)'), 'Feedback exposes separate telemetry saved-row projection');
ok(feedback.includes("telemetrySummary.textContent = 'Anonymous telemetry JSONL · separate privacy-minimal row'"), 'Feedback inspect exposes separate telemetry JSONL view without another toolbar button');
ok(src.includes("_dedup_key: feedbackId ? (String(feedbackId) + ':feedback') : null"), 'Telemetry preview uses identifier-first dedup convention');
ok(src.includes("trainingStatus: 'telemetry'"), 'Telemetry preview remains non-training');
ok(src.includes("application/x-ndjson"), 'saved projections download as JSONL rather than request JSON');
ok(src.includes("function _contributionArtifactFilename(scope, role)"),'Contribution filenames are generated by a lifecycle-role helper');
ok(src.includes("ai-contribution-' + scopeSlug + '-request-json-'")&&src.includes("ai-contribution-' + scopeSlug + '-cloud-projection-jsonl-'"),'Contribution filenames include scope plus request/projection role');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
