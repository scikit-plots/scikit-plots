import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let passed=0,failed=0;
function ok(v,n){if(v){passed++;}else{failed++;console.error('FAIL '+n);}}
function eq(g,w,n){ok(JSON.stringify(g)===JSON.stringify(w),n);if(JSON.stringify(g)!==JSON.stringify(w))console.error(' got='+JSON.stringify(g)+' want='+JSON.stringify(w));}
function slice(a,b){const i=src.indexOf(a),j=src.indexOf(b,i+1);if(i<0||j<0)throw new Error('missing '+a);return src.slice(i,j);}

const lineage=slice("    var _FEEDBACK_LINEAGE_MAX_IDS = 1000;", "    /**\n     * Unique session id");
const rt=new Function(`${lineage}\nreturn {_feedbackLineageFromPrior,_feedbackRetractionLineage};`)();
const f1=rt._feedbackLineageFromPrior(null,'f1');
eq(f1,{feedbackChainId:'f1',prevFeedbackId:null,prevFeedbackIds:[],editCount:0},'first rating starts chain');
const p1={feedbackId:'f1',feedbackChainId:'f1',prevFeedbackId:null,prevFeedbackIds:[],editCount:0};
const f2=rt._feedbackLineageFromPrior(p1,'f2');
eq(f2,{feedbackChainId:'f1',prevFeedbackId:'f1',prevFeedbackIds:['f1'],editCount:1},'second rating keeps root');
const p2={feedbackId:'f2',feedbackChainId:'f1',prevFeedbackId:'f1',prevFeedbackIds:['f1'],editCount:1};
const f3=rt._feedbackLineageFromPrior(p2,'f3');
eq(f3,{feedbackChainId:'f1',prevFeedbackId:'f2',prevFeedbackIds:['f1','f2'],editCount:2},'third rating keeps full ordered ancestry');
const malformed=rt._feedbackLineageFromPrior({feedbackId:'old',feedbackChainId:'old',prevFeedbackId:'older',editCount:1},'new');
eq(malformed,{feedbackChainId:'new',prevFeedbackId:null,prevFeedbackIds:[],editCount:0},'scalar-only prior is not heuristically upgraded');
const retract=rt._feedbackRetractionLineage({feedbackId:'f3',feedbackChainId:'f1',prevFeedbackId:'f2',prevFeedbackIds:['f1','f2'],editCount:2});
eq(retract,{feedbackChainId:'f1',prevFeedbackId:'f3',prevFeedbackIds:['f1','f2','f3']},'retraction targets current terminal with full ancestry');

ok(src.includes("var _FEEDBACK_STATE_KEY = 'ai-assistant-feedback-state-v2'"),'feedback lineage has same-tab companion persistence');
ok(src.includes('_loadTranscript();\n        _loadFeedbackState();'),'feedback state restores before transcript replay');
ok(src.includes('feedbackChainId: detail.feedbackChainId || null'),'telemetry/review serializers carry chain root');
ok(src.includes('prevFeedbackIds: Array.isArray(detail.prevFeedbackIds)'),'serializers carry ordered ancestry');
ok(src.includes('feedbackId: fb ? (fb.feedbackId || null) : null'),'Q&A contribution carries feedback event identity');
ok(src.includes('feedbackChainId: fb.feedbackChainId || null'),'whole-conversation feedback carries same lineage');
ok(src.includes("_ssDel(_FEEDBACK_STATE_KEY)"),'remember-off/clear can erase persisted lineage companion state');
ok(src.includes('_saveTranscript();\n            _saveFeedbackState();'),'remember-on immediately persists transcript plus lineage state');
ok(src.includes('if (!_contributionQaAtIndex(idx)) return;'),'restored feedback cannot attach to a stale/non-Q&A answer slot');
ok(src.includes('restoredPrev !== ids[ids.length - 1] || restoredChain !== ids[0] || restoredEdit !== ids.length'),'restored lineage must satisfy parent/root/revision invariants');
ok(src.includes("function _postFeedbackRetract(url, token, priorEntry, answerIndex)"),'retraction uses canonical prior entry rather than legacy scalar signature');

console.log(`${passed} passed, ${failed} failed`);if(failed)process.exit(1);
