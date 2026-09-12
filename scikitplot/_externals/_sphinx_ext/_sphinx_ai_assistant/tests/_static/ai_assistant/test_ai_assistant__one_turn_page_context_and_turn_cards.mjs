// Run 109 regression: PAGE / pinned MD / uploaded FILE transport is strictly
// one-turn. Current-page settings and pins are source/default conveniences only;
// saved state must never imply silent context reuse on a later question.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let passed=0, failed=0;
function ok(cond,name){if(cond){passed++;}else{failed++;console.error('FAIL '+name);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,started=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;started=true;}else if(src[j]==='}'){d--;if(started&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}

const shelf = extract('_contextShelfItems');
const prepare = extract('_privacyPrepareDocumentationContext');
const consume = extract('_consumePreparedPageContexts');
const setConsumed = extract('_setPageContextConsumed');
const loadConsumed = extract('_loadConsumedPageContexts');
const saveConsumed = extract('_saveConsumedPageContexts');
const setCurrent = extract('_setCurrentPageContextInTab');
const pin = extract('_pinCurrentPageContext');
const renderComposer = extract('_renderComposerAttachments');
const removeResource = extract('_removeComposerResourceItem');
const snapshot = extract('_composerTurnAttachmentSnapshot');
const renderBubble = extract('_renderBubble');
const submit = extract('handleAIPanelSubmit');
const remember = extract('_setRememberConversationInTab');
const refresh = extract('_refreshPinnedPageContextShelf');
const clear = extract('clearConversation');
const slashDefs = extract('_localSlashCommandDefinitions');

const getActiveModelId = extract('_getActiveModelId');
const setActiveModelId = extract('_setActiveModelId');
const apiCall = extract('_panelApiCall');
ok(src.includes('var _activeModelIdMemory = null'), 'model selection has an in-memory correctness authority independent of sessionStorage');
ok(setActiveModelId.includes('_activeModelIdMemory = id'), 'explicit model selection updates runtime authority before attempting storage');
ok(getActiveModelId.indexOf('_activeModelIdMemory') < getActiveModelId.indexOf('sessionStorage.getItem'), 'request resolution prefers the latest runtime selection over stale/blocked storage');
ok(apiCall.includes('modelName = activeModel.model || activeModel.id'), 'client request preserves the effective selected model wire id');

ok(src.includes("var _CONSUMED_PAGE_CONTEXT_KEY = 'ai-assistant-consumed-page-contexts-v1'"), 'consumed page transport has an explicit versioned same-tab key');
ok(src.includes('var _CONSUMED_PAGE_CONTEXT_MAX_ITEMS = 64'), 'consumed page transport state is bounded');
ok(shelf.includes('!_pageContextConsumed(currentUrl)'), 'current PAGE disappears from composer after its one-turn use');
ok(shelf.includes('if (!sourceUrl || _pageContextConsumed(sourceUrl)) return'), 'saved pinned MD source stays stored but is not visibly armed after consumption');
ok(prepare.includes('!_pageContextConsumed(currentUrl)') && prepare.includes('_pageContextConsumed(sourceUrl)'), 'outbound context uses the same consumed authority as visible UI');
ok(consume.includes('_setPageContextConsumed(item.sourceUrl, true, false)'), 'Send consumes every prepared PAGE/MD source');
ok(setCurrent.includes('_stagePageContextForNextTurn(currentUrl, false)'), 'current-page setting stages the current source once instead of granting perpetual transport authority');
ok(pin.includes('_stagePageContextForNextTurn(current.sourceUrl, false)'), 'pinning saves a source and stages exactly one use');
ok(renderComposer.includes('_removeComposerResourceItem(item)') && removeResource.includes('_setPageContextConsumed(item.sourceUrl, true, false)'), 'PAGE/MD remove only unstages the next message through shared resource helper');
ok(!renderComposer.includes('_removePinnedPageContext(item.sourceUrl)'), 'PAGE/MD card remove never deletes the saved pin as a side effect');
ok(src.includes("attachmentTray.setAttribute('aria-label', 'Pages and files staged for the next message only')"), 'composer tray states one-turn transport semantics');
ok(src.includes("parts.push('Next message')") && src.includes("parts.push('One turn')"), 'composer PAGE/FILE metadata declares one-turn lifecycle');
ok(src.includes('Current-page Markdown staged for the next question only.'), 'current PAGE preview explains consumption');
ok(src.includes('Saved pinned-page Markdown staged for the next question only.'), 'pinned MD preview separates saved source from staged use');
ok(snapshot.includes("status: item.contextRole === 'current' ? 'Current page · used once' : 'Pinned page · used once'"), 'turn provenance records PAGE and pinned MD as one-use resources');
ok(snapshot.includes("'Included once · bounded excerpt'") && snapshot.includes("'Not sent · shared context budget'") && snapshot.includes("'Local only · not sent'"), 'turn provenance records uploaded file send status honestly');
ok(renderBubble.includes("files.className = 'ai-assistant-panel-attachments ai-assistant-panel-user-turn-attachments'"), 'user turn reuses the composer attachment-strip structure');
ok(renderBubble.includes("preview.className = 'ai-assistant-panel-attachment-card'"), 'historical resources reuse composer attachment-card anatomy');
ok(renderBubble.includes('_openAttachmentPreview(item, preview)'), 'historical resource cards support the same preview surface');
ok(renderBubble.includes('ai-assistant-panel-user-turn-question-wrap') && renderBubble.includes('data-collapsed'), 'question text is separately expandable below its resources');
ok(renderBubble.indexOf("bubble.appendChild(files)") < renderBubble.indexOf("questionWrap.className = 'ai-assistant-panel-user-turn-question-wrap'"), 'resource cards render above the question text');
ok(submit.indexOf('_composerTurnAttachmentSnapshot(attachmentText, preparedPageContext)') < submit.indexOf('_consumePreparedPageContexts(preparedPageContext, false)'), 'turn provenance is captured before PAGE/MD state is consumed');
ok(submit.indexOf('_consumePreparedPageContexts(preparedPageContext, false)') < submit.indexOf('_clearComposerAttachments()'), 'PAGE/MD and uploaded FILEs are consumed together immediately after turn commit');
ok(remember.includes('_loadConsumedPageContexts(false)') && remember.includes('_saveConsumedPageContexts()'), 'Remember ON preserves consumed state so restored saved pins cannot silently re-arm');
ok(remember.includes('_ssDel(_CONSUMED_PAGE_CONTEXT_KEY)'), 'Remember OFF removes persisted consumed-state history');
ok(refresh.includes('_loadConsumedPageContexts(true)'), 'BFCache/navigation refresh rehydrates consumed-state authority before shelf rendering');
ok(clear.includes('_consumedPageContextUrls = Object.create(null)') && clear.includes('_ssDel(_CONSUMED_PAGE_CONTEXT_KEY)'), 'New Chat starts a fresh one-turn staging lifecycle');
ok(slashDefs.includes('var currentConsumed = _pageContextConsumed(currentUrl)'), 'slash command state distinguishes saved/default source from currently staged transport');
ok(slashDefs.includes("current.badge = currentStaged ? 'Staged' : 'Context'"), 'slash current-page badge reflects one-turn staging');
ok(slashDefs.includes("currentPinned ? (currentConsumed ? 'Saved' : 'Staged') : 'Save'"), 'slash pin badge distinguishes saved source from staged use');
ok(css.includes('.ai-assistant-panel-user-turn-attachments'), 'user-turn resource tray has dedicated responsive styling');
ok(css.includes('.ai-assistant-panel-bubble--user[data-has-turn-attachments="true"]'), 'attachment-bearing user bubble receives bounded responsive width');

// ----- Runtime: consumed transport survives same-tab restore when Remember is ON -----
var _CONSUMED_PAGE_CONTEXT_KEY='consumed';
var _CONSUMED_PAGE_CONTEXT_SCHEMA=1;
var _CONSUMED_PAGE_CONTEXT_MAX_ITEMS=64;
var _consumedPageContextUrls=Object.create(null);
var _consumedPageContextUrlsLoaded=false;
let persist=false;
let raw=null;
let deleted=0;
function _persistEnabled(){return persist;}
function _normalizeContextPageUrl(v){return String(v||'').split('#')[0].split('?')[0];}
function _ssGet(){return raw;}
function _ssSet(_k,v){raw=String(v);}
function _ssDel(){raw=null;deleted++;}
function _renderComposerAttachments(){}
const _loadConsumedPageContexts=eval('('+loadConsumed+')');
const _saveConsumedPageContexts=eval('('+saveConsumed+')');
const _setPageContextConsumed=eval('('+setConsumed+')');
const _pageContextConsumed=eval('('+extract('_pageContextConsumed')+')');
const _consumePreparedPageContexts=eval('('+consume+')');

const A='https://docs.test/A.html';
const B='https://docs.test/B.html';
ok(_setPageContextConsumed(A,true,false)===true && _pageContextConsumed(A), 'runtime: first Send consumes page A in memory');
ok(raw===null, 'runtime: Remember OFF does not persist consumed transport state');

// Reader enables Remember after the send: live consumed intent must be preserved
// during first hydration, then persisted.
persist=true;
_loadConsumedPageContexts(false);
_saveConsumedPageContexts();
ok(JSON.parse(raw).items.includes(A), 'runtime: enabling Remember persists a page already consumed earlier in the tab');

// Simulate navigation/new document in the same remembered conversation.
_consumedPageContextUrls=Object.create(null);
_consumedPageContextUrlsLoaded=false;
_loadConsumedPageContexts(false);
ok(_pageContextConsumed(A)===true, 'runtime: navigation restore does not silently re-arm previously consumed saved page A');

// Explicit re-stage removes only A from consumed authority.
ok(_setPageContextConsumed(A,false,false)===true && !_pageContextConsumed(A), 'runtime: explicit restage re-arms A for exactly one later question');
ok(JSON.parse(raw).items.length===0, 'runtime: explicit restage persists removal from consumed set');

// A send with two staged PAGE/MD sources consumes both atomically from transport perspective.
_consumePreparedPageContexts({sources:[{sourceUrl:A},{sourceUrl:B}]},false);
ok(_pageContextConsumed(A) && _pageContextConsumed(B), 'runtime: Send consumes every PAGE/MD source that actually participated');

// Forced BFCache refresh: persisted state wins over stale in-memory memory.
raw=JSON.stringify({schemaVersion:1,items:[B]});
_consumedPageContextUrls[A]=true;
_consumedPageContextUrls[B]=true;
_consumedPageContextUrlsLoaded=true;
_loadConsumedPageContexts(true);
ok(!_pageContextConsumed(A) && _pageContextConsumed(B), 'runtime: forced restore cannot resurrect stale consumed-memory entries absent from same-tab storage');

// Bounds and malformed storage.
raw=JSON.stringify({schemaVersion:1,items:Array.from({length:100},(_,i)=>`https://docs.test/${i}.html`)});
_consumedPageContextUrls=Object.create(null); _consumedPageContextUrlsLoaded=false;
_loadConsumedPageContexts(false);
ok(Object.keys(_consumedPageContextUrls).length===64, 'runtime: restored consumed-state list is bounded to 64 URLs');
raw='{broken'; deleted=0; _consumedPageContextUrls=Object.create(null); _consumedPageContextUrlsLoaded=false;
_loadConsumedPageContexts(false);
ok(deleted===1 && Object.keys(_consumedPageContextUrls).length===0, 'runtime: malformed consumed-state storage is quarantined without arming context');



// ----- Runtime: request preparation itself is one-shot across consecutive questions -----
{
  // Use the production _privacyPrepareDocumentationContext body with a tiny
  // environment so this test proves outbound source selection, not merely the
  // consumed-state map in isolation.
  let _pinnedPageContexts=[
    {kind:'page',contextRole:'pinned',title:'Pinned B',sourceUrl:B,text:'PIN-B'}
  ];
  function _loadPinnedPageContexts(){}
  function _currentContextPageUrl(){return A;}
  function _currentPageContextActive(){return true;}
  async function _prepareCurrentPageContextItem(){return {kind:'page',contextRole:'current',title:'Current A',sourceUrl:A,text:'CUR-A',redactionFindings:[],invisibleRemoved:0};}
  function _log(){}
  function _assembleDocumentationContextSources(items){return items.map(x=>x.text).join('|');}
  function _effectivePanelContextLimit(){return 8000;}
  function _cfg(){return {};}
  const prepareRequestContext=eval('('+prepare.replace(/^function /,'async function ')+')');

  // Fresh explicit staging for A+B.
  _setPageContextConsumed(A,false,false);
  _setPageContextConsumed(B,false,false);
  let first=await prepareRequestContext({});
  ok(first.sources.length===2 && first.text==='CUR-A|PIN-B', 'runtime request: Q1 includes current PAGE and pinned MD exactly while staged');
  _consumePreparedPageContexts(first,false);

  let second=await prepareRequestContext({});
  ok(second.sources.length===0 && second.text==='', 'runtime request: Q2 contains zero PAGE/MD context after Q1 consumed them');

  _setPageContextConsumed(B,false,false);
  let third=await prepareRequestContext({});
  ok(third.sources.length===1 && third.sources[0].sourceUrl===B && third.text==='PIN-B', 'runtime request: explicit re-stage makes only pinned MD B participate in Q3');
  _consumePreparedPageContexts(third,false);
  let fourth=await prepareRequestContext({});
  ok(fourth.sources.length===0, 'runtime request: explicitly reused pinned MD is consumed again after Q3');
}

// ----- Runtime: blocked sessionStorage cannot make Mirror UI send Echo -----
{
  var _PANEL_MODEL_KEY='active-model';
  var _activeModelIdMemory=null;
  var sessionStorage={
    getItem(){ return 'stub-echo'; },
    setItem(){ throw new Error('blocked'); }
  };
  const getId=eval('('+getActiveModelId+')');
  const setId=eval('('+setActiveModelId+')');
  const models=[
    {id:'stub-echo',model:'stub/echo',default:true},
    {id:'stub-mirror',model:'stub/mirror'}
  ];
  ok(getId(models)==='stub-echo', 'runtime model authority: stored/default Echo is initial selection');
  setId('stub-mirror');
  ok(_activeModelIdMemory==='stub-mirror', 'runtime model authority: Mirror click survives failed storage write in memory');
  ok(getId(models)==='stub-mirror', 'runtime model authority: next request resolves Mirror, never stale Echo');
}

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
