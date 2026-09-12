// Run 98 regression: exhaustive saved-source/current-default state matrix plus
// idempotency/race hardening. Run 109 supersedes persistent transport: saved
// pins may persist, while PAGE/MD staging is consumed after each Send.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('missing ' + name);
  let d = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { d++; started = true; }
    else if (src[j] === '}') { d--; if (started && d === 0) return src.slice(i, j + 1); }
  }
  throw new Error('unbalanced ' + name);
}

const enabledSrc = extract('_currentPageContextEnabled');
const setCurrentSrc = extract('_setCurrentPageContextInTab');
const shelfSrc = extract('_contextShelfItems');
const outboundSrc = extract('_privacyPrepareDocumentationContext');
const pinRevisionSrc = extract('_pageContextPinRevision');
const invalidatePinSrc = extract('_invalidatePageContextPin');
const pinSrc = extract('_pinCurrentPageContext');
const removePinSrc = extract('_removePinnedPageContext');
const removeCurrentSrc = extract('_removeCurrentPageFromConversationContext');
const clearSrc = extract('clearConversation');

ok(src.includes('var _currentPageContextPermissionMemory = null'), 'current-page preference has an in-memory fallback');
ok(enabledSrc.includes('_currentPageContextPermissionMemory'), 'current-page enabled resolver consults the in-memory fallback');
ok(setCurrentSrc.includes('_currentPageContextPermissionMemory = on'), 'every explicit current-page toggle updates in-memory authority first');
ok(src.includes('var _pageContextConversationGeneration = 0'), 'page-context mutations have a conversation generation');
ok(src.includes('var _pageContextPinRevisions = Object.create(null)'), 'page pinning has URL-scoped mutation revisions');
ok(pinSrc.includes('expectedConversationGeneration') && pinSrc.includes('expectedPinRevision'), 'async pin captures both conversation and URL revision guards');
ok(pinSrc.includes("throw new Error('PAGE_CONTEXT_PIN_STALE')"), 'stale async pin completion is explicitly cancellable');
ok(removePinSrc.includes('_invalidatePageContextPin(target)'), 'unpin invalidates an in-flight pin for the same page');
ok(!removeCurrentSrc.includes('_invalidatePageContextPin(target)') && removeCurrentSrc.includes('_setPageContextConsumed(target, true, true)'), 'one-turn page unstage does not mutate or invalidate the saved pin lifecycle');
ok(clearSrc.includes('_pageContextConversationGeneration += 1'), 'New Chat invalidates every prior async page-context mutation');
ok(clearSrc.includes('_pageContextPinRevisions = Object.create(null)'), 'New Chat resets URL-scoped pin revisions');
ok(src.includes("err.message === 'PAGE_CONTEXT_PIN_STALE'"), 'stale pin cancellation does not produce a false failure toast');
ok(src.includes('Conversation persistence is disabled by site configuration.'), 'pin notification does not tell readers to enable a site-disabled persistence feature');
ok(src.includes('Save a bounded Markdown snapshot and stage it once. Turn on Remember conversation to retain the saved source after navigation.'), 'pin menu copy distinguishes saved-source persistence from one-turn staging');
ok(src.includes('Conversation persistence is disabled by site configuration.'), 'pin menu copy reflects site-disabled persistence authority');
ok(shelfSrc.includes('_currentPageContextActive()') && outboundSrc.includes('_currentPageContextActive()'), 'visible shelf and outbound context share one active-page authority');
ok(shelfSrc.includes('if (currentActive && sourceUrl === currentUrl) return'), 'visible shelf deduplicates current+pin identity after consumed-state filtering');
ok(outboundSrc.includes('if (currentActive && sourceUrl === currentUrl) return'), 'outbound context deduplicates current+pin identity identically after consumed-state filtering');

// ----- Storage-unavailable preference fallback -----
{
let storedValue = null;
let storageThrows = true;
var sessionStorage = {
  getItem() { if (storageThrows) throw new Error('blocked'); return storedValue; },
  setItem(_k, v) { if (storageThrows) throw new Error('blocked'); storedValue = v; }
};
var _CURRENT_PAGE_CONTEXT_PERMISSION_KEY = 'ctx';
var _currentPageContextPermissionMemory = null;
function _cfg(){ return { panelCurrentPageContext: true }; }
function _currentContextPageUrl(){ return 'https://docs.test/A.html'; }
function _setCurrentPageContextExcluded(){ return false; }
function _stagePageContextForNextTurn(){}
function _prepareCurrentPageContextItem(){ return Promise.resolve({}); }
function _renderComposerAttachments(){}
var document = { getElementById(){ return null; } };
const enabledFn = eval('(' + enabledSrc + ')');
const setCurrentFn = eval('(' + setCurrentSrc + ')');
ok(enabledFn() === true, 'storage blocked: site default still initializes current-page context');
setCurrentFn(false);
ok(enabledFn() === false, 'storage blocked: explicit OFF remains effective in memory');
setCurrentFn(false);
ok(enabledFn() === false, 'storage blocked: repeated OFF is idempotent');
setCurrentFn(true);
ok(enabledFn() === true, 'storage blocked: explicit ON remains effective in memory');
setCurrentFn(true);
ok(enabledFn() === true, 'storage blocked: repeated ON is idempotent');
storageThrows = false;
storedValue = 'false';
ok(enabledFn() === false, 'when storage becomes readable, explicit stored tab preference becomes authoritative');
}

// ----- Remember ON/OFF authority and idempotency -----
{
  const rememberSrc = extract('_setRememberConversationInTab');
  const storage = new Map([['ctx-pref','false']]);
  var sessionStorage = {
    setItem(k,v){ storage.set(k,String(v)); },
    removeItem(k){ storage.delete(k); },
    getItem(k){ return storage.has(k) ? storage.get(k) : null; }
  };
  var _TRANSCRIPT_PERSIST_PERMISSION_KEY = 'remember-pref';
  var _CONVERSATION_ID_KEY='conv';
  var _TRANSCRIPT_KEY='transcript', _FEEDBACK_STATE_KEY='feedback';
  var _TRANSCRIPT_COUNT_KEY = 'ai-assistant-transcript-count';
  var _PINNED_PAGE_CONTEXT_KEY='pins', _CURRENT_PAGE_CONTEXT_EXCLUSIONS_KEY='exclusions', _CONSUMED_PAGE_CONTEXT_KEY='consumed';
  var _conversationId='conv-live';
  let loads=0, saves=0, deletes=[];
  function _cfg(){ return {panelPersist:true}; }
  function _loadPinnedPageContexts(){ loads++; return [{sourceUrl:'A'}]; }
  function _loadCurrentPageContextExclusions(){ loads++; return []; }
  function _loadConsumedPageContexts(){ loads++; return {}; }
  function _saveTranscript(){ saves++; }
  function _saveFeedbackState(){ saves++; }
  function _savePinnedPageContexts(){ saves++; }
  function _saveCurrentPageContextExclusions(){ saves++; }
  function _saveConsumedPageContexts(){ saves++; }
  function _ssSet(k,v){ storage.set(k,String(v)); }
  function _ssDel(k){ storage.delete(k); deletes.push(k); }
  var document={getElementById(){return null;}};
  function _renderComposerAttachments(){}
  const setRememberFn=eval('('+rememberSrc+')');
  setRememberFn(true);
  ok(storage.get('remember-pref')==='true', 'Remember OFF→ON stores explicit ON permission');
  ok(loads===3 && saves===5, 'Remember ON hydrates saved pins, exclusions, and consumed one-turn state before saving conversation state');
  ok(storage.get('ctx-pref')==='false', 'Remember toggle does not mutate independent current-page ON/OFF preference');
  const savesAfterFirstOn=saves;
  setRememberFn(true);
  ok(storage.get('remember-pref')==='true' && saves===savesAfterFirstOn+5, 'repeated Remember ON is state-idempotent and safely refreshes persistence');
  setRememberFn(false);
  ok(storage.get('remember-pref')==='false', 'Remember ON→OFF stores explicit OFF permission');
  ok(!storage.has('pins') && !storage.has('exclusions') && !storage.has('consumed'), 'Remember OFF removes persisted page-source and consumed-transport state');
  ok(storage.get('ctx-pref')==='false', 'Remember OFF leaves current-page preference untouched');
  const deleteCount=deletes.length;
  setRememberFn(false);
  // Eight, not seven: the transcript integrity marker is pruned with the
  // transcript it describes. A marker left behind would make the next restore
  // compare a fresh conversation against a stale expected length and report a
  // shortfall that never happened.
  ok(storage.get('remember-pref')==='false' && deletes.length===deleteCount+8, 'repeated Remember OFF remains state-idempotent while pruning persisted conversation state');
  ok(deletes.includes('ai-assistant-transcript-count'), 'the transcript integrity marker is pruned alongside the transcript');
  setRememberFn(true);
  ok(storage.get('remember-pref')==='true', 'Remember OFF→ON can be re-enabled after repeated OFF without corrupting control state');
}

// ----- Four-state visibility/context matrix -----
function norm(v){ return String(v || '').split('#')[0].split('?')[0]; }
function matrixState({remember, current, page='B', storedPins=['A'], livePins=[]}) {
  // A fresh navigation document has only persisted pins when Remember is ON.
  const pins = (remember ? storedPins : livePins).map(name => ({
    kind:'page', contextRole:'pinned', name,
    sourceUrl:`https://docs.test/${name}.html`, text:`${name} context`
  }));
  const currentUrl = `https://docs.test/${page}.html`;
  const visible = [];
  if (current) visible.push({kind:'page', contextRole:'current', name:page, sourceUrl:currentUrl});
  for (const p of pins) {
    if (current && norm(p.sourceUrl) === norm(currentUrl)) continue;
    visible.push(p);
  }
  const outbound = visible.filter(x => x.kind === 'page').map(x => `${x.contextRole}:${x.name}`);
  return {visible, outbound};
}
let m = matrixState({remember:false,current:false,storedPins:['A']});
ok(m.visible.length === 0, 'matrix OFF/OFF: no current page and no previous persisted pins after navigation');
m = matrixState({remember:false,current:true,storedPins:['A'],page:'B'});
ok(m.visible.map(x=>x.contextRole+':'+x.name).join(',') === 'current:B', 'matrix Remember OFF / Current ON: only new live current page is present after navigation');
m = matrixState({remember:true,current:false,storedPins:['A','C'],page:'B'});
ok(m.visible.map(x=>x.name).join(',') === 'A,C', 'matrix Remember ON / Current OFF: all previous pins are visible without a PAGE card');
m = matrixState({remember:true,current:true,storedPins:['A','C'],page:'B'});
ok(m.visible.map(x=>x.contextRole+':'+x.name).join(',') === 'current:B,pinned:A,pinned:C', 'matrix ON/ON: current page plus all previous pins are visible');
m = matrixState({remember:true,current:true,storedPins:['B','A'],page:'B'});
ok(m.visible.map(x=>x.contextRole+':'+x.name).join(',') === 'current:B,pinned:A', 'matrix ON/ON: current page already pinned is represented once');
ok(m.outbound.join(',') === 'current:B,pinned:A', 'matrix ON/ON: outbound context matches visible deduplicated shelf');

// ----- Async pin race runtime using the production functions -----
{
var _pageContextConversationGeneration = 0;
var _pageContextPinRevisions = Object.create(null);
var _pinnedPageContexts = [];
var _PINNED_PAGE_CONTEXT_MAX_ITEMS = 6;
var _currentPageContextCache = null;
let currentUrl = 'https://docs.test/A.html';
let deferredResolve = null;
let renderCount = 0;
let saveCount = 0;
function _normalizeContextPageUrl(v){ return norm(v); }
function _loadPinnedPageContexts(){ return _pinnedPageContexts; }
function _currentContextPageUrl(){ return currentUrl; }
function _prepareCurrentPageContextItem(){ return new Promise(resolve => { deferredResolve = resolve; }); }
function _findPinnedPageContext(url){ return _pinnedPageContexts.find(x=>norm(x.sourceUrl)===norm(url)) || null; }
function _sanitizePinnedPageContext(raw){
  if (!raw || !raw.sourceUrl || !raw.text) return null;
  return { ...raw, id:'page:'+norm(raw.sourceUrl), kind:'page', contextRole:'pinned', sourceUrl:norm(raw.sourceUrl), previewText:raw.text };
}
function _savePinnedPageContexts(){ saveCount++; }
function _stagePageContextForNextTurn(){}
function _renderComposerAttachments(){ renderCount++; }
const _pageContextPinRevision = eval('(' + pinRevisionSrc + ')');
const _invalidatePageContextPin = eval('(' + invalidatePinSrc + ')');
const pinFn = eval('(async ' + pinSrc + ')');
function preparedA(text='A markdown') { return {name:'A',title:'A',sourceUrl:'https://docs.test/A.html',markdownUrl:'https://docs.test/A.md',text}; }

// Normal completion.
let p = pinFn();
deferredResolve(preparedA());
let result = await p;
ok(result && _pinnedPageContexts.length === 1, 'async pin: normal completion adds exactly one pin');
ok(saveCount === 1 && renderCount === 1, 'async pin: normal completion saves/renders once');

// Repeated pin is idempotent by URL (replace, never duplicate).
deferredResolve = null;
p = pinFn();
deferredResolve(preparedA('A refreshed'));
await p;
ok(_pinnedPageContexts.length === 1 && _pinnedPageContexts[0].text === 'A refreshed', 'async pin: repeated pin refreshes same URL without duplicate');

// In-flight pin invalidated by later remove/unpin revision.
_pinnedPageContexts = [];
deferredResolve = null;
p = pinFn();
_invalidatePageContextPin('https://docs.test/A.html');
deferredResolve(preparedA('stale remove race'));
let staleRemove = null;
try { await p; } catch (e) { staleRemove = e; }
ok(staleRemove?.message === 'PAGE_CONTEXT_PIN_STALE' && _pinnedPageContexts.length === 0, 'async pin: later remove/unpin wins and stale completion cannot resurrect page');

// In-flight pin invalidated by New Chat conversation generation.
deferredResolve = null;
p = pinFn();
_pageContextConversationGeneration += 1;
_pageContextPinRevisions = Object.create(null);
_pinnedPageContexts = [];
deferredResolve(preparedA('stale new chat race'));
let staleChat = null;
try { await p; } catch (e) { staleChat = e; }
ok(staleChat?.message === 'PAGE_CONTEXT_PIN_STALE' && _pinnedPageContexts.length === 0, 'async pin: New Chat wins and stale completion cannot mutate fresh conversation');

// Latest of two overlapping requests owns the URL revision.
_pinnedPageContexts = [];
let resolveFirst, resolveSecond;
let call = 0;
_prepareCurrentPageContextItem = function(){
  call++;
  return new Promise(resolve => { if (call === 1) resolveFirst = resolve; else resolveSecond = resolve; });
};
const first = pinFn();
const second = pinFn();
resolveFirst(preparedA('older'));
resolveSecond(preparedA('newer'));
let firstErr = null;
try { await first; } catch(e) { firstErr = e; }
await second;
ok(firstErr?.message === 'PAGE_CONTEXT_PIN_STALE', 'async pin: overlapping older request is cancelled by newer request');
ok(_pinnedPageContexts.length === 1 && _pinnedPageContexts[0].text === 'newer', 'async pin: newest overlapping request is the sole committed snapshot');
}

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
