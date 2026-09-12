// Run 96 regression: pinned documentation context uses a reconciled state
// machine across Remember/current-page toggles, navigation hydration, multiple
// pins, duplicate current-page identity, capacity bounds, and malformed storage.
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

const mergeSrc = extract('_mergePinnedPageContextSets');
const loadSrc = extract('_loadPinnedPageContexts');
const shelfSrc = extract('_contextShelfItems');
const refreshSrc = extract('_refreshPinnedPageContextShelf');
const rememberSrc = extract('_setRememberConversationInTab');
const clearSrc = extract('clearConversation');

ok(mergeSrc.includes('(memoryItems || []).forEach(addUnique)'), 'reconciliation gives live explicit pins first capacity priority');
ok(mergeSrc.includes('(restoredItems || []).forEach(addUnique)'), 'persisted pins fill remaining reconciliation capacity');
ok(mergeSrc.includes('_normalizeContextPageUrl(item.sourceUrl)'), 'source URL is the reconciliation identity');
ok(mergeSrc.includes('_PINNED_PAGE_CONTEXT_MAX_ITEMS'), 'reconciliation reapplies item-count bound');
ok(mergeSrc.includes('_PINNED_PAGE_CONTEXT_TOTAL_CHARS'), 'reconciliation reapplies total-text bound');
ok(loadSrc.includes('var memory = _pinnedPageContexts.slice()'), 'hydration snapshots live pins before reading persistence');
ok(loadSrc.includes('_mergePinnedPageContextSets(restored, (force && wasLoaded) ? [] : memory)'), 'first hydration reconciles live state while forced refresh rejects stale BFCache memory');
ok(refreshSrc.includes('_loadPinnedPageContexts(true)'), 'navigation/tab refresh still force-reads persisted state');
ok(rememberSrc.includes('_loadPinnedPageContexts(false)') && rememberSrc.includes('_savePinnedPageContexts()'), 'Remember enable hydrates/reconciles before first save');
ok(clearSrc.includes('_pinnedPageContexts = []') && clearSrc.includes('_ssDel(_PINNED_PAGE_CONTEXT_KEY)'), 'New Chat clears both memory and persisted pin planes');

// ----- Actual reconciliation helper runtime -----
var _PINNED_PAGE_CONTEXT_MAX_ITEMS = 6;
var _PINNED_PAGE_CONTEXT_MAX_CHARS = 24000;
var _PINNED_PAGE_CONTEXT_TOTAL_CHARS = 96000;
function _normalizeContextPageUrl(value) { return String(value || '').split('#')[0].split('?')[0]; }
function _attachmentLineCount(text) { return String(text || '').split('\n').length; }
function _sanitizePinnedPageContext(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const sourceUrl = _normalizeContextPageUrl(raw.sourceUrl || '');
  let text = typeof raw.text === 'string' ? raw.text.slice(0, _PINNED_PAGE_CONTEXT_MAX_CHARS) : '';
  if (!sourceUrl || !text) return null;
  return {
    id: 'page:' + sourceUrl, kind: 'page', contextRole: 'pinned',
    name: String(raw.name || raw.title || 'Documentation page'),
    title: String(raw.title || raw.name || ''), sourceUrl,
    markdownUrl: raw.markdownUrl || sourceUrl.replace(/\.html?$/i, '.md'),
    text, previewText: text, lineCount: _attachmentLineCount(text), size: text.length,
    pinnedAt: Math.max(0, Number(raw.pinnedAt) || 1), localOnly: false
  };
}
const mergeFn = eval('(' + mergeSrc + ')');
function page(name, n, chars = 32, suffix = '') {
  return { name, title: name, sourceUrl: `https://docs.test/${name}${suffix}.html`, text: name[0].repeat(chars), pinnedAt: n };
}

let merged = mergeFn([], [page('A', 10)]);
ok(merged.length === 1 && merged[0].name === 'A', 'matrix: in-memory-only pin survives empty persisted hydration');

merged = mergeFn([page('B', 5)], [page('A', 10)]);
ok(merged.map(x => x.name).join(',') === 'B,A', 'matrix: persisted previous page + live current pin coexist in chronological display order');

merged = mergeFn([{...page('A', 1), text:'OLD'}], [{...page('A', 2), text:'FRESH'}]);
ok(merged.length === 1 && merged[0].text === 'FRESH', 'matrix: duplicate URL keeps freshest live-document snapshot');

merged = mergeFn([page('S1',1),page('S2',2),page('S3',3),page('S4',4),page('S5',5),page('S6',6)], [page('M1',20),page('M2',21)]);
ok(merged.length === 6, 'matrix: merged set remains bounded to six pins');
ok(merged.some(x=>x.name==='M1') && merged.some(x=>x.name==='M2'), 'matrix: current-document pins cannot be evicted by six older stored pins');

merged = mergeFn([page('B', 2, 24000), page('C', 3, 24000), page('D', 4, 24000), page('E', 5, 24000)], [page('A', 1, 24000)]);
ok(merged.reduce((n,x)=>n+x.text.length,0) <= 96000, 'matrix: merged Markdown never exceeds total persistence budget');
ok(merged.every(x=>x.text.length <= 24000), 'matrix: merged Markdown keeps per-page cap');

// ----- Actual loader transition runtime -----
var _PINNED_PAGE_CONTEXT_KEY = 'ai-assistant-pinned-page-contexts-v1';
var _PINNED_PAGE_CONTEXT_SCHEMA = 1;
var _pinnedPageContexts = [];
var _pinnedPageContextsLoaded = false;
let persist = false;
let storageRaw = null;
let deleted = 0;
function _persistEnabled(){ return persist; }
function _ssGet(){ return storageRaw; }
function _ssDel(){ storageRaw = null; deleted++; }
const _mergePinnedPageContextSets = mergeFn;
const loadFn = eval('(' + loadSrc + ')');

_pinnedPageContexts = [_sanitizePinnedPageContext(page('A', 10))];
_pinnedPageContextsLoaded = false;
persist = false;
let loaded = loadFn(false);
ok(loaded.length === 1 && loaded[0].name === 'A', 'matrix: Remember OFF keeps current-document pin in memory');
ok(_pinnedPageContextsLoaded === false, 'matrix: Remember OFF does not poison loader as hydrated-empty');

persist = true;
storageRaw = null;
loaded = loadFn(false);
ok(loaded.length === 1 && loaded[0].name === 'A', 'matrix: OFF → pin A → ON preserves A before first persistence save');
ok(_pinnedPageContextsLoaded === true, 'matrix: successful ON hydration marks loader complete');

_pinnedPageContexts = [_sanitizePinnedPageContext(page('A', 10))];
_pinnedPageContextsLoaded = false;
storageRaw = JSON.stringify({schemaVersion:1,items:[page('B',5)]});
loaded = loadFn(false);
ok(loaded.map(x=>x.name).join(',') === 'B,A', 'matrix: enabling/restoring merges previous persisted B with live A');

_pinnedPageContexts = [_sanitizePinnedPageContext(page('A', 10))];
_pinnedPageContextsLoaded = false;
storageRaw = '{broken json';
deleted = 0;
loaded = loadFn(false);
ok(loaded.length === 1 && loaded[0].name === 'A', 'matrix: malformed persisted payload cannot erase a valid live pin');
ok(deleted === 1, 'matrix: malformed persisted payload is quarantined by deletion');

// ----- Shelf projection combinations -----
function shelfFor({currentOn, currentUrl='https://docs.test/C.html', pins=[], files=[]}) {
  _pinnedPageContexts = pins.map(x => _sanitizePinnedPageContext(x));
  _pinnedPageContextsLoaded = true;
  function _loadPinnedPageContexts(){ return _pinnedPageContexts; }
  function _currentPageContextEnabled(){ return currentOn; }
  function _currentPageContextActive(){ return currentOn; }
  function _currentContextPageUrl(){ return currentUrl; }
  var _currentPageContextCache = null;
  function _currentPageContextPlaceholder(){ return {kind:'page', contextRole:'current', name:'CURRENT', sourceUrl:currentUrl, text:'', loading:true}; }
  function _findPinnedPageContext(url){ return _pinnedPageContexts.find(x=>_normalizeContextPageUrl(x.sourceUrl)===_normalizeContextPageUrl(url)) || null; }
  function _pageContextConsumed(){ return false; }
  var _composerAttachments = files.slice();
  return eval('(' + shelfSrc + ')')();
}

let shelf = shelfFor({currentOn:false,pins:[]});
ok(shelf.length === 0, 'matrix: current OFF + zero pins => empty page shelf');
shelf = shelfFor({currentOn:false,pins:[page('A',1),page('B',2)]});
ok(shelf.map(x=>x.name).join(',') === 'A,B', 'matrix: current OFF + previous A/B => both pinned MD cards visible');
shelf = shelfFor({currentOn:true,pins:[page('A',1),page('B',2)]});
ok(shelf.length === 3 && shelf[0].contextRole === 'current' && shelf.slice(1).map(x=>x.name).join(',')==='A,B', 'matrix: current ON + unrelated pins => PAGE current + all previous MD pins');
shelf = shelfFor({currentOn:true,currentUrl:'https://docs.test/A.html',pins:[page('A',1),page('B',2)]});
ok(shelf.length === 2 && shelf[0].contextRole === 'current' && shelf[1].name === 'B', 'matrix: current ON + current page already pinned => one PAGE identity plus other pins, no duplicate A');
shelf = shelfFor({currentOn:false,currentUrl:'https://docs.test/A.html',pins:[page('A',1),page('B',2)]});
ok(shelf.length === 2 && shelf.map(x=>x.name).join(',')==='A,B', 'matrix: toggling current OFF reveals pinned current A as MD alongside B');
shelf = shelfFor({currentOn:false,pins:[page('A',1)],files:[{kind:'text',name:'notes.md'}]});
ok(shelf.length === 2 && shelf[0].name === 'A' && shelf[1].name === 'notes.md', 'matrix: pinned page context remains visible with ordinary one-turn files');

// ----- End-to-end state-sequence contract (pure transition model) -----
// Mirrors the user-visible lifecycle while keeping transport/browser machinery
// out of this focused state test.
let rememberedStore = [];
let livePins = [];
let rememberOn = false;
function persistModel() { rememberedStore = rememberOn ? livePins.map(x => ({...x})) : []; }
function navigateModel() {
  // A new document starts with no live JS state, then hydrates only when
  // Remember conversation is enabled.
  livePins = rememberOn ? mergeFn(rememberedStore, []) : [];
}
function pinModel(item) {
  livePins = mergeFn([], livePins.concat([item]));
  persistModel();
}
function unpinModel(url) {
  livePins = livePins.filter(x => _normalizeContextPageUrl(x.sourceUrl) !== _normalizeContextPageUrl(url));
  persistModel();
}

// Page A: pin while persistence is OFF.
pinModel(page('A', 1));
ok(livePins.map(x=>x.name).join(',') === 'A' && rememberedStore.length === 0, 'sequence: pin A while Remember OFF is visible now but not yet persisted');
// Enable persistence: reconciliation must keep A and then save it.
rememberOn = true;
livePins = mergeFn(rememberedStore, livePins);
persistModel();
ok(rememberedStore.map(x=>x.name).join(',') === 'A', 'sequence: enabling Remember persists the already-visible A pin');
// Page B: A must restore before B is pinned.
navigateModel();
ok(livePins.map(x=>x.name).join(',') === 'A', 'sequence: Page B immediately restores previous A before any current-page pin action');
pinModel(page('B', 2));
ok(livePins.map(x=>x.name).join(',') === 'A,B', 'sequence: pinning B adds to already-visible A');
// Page C: both previous pins restore without pinning C.
navigateModel();
ok(livePins.map(x=>x.name).join(',') === 'A,B', 'sequence: Page C immediately restores A+B with no current-page pin required');
// Current-page ON projects C in addition to A+B; OFF returns to A+B.
let sequenceShelf = shelfFor({currentOn:true,currentUrl:'https://docs.test/C.html',pins:livePins});
ok(sequenceShelf.length === 3 && sequenceShelf[0].contextRole === 'current' && sequenceShelf.slice(1).map(x=>x.name).join(',')==='A,B', 'sequence: current-page ON adds live C without hiding previous A+B');
sequenceShelf = shelfFor({currentOn:false,currentUrl:'https://docs.test/C.html',pins:livePins});
ok(sequenceShelf.map(x=>x.name).join(',') === 'A,B', 'sequence: current-page OFF removes only C automatic context, preserving A+B');
// Explicit unpin and New Chat semantics.
unpinModel('https://docs.test/A.html');
ok(livePins.map(x=>x.name).join(',') === 'B' && rememberedStore.map(x=>x.name).join(',') === 'B', 'sequence: unpin A removes only A from live and persisted state');
livePins = []; rememberedStore = [];
ok(livePins.length === 0 && rememberedStore.length === 0, 'sequence: New Chat leaves no pinned context in either plane');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
