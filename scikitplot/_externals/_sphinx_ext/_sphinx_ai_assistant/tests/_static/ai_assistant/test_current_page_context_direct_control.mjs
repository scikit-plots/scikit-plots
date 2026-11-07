// Run 97 regression: the current-page setting is a default staging convenience.
// PAGE context is directly removable/restageable for the next question and is
// consumed after Send, independently of the saved pin/bookmark lifecycle.
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

const activeSrc = extract('_currentPageContextActive');
const setExcludedSrc = extract('_setCurrentPageContextExcluded');
const loadExcludedSrc = extract('_loadCurrentPageContextExclusions');
const saveExcludedSrc = extract('_saveCurrentPageContextExclusions');
const shelfSrc = extract('_contextShelfItems');
const prepareSrc = extract('_privacyPrepareDocumentationContext');
const removeCurrentSrc = extract('_removeCurrentPageFromConversationContext');
const renderSrc = extract('_renderComposerAttachments');
const removeResource = extract('_removeComposerResourceItem');
const setCurrentSrc = extract('_setCurrentPageContextInTab');
const rememberSrc = extract('_setRememberConversationInTab');
const clearSrc = extract('clearConversation');
const panelSrc = extract('createAIPanel');

ok(src.includes("var _CURRENT_PAGE_CONTEXT_EXCLUSIONS_KEY = 'ai-assistant-current-page-context-exclusions-v1'"), 'current-page exclusions use an explicit versioned storage key');
ok(src.includes('var _CURRENT_PAGE_CONTEXT_EXCLUSIONS_MAX_ITEMS = 32'), 'current-page exclusion list is bounded');
ok(activeSrc.includes('_currentPageContextEnabled()') && activeSrc.includes('!_isCurrentPageContextExcluded'), 'active current-page context requires global ON and no page exclusion');
ok(loadExcludedSrc.includes("if (!_persistEnabled()) return _currentPageContextExclusions"), 'exclusions persist only under Remember conversation authority');
ok(loadExcludedSrc.includes('memory.forEach'), 'hydration reconciles live exclusion intent instead of replacing it');
ok(loadExcludedSrc.includes('if (!(force && wasLoaded))'), 'forced refresh of an already-hydrated document rejects stale BFCache exclusion memory');
ok(saveExcludedSrc.includes('_CURRENT_PAGE_CONTEXT_EXCLUSIONS_MAX_ITEMS'), 'saved exclusion list keeps the item bound');
ok(setCurrentSrc.includes('_stagePageContextForNextTurn(currentUrl, false)'), 'explicitly switching current-page default ON stages the current page once');
ok(shelfSrc.includes('var currentActive = _currentPageContextActive()'), 'visible shelf uses the per-page active state');
ok(prepareSrc.includes('var currentActive = _currentPageContextActive()'), 'outbound documentation context uses the same per-page active state');
ok(renderSrc.includes("item.kind === 'page' && item.contextRole === 'current'"), 'renderer recognizes automatic current PAGE cards');
ok(renderSrc.includes('_removeComposerResourceItem(item)') && removeResource.includes('_setPageContextConsumed(item.sourceUrl, true, false)'), 'automatic current PAGE card Remove routes through canonical one-turn unstage state');
ok(!removeCurrentSrc.includes('_pinnedPageContexts = _pinnedPageContexts.filter'), 'compatibility Remove helper never deletes a saved same-page pin');
ok(removeCurrentSrc.includes('_setPageContextConsumed(target, true, true)'), 'compatibility Remove helper only consumes/unstages the page for the next message');
ok(removeCurrentSrc.includes('return _setPageContextConsumed(target, true, true)'), 'compatibility Remove helper delegates rendering to canonical one-turn state');
ok(panelSrc.includes('ai-assistant-panel-attach-menu-item--current-context'), '+ menu has a dedicated automatic current-page control');
ok(panelSrc.includes('ai-assistant-panel-attach-menu-item--pin'), '+ menu keeps pin/unpin as a separate lifecycle control');
ok(panelSrc.includes("'Remove current page from next message'") && panelSrc.includes("'Add current page to next message'"), '+ menu exposes reversible one-turn current-page staging actions');
ok(panelSrc.includes("'Pin current page'") && panelSrc.includes("'Unpin current page'"), '+ menu retains navigation-persistence pin/unpin actions');
ok(rememberSrc.includes('_saveCurrentPageContextExclusions()'), 'Remember ON persists conversation page exclusions');
ok(rememberSrc.includes('_ssDel(_CURRENT_PAGE_CONTEXT_EXCLUSIONS_KEY)'), 'Remember OFF deletes persisted exclusions');
ok(clearSrc.includes('_currentPageContextExclusions = []') && clearSrc.includes('_ssDel(_CURRENT_PAGE_CONTEXT_EXCLUSIONS_KEY)'), 'New Chat clears current-page exclusions in memory and storage');
ok(src.includes('Sending consumes that PAGE selection; it is not silently resent on later questions.'), 'Endpoint Configuration explains one-turn current-page transport');

// ----- Actual exclusion-loader authority transitions -----
var _CURRENT_PAGE_CONTEXT_EXCLUSIONS_KEY = 'ai-assistant-current-page-context-exclusions-v1';
var _CURRENT_PAGE_CONTEXT_EXCLUSIONS_SCHEMA = 1;
var _CURRENT_PAGE_CONTEXT_EXCLUSIONS_MAX_ITEMS = 32;
var _currentPageContextExclusions = ['https://docs.test/A.html'];
var _currentPageContextExclusionsLoaded = false;
let loaderPersist = false;
let loaderRaw = null;
function _persistEnabled(){ return loaderPersist; }
function _ssGet(){ return loaderRaw; }
function _ssDel(){ loaderRaw = null; }
function _normalizeContextPageUrl(v){ return String(v || '').split('#')[0].split('?')[0]; }
const exclusionLoadFn = eval('(' + loadExcludedSrc + ')');
let exclusionResult = exclusionLoadFn(false);
ok(exclusionResult.length === 1 && _currentPageContextExclusionsLoaded === false, 'runtime loader: Remember OFF preserves live exclusion without falsely marking hydration complete');
loaderPersist = true;
loaderRaw = JSON.stringify({schemaVersion:1,items:[]});
exclusionResult = exclusionLoadFn(false);
ok(exclusionResult.includes('https://docs.test/A.html'), 'runtime loader: first Remember ON hydration preserves an exclusion created while persistence was off');
_currentPageContextExclusions = ['https://docs.test/A.html'];
_currentPageContextExclusionsLoaded = true;
loaderRaw = JSON.stringify({schemaVersion:1,items:[]});
exclusionResult = exclusionLoadFn(true);
ok(exclusionResult.length === 0, 'runtime loader: forced BFCache/tab refresh accepts persisted removal and does not resurrect stale exclusion memory');

// ----- Runtime state model for exact UI semantics -----
let currentUrl = 'https://docs.test/A.html';
let currentOn = true;
let consumed = new Set();
let pins = [
  {kind:'page', contextRole:'pinned', name:'A', sourceUrl:'https://docs.test/A.html', text:'A pinned'},
  {kind:'page', contextRole:'pinned', name:'Previous', sourceUrl:'https://docs.test/previous.html', text:'Previous pinned'}
];
function norm(v){ return String(v || '').split('#')[0].split('?')[0]; }
function isConsumed(url){ return consumed.has(norm(url)); }
function shelf(){
  const out=[];
  if(currentOn && !isConsumed(currentUrl)) out.push({kind:'page',contextRole:'current',name:'CURRENT',sourceUrl:currentUrl,pinned:pins.some(x=>norm(x.sourceUrl)===norm(currentUrl))});
  for(const item of pins){
    if(isConsumed(item.sourceUrl)) continue;
    if(currentOn && !isConsumed(currentUrl) && norm(item.sourceUrl)===norm(currentUrl)) continue;
    out.push(item);
  }
  return out;
}

let visible = shelf();
ok(visible.map(x=>x.contextRole+':'+x.name).join(',') === 'current:CURRENT,pinned:Previous', 'runtime: initial current-page default stages PAGE once and deduplicates same saved pin');

// Removing/consuming current PAGE leaves the saved pin source intact but unstaged.
consumed.add(norm(currentUrl));
visible = shelf();
ok(visible.map(x=>x.name).join(',') === 'Previous', 'runtime: one-turn Remove hides both PAGE projection and same-URL saved source without deleting the pin');
ok(pins.some(x=>x.name==='A'), 'runtime: saved same-page pin survives one-turn unstage');
ok(currentOn === true, 'runtime: one-turn Remove does not switch the current-page default off');

// Explicit restage re-enables the same source for one question and dedupes its pin.
consumed.delete(norm(currentUrl));
visible = shelf();
ok(visible.map(x=>x.contextRole+':'+x.name).join(',') === 'current:CURRENT,pinned:Previous', 'runtime: explicit Add current page restages PAGE once without duplicate pinned A');

// Send consumes every staged page source for this turn.
for(const item of visible) consumed.add(norm(item.sourceUrl));
visible = shelf();
ok(visible.length === 0, 'runtime: after Send no PAGE/MD source remains armed for the next ordinary question');
ok(pins.map(x=>x.name).join(',') === 'A,Previous', 'runtime: Send consumption never deletes saved pin/bookmark sources');

// Explicitly restaging one saved pin arms only that source.
consumed.delete(norm('https://docs.test/previous.html'));
visible = shelf();
ok(visible.map(x=>x.name).join(',') === 'Previous', 'runtime: a saved pin can be explicitly restaged without rearming other consumed sources');

// Navigation creates a distinct current-page default source for the new URL.
currentUrl = 'https://docs.test/B.html';
visible = shelf();
ok(visible[0]?.contextRole === 'current' && visible[0]?.sourceUrl === currentUrl, 'runtime: navigation can stage the new current page while old consumed pages stay unarmed');

// Global OFF prevents automatic staging but does not delete pins.
currentOn = false;
visible = shelf();
ok(!visible.some(x=>x.contextRole==='current') && pins.length === 2, 'runtime: global OFF suppresses current PAGE default while saved pins remain stored');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
