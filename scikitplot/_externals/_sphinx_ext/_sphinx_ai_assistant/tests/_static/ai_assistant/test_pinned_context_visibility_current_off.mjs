// Run 93 regression: explicit pinned documentation context must remain visible
// and restorable when automatic current-page context is OFF.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed=0, failed=0;
function ok(cond,name){if(cond){passed++;}else{failed++;console.error('FAIL '+name);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,started=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;started=true;}else if(src[j]==='}'){d--;if(started&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}

const mergeSrc = extract('_mergePinnedPageContextSets');
const loadSrc = extract('_loadPinnedPageContexts');
const shelfSrc = extract('_contextShelfItems');
const setCurrent = extract('_setCurrentPageContextInTab');
const remember = extract('_setRememberConversationInTab');
const createPanel = extract('createAIPanel');
const refresh = extract('_refreshPinnedPageContextShelf');

ok(loadSrc.includes('if (_pinnedPageContextsLoaded && !force)'), 'loader supports explicit refresh without discarding normal cache');
ok(loadSrc.indexOf('if (!_persistEnabled()) return _pinnedPageContexts;') < loadSrc.indexOf('_pinnedPageContextsLoaded = true'), 'persistence-ineligible read does not poison loaded-empty state');
ok(loadSrc.includes('var restored = []'), 'restoration is assembled off to the side before replacing visible state');
ok(loadSrc.includes('_mergePinnedPageContextSets(restored, (force && wasLoaded) ? [] : memory)'), 'first hydration reconciles live pins while forced BFCache/tab refresh uses persisted state as authority');
ok(refresh.includes('_loadPinnedPageContexts(true)'), 'shelf refresh force-rehydrates persisted pins');
ok(refresh.includes('_renderComposerAttachments()'), 'shelf refresh immediately rerenders visible context cards');
const lifecycleSrc = extract('_bindPinnedPageContextLifecycle');
ok(createPanel.includes('_bindPinnedPageContextLifecycle()'), 'panel installs the canonical pinned-context lifecycle');
ok(lifecycleSrc.includes("window.addEventListener('pageshow'"), 'BFCache/page restoration refreshes the pinned context shelf');
ok(lifecycleSrc.includes('_schedulePinnedPageContextShelfRefresh'), 'pageshow routes through the canonical activation refresh scheduler');
ok(remember.includes('_loadPinnedPageContexts(false)') && remember.includes('_savePinnedPageContexts()'), 'enabling Remember hydrates before saving so existing tab pins are not overwritten');
ok(remember.includes('_renderComposerAttachments()'), 'Remember toggle immediately reconciles visible shelf state');
ok(setCurrent.includes('_renderComposerAttachments()'), 'current-page toggle rerenders shelf instead of hiding the entire tray');
ok(!setCurrent.includes('_pinnedPageContexts = []'), 'turning automatic current context off never clears explicit pins');
ok(shelfSrc.includes('var currentActive = _currentPageContextActive()') && shelfSrc.includes('_pinnedPageContexts.forEach'), 'automatic PAGE active-state gating and pinned MD enumeration are separate branches');

// Runtime regression for the exact reported sequence: an early read while
// persistence is unavailable must remain retryable; once Remember is active,
// the persisted pin must hydrate and remain visible with current-page context OFF.
var _pinnedPageContexts = [];
var _pinnedPageContextsLoaded = false;
var persistence = false;
const storedPage = {
  schemaVersion: 1,
  items: [{
    name: 'Bayesian Inference', title: 'Bayesian Inference',
    sourceUrl: 'https://docs.test/003-bayesian-inference.html',
    markdownUrl: 'https://docs.test/003-bayesian-inference.md',
    text: '# Bayesian Inference\nPinned content', pinnedAt: 123
  }]
};
function _persistEnabled(){ return persistence; }
function _ssGet(key){ return key === 'ai-assistant-pinned-page-contexts-v1' ? JSON.stringify(storedPage) : null; }
function _ssDel(_key){}
const _PINNED_PAGE_CONTEXT_KEY = 'ai-assistant-pinned-page-contexts-v1';
const _PINNED_PAGE_CONTEXT_SCHEMA = 1;
const _PINNED_PAGE_CONTEXT_MAX_ITEMS = 6;
const _PINNED_PAGE_CONTEXT_MAX_CHARS = 24000;
const _PINNED_PAGE_CONTEXT_TOTAL_CHARS = 96000;
function _attachmentLineCount(text){ return String(text||'').split(/\r?\n/).length; }
function _normalizeContextPageUrl(v){ return String(v||'').split('#')[0].split('?')[0]; }
function _sanitizePinnedPageContext(raw){
  if(!raw || !raw.sourceUrl || !raw.text) return null;
  const text=String(raw.text).slice(0,_PINNED_PAGE_CONTEXT_MAX_CHARS);
  return {id:'page:'+raw.sourceUrl,kind:'page',contextRole:'pinned',name:raw.name,title:raw.title,sourceUrl:raw.sourceUrl,markdownUrl:raw.markdownUrl,text,previewText:text,lineCount:_attachmentLineCount(text),size:text.length,pinnedAt:raw.pinnedAt,localOnly:false};
}
const mergeFn = eval('(' + mergeSrc + ')');
const _mergePinnedPageContextSets = mergeFn;
const loadFn = eval('(' + loadSrc + ')');
ok(loadFn(false).length === 0 && _pinnedPageContextsLoaded === false, 'runtime: unavailable persistence leaves loader retryable');
persistence = true;
ok(loadFn(false).length === 1 && _pinnedPageContextsLoaded === true, 'runtime: later Remember-enabled read restores persisted pin');
ok(_pinnedPageContexts[0].name === 'Bayesian Inference', 'runtime: restored pin preserves page identity');

const _loadPinnedPageContexts = loadFn;
function _currentPageContextEnabled(){ return false; }
function _currentPageContextActive(){ return false; }
function _currentContextPageUrl(){ return 'https://docs.test/004-next.html'; }
var _currentPageContextCache = null;
function _currentPageContextPlaceholder(){ throw new Error('must not build PAGE card while auto current context is off'); }
function _findPinnedPageContext(){ return null; }
var _composerAttachments = [];
function _pageContextConsumed(){ return false; }
const shelfFn = eval('(' + shelfSrc + ')');
const visible = shelfFn();
ok(visible.length === 1, 'runtime: current-page OFF still leaves persisted pinned page visible');
ok(visible[0].contextRole === 'pinned' && visible[0].kind === 'page', 'runtime: restored item renders as pinned MD context, not automatic PAGE');
ok(!visible.some(x => x.contextRole === 'current'), 'runtime: automatic current PAGE stays hidden while its toggle is OFF');

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
