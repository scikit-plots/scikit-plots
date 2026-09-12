// Run 94 regression: explicit pinned documentation context must rehydrate when
// returning to an already-live browser tab. `pageshow` alone is insufficient
// because ordinary tab switching uses visibilitychange/focus.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed=0, failed=0;
function ok(cond,name){if(cond){passed++;}else{failed++;console.error('FAIL '+name);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,started=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;started=true;}else if(src[j]==='}'){d--;if(started&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}

const bindSrc = extract('_bindPinnedPageContextLifecycle');
const scheduleSrc = extract('_schedulePinnedPageContextShelfRefresh');
const refreshSrc = extract('_refreshPinnedPageContextShelf');
const createPanel = extract('createAIPanel');

ok(bindSrc.includes("window.addEventListener('pageshow'"), 'same-tab navigation/BFCache remains covered');
ok(bindSrc.includes("document.addEventListener('visibilitychange'"), 'browser-tab activation listens for visibility changes');
ok(bindSrc.includes("document.visibilityState === 'visible'"), 'hidden-tab transitions do not trigger shelf hydration');
ok(bindSrc.includes("window.addEventListener('focus'"), 'focus activation fallback covers browsers with incomplete visibility events');
ok(bindSrc.includes('_pinnedPageContextLifecycleBound'), 'lifecycle listeners are singleton-bound');
ok(scheduleSrc.includes('_pinnedPageContextRefreshQueued'), 'activation refreshes are coalesced');
ok(scheduleSrc.includes("document.visibilityState === 'hidden'"), 'queued work aborts if tab became hidden again');
ok(scheduleSrc.includes('_refreshPinnedPageContextShelf()'), 'all activation paths converge on canonical shelf refresh');
ok(refreshSrc.includes('_loadPinnedPageContexts(true)'), 'canonical refresh force-rehydrates persisted pins');
ok(refreshSrc.includes('_renderComposerAttachments()'), 'canonical refresh redraws the visible context shelf');
ok(createPanel.includes('_bindPinnedPageContextLifecycle()'), 'panel initialization installs the activation lifecycle');
ok(!createPanel.includes('_aiAssistantPinnedPageShelfPageshowBound'), 'obsolete pageshow-only global binding is removed');

// Runtime event simulation. Both visibilitychange and focus may fire during one
// activation; they should yield one refresh. A later activation must refresh
// again, proving the coalescing flag resets rather than suppressing future tabs.
var _pinnedPageContextLifecycleBound = false;
var _pinnedPageContextRefreshQueued = false;
let refreshes = 0;
function _refreshPinnedPageContextShelf(){ refreshes++; }
const windowListeners = {};
const documentListeners = {};
const window = { addEventListener(type, fn){ (windowListeners[type] ||= []).push(fn); } };
const document = {
  visibilityState: 'visible',
  addEventListener(type, fn){ (documentListeners[type] ||= []).push(fn); }
};
const pending = [];
function queueMicrotask(fn){ pending.push(fn); }
function flush(){ while(pending.length) pending.shift()(); }
const scheduleFn = eval('(' + scheduleSrc + ')');
const _schedulePinnedPageContextShelfRefresh = scheduleFn;
const bindFn = eval('(' + bindSrc + ')');
bindFn();
bindFn();
ok((windowListeners.pageshow||[]).length === 1, 'runtime: pageshow listener is bound once');
ok((windowListeners.focus||[]).length === 1, 'runtime: focus listener is bound once');
ok((documentListeners.visibilitychange||[]).length === 1, 'runtime: visibility listener is bound once');

document.visibilityState = 'hidden';
documentListeners.visibilitychange[0]();
flush();
ok(refreshes === 0, 'runtime: hiding the tab does not refresh the shelf');

document.visibilityState = 'visible';
documentListeners.visibilitychange[0]();
windowListeners.focus[0]();
ok(pending.length === 1, 'runtime: visible+focus activation is coalesced into one queued refresh');
flush();
ok(refreshes === 1, 'runtime: returning to tab performs one shelf rehydrate/render');

windowListeners.focus[0]();
flush();
ok(refreshes === 2, 'runtime: a later activation can refresh again');

document.visibilityState = 'visible';
windowListeners.pageshow[0]();
flush();
ok(refreshes === 3, 'runtime: BFCache/pageshow path uses the same refresh lifecycle');

// Queued activation followed immediately by hiding must not touch DOM/storage.
document.visibilityState = 'visible';
windowListeners.focus[0]();
document.visibilityState = 'hidden';
flush();
ok(refreshes === 3, 'runtime: queued activation is cancelled when tab becomes hidden before execution');

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
