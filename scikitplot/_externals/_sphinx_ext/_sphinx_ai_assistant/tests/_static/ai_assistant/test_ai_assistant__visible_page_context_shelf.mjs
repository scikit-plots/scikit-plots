// Run 92 regression: documentation context must be visible, source-aware,
// bounded, deduplicated, previewable, and saved only under the existing same-tab
// authority. Run 109 intentionally makes active PAGE/MD transport one-turn: a
// saved/default source may persist, but it must be explicitly staged again after Send.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let passed=0, failed=0;
function ok(cond,name){if(cond){passed++;}else{failed++;console.error('FAIL '+name);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,started=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;started=true;}else if(src[j]==='}'){d--;if(started&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}

const shelf = extract('_contextShelfItems');
const assemble = extract('_assembleDocumentationContextSources');
const prepare = extract('_privacyPrepareDocumentationContext');
const pin = extract('_pinCurrentPageContext');
const save = extract('_savePinnedPageContexts');
const render = extract('_renderComposerAttachments');
const removeResource = extract('_removeComposerResourceItem');
const submit = extract('handleAIPanelSubmit');
const api = extract('_panelApiCall');
const clear = extract('clearConversation');
const panel = extract('createAIPanel');

ok(src.includes("var _CURRENT_PAGE_CONTEXT_PERMISSION_KEY = 'ai-assistant-current-page-context-this-tab-v1'"),'current-page choice is per-tab');
ok(src.includes("var _PINNED_PAGE_CONTEXT_KEY = 'ai-assistant-pinned-page-contexts-v1'"),'pinned-page store is explicit and versioned');
ok(src.includes("var _CURRENT_PAGE_CONTEXT_EXCLUSIONS_KEY = 'ai-assistant-current-page-context-exclusions-v1'"),'per-conversation current-page exclusions have an explicit bounded store');
ok(src.includes('var _PINNED_PAGE_CONTEXT_MAX_ITEMS = 6'),'pinned-page count is bounded');
ok(src.includes('var _PINNED_PAGE_CONTEXT_MAX_CHARS = 24000'),'each pinned snapshot is bounded');
ok(src.includes('var _PINNED_PAGE_CONTEXT_TOTAL_CHARS = 96000'),'persistent page context has a total bound');
ok(extract('_normalizeContextPageUrl').includes('/^(?:https?:|file:)$/'),'page-context URL normalization rejects script/data schemes');
ok(src.includes('return _cfg().panelCurrentPageContext !== false'),'site default for current page is on unless configured off');
ok(src.includes("'Use current page as context'"),'Endpoint Configuration owns a visible current-page toggle');
ok(src.includes("'ai-assistant-current-page-context-toggle'"),'current-page toggle has stable DOM identity');
ok(panel.includes("pageLabel.textContent = 'Pages'"),'attach menu has a dedicated Pages section');
ok(panel.includes("'Pin current page'" ) && panel.includes("'Unpin current page'"),'attach menu exposes pin/unpin lifecycle');
ok(panel.includes('Current page is already pinned and staged for the next question.') || panel.includes('Current page saved as a pin and staged'), 'pin copy explains saved-source vs one-turn staged distinction');
ok(pin.includes('_prepareCurrentPageContextItem(true)') && extract('_prepareCurrentPageContextItem').includes('_privacyPreparePageContext'),'pin snapshot starts from privacy-prepared live page Markdown');
ok(pin.includes('_sanitizePinnedPageContext'),'pin snapshot passes bounded persistence sanitizer');
ok(save.includes('if (!_persistEnabled())'),'pinned-page persistence is gated by Remember conversation authority');
ok(clear.includes('_ssDel(_PINNED_PAGE_CONTEXT_KEY)'),'new conversation removes persisted pinned-page context');
ok(clear.includes('_pinnedPageContexts = []'),'new conversation removes in-memory pinned pages');
ok(shelf.includes("contextRole: 'pinned'") && shelf.includes('_currentPageContextActive()'),'shelf distinguishes active automatic current and pinned page roles');
ok(shelf.includes('if (currentActive && sourceUrl === currentUrl) return'),'current page and same pinned page are visually deduplicated after one-turn consumed gating');
ok(prepare.includes('if (currentActive && sourceUrl === currentUrl) return'),'outbound context also deduplicates a pinned current page after one-turn consumed gating');
ok(prepare.includes('_assembleDocumentationContextSources'),'all visible page sources use one documentation assembler');
ok(assemble.includes('Current page · automatic') && assemble.includes('Pinned page · reader-selected'),'outbound context labels source authority explicitly');
ok(assemble.includes('Math.floor(bodyBudget / list.length)'),'shared context budget starts fairly across sources');
ok(assemble.includes('[context source truncated to shared budget]'),'per-source truncation is explicit');
const assembleFn = (0,eval)('(' + assemble + ')');
const assembled = assembleFn([
  {contextRole:'current',title:'Current',sourceUrl:'https://docs.test/current.html',text:'A'.repeat(500)},
  {contextRole:'pinned',title:'Related',sourceUrl:'https://docs.test/related.html',text:'B'.repeat(500)}
], 700);
ok(assembled.length <= 700,'runtime assembler respects the shared character budget');
ok(assembled.includes('Current page · automatic') && assembled.includes('Pinned page · reader-selected'),'runtime assembler preserves both source labels');
ok(assembled.includes('AAAA') && assembled.includes('BBBB'),'runtime assembler gives both long sources body budget');
ok(submit.includes('_privacyPrepareDocumentationContext(cfg)'),'privacy preflight reviews the assembled multi-page context');
ok(api.includes('_privacyPrepareDocumentationContext(cfg)'),'direct/internal API callers use the same assembler');
ok(api.includes('_effectivePanelContextLimit(cfg)'),'assembler and final API path share the same context limit');
ok(api.includes('documentation context visibly selected in the assistant panel'),'untrusted fence names the visible multi-source context');
ok(render.includes("item.kind === 'page' && item.contextRole === 'current'"),'current page has dedicated preview lifecycle');
ok(render.includes('_removeComposerResourceItem(item)') && removeResource.includes("if (item.kind === 'page')") && removeResource.includes('_setPageContextConsumed(item.sourceUrl, true, false)'), 'visible page cards can be unstaged without deleting the saved pin/default source');
ok(render.includes('Current page removed from the next message. The automatic setting is unchanged.'), 'current PAGE Remove is one-turn unstage and leaves the automatic default unchanged');
ok(src.includes("if (item && item.kind === 'page') return item.contextRole === 'current' ? 'PAGE' : 'MD';"),'cards clearly distinguish PAGE current from pinned MD');
ok(src.includes('Current-page Markdown staged for the next question only.'),'preview explains current-page one-turn staging');
ok(src.includes('Saved pinned-page Markdown staged for the next question only.'),'preview explains saved-pin versus one-turn transport lifecycle');
ok(panel.includes("attachmentTray.setAttribute('aria-label', 'Pages and files staged for the next message only')"),'composer shelf explicitly declares all visible PAGE/MD/FILE context one-turn');
ok(css.includes('.ai-assistant-panel-attachment-tile[data-kind="page"]'),'page-context cards have a distinct visual treatment');
ok(css.includes('.ai-assistant-panel-attachment-tile[data-context-role="current"]'),'automatic current page has its own visual state');

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
