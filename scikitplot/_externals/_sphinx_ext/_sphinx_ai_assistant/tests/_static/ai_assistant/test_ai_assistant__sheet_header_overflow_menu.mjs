// Run 81 regression: constrained sheet headers stay one row and replace the
// secondary action toolbar atomically with a panel-bounded vertical ⋮ menu.
// Compactness is fit-based, not a device-name/width lookup.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let pass = 0, fail = 0;
function t(name, got, want) {
  if (got === want) pass++;
  else { fail++; console.log(`FAIL ${name}\n  got: ${JSON.stringify(got)}\n want: ${JSON.stringify(want)}`); }
}
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { depth++; started = true; }
    else if (src[j] === '}') {
      depth--;
      if (started && depth === 0) return src.slice(i, j + 1);
    }
  }
  throw new Error('unbalanced: ' + name);
}

const panel = extract('createAIPanel');
const sync = extract('_syncSheetHeaderOverflow');
const buildOverflow = extract('_buildSheetActionOverflow');
const buildExport = extract('_buildExportDropdownBtn');

t('regular header remains one row with full toolbar',
  /grid-template-areas:\s*"menu title toolbar close";/.test(css), true);
t('compact header is one row with overflow replacing toolbar',
  /data-sheet-head-compact="true"\][\s\S]*?grid-template-areas:\s*"menu title overflow close";/.test(css), true);
t('old compact second-row toolbar is gone',
  /"toolbar toolbar toolbar"/.test(css), false);
t('compact mode hides wide toolbar',
  /data-sheet-head-compact="true"\][\s\S]*?ai-assistant-panel-sheet-head-toolbar\s*\{\s*display:\s*none;/.test(css), true);
t('compact mode reveals dedicated overflow anchor',
  /data-sheet-head-compact="true"\][\s\S]*?ai-assistant-panel-sheet-head-overflow\s*\{\s*display:\s*block;/.test(css), true);
t('header overflow trigger reuses vertical-three-dot control language',
  buildOverflow.includes("'ai-assistant-panel-subbar-overflow-btn ai-assistant-panel-sheet-overflow-btn'"), true);
t('overflow trigger exposes menu semantics',
  buildOverflow.includes("setAttribute('aria-haspopup', 'menu')") && buildOverflow.includes("setAttribute('aria-expanded', 'false')"), true);
t('fit controller reserves readable title budget',
  panel.includes('var _SHEET_HEAD_TITLE_BUDGET_PX = 112;'), true);
t('compact decision measures actual wide toolbar',
  sync.includes('toolbar.scrollWidth || toolbar.offsetWidth || 0'), true);
t('wide toolbar measurement is cached across hidden compact state',
  sync.includes('entry.toolbarWideWidth'), true);
t('compact decision uses live head geometry rather than fixed phone width',
  sync.includes('head.getBoundingClientRect().width') && !sync.includes('w < 340'), true);
t('resize observer routes live width into fit controller',
  panel.includes('_syncSheetHeaderOverflow(w);'), true);
t('overflow menu height is bounded from live panel and trigger rects',
  panel.includes('panelRect.bottom - btnRect.bottom - 10'), true);
t('overflow menu scrolls on short panels',
  /\.ai-assistant-panel-sheet-overflow-menu\s*\{[\s\S]*?overflow-y:\s*auto;/.test(css), true);
t('compact menu owns canonical new chat action',
  buildOverflow.includes('clearConversation();'), true);
t('compact export reuses canonical export dropdown builder',
  buildOverflow.includes("_buildExportDropdownBtn({") && buildOverflow.includes("triggerLabel: 'Export conversation'"), true);
t('export builder supports optional visible trigger label without changing wide callers',
  buildExport.includes("triggerLabelText") && buildExport.includes("ai-assistant-export-trigger-label"), true);
t('compact export closes parent after a real format selection',
  buildExport.includes('if (onSelect) { onSelect(fmt); }') && buildOverflow.includes('onSelect: function () { _closeSheetActionOverflow(entry, false); }'), true);
t('compact export submenu opens inline instead of sideways off a narrow panel',
  /sheet-overflow-menu \.ai-assistant-export-menu\s*\{[\s\S]*?position:\s*static;/.test(css), true);
t('minimize stays in compact action menu',
  buildOverflow.includes("'minimize', 'Minimize panel'"), true);
t('maximize and collapse remain state-synchronised in compact menu',
  buildOverflow.includes("_extraMaxPairs.push({ max: maximize, col: collapse })"), true);
t('Escape closes sheet action overflow before closing sheet/panel',
  panel.indexOf('_closeAnySheetActionOverflow(true)') >= 0 &&
  panel.indexOf('_closeAnySheetActionOverflow(true)') < panel.indexOf("hamburgerMenuEl && typeof hamburgerMenuEl._exit"), true);
t('one document outside-click closer handles all sheet overflow menus',
  panel.includes("_sheetActionOverflowEntries.forEach(function (entry) {") &&
  panel.includes("if (entry.anchor.contains(e.target)) return;"), true);
t('every registry sheet receives both toolbar and overflow representations centrally',
  panel.includes('_buildSheetActionOverflow(\n                entry.toolbarId, head, sheetToolbar)') &&
  panel.includes('head.insertBefore(sheetOverflow, closeBtnEl)'), true);
t('safe-area padding remains independent from compact layout',
  /@media\s*\(max-width:\s*480px\)[\s\S]*?env\(safe-area-inset-left\)[\s\S]*?env\(safe-area-inset-right\)/.test(css), true);

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
