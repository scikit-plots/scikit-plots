// Run 80 regression: every slide-over sheet header must use one deterministic
// semantic grid so title length cannot orphan toolbar/close controls. Run 81
// later replaces the original two-row compact presentation with a one-row ⋮ menu,
// while preserving Run 80's semantic-region normalization.
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

const panelCreate = extract('createAIPanel');

t('panel avoids new size-containment side effects',
  /container-name:\s*ai-assistant-panel-shell/.test(css), false);
const baseHeadMatch = css.match(/\.ai-assistant-panel-privacy-head\s*\{([^}]*)\}/);
t('legacy semantic header no longer uses flex-wrap fallback',
  !!baseHeadMatch && /flex-wrap:\s*wrap;/.test(baseHeadMatch[1]), false);
t('normalized sheet head uses four-area regular grid',
  /\.ai-assistant-panel-privacy-head\.ai-assistant-panel-sheet-head-layout\s*\{[\s\S]*?grid-template-areas:\s*"menu title toolbar close";/.test(css), true);
t('title column is shrink-safe',
  /grid-template-columns:\s*28px\s+minmax\(0,\s*1fr\)\s+max-content\s+28px;/.test(css), true);
t('toolbar is atomic and non-wrapping',
  /\.ai-assistant-panel-sheet-head-toolbar\s*\{[\s\S]*?min-width:\s*max-content;[\s\S]*?flex-wrap:\s*nowrap;/.test(css), true);
t('compact evolution no longer consumes a second header row',
  /"toolbar toolbar toolbar"/.test(css), false);
t('mobile header respects horizontal safe areas',
  /padding-left:\s*max\(0\.75rem,\s*env\(safe-area-inset-left\)\);[\s\S]*?padding-right:\s*max\(0\.75rem,\s*env\(safe-area-inset-right\)\);/.test(css), true);
t('compact state now belongs to each normalized sheet head',
  /ai-assistant-panel-sheet-head-layout\[data-sheet-head-compact=\"true\"\]/.test(css), true);
t('existing resize observer delegates to adaptive header sync',
  panelCreate.includes('_syncSheetHeaderOverflow(w);'), true);
t('central registry marks every sheet head once',
  panelCreate.includes("head.classList.add('ai-assistant-panel-sheet-head-layout')"), true);
t('central registry marks menu region',
  panelCreate.includes("menuBtnEl.classList.add('ai-assistant-panel-sheet-head-menu')"), true);
t('central registry marks title region atomically',
  panelCreate.includes("titleRegionEl.classList.add('ai-assistant-panel-sheet-head-title')"), true);
t('central registry marks close region',
  panelCreate.includes("closeBtnEl.classList.add('ai-assistant-panel-sheet-head-close')"), true);
t('central registry marks toolbar region before insertion',
  panelCreate.includes("sheetToolbar.classList.add('ai-assistant-panel-sheet-head-toolbar')"), true);
t('toolbar remains inserted immediately before close',
  panelCreate.includes('head.insertBefore(sheetToolbar, closeBtnEl)'), true);
t('share title+badge wrapper remains supported as atomic title region',
  src.includes("headLeft.className = 'ai-assistant-conv-share-head-left'"), true);

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
