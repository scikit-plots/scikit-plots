// Run 82 regression: the main AI Assistant header must consume the same
// fit-based single-row overflow architecture as every slide-over sheet.
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
const overflow = extract('_buildSheetActionOverflow');

t('main header regular state is four semantic regions',
  /\.ai-assistant-panel-header\.ai-assistant-panel-header--adaptive\s*\{[\s\S]*?grid-template-areas:\s*"menu title toolbar close";/.test(css), true);
t('main compact state remains one row',
  /data-panel-head-compact="true"\][\s\S]*?grid-template-areas:\s*"menu title overflow close";/.test(css), true);
t('main compact state hides full toolbar atomically',
  /data-panel-head-compact="true"\][\s\S]*?ai-assistant-panel-header-toolbar\s*\{\s*display:\s*none;/.test(css), true);
t('main compact state reveals vertical overflow atomically',
  /data-panel-head-compact="true"\][\s\S]*?ai-assistant-panel-header-overflow\s*\{\s*display:\s*block;/.test(css), true);
t('hamburger is a stable header region outside branded title',
  panel.includes("hamburgerBtn.classList.add('ai-assistant-panel-header-menu')") &&
  panel.includes('header.appendChild(hamburgerBtn);') &&
  !panel.includes('headerTitle.appendChild(hamburgerBtn);'), true);
t('brand title keeps logo and text together',
  panel.includes('headerTitle.appendChild(logo);') &&
  panel.includes('headerTitle.appendChild(titleSpan);') &&
  panel.includes("headerTitle.classList.add('ai-assistant-panel-header-brand')"), true);
t('close is removed from collapsible wide toolbar',
  !/headerActions\.appendChild\(closeBtn\)/.test(panel), true);
t('close remains a direct persistent header control',
  panel.includes("closeBtn.classList.add('ai-assistant-panel-header-close')") &&
  panel.includes('header.appendChild(closeBtn);'), true);
t('main overflow reuses canonical sheet overflow factory',
  panel.includes("_buildSheetActionOverflow(\n            'main', header, headerActions"), true);
t('main entry declares its own compact attribute',
  panel.includes("compactAttr: 'data-panel-head-compact'"), true);
t('fit controller supports per-header compact attribute',
  sync.includes("entry.compactAttr || 'data-sheet-head-compact'"), true);
t('fit controller accounts for explicit main menu and close controls',
  sync.includes('entry.leadingMenu') && sync.includes('entry.close'), true);
t('main compact menu gets same canonical New/Export/Min/Resize actions',
  overflow.includes('clearConversation();') &&
  overflow.includes('_buildExportDropdownBtn({') &&
  overflow.includes("'minimize', 'Minimize panel'") &&
  overflow.includes("'maximize', 'Maximize panel'") &&
  overflow.includes("'collapse', 'Collapse full screen'"), true);
t('fit transition closes hidden wide export picker',
  sync.includes("'.ai-assistant-export-menu[data-open=\"true\"]'") &&
  sync.includes('_closeExportMenu(wideExportMenu, wideExportTrigger);'), true);
t('main overflow trigger uses established vertical-three-dot visual',
  /\.ai-assistant-panel-header-overflow \.ai-assistant-panel-subbar-overflow-btn\s*\{/.test(css), true);
t('hamburger-off feature flag does not reserve ghost column',
  /ai-assistant-panel-header--no-menu\s*\{[\s\S]*?grid-template-columns:\s*minmax\(0,\s*1fr\)\s+max-content\s+28px;/.test(css), true);
t('main header shares phone safe-area treatment with sheets',
  /@media\s*\(max-width:\s*480px\)[\s\S]*?ai-assistant-panel-header\.ai-assistant-panel-header--adaptive,[\s\S]*?env\(safe-area-inset-left\)/.test(css), true);
t('no compact second row is introduced for main header',
  /data-panel-head-compact="true"\][\s\S]{0,500}?grid-template-areas:\s*"[^\n"]*"\s*"/.test(css), false);

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
