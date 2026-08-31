// Contract for the two-level hamburger + Usage Policy / Keyboard shortcuts.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let pass=0, fail=0;
function t(name, got, want=true){ if(got===want) pass++; else {fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`);} }

function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth=0, started=false;
  for (let j=i;j<src.length;j++) {
    if(src[j]==='{'){depth++;started=true;}
    else if(src[j]==='}'){depth--;if(started&&depth===0)return src.slice(i,j+1);}
  }
  throw new Error('unbalanced: ' + name);
}

const menu = extract('_buildHamburgerMenu');
const usage = extract('_buildUsagePolicySheet');
const shortcuts = extract('_buildKeyboardShortcutsSheet');

t('primary group built first', /addRegistryGroup\('primary', pop\)/.test(menu));
t('More is a real menuitem', /moreBtn\.setAttribute\('role', 'menuitem'\)/.test(menu));
t('More advertises a menu submenu', /aria-haspopup', 'menu'/.test(menu));
t('More owns an aria-controlled region', /aria-controls', 'ai-assistant-panel-hamburger-more'/.test(menu));
t('More starts collapsed', /moreRegion\.hidden = true/.test(menu));
t('secondary group is built inside More', /addRegistryGroup\('more', moreRegion\)/.test(menu));
t('More click toggles disclosure', /moreBtn\.addEventListener\('click'/.test(menu));
t('ArrowRight opens More', /e\.key === 'ArrowRight'/.test(menu));
t('ArrowLeft closes More', /e\.key === 'ArrowLeft'/.test(menu));
t('popover exposes close-More helper', /pop\._closeMore = function/.test(menu));
t('popover exposes reset-More helper', /pop\._resetMore = function/.test(menu));
t('E footer hint is rendered', /kbdExit[\s\S]*?_createShortcutCaps\(\['E'\]\)/.test(menu));
t('E footer has no visible Exit label', !/kbdExitText/.test(menu));

t('Usage Policy sheet has stable id', /ai-assistant-panel-usage-policy/.test(usage));
t('Usage Policy supports trusted conf override', /panelUsagePolicyHtml/.test(usage));
t('Usage Policy defaults fail-safe around sensitive data', /passwords, API keys, access tokens/.test(usage));

t('Keyboard shortcut sheet has stable id', /ai-assistant-panel-shortcuts-sheet/.test(shortcuts));
t('shortcut sheet is generated from live registry', /_MENU_ITEMS\.forEach/.test(shortcuts));
t('shortcut sheet filters disabled optional features', /cfg\.panelUsagePolicy === false/.test(shortcuts));
t('shortcut sheet maps E to Exit', /shortcutRow\('Exit current menu or sheet', \['E'\]\)/.test(shortcuts));
t('shortcut sheet maps Escape to model stop', /shortcutRow\('Stop model response', \['Escape'\]/.test(shortcuts));
t('shortcut sheet includes composer Send', /Send message/.test(shortcuts));
t('shortcut sheet includes Shift+Enter newline', /New line/.test(shortcuts));

t('createAIPanel wires Usage Policy hook', /onUsagePolicy:\s*usagePolicySheet/.test(src));
t('createAIPanel wires Keyboard shortcuts hook', /onKeyboardShortcuts:\s*function/.test(src));
t('Escape is reserved for active model stop', /if \(!_stopActivePanelResponse\(\)\) return;[\s\S]{0,100}preventDefault/.test(src));
t('Escape is not advertised as Exit', /Escape: exit current menu or sheet/.test(src), false);
t('canonical sheet registry contains Usage Policy', /key:\s*'usage-policy'[\s\S]{0,80}sheet:\s*usagePolicySheet/.test(src));
t('open-sheet sweep consumes canonical registry', /function _openSheet\(target, openerOverride\)[\s\S]{0,250}_allPanelSheets\(\)\.forEach/.test(src));
t('close-button wiring consumes canonical registry', /_allPanelSheets\(\)\.forEach\(function \(s\)[\s\S]{0,220}button\[id\$=/.test(src));

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
