import fs from 'node:fs';
import path from 'node:path';

const jsPath = process.argv[2];
const src = fs.readFileSync(jsPath, 'utf8');
const css = fs.readFileSync(path.join(path.dirname(jsPath), 'ai-assistant.css'), 'utf8');
let pass = 0, fail = 0;
const ok = (cond, name) => {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
};

ok(/function _makeEpDisclosureToggle\(/.test(src), 'endpoint launchers share one builder');
ok(/className = 'ai-assistant-panel-ep-add-toggle'/.test(src), 'compatibility class is retained');
ok(/'Add custom profile', ICONS\.plus/.test(src), 'add profile uses plus icon');
ok(/'Import profiles from JSON', ICONS\.upload/.test(src), 'import uses upload icon');
ok(/'conf\.py helper', ICONS\.exportHtml/.test(src), 'conf helper uses code icon');
ok(/ai-assistant-panel-ep-add-profile-form/.test(src), 'add form is aria-addressable');
ok(/ai-assistant-panel-ep-import-profiles-form/.test(src), 'import form is aria-addressable');
ok(/_setEpDisclosureState\(addToggleBtn, !isOpen\)/.test(src), 'add disclosure state is synchronized');
ok(/_setEpDisclosureState\(importToggle, !isOpen\)/.test(src), 'import disclosure state is synchronized');
ok(/_setEpDisclosureState\(snippetToggle, !isOpen\)/.test(src), 'conf helper disclosure state is synchronized');
ok(!/textContent = '\+ Add custom profile'/.test(src), 'add profile no longer swaps raw text labels');
ok(!/textContent = '↑ Import profiles from JSON'/.test(src), 'import no longer uses a text glyph as its icon');
ok(!/textContent = '\{ \} conf\.py helper'/.test(src), 'conf helper no longer embeds a text glyph icon');

ok(/\.ai-assistant-panel-ep-add-toggle\s*\{[\s\S]*border:\s*1px dashed/.test(css), 'launcher uses dashed Add-model-style edge');
ok(/\.ai-assistant-panel-ep-add-toggle-icon/.test(css), 'launcher has dedicated icon sizing');
ok(/\.ai-assistant-panel-ep-add-toggle-chevron/.test(css), 'launcher has disclosure chevron');
ok(/\[aria-expanded="true"\][\s\S]*ai-assistant-panel-ep-add-toggle-chevron/.test(css), 'expanded state rotates disclosure chevron');
ok(/@media \(max-width: 500px\)[\s\S]*\.ai-assistant-panel-ep-add-toggle\s*\{[\s\S]*width:\s*100%/.test(css), 'narrow panels use full-width launchers');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
