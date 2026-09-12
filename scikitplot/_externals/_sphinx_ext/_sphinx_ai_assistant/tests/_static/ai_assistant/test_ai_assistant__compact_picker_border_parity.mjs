// Run 173 T82: a fit-contracted model picker remains one outlined segmented
// control: [model icon + effort] | [quick-model chevron].
// Dependency-free static regression; no browser/jsdom required.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let passed = 0, failed = 0;
const ok = (cond, name) => cond ? passed++ : (failed++, console.error('FAIL ' + name));

function rule(selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = css.match(new RegExp('^' + escaped + ' \\{[^}]*\\}', 'm'));
  return match ? match[0] : '';
}

const compactSelector = '.ai-assistant-panel[data-footer-compact="true"] .ai-assistant-panel-inline-model-picker';
const compact = rule(compactSelector);
const chevron = rule('.ai-assistant-panel-inline-picker-more');
const joined = rule('.ai-assistant-panel-inline-picker-wrapper .ai-assistant-panel-inline-model-picker');

ok(Boolean(compact), 'compact primary picker rule exists');
ok(/border:\s*1px solid var\(\s*--pst-color-border/.test(compact),
   'compact icon+effort half keeps the family border token');
ok(!/border:\s*(?:none|0)\s*;/.test(compact),
   'compact icon+effort half never drops its outer outline');
ok(/border-radius:\s*6px/.test(compact),
   'compact primary half keeps the outer left corner radius');

ok(/border:\s*1px solid var\(\s*--pst-color-border/.test(chevron),
   'chevron half uses the same border token');
ok(/border-inline-start:\s*0/.test(chevron),
   'chevron drops only the shared middle edge');
ok(/\.ai-assistant-panel-inline-picker-wrapper \.ai-assistant-panel-inline-model-picker \{[^}]*border-inline-end:\s*1px solid/.test(css),
   'primary half supplies the single middle separator');
ok(/border-start-end-radius:\s*0/.test(joined) && /border-end-end-radius:\s*0/.test(joined),
   'primary half is square only on the joined edge');
ok(/\.ai-assistant-panel-inline-picker-more::before \{ display:\s*none; \}/.test(css),
   'legacy pseudo-divider stays disabled so the separator is not doubled');

ok(src.includes("pickerWrap.appendChild(inlinePicker);") && src.includes("pickerWrap.appendChild(quickModelBtn);"),
   'primary and chevron remain sibling buttons in one wrapper');
ok(src.includes("panel.setAttribute('data-footer-compact'"),
   'compact representation remains panel-fit owned');
ok(src.includes("_attachEffortChip(btn, 'ai-assistant-panel-inline-picker-effort')"),
   'compact primary half still carries effort state');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
