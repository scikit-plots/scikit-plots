// R173T85 — PDF/Panel/Copy checked-thumb geometry parity.
// Dependency-free static regression: no browser or jsdom required.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
const cssRaw = fs.readFileSync(process.argv[3], 'utf8');
const css = cssRaw.replace(/\/\*[\s\S]*?\*\//g, '');
let passed = 0, failed = 0;
const ok = (cond, name) => cond ? passed++ : (failed++, console.error('FAIL ' + name));

function rule(selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = css.match(new RegExp('^' + escaped + ' \\{[^}]*\\}', 'm'));
  return match ? match[0] : '';
}

const pdfSwitch = rule('.ai-assistant-pdf-mode-switch');
const copySwitch = rule('.ai-assistant-copy-mode-switch');
const panelSwitch = rule('.ai-assistant-panel-mode-switch');
const genericChecked = rule('.ai-assistant-mic-popup-toggle[aria-checked="true"] .ai-assistant-mic-toggle-thumb');
const pdfThumb = rule('.ai-assistant-pdf-mode-switch .ai-assistant-pdf-toggle-thumb');
const panelThumb = rule('.ai-assistant-panel-mode-switch .ai-assistant-panel-toggle-thumb');
const pdfTrack = rule('.ai-assistant-pdf-mode-switch .ai-assistant-pdf-toggle-track');
const panelTrack = rule('.ai-assistant-panel-mode-switch .ai-assistant-panel-toggle-track');

ok(/--ai-assistant-toggle-thumb-travel:\s*16px/.test(pdfSwitch), 'PDF uses the canonical large-toggle thumb travel');
ok(/--ai-assistant-toggle-thumb-travel:\s*16px/.test(copySwitch), 'Copy uses the canonical large-toggle thumb travel');
ok(/--ai-assistant-toggle-thumb-travel:\s*16px/.test(panelSwitch), 'Panel reference uses the canonical large-toggle thumb travel');
ok(/translateX\(var\(--ai-assistant-toggle-thumb-travel,\s*12px\)\)/.test(genericChecked), 'shared checked rule consumes the per-control travel token with 12px Mic fallback');
ok(!/translateX\(17px\)/.test(cssRaw), 'no stale PDF-only 17px optical exception remains');
ok(!/\.ai-assistant-(?:pdf|copy|panel)-mode-switch\[aria-checked="true"\][^{]*toggle-thumb\s*\{[^}]*transform:/s.test(css), 'large toggles do not compete with the shared checked transform authority');
ok(/top:\s*1px/.test(pdfThumb) && /left:\s*1px/.test(pdfThumb) && /width:\s*14px/.test(pdfThumb) && /height:\s*14px/.test(pdfThumb), 'PDF thumb base geometry is canonical');
ok(/top:\s*1px/.test(panelThumb) && /left:\s*1px/.test(panelThumb) && /width:\s*14px/.test(panelThumb) && /height:\s*14px/.test(panelThumb), 'Panel reference thumb base geometry is canonical');
ok(/width:\s*34px/.test(pdfTrack) && /height:\s*18px/.test(pdfTrack) && /width:\s*34px/.test(panelTrack) && /height:\s*18px/.test(panelTrack), 'PDF and Panel track geometry remains identical');
ok(src.includes("modeSwitch.setAttribute('aria-checked', normalized === 'url' ? 'true' : 'false')") || src.includes("modeSwitch.setAttribute('aria-checked', mode === 'url' ? 'true' : 'false')"), 'aria-checked=true still means prepared-PDF URL mode');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
