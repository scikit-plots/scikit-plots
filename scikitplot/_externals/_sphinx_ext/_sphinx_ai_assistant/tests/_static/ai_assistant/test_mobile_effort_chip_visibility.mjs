// Run 85 regression: effort is persistent footer state. Mobile/tablet must not
// hide it by viewport, and compact panel-fit mode must preserve a bounded badge.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let pass = 0, fail = 0;
function t(name, got, want=true) {
  if (got === want) pass++;
  else { fail++; console.log(`FAIL ${name}\n  got: ${JSON.stringify(got)}\n want: ${JSON.stringify(want)}`); }
}

// DOM/state contract: chip exists on the inline model control and precedes the
// chevron so compact mode naturally reads [model icon] [effort].
const attach = src.indexOf("_attachEffortChip(btn, 'ai-assistant-panel-inline-picker-effort')");
const chev = src.indexOf("chev.className = 'ai-assistant-panel-inline-picker-chev'", attach);
t('inline model picker owns effort chip', attach >= 0);
t('effort chip precedes chevron', attach >= 0 && chev > attach);
t('effort chip is kept live by shared helper', src.includes("addEventListener('ai-assistant-effort-change'") && src.includes("addEventListener('ai-assistant-model-change'"));
t('model button accessible label includes effort', src.includes("+ ', effort: ' + word"));

// Old viewport authority must not suppress the footer effort badge anymore.
const mobileBlocks = [...css.matchAll(/@media \(max-width: 575px\) \{[\s\S]*?\n\}/g)].map(m => m[0]);
t('575px rules do not hide inline effort badge', !mobileBlocks.some(b => /ai-assistant-panel-inline-picker-effort[\s\S]*?display:\s*none/.test(b)));
t('subbar effort may retain its own viewport policy', mobileBlocks.some(b => /ai-assistant-panel-model-link-effort[\s\S]*?display:\s*none/.test(b)));

// Compact mode preserves state but bounds administrator-defined labels.
const compact = css.match(/\.ai-assistant-panel\[data-footer-compact="true"\] \.ai-assistant-panel-inline-picker-effort\s*\{([\s\S]*?)\}/);
t('compact effort rule exists', Boolean(compact));
const compactCss = compact ? compact[1] : '';
t('compact effort stays visible', /display:\s*inline-block/.test(compactCss));
t('compact effort can shrink', /min-width:\s*0/.test(compactCss) && /flex:\s*1\s+1\s+auto/.test(compactCss));
t('compact effort has bounded width', /max-width:\s*4\.25rem/.test(compactCss));
t('compact effort ellipsis long custom labels', /overflow:\s*hidden/.test(compactCss) && /text-overflow:\s*ellipsis/.test(compactCss) && /white-space:\s*nowrap/.test(compactCss));

// Compact mode still hides verbose model metadata, not the state badge.
t('compact hides model label', /data-footer-compact="true"\][\s\S]*?ai-assistant-panel-inline-picker-label,[\s\S]*?display:\s*none/.test(css));
t('compact exposes model icon', /data-footer-compact="true"\][\s\S]*?ai-assistant-panel-inline-picker-icon\s*\{[\s\S]*?display:\s*inline-flex/.test(css));

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
