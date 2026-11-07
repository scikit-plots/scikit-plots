// R173T96 - speak disclosure keeps a stable paint box and semantic contrast on real mobile hardware.
import fs from 'node:fs';
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};
const rule=(sel)=>{const q=sel.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');const m=css.match(new RegExp('^'+q+'\\s*\\{([^}]*)\\}','m'));return m?m[1]:'';};
const row=rule('.ai-assistant-panel-speak-row');
const toggle=rule('.ai-assistant-panel-speak-toggle');
const svg=rule('.ai-assistant-panel-speak-toggle svg');
const expanded=rule('.ai-assistant-panel-speak-toggle[aria-expanded="true"] svg');

ok(!!row,'speak row exists');
ok(/min-height:\s*2rem/.test(row) && /height:\s*auto/.test(row),'row owns a positive paint box');
ok(/margin:\s*-2rem\s+0\.75rem\s+0/.test(row),'negative flow margin reclaims the row without collapsing its paint box');
ok(!/height:\s*0/.test(row) && !/min-height:\s*0/.test(row),'row never returns to zero-height compositing');
ok(/isolation:\s*isolate/.test(row),'floating row owns an isolated stacking context');
ok(/\.ai-assistant-panel-speak-row\s*>\s*\*\s*\{\s*transform:\s*none/.test(css),'children are not translated out of a zero-height row');
ok(!/\.ai-assistant-panel-speak-row[^\n{]*[\s\S]{0,180}?translateY\(-100%\)/.test(css),'speak-row positioning never depends on -100% transform paint');

ok(/--ai-speak-toggle-surface:\s*var\(\s*--pst-color-surface/.test(toggle),'toggle surface is locally owned');
ok(/--ai-speak-toggle-ink:\s*var\(\s*--pst-color-on-surface/.test(toggle),'toggle ink uses the semantic on-surface token');
ok(/background-color:\s*var\(--ai-speak-toggle-surface\)/.test(toggle),'button paints its owned surface');
ok(/color:\s*var\(--ai-speak-toggle-ink\)/.test(toggle),'button paints its owned ink');
ok(/opacity:\s*1/.test(toggle) && /visibility:\s*visible/.test(toggle),'button cannot remain clickable while hidden by opacity/visibility');

ok(/color:\s*var\(--ai-speak-toggle-ink\)/.test(svg) && /stroke:\s*var\(--ai-speak-toggle-ink\)/.test(svg),'SVG paints directly from the same contrast-safe ink');
ok(/stroke-width:\s*2\.25/.test(svg),'small mobile chevron gets a robust raster stroke');
ok(/opacity:\s*1/.test(svg) && /visibility:\s*visible/.test(svg),'SVG cannot inherit a hidden paint state');
ok(/vector-effect:\s*non-scaling-stroke/.test(css),'chevron stroke survives CSS scaling/rotation');
ok(!/(?:color|background|opacity|visibility)\s*:/.test(expanded),'aria-expanded changes direction only, never paint visibility');

ok(/@media \(hover: none\), \(pointer: coarse\)[\s\S]*?\.ai-assistant-panel-speak-toggle\s*\{[^}]*min-width:\s*2rem[^}]*min-height:\s*2rem[^}]*touch-action:\s*manipulation/.test(css),'touch gets a stable target without hover dependence');
ok(/:is\(\[data-theme="dark"\], \[data-mode="dark"\], \[data-bs-theme="dark"\], \.dark\)\n\s*\.ai-assistant-panel-speak-toggle:hover,[\s\S]*?\.ai-assistant-panel-speak-toggle:active/.test(css),'interaction fallback recognizes the real PyData data-theme/data-mode contract');
ok(/:is\(\[data-theme="dark"\], \[data-mode="dark"\], \[data-bs-theme="dark"\], \.dark\)\n\s*\.ai-assistant-panel-speak-row\[data-collapsed="true"\]/.test(css),'collapsed floating surface recognizes the same host dark contract');
ok(/@media \(forced-colors: active\)[\s\S]*?--ai-speak-toggle-surface:\s*Canvas[\s\S]*?--ai-speak-toggle-ink:\s*ButtonText[\s\S]*?stroke:\s*ButtonText/.test(css),'forced-colors owns both surface and SVG ink');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
