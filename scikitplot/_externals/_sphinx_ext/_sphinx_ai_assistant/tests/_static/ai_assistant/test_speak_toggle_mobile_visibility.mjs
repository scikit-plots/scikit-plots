// R173T89 + R173T96 - speak-hint chevron stays visible on touch/mobile.
import fs from 'node:fs';
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};
const base = (css.match(/^\.ai-assistant-panel-speak-toggle \{[^}]*\}/m) || [''])[0];
const svg = (css.match(/^\.ai-assistant-panel-speak-toggle svg \{[^}]*\}/m) || [''])[0];

ok(!!base, 'speak toggle has one base rule');
ok(/--ai-speak-toggle-surface:\s*var\([\s\S]*?--pst-color-surface/.test(base), 'resting toggle owns a semantic surface');
ok(/background-color:\s*var\(--ai-speak-toggle-surface\)/.test(base), 'resting toggle paints that surface');
ok(!/background\s*:\s*transparent/.test(base), 'background shorthand cannot erase that surface');
ok(/-webkit-appearance:\s*none/.test(base) && /appearance:\s*none/.test(base), 'native mobile button chrome cannot recolor the icon');
ok(/--ai-speak-toggle-ink:\s*var\([\s\S]*?--pst-color-on-surface/.test(base), 'resting ink is paired to its surface');
ok(/color:\s*var\(--ai-speak-toggle-ink\)/.test(base), 'resting button uses its owned ink');
ok(!!svg, 'speak chevron has an explicit svg rule');
ok(/display:\s*block/.test(svg), 'svg is not baseline-rendered like inline text');
ok(/color:\s*var\(--ai-speak-toggle-ink\)/.test(svg) && /stroke:\s*var\(--ai-speak-toggle-ink\)/.test(svg), 'svg stroke is painted explicitly, not indirectly through host inheritance');
ok(/@media \(hover: none\), \(pointer: coarse\) \{\s*\.ai-assistant-panel-speak-toggle \{\s*color:\s*var\(--ai-speak-toggle-ink\)/.test(css), 'touch keeps explicit high-contrast ink without hover');
ok(/@media \(hover: none\), \(pointer: coarse\)[\s\S]*?\.ai-assistant-panel-speak-toggle svg \{[^}]*width:\s*1rem[^}]*height:\s*1rem/.test(css), 'touch chevron is enlarged to one rem');
ok(/\[data-theme="dark"\]/.test(css) && /\[data-mode="dark"\]/.test(css), 'real PyData dark-theme attributes are recognized');
ok(/\.ai-assistant-panel-speak-toggle\[aria-expanded="true"\] svg \{ transform:\s*rotate\(-90deg\); \}/.test(css), 'visibility fix preserves state-driven rotation');
ok(/@media \(forced-colors: active\)[\s\S]*?--ai-speak-toggle-ink:\s*ButtonText/.test(css), 'forced-colors keeps system text authority');
ok(/opacity:\s*1/.test(base) && /visibility:\s*visible/.test(base), 'resting toggle cannot be present-but-unpainted');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
