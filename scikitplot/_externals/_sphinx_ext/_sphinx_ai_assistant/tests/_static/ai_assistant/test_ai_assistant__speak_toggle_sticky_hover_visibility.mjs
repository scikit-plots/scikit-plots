// R173T91 + R173T96 - speak icons remain visible through touch/sticky-hover state changes.
import fs from 'node:fs';
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};
const block=(sel)=>{const q=sel.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');const m=css.match(new RegExp('^'+q+'\\s*\\{([^}]*)\\}','m'));return m?m[1]:'';};
const toggleBase=block('.ai-assistant-panel-speak-toggle');
const bannerBase=block('.ai-assistant-panel-speak-banner');
const bannerSvg=block('.ai-assistant-panel-speak-banner svg');
const toggleSvg=block('.ai-assistant-panel-speak-toggle svg');
const expandedSvg=block('.ai-assistant-panel-speak-toggle[aria-expanded="true"] svg');
const interactionMatch=css.match(/\.ai-assistant-panel-speak-toggle:hover,\s*\.ai-assistant-panel-speak-toggle:focus-visible,\s*\.ai-assistant-panel-speak-toggle:active\s*\{([^}]*)\}/m);
const interaction=interactionMatch?interactionMatch[1]:'';

ok(!!toggleBase,'toggle base rule exists');
ok(/--ai-speak-toggle-surface:\s*var\(\s*--pst-color-surface/.test(toggleBase),'toggle surface derives from the actual surface token');
ok(/--ai-speak-toggle-ink:\s*var\(\s*--pst-color-on-surface/.test(toggleBase),'toggle foreground derives from the paired on-surface token');
ok(!/--pst-color-on-background/.test(toggleBase),'toggle does not misuse on-background as a surface');
ok(!!bannerBase && /background-color:\s*var\(\s*--pst-color-surface/.test(bannerBase),'speak banner uses the same surface family');
ok(!/--pst-color-on-background/.test(bannerBase),'speak banner does not misuse on-background as a surface');
ok(!!interaction,'hover/focus/active share one interaction rule');
ok(/color:\s*var\(--ai-speak-toggle-ink\)/.test(interaction),'post-tap interaction keeps the owned contrast-safe foreground');
ok(!/color:\s*inherit/.test(interaction),'sticky hover can never fall back to inherited color');
ok(/--ai-speak-toggle-surface/.test(interaction) && /--ai-speak-toggle-ink/.test(interaction),'interaction background mixes the owned surface/ink pair');
ok(/stroke:\s*var\(--ai-speak-toggle-ink\)/.test(toggleSvg),'chevron stroke is independent of inherited host color');
ok(/stroke:\s*currentColor/.test(bannerSvg),'speak/mic icon still follows its explicit banner foreground');
ok(/color:\s*var\(--pst-color-text-base/.test(bannerSvg),'speak/mic icon uses readable text foreground rather than a theme accent');
ok(!/--pst-color-primary/.test(bannerSvg),'speak/mic icon cannot disappear when host primary equals its surface');
ok(/transform:\s*rotate\(-90deg\)/.test(expandedSvg),'expanded state still rotates the chevron');
ok(!/(?:color|background|opacity|visibility)\s*:/.test(expandedSvg),'expanded state cannot change visibility paint');
ok(!/\.ai-assistant-panel-speak-toggle\[aria-expanded="(?:true|false)"\][^{]*\{[^}]*(?:color|background|opacity|visibility)\s*:/.test(css),'neither aria state owns a competing visibility style');
ok(/@media \(hover: none\), \(pointer: coarse\)[\s\S]*?\.ai-assistant-panel-speak-toggle \{[\s\S]*?color:\s*var\(--ai-speak-toggle-ink\)/.test(css),'touch resting state remains explicitly high contrast');
ok(/:is\(\[data-theme="dark"\], \[data-mode="dark"\], \[data-bs-theme="dark"\], \.dark\)[\s\S]*?\.ai-assistant-panel-speak-toggle:active/.test(css),'dark sticky-hover/active state recognizes PyData and compatibility theme contracts');
ok(/@media \(forced-colors: active\)[\s\S]*?--ai-speak-toggle-ink:\s*ButtonText[\s\S]*?stroke:\s*ButtonText/.test(css),'forced-colors remains system-authoritative through the SVG');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
