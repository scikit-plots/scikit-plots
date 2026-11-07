// Run 173 T79 - workspace tabs are not format tabs.
//
// The feedback tablist borrows `.ai-assistant-conv-share-format-switcher` for
// its visual tokens and inherited its LAYOUT with them. R173T41 made that
// switcher an even grid -- right for five interchangeable formats, wrong for
// two or three named sections, which were stretched to equal fractions of the
// row with the active edge drawn across the whole of each cell.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let n = 0, f = 0;
const ok = (c, m) => { c ? n++ : (f++, console.error('FAIL ' + m)); };

// The shared class is kept: the tokens are worth sharing, the layout is not.
ok(src.includes("workspaceTabs.className = 'ai-assistant-conv-share-format-switcher ai-assistant-panel-feedback-workspace-tabs';"),'the tablist still takes the shared visual tokens');

const tabs = (css.match(/^\.ai-assistant-panel-feedback-workspace-tabs \{[^}]*\}/m) || [''])[0];
ok((css.match(/^\.ai-assistant-panel-feedback-workspace-tabs \{/gm) || []).length === 1,'the tablist is defined once, not by two rules read together');
ok(/grid-template-columns:\s*none/.test(tabs),'the inherited grid is switched off');
ok(/display:\s*flex/.test(tabs),'and replaced by a row');
ok(/justify-content:\s*flex-start/.test(tabs),'which starts where a tablist starts');
ok(/flex-wrap:\s*wrap/.test(tabs),'and wraps rather than clipping a section name');

// Content-sized, or the active edge spans an empty fraction of the row.
const btn = (css.match(/\.ai-assistant-panel-feedback-workspace-tabs > \.ai-assistant-conv-share-format-btn \{[^}]*\}/) || [''])[0];
ok(/flex:\s*0 1 auto/.test(btn),'a tab sizes to its label rather than to a share of the row');
ok(/min-width:\s*0/.test(btn),'and can still shrink when the panel is narrow');

// T86 supersedes T79's extra active-edge treatment. The workspace keeps only
// its different layout; all visual states now come from the same canonical
// button rule as Conversation export-format tabs.
ok(!/\.ai-assistant-panel-feedback-workspace-tabs > \.ai-assistant-conv-share-format-btn\[aria-selected="true"\]\s*\{/.test(css),'the workspace does not fork selected-state colourization');
ok(!/(?:font|color|background|border|box-shadow|opacity)\s*:/.test(btn),'the workspace button override remains layout-only');
// Driven from aria, not a class, so the shared selected state and announced
// state cannot disagree.
ok(!/workspace-tabs[^{]*\.is-active/.test(css),'selection is read from aria-selected, not a parallel class');

console.log(`${n} passed, ${f} failed`); if (f) process.exit(1);
