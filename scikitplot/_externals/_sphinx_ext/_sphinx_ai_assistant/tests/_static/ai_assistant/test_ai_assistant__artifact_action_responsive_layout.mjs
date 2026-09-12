// Run 173 T87 — managed Share artifacts preserve readable metadata and move
// their actions as a group below it when the actual Share surface is narrow.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let n = 0, f = 0;
const ok = (c, m) => { c ? n++ : (f++, console.error('FAIL ' + m)); };

ok(src.includes("actions.className = 'ai-assistant-conv-share-artifact-actions';"),
   'artifact actions have an explicit layout wrapper');
ok(src.includes('row.appendChild(actions);'),
   'action wrapper is a sibling after the metadata group');
for (const name of ['copyLink', 'open', 'check', 'remove', 'forgetUnavailable']) {
  ok(src.includes(`actions.appendChild(${name});`), `${name} is rendered inside the action group`);
}
const renderArtifacts = src.slice(src.indexOf('function _renderArtifacts()'), src.indexOf('function _renderResult()'));
ok(!/row\.appendChild\((?:copyLink|open|check|remove|forgetUnavailable)\)/.test(renderArtifacts),
   'managed artifact action buttons no longer participate as independent row flex children');

const listRule = (css.match(/\.ai-assistant-conv-share-artifacts\s*\{[^}]*\}/) || [''])[0];
ok(/container-type\s*:\s*inline-size/.test(listRule), 'responsive behavior measures the real artifact-list width');
ok(/container-name\s*:\s*conv-share-artifacts/.test(listRule), 'artifact container query is explicitly scoped');

const textRules = css.match(/\.ai-assistant-conv-share-artifact-text\s*\{[^}]*\}/g) || [];
ok(textRules.length === 1, 'artifact metadata has one flex authority instead of duplicate shorthand rules');
ok(/flex\s*:\s*1 1 12rem/.test(textRules[0] || ''), 'metadata keeps a meaningful wide-row flex basis');
ok(!/flex\s*:\s*1\s*(?:;|$)/.test(textRules[0] || ''), 'metadata basis cannot be silently reset to zero percent');

const narrow = (css.match(/@container\s+conv-share-artifacts\s*\(max-width:\s*30rem\)\s*\{[\s\S]*?\n\}/) || [''])[0];
ok(/flex-direction\s*:\s*column/.test(narrow), 'narrow cards put metadata on its own first row');
ok(/align-items\s*:\s*stretch/.test(narrow), 'narrow card rows expose full content width to the action group');
ok(/\.ai-assistant-conv-share-artifact-actions\s*\{[\s\S]*?width\s*:\s*100%/.test(narrow),
   'narrow action group claims the second row');
ok(/flex-wrap\s*:\s*wrap/.test(narrow), 'three actions can stay together and wrap only when truly necessary');

const tight = (css.match(/@container\s+conv-share-artifacts\s*\(max-width:\s*21rem\)\s*\{[\s\S]*?\n\}/) || [''])[0];
ok(/:has\(> \.ai-assistant-conv-share-action-btn:nth-child\(4\)\)/.test(tight),
   'balanced tight layout is activated only for four-or-more action rows');
ok(/grid-template-columns\s*:\s*repeat\(2,minmax\(0,1fr\)\)/.test(tight),
   'four-or-more actions become a stable two-column grid at tight widths');

ok(!/@media\s*\(max-width:\s*720px\)[\s\S]*?\.ai-assistant-conv-share-artifact\s*\{[^}]*flex-wrap/.test(css),
   'artifact responsiveness no longer guesses from viewport width');

console.log(`${n} passed, ${f} failed`); if (f) process.exit(1);
