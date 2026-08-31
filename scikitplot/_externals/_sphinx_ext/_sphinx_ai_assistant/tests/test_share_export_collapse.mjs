import fs from 'node:fs';
import assert from 'node:assert/strict';

const js = fs.readFileSync(new URL('../_static/ai-assistant.js', import.meta.url), 'utf8');
const css = fs.readFileSync(new URL('../_static/ai-assistant.css', import.meta.url), 'utf8');

let n = 0;
function ok(cond, msg) { assert.ok(cond, msg); n += 1; }

ok(js.includes("bodyInner.className = 'ai-assistant-share-export-body-inner'"), 'single inner wrapper is created');
ok(js.includes('body.appendChild(bodyInner);'), 'body receives the inner wrapper');
ok(js.includes('bodyInner.appendChild(cardsGrid);'), 'cards live inside inner wrapper');
ok(js.includes('bodyInner.appendChild(cardsLive);'), 'live region lives inside inner wrapper');
ok(js.includes('bodyInner.appendChild(modeSep);'), 'separator lives inside inner wrapper');
ok(js.includes('bodyInner.appendChild(modeRow);'), 'mode row lives inside inner wrapper');
ok(!js.includes('body.appendChild(cardsGrid);'), 'cards are not an implicit grid row');
ok(!js.includes('body.appendChild(modeSep);'), 'mode separator is not an implicit grid row');
ok(!js.includes('body.appendChild(modeRow);'), 'mode row is not an implicit grid row');
ok(css.includes('.ai-assistant-share-export-body-inner {'), 'inner wrapper has collapse CSS');
ok(/\.ai-assistant-share-export-body-inner\s*\{[^}]*overflow:\s*hidden;[^}]*min-height:\s*0;/s.test(css), 'inner wrapper can collapse to zero');
ok(/\.ai-assistant-share-export-body\s*\{[^}]*grid-template-rows:\s*0fr;/s.test(css), 'closed body uses zero grid track');
ok(/\.ai-assistant-share-export-body\[data-open="true"\]\s*\{[^}]*grid-template-rows:\s*1fr;/s.test(css), 'open body uses expanded grid track');
ok(js.includes("body.setAttribute('data-open', willOpen ? 'true' : 'false');"), 'toggle still drives data-open');
ok(js.includes("triggerBtn.setAttribute('aria-expanded', willOpen ? 'true' : 'false');"), 'ARIA expanded state stays synchronized');
ok(js.includes("triggerBtn.setAttribute('aria-controls', 'ai-assistant-share-export-body');"), 'trigger points to collapsible body');
ok(js.includes("body.setAttribute('aria-hidden', willOpen ? 'false' : 'true');"), 'collapsed state is hidden from accessibility tree');
ok(js.includes("body.removeAttribute('inert');"), 'expanded body becomes interactive');
ok(js.includes("body.setAttribute('inert', '');"), 'collapsed body leaves tab order');

console.log(`${n} passed, 0 failed`);
