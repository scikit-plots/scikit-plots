// PDF mode toggle layout parity with the working Panel/Copy switches.
// Dependency-free static regression: no browser or jsdom required.
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

const row = rule('.ai-assistant-pdf-row');
const pdf = rule('.ai-assistant-pdf-mode-switch');
const panel = rule('.ai-assistant-panel-mode-switch');
const copy = rule('.ai-assistant-copy-mode-switch');

ok(/grid-template-columns:\s*minmax\(0, 1fr\) auto;/.test(row),
   'PDF row has exactly one action column and one switch column');
ok(!/minmax\(0, 1fr\) auto auto/.test(row),
   'PDF row does not reserve an orphan third grid column');

for (const [name, block] of [['PDF', pdf], ['Panel', panel], ['Copy', copy]]) {
  ok(/width:\s*48px/.test(block) && /min-width:\s*48px/.test(block),
     name + ' switch owns the same 48px gutter');
  ok(/display:\s*flex/.test(block) && /align-items:\s*center/.test(block),
     name + ' switch centers the track explicitly');
  ok(/background:\s*transparent/.test(block) && /cursor:\s*pointer/.test(block),
     name + ' switch has the same neutral interactive surface');
}

ok(!/padding:/.test(pdf), 'PDF switch adds no horizontal padding outside the 48px gutter');
ok(!/min-height:/.test(pdf), 'PDF switch height follows the row instead of forcing stale geometry');

// JS semantics remain independent from the CSS fix: checked means prepared URL.
ok(src.includes("modeSwitch.setAttribute('aria-checked', initialMode === 'url' ? 'true' : 'false')"),
   'checked state still means prepared-PDF URL mode');
ok(src.includes("modeText.textContent = initialDef.label"),
   'visible/accessible state text is initialized from the canonical mode definition');
ok(src.includes("modeText.textContent = def.label"),
   'state text stays synchronized after mode changes');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
