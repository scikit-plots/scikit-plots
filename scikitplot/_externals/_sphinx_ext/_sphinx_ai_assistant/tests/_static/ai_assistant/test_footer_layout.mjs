// Run 173 T50 - the footer spends one row, not two.
//
// The disclaimer and the credit both carried `width: 100%`, so each claimed a
// full line of a wrapping flex row: two rows of small print at the bottom of a
// panel whose vertical space is the scarce dimension.
//
// The disclaimer is also the one piece of text in the panel a reader most
// needs to have seen, so this harness asserts what it must NOT become --
// truncated, or revealed on hover -- as firmly as it asserts the layout.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let n = 0, f = 0;
const ok = (c, m) => { c ? n++ : (f++, console.error('FAIL ' + m)); };

const noteRules = css.match(/\.ai-assistant-panel-footer-note \{[^}]*\}/g) || [];
const creditRules = css.match(/\.ai-assistant-panel-footer-credit \{[^}]*\}/g) || [];
const note = noteRules.join('\n');
const credit = creditRules.join('\n');

// ── One row ──────────────────────────────────────────────────────────────
ok(/width:\s*auto/.test(note) && /width:\s*auto/.test(credit),'neither element claims a full line by default');
ok(/flex:\s*1 1 10rem/.test(note),'the note takes the room left over');
ok(/flex:\s*0 1 auto/.test(credit),'the credit takes only what it needs');
ok(/margin-inline-start:\s*auto/.test(credit),'the credit sits at the trailing edge');
ok(/min-width:\s*0/.test(note),'the note can shrink rather than forcing a wrap');

// ── The disclaimer is never abbreviated ──────────────────────────────────
// Hover reveals nothing on a touch screen, and clipping a correctness notice
// to one line makes it look tidier and mean less.
ok(!/text-overflow:\s*ellipsis/.test(note),'the note is never truncated');
ok(!/white-space:\s*nowrap/.test(note),'it wraps rather than running off the edge');
// Three regimes, two of them side by side: wide (one line each), medium (both
// wrapped into columns), tiny (stacked). The stack is the last resort.
ok(/white-space:\s*normal/.test(credit),'the credit wraps too, so the pair survives a medium panel');
ok(!/\.ai-assistant-panel-footer-credit \{[^}]*white-space:\s*nowrap/.test(css),'it is no longer an unbreakable block deciding the layout for both');
ok(/\.ai-assistant-panel-footer-credit-link \{ white-space: nowrap; \}/.test(css),'but the project name never breaks across lines');
ok(/@container ai-panel-footer \(max-width: 20rem\)/.test(css),'stacking waits until two columns are genuinely too narrow');
ok(!/@container ai-panel-footer \(max-width: 26rem\)/.test(css),'the earlier threshold is gone, not merely overridden');
ok(/white-space:\s*normal/.test(note),'wrapping is stated, not left to inheritance');
ok(!/\.ai-assistant-panel-footer-note[^{]*:hover[^{]*\{[^}]*(display|visibility|opacity)/.test(css),'it is not revealed on hover');
ok(!/\.ai-assistant-panel-footer-note \{[^}]*display:\s*none/.test(css),'and never hidden outright');

// ── Measured on the panel, not the window ────────────────────────────────
// The panel is resizable and can be docked or maximized, so its width and the
// viewport's are different numbers.
ok(/\.ai-assistant-panel-footer \{[^}]*container-type:\s*inline-size/.test(css),'the footer is a query container');
ok(/container-name:\s*ai-panel-footer/.test(css),'named, so nested descendants cannot mis-fire');
const stack = (css.match(/@container ai-panel-footer \(max-width: 20rem\) \{[\s\S]*?\n\}/) || [''])[0];
ok(stack.length > 0,'there is a stacked fallback for narrow panels');
ok(/flex-basis:\s*100%/.test(stack),'where the two stack rather than clipping');
ok(/text-align:\s*center/.test(stack),'and re-centre, since neither shares the line any more');
ok(!/@media[^{]*\)\s*\{[^{}]*ai-assistant-panel-footer-note/.test(css),'no viewport media query decides the footer layout');

// ── The composer owns the first row ──────────────────────────────────────
//
// It only had that row because the note and credit each carried width:100% and
// were pushed off it. Making those two shrinkable (R173T50) let all three fit
// on one line on a wide panel, putting the composer beside its own small print.
const groupRule = (css.match(/\.ai-assistant-panel-input-group \{[^}]*\}/) || [''])[0];
ok(/flex:\s*1 1 100%/.test(groupRule),'the composer claims the whole first line');
ok(!/flex:\s*1 1 auto/.test(groupRule),'its row no longer depends on what follows it being unshrinkable');
ok(/min-width:\s*0/.test(groupRule),'it can still shrink internally, so the attachment strip scrolls');
// And the two footnotes still share the row below rather than taking one each.
ok(/flex:\s*1 1 10rem/.test(note) && /flex:\s*0 1 auto/.test(credit),'the note and credit still share the second row');

console.log(`${n} passed, ${f} failed`); if (f) process.exit(1);
