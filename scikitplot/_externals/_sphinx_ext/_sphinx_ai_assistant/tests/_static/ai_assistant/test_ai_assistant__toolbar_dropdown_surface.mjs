// R173T98 - top toolbar dropdown must own a valid opaque menu surface.
import fs from 'node:fs';

const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let n = 0, f = 0;
const ok = (cond, msg) => cond ? n++ : (f++, console.error('FAIL ' + msg));

const base = (css.match(/^\.ai-assistant-dropdown \{[\s\S]*?^\}/m) || [''])[0];
const dark = (css.match(/^:is\(\[data-theme="dark"\], \[data-mode="dark"\], \[data-bs-theme="dark"\], \.dark\)[\s\S]*?^\}/m) || [''])[0];
const bgMatch = base.match(/background-color\s*:\s*([^;]+);/);
const bgValue = bgMatch ? bgMatch[1].replace(/\s+/g, ' ').trim() : '';

ok(!!base, 'toolbar dropdown has a base rule');
ok(/--ai-assistant-dropdown-surface\s*:\s*var\(\s*--pst-color-surface,\s*var\(\s*--color-background-primary,\s*#fff\s*\)\s*\)/.test(base),
   'dropdown owns a light semantic surface with nested fallback');
ok(bgValue === 'var(--ai-assistant-dropdown-surface, #fff)',
   'background-color is exactly one valid dropdown-surface value');
ok(!/--ai-speak-toggle-surface/.test(base),
   'toolbar dropdown never borrows the speak-toggle scoped token');
ok(!/background(?:-color)?\s*:\s*transparent/.test(base),
   'base dropdown cannot regress to a transparent surface');
ok(!!dark, 'real PyData/Bootstrap dark-theme selectors own a dropdown rule');
ok(/--ai-assistant-dropdown-surface\s*:\s*var\(\s*--pst-color-surface,\s*#29313d\s*\)/.test(dark),
   'dark dropdown keeps a single #29313d fallback inside var()');
ok(!/background-color\s*:[^;]*\)\s*#[0-9a-f]{3,8}\s*;/i.test(base),
   'a second color cannot be appended after a var() background value');
ok(/pointer-events\s*:\s*auto/.test(base) && /z-index\s*:\s*1021/.test(base),
   'surface repair preserves dropdown interaction/stacking contract');

console.log(`${n} passed, ${f} failed`);
if (f) process.exit(1);
