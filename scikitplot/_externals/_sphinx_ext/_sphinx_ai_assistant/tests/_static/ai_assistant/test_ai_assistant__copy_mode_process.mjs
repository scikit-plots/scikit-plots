import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let d = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { d++; started = true; }
    else if (src[j] === '}') { d--; if (started && d === 0) return src.slice(i, j + 1); }
  }
  throw new Error('unbalanced ' + name);
}
const F = {};
for (const n of ['_copyModeProcess','_copyModeDescription','_copySwitchAccessibleLabel'])
  F[n] = (0, eval)('(' + extract(n) + ')');
globalThis._copyModeProcess = F._copyModeProcess;
const { _copyModeProcess, _copyModeDescription, _copySwitchAccessibleLabel } = F;

let pass=0, fail=0;
const t=(n,g,w)=>{ if(g===w){pass++;} else {fail++;console.log(`  FAIL ${n}\n    got  ${JSON.stringify(g)}\n    want ${JSON.stringify(w)}`);} };
const ok=(n,c)=>{ if(c){pass++;} else {fail++;console.log(`  FAIL ${n}`);} };

t('browser label', _copyModeProcess('browser').label, 'Rendered page');
t('static label',  _copyModeProcess('static').label,  'Published file');
ok('descriptions differ', _copyModeDescription('browser') !== _copyModeDescription('static'));
ok('browser names the browser', /browser/i.test(_copyModeDescription('browser')));
ok('browser says it always works', /always available|no download/i.test(_copyModeDescription('browser')));
ok('static names the file', /page\.md/.test(_copyModeDescription('static')));
ok('static names the consumers', /View/.test(_copyModeDescription('static')) && /Ask AI/.test(_copyModeDescription('static')));
ok('static claims byte equality', /byte-for-byte/i.test(_copyModeDescription('static')));
ok('labels differ', _copySwitchAccessibleLabel('browser') !== _copySwitchAccessibleLabel('static'));
ok('a11y label states the action', /Activate/.test(_copySwitchAccessibleLabel('browser')));
ok('a11y label names both processes', /fetch/i.test(_copySwitchAccessibleLabel('browser')) && /convert/i.test(_copySwitchAccessibleLabel('browser')));
ok('unknown mode falls back to browser', _copyModeProcess('nonsense').label === 'Rendered page');

// the selector bug that started this
ok('setCopyMode targets the real class',
   /querySelector\('\.ai-assistant-menu-item-description'\)/.test(src));
ok('no stale -desc selector', !/querySelector\('\.ai-assistant-menu-item-desc'\)/.test(src));
ok('switch label comes from one source',
   (src.match(/'Published file'/g) || []).length === 1);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail?1:0);
