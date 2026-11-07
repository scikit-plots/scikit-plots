// Behavioural harness for the Copy mode logic, exercised outside a browser.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');

// Extract the pure pieces under test rather than booting the whole IIFE.
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { depth++; started = true; }
    else if (src[j] === '}') { depth--; if (started && depth === 0) return src.slice(i, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

let store = {};
globalThis.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
};
let CFG = {};
globalThis._cfg = () => CFG;
const _COPY_MODES = ['browser', 'static'];
globalThis._COPY_MODES = _COPY_MODES;
globalThis._COPY_MODE_KEY = 'ai-assistant-copy-mode';
const F = {};
for (const n of ['_copyModeProcess','_copyMode','_setCopyMode','_copyModeDescription','_copySwitchAccessibleLabel'])
  F[n] = (0, eval)('(' + extract(n) + ')');
globalThis._copyModeProcess = F._copyModeProcess;
const { _copyMode, _setCopyMode, _copyModeDescription, _copySwitchAccessibleLabel } = F;

let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) { pass++; }
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
};

// default
store = {}; CFG = {};
t('default is browser', _copyMode(), 'browser');

// configured default honoured
store = {}; CFG = { copyMode: 'static' };
t('configured static', _copyMode(), 'static');

// invalid configured value falls back
store = {}; CFG = { copyMode: 'Static' };
t('invalid config -> browser', _copyMode(), 'browser');

// reader preference wins over config
store = {}; CFG = { copyMode: 'browser' };
_setCopyMode('static');
t('stored pref wins', _copyMode(), 'static');

// invalid stored value ignored
store = { 'ai-assistant-copy-mode': 'nonsense' }; CFG = { copyMode: 'browser' };
t('invalid stored ignored', _copyMode(), 'browser');

// toggle disabled: config wins outright even with a stored pref
store = { 'ai-assistant-copy-mode': 'static' }; CFG = { copyMode: 'browser', copyModeToggle: false };
t('toggle off pins config', _copyMode(), 'browser');

// _setCopyMode rejects junk
store = {}; CFG = {};
_setCopyMode('nope');
t('setter rejects junk', store['ai-assistant-copy-mode'], undefined);

// storage failure is non-fatal
globalThis.localStorage = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } };
CFG = { copyMode: 'static' };
t('storage failure -> config', _copyMode(), 'static');
_setCopyMode('browser');
t('setter survives storage failure', true, true);

// labels differ and name both sides
globalThis.localStorage = { getItem: () => null, setItem: () => {} };
t('browser desc names the process', /browser/i.test(_copyModeDescription('browser')), true);
t('static desc names the file', /page\.md/.test(_copyModeDescription('static')), true);
t('labels differ', _copySwitchAccessibleLabel('browser') === _copySwitchAccessibleLabel('static'), false);
t('label states action', /Activate/.test(_copySwitchAccessibleLabel('browser')), true);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
