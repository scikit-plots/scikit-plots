// Contract harness for per-model reasoning declarations and runtime overrides.
//
//   node tests/test_model_overrides.mjs _static/ai-assistant.js
//
// Two features, one purpose: correcting a model's configuration from the
// browser, without editing conf.py and rebuilding the whole documentation set
// to find out whether a value was right.
//
// The security angle is not incidental. Both a reasoning declaration and an
// override arrive from a text field or from localStorage, and what they
// influence is the endpoint a request goes to and the shape of its body. They
// get the same discipline as the capability-discovery document.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');

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

// ── fakes for the store's scope ────────────────────────────────────────────
globalThis._trim = (v) => (typeof v === 'string' ? v.trim() : '');
globalThis._MAX_LABEL = 100;
globalThis._MAX_DESC = 500;
globalThis._MAX_SIZE = 20;
globalThis._MAX_URL = 2048;
globalThis._SCHEMA_VER = 1;
globalThis._SAFE_ID_RE = /^[a-zA-Z][a-zA-Z0-9_.-]{0,63}$/;
globalThis._EFFORT_LEVELS = [
  { id: 'low' }, { id: 'medium' }, { id: 'high' },
  { id: 'extra' }, { id: 'max' },
];
globalThis._ALLOWED_PROVIDERS = [
  'openai', 'anthropic', 'huggingface', 'mistral', 'groq',
  'cerebras', 'togetherai', 'deepseek', 'custom',
];

let store = {};
globalThis.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
};

globalThis._sanitizeReasoning = (0, eval)('(' + extract('_sanitizeReasoning') + ')');
const _sanitizeReasoning = globalThis._sanitizeReasoning;
globalThis._sanitizeCustomFields = (0, eval)('(' + extract('_sanitizeCustomFields') + ')');
globalThis._sanitizeModel = (0, eval)('(' + extract('_sanitizeModel') + ')');
const _sanitizeModel = globalThis._sanitizeModel;

globalThis._OVERRIDE_KEY = 'ai-assistant-model-overrides';
{
  const m = src.match(/var _OVERRIDABLE = \[([\s\S]*?)\];/);
  globalThis._OVERRIDABLE = (0, eval)('([' + m[1] + '])');
}
globalThis._overrides = Object.create(null);
for (const name of [
  '_sanitizePatch', '_persistOverrides', 'setOverride',
  'clearOverride', 'getOverride', 'listOverrides', 'applyOverrides',
]) {
  globalThis[name] = (0, eval)('(' + extract(name) + ')');
}
const { setOverride, clearOverride, getOverride, applyOverrides } = globalThis;

let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) { pass++; }
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
};
const reset = () => { store = {}; globalThis._overrides = Object.create(null); };

// ── per-model reasoning declaration ────────────────────────────────────────
{
  t('true is kept', _sanitizeReasoning(true), true);
  t('false is kept', _sanitizeReasoning(false), false);
  // Inherit is the answer that changes nothing, so it is the fallback for
  // everything unrecognised: a malformed stored value must not make a model
  // unselectable.
  for (const bad of [undefined, null, 0, 1, 'true', 'yes', [], NaN]) {
    t('inherits on: ' + JSON.stringify(bad), _sanitizeReasoning(bad), undefined);
  }

  const full = _sanitizeReasoning({
    effortParam: 'reasoning_effort',
    effortValues: { low: 'low', medium: 'medium', high: 'high', extra: 'high', max: 'high' },
    thinkingParam: 'thinking',
    budgetMin: 1024, budgetMax: 8192,
  });
  t('dict declaration is accepted', !!full, true);
  t('effort param kept', full.effortParam, 'reasoning_effort');
  t('thinking param kept', full.thinkingParam, 'thinking');
  t('budget range kept', full.budgetMin + '-' + full.budgetMax, '1024-8192');

  // Field-name injection: a declaration may INTRODUCE a wire field, never
  // override one that decides what is sent or to whom.
  for (const evil of ['messages', 'model', 'system', 'max_tokens', 'endpoint',
                      'api_key', '__proto__', 'constructor']) {
    t('reserved effort param refused: ' + evil,
      _sanitizeReasoning({ effortParam: evil, thinkingParam: null }), undefined);
    const withThinking = _sanitizeReasoning({ thinkingParam: evil });
    t('reserved thinking param refused: ' + evil, withThinking, undefined);
  }
  for (const bad of ['Reasoning', 'has space', '1lead', 'has-dash', '']) {
    t('malformed param refused: ' + JSON.stringify(bad),
      _sanitizeReasoning({ thinkingParam: bad }), undefined);
  }

  // A partial effort map would send nothing for unmapped levels.
  t('partial effort map is dropped',
    _sanitizeReasoning({ effortParam: 'x', effortValues: { low: 'l' } }), undefined);
  t('effort param without a map is dropped',
    _sanitizeReasoning({ effortParam: 'x' }), undefined);

  // Budgets clamped into the panel's own bounds.
  const wide = _sanitizeReasoning({ thinkingParam: 'thinking', budgetMin: 1, budgetMax: 999999 });
  t('budget min floored', wide.budgetMin, 500);
  t('budget max capped', wide.budgetMax, 16000);
  const junk = _sanitizeReasoning({ thinkingParam: 'thinking', budgetMin: 'abc' });
  t('non-numeric budget falls back', junk.budgetMin, 500);

  t('prototype is not polluted by a declaration',
    (() => {
      _sanitizeReasoning(JSON.parse('{"thinkingParam":"t","__proto__":{"x":1}}'));
      return ({}).x;
    })(), undefined);

  // `false` forces one field off while the other still falls through to the
  // provider-shape default -- this is what lets effort and thinking be
  // declared independently instead of as one combined on/off.
  const effortOnly = _sanitizeReasoning({ thinkingParam: false });
  t('thinking forced off is kept as false', effortOnly.thinkingParam, false);
  t('effort key left absent so it inherits the shape default',
    Object.prototype.hasOwnProperty.call(effortOnly, 'effortParam'), false);

  const thinkingOnly = _sanitizeReasoning({ effortParam: false });
  t('effort forced off is kept as false', thinkingOnly.effortParam, false);
  t('thinking key left absent so it inherits the shape default',
    Object.prototype.hasOwnProperty.call(thinkingOnly, 'thinkingParam'), false);
}

// ── the model sanitiser carries it ─────────────────────────────────────────
{
  const withDecl = _sanitizeModel('m1', { model: 'a/b', reasoning: true });
  t('declaration survives sanitisation', withDecl.reasoning, true);

  // Absent stays absent, so _reasoningSupport falls through to the build-wide
  // default instead of seeing an explicit undefined it must special-case.
  const without = _sanitizeModel('m2', { model: 'a/b' });
  t('absent stays absent', 'reasoning' in without, false);
  const rejected = _sanitizeModel('m3', { model: 'a/b', reasoning: 'yes' });
  t('rejected declaration fails closed', rejected.reasoning, false);
  const off = _sanitizeModel('m4', { model: 'a/b', reasoning: false });
  t('explicit false is preserved', off.reasoning, false);

  const meta = _sanitizeModel('m5', {
    model: 'a/b',
    custom_fields: [
      { label: 'Context window', value: '128K', display: 'badge' },
      { key: '__proto__', label: 'Region', value: 'EU', display: 'detail' },
    ],
  });
  t('custom metadata survives sanitisation', meta.custom_fields.length, 2);
  t('metadata keys are safe/generated', meta.custom_fields[0].key, 'context_window');
  t('bad metadata key is regenerated', meta.custom_fields[1].key, 'region');
  t('metadata display is bounded', meta.custom_fields[0].display, 'badge');
}

// ── overrides: a diff, never a replacement ─────────────────────────────────
{
  reset();
  const builtin = {
    id: 'prod', label: 'Production', provider: 'openai',
    model: 'gpt-4', endpoint: 'https://old.example.org/v1/chat/completions',
    description: 'The main model.',
  };

  t('no override leaves the entry untouched',
    applyOverrides([builtin])[0], builtin);

  const res = setOverride('prod', {
    endpoint: 'https://new.example.org/v1/chat/completions',
  });
  t('override accepted', res.ok, true);

  const merged = applyOverrides([builtin])[0];
  t('the corrected field wins', merged.endpoint,
    'https://new.example.org/v1/chat/completions');
  // Only what was changed is stored, so a later conf.py edit still lands on
  // every field the reader did not touch.
  t('untouched fields keep the build value', merged.model, 'gpt-4');
  t('untouched label keeps the build value', merged.label, 'Production');
  t('only the changed key is stored',
    Object.keys(getOverride('prod')).join(','), 'endpoint');
  t('the merge is marked', merged._overridden, true);

  // The build-time entry is never mutated, so the diff can be shown against it.
  const metaOverride = setOverride('prod', {
    endpoint: 'https://new.example.org/v1/chat/completions',
    custom_fields: [{ label: 'Context window', value: '128K', display: 'badge' }],
  });
  t('custom metadata override is accepted', metaOverride.ok, true);
  t('custom metadata override is persisted as UI metadata',
    getOverride('prod').custom_fields[0].value, '128K');

  t('the source object is not mutated', builtin.endpoint,
    'https://old.example.org/v1/chat/completions');

  // A later build changing an untouched field must reach the reader.
  const rebuilt = Object.assign({}, builtin, { model: 'gpt-5' });
  t('a later build lands on untouched fields',
    applyOverrides([rebuilt])[0].model, 'gpt-5');
  t('and the override still applies',
    applyOverrides([rebuilt])[0].endpoint,
    'https://new.example.org/v1/chat/completions');

  // Reset restores the build-time definition exactly.
  t('clear reports removal', clearOverride('prod'), true);
  t('clear is idempotent', clearOverride('prod'), false);
  t('build value is restored', applyOverrides([builtin])[0].endpoint,
    'https://old.example.org/v1/chat/completions');
  t('and the marker is gone', applyOverrides([builtin])[0]._overridden, undefined);
}

// ── overrides are validated, not trusted ───────────────────────────────────
{
  reset();
  const builtin = { id: 'prod', label: 'Prod', provider: 'openai', model: 'm' };

  // id is not overridable: a patch must not be able to impersonate another
  // entry, which would silently redirect a selection.
  setOverride('prod', { id: 'other', label: 'Renamed' });
  t('id cannot be overridden', getOverride('prod').id, undefined);
  t('but a legitimate field in the same patch still applies',
    getOverride('prod').label, 'Renamed');

  // A non-http endpoint is refused by the shared sanitiser.
  reset();
  t('javascript: endpoint refused',
    setOverride('prod', { endpoint: 'javascript:alert(1)' }).ok, false);
  reset();
  t('non-http info_url refused',
    setOverride('prod', { info_url: 'data:text/html,x' }).ok, false);

  // An unknown provider falls back rather than being stored verbatim.
  reset();
  setOverride('prod', { provider: 'not-a-provider' });
  t('unknown provider normalised', getOverride('prod').provider, 'custom');

  // Fields outside the overridable set are dropped entirely.
  reset();
  setOverride('prod', { label: 'ok', tags: ['x'], _isCustom: false, group: 'g' });
  t('non-overridable keys dropped',
    Object.keys(getOverride('prod')).sort().join(','), 'label');

  // A reasoning declaration can be corrected at runtime -- the point of the
  // whole feature for this field: discovering an endpoint does not accept
  // reasoning parameters should not need a documentation rebuild.
  reset();
  t('reasoning can be overridden off',
    setOverride('prod', { reasoning: false }).ok, true);
  t('and it merges', applyOverrides([builtin])[0].reasoning, false);

  reset();
  t('malformed custom metadata cannot erase compiled fields',
    setOverride('prod', { custom_fields: 'not-an-array' }).ok, false);
  t('explicit empty custom metadata list is a valid clear',
    setOverride('prod', { custom_fields: [] }).ok, true);
  t('empty metadata clear is preserved',
    Array.isArray(getOverride('prod').custom_fields) && getOverride('prod').custom_fields.length, 0);

  reset();
  t('an empty patch is refused', setOverride('prod', {}).ok, false);
  t('an all-invalid patch is refused',
    setOverride('prod', { endpoint: 'ftp://x' }).ok, false);
  t('a bad id is refused', setOverride('../etc', { label: 'x' }).ok, false);
  t('a non-string id is refused', setOverride(null, { label: 'x' }).ok, false);

  // Built-in ids embed version numbers (e.g. "Qwen2.5-Coder-7B-Instruct-hf")
  // and must remain editable -- a dot is not a path separator.
  reset();
  t('a dotted built-in id is accepted',
    setOverride('Qwen2.5-Coder-7B-Instruct-hf', { label: 'x' }).ok, true);
  t('an id with a path separator is still refused',
    setOverride('a/b.5', { label: 'x' }).ok, false);
}

// ── persistence and corruption ─────────────────────────────────────────────
{
  reset();
  setOverride('prod', { label: 'Renamed' });
  t('the diff is persisted', store[globalThis._OVERRIDE_KEY].includes('Renamed'), true);
  t('persisted under a versioned envelope',
    JSON.parse(store[globalThis._OVERRIDE_KEY])._v, 1);

  // Storage failure must not break model selection.
  reset();
  globalThis.localStorage = {
    getItem() { throw new Error('denied'); },
    setItem() { throw new Error('denied'); },
  };
  t('a write failure does not throw', setOverride('prod', { label: 'x' }).ok, true);
  globalThis.localStorage = {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
  };
}

// ── every read point merges ────────────────────────────────────────────────
{
  // If the request path used the un-overridden entry it would send to the
  // endpoint the reader just corrected away from -- the exact bug the feature
  // exists to fix, reintroduced one layer down.
  t('the active model merges overrides',
    /var models = _MODEL_STORE\.applyOverrides\(builtins\)\.filter/.test(src), true);
  t('the active model includes reader-added custom models',
    /concat\(_MODEL_STORE\.applyOverrides\(_MODEL_STORE\.listCustom\(\)\)\)/.test(src), true);
  t('the active model excludes locally removed compiled models',
    /!_MODEL_STORE\.isHiddenBuiltin\(m\.id\)/.test(src), true);
  t('the sheet list merges overrides',
    /var allModels = _MODEL_STORE\.applyOverrides\(models\)/.test(src), true);
  t('no read point uses the raw config list',
    /var id = _getActiveModelId\(cfg\.panelApiModels\)/.test(src), false);
  t('the merge helper has one definition',
    (src.match(/function applyOverrides\(/g) || []).length, 1);
}

// ── the editor UI ──────────────────────────────────────────────────────────
{
  const section = extract('_appendModelCustomSection');
  const row = extract('_buildModelRowV2');

  // The custom-model form describes MODEL capabilities only. Provider/proxy
  // field names and payload shapes live in the dedicated Effort/Thinking
  // editors, so model identity does not become a transport configuration form.
  t('effort and thinking are boolean capability selects',
    /function _boolCapabilitySelect\(\)/.test(section) &&
    /var effortSel\s*=\s*_boolCapabilitySelect\(\);/.test(section) &&
    /var thinkingSel\s*=\s*_boolCapabilitySelect\(\);/.test(section), true);
  t('capability selects expose only supported/not supported',
    /\['false', 'Not supported'\]/.test(section) &&
    /\['true',\s+'Supported'\]/.test(section), true);
  t('custom form labels capability intent clearly',
    /_frow\('Effort support', effortSel\)/.test(section) &&
    /_frow\('Thinking support', thinkingSel\)/.test(section), true);
  t('raw Thinking field is not in the custom-model form',
    /_frow\('Thinking field'/.test(section), false);

  // Execute the capability conversion for real. Existing wire settings are
  // preserved while the two booleans are edited independently.
  function fakeEl() {
    return {
      className: '', value: '', textContent: '', children: [],
      appendChild(c) { this.children.push(c); },
    };
  }
  const fakeDocument = { createElement: () => fakeEl() };
  const reasonFns = (0, eval)(
    '(function(document){ ' +
    'function _reasoningSupport(){ return { effort:true, thinking:true }; } ' +
    'function _cfg(){ return {}; } ' +
    'var _editingReasoningBase = undefined; ' +
    extract('_boolCapabilitySelect') + ' ' +
    'var effortSel = _boolCapabilitySelect(); var thinkingSel = _boolCapabilitySelect(); ' +
    extract('_reasonValue') + ' ' + extract('_setReason') + ' ' +
    'return { effortSel, thinkingSel, _reasonValue, _setReason }; })'
  )(fakeDocument);
  const {
    effortSel: eSel, thinkingSel: tSel,
    _reasonValue: reasonValue, _setReason: setReason,
  } = reasonFns;

  eSel.value = 'true'; tSel.value = 'true';
  let rv = reasonValue();
  t('both supported writes two booleans', rv.effort + '/' + rv.thinking, 'true/true');

  eSel.value = 'true'; tSel.value = 'false';
  rv = reasonValue();
  t('effort and thinking are independent', rv.effort + '/' + rv.thinking, 'true/false');

  setReason(true);
  t('legacy true maps to both supported', eSel.value + '/' + tSel.value, 'true/true');
  setReason(false);
  t('legacy false maps to both unsupported', eSel.value + '/' + tSel.value, 'false/false');
  setReason(undefined);
  t('new/undeclared model defaults fail closed', eSel.value + '/' + tSel.value, 'false/false');

  setReason({
    effort: false, thinking: true,
    thinkingParam: 'thinking', thinkingMode: 'adaptive',
  });
  t('explicit object capability booleans populate the form',
    eSel.value + '/' + tSel.value, 'false/true');
  eSel.value = 'true';
  rv = reasonValue();
  t('editing capability preserves Thinking field', rv.thinkingParam, 'thinking');
  t('editing capability preserves Thinking mode', rv.thinkingMode, 'adaptive');

  // One form for add and edit: a separate dialog is a second place for the
  // identity fields and validation messages to drift.
  t('add and edit share one form',
    (section.match(/function _loadIntoForm\(/g) || []).length, 1);
  t('the form is pre-filled from the existing reasoning declaration',
    /_setReason\(m\.reasoning, m\)/.test(section), true);
  // The id keys the override and matches the radio row; editing it would
  // silently create a second entry instead of correcting the first.
  t('the id is locked while editing', /idInp\.disabled = true;/.test(section), true);
  t('and unlocked on exit', /idInp\.disabled = false;/.test(section), true);

  // Two storage paths, one form.
  t('a built-in model is corrected with a diff',
    /_editingBuiltin\s*\n?\s*\? _MODEL_STORE\.setOverride\(_editingId, patch\)/
      .test(section), true);
  t('a custom model is rewritten',
    /: _MODEL_STORE\.addModel\(_editingId, patch\)/.test(section), true);

  // Reset is only offered where there is something to reset.
  t('reset appears only for an overridden built-in',
    /\(isBuiltin && _MODEL_STORE\.getOverride\(m\.id\)\)/.test(section), true);
  t('reset clears the diff',
    /var resetId = _editingId;[\s\S]*?_MODEL_STORE\.clearOverride\(resetId\)/.test(section), true);
  t('editing/resetting clears a reasoning fallback circuit',
    /_clearReasoningCircuit\(\{ id: editedId \}\)/.test(section) &&
    /_clearReasoningCircuit\(\{ id: resetId \}\)/.test(section), true);

  // A correction must land everywhere without reopening the sheet.
  t('saving announces a model change', /reason: 'model-edited'/.test(section), true);
  t('resetting announces one too', /reason: 'override-reset'/.test(section), true);

  // Edit is offered on EVERY row: a build-time model is exactly the case that
  // cannot be corrected any other way.
  t('every row gets an edit button', /ai-assistant-panel-model-edit-btn/.test(row), true);
  t('the edit button is not limited to custom rows',
    /if \(m\._isCustom\)[\s\S]{0,120}model-edit-btn/.test(row), false);
  t('editing does not select the model',
    /function _requestModelEdit\(ev\)[\s\S]{0,120}ev\.preventDefault\(\);[\s\S]{0,80}ev\.stopPropagation\(\);/.test(row), true);
  t('Edit and Edited reuse the exact same edit request path',
    /editBtn\.addEventListener\('click', _requestModelEdit\)/.test(row) &&
    /editedStatus\.addEventListener\('click', _requestModelEdit\)/.test(row), true);
  t('the row requests an edit by event', /'ai-assistant-model-edit'/.test(row), true);
  t('the form listens for that event',
    /addEventListener\('ai-assistant-model-edit'/.test(section), true);
  t('an overridden row is marked',
    /row\.setAttribute\('data-overridden', 'true'\)/.test(row), true);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
