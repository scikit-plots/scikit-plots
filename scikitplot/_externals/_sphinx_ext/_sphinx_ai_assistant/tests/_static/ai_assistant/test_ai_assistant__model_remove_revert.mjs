// Contract harness for row-level model removal + global compiled-state Revert.
//
//   node tests/test_model_remove_revert.mjs _static/ai-assistant.js
//
// Built-in models are immutable site config. "Remove" therefore means a
// persisted local tombstone for a compiled model and an actual local deletion
// for a reader-added custom model. Revert clears tombstones, overrides, and
// custom models, returning model management to the compiled starting point.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');

function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { depth++; started = true; }
    else if (src[j] === '}') {
      depth--;
      if (started && depth === 0) return src.slice(i, j + 1);
    }
  }
  throw new Error('unbalanced: ' + name);
}

let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) pass++;
  else {
    fail++;
    console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`);
  }
};

// ── local tombstone storage executes for real ─────────────────────────────
let store = {};
globalThis.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
  removeItem: (k) => { delete store[k]; },
};
globalThis._HIDDEN_BUILTIN_KEY = 'ai-assistant-hidden-builtin-models';
globalThis._SCHEMA_VER = 1;
globalThis._builtin = Object.create(null);
globalThis._builtin.site = true;
globalThis._hiddenBuiltin = Object.create(null);
globalThis._trim = (v) => typeof v === 'string' ? v.trim() : '';
globalThis._isStr = (v) => typeof v === 'string';
globalThis._safeId = (id) => /^[a-zA-Z][a-zA-Z0-9_.-]{0,63}$/.test(globalThis._trim(id));

for (const name of [
  '_persistHiddenBuiltin', 'hideBuiltin', 'isHiddenBuiltin',
  'listHiddenBuiltins', 'clearHiddenBuiltins',
]) {
  globalThis[name] = (0, eval)('(' + extract(name) + ')');
}

t('compiled row can be locally hidden', hideBuiltin('site').ok, true);
t('hidden compiled row is queryable', isHiddenBuiltin('site'), true);
t('tombstone is persisted', JSON.parse(store[globalThis._HIDDEN_BUILTIN_KEY]).ids[0], 'site');
t('non-compiled row cannot use builtin tombstone path', hideBuiltin('custom').ok, false);
t('hidden list reports the compiled row', listHiddenBuiltins().join(','), 'site');
t('clear returns tombstone count', clearHiddenBuiltins(), 1);
t('clear reveals the compiled row', isHiddenBuiltin('site'), false);

// ── reset aggregate executes with isolated component fakes ────────────────
globalThis.clearCustom = () => 3;
globalThis.clearOverrides = () => 2;
globalThis.clearHiddenBuiltins = () => 4;
const resetToCompiled = (0, eval)('(' + extract('resetToCompiled') + ')');
const reset = resetToCompiled();
t('revert clears added models', reset.custom, 3);
t('revert clears built-in edits', reset.overrides, 2);
t('revert clears hidden compiled rows', reset.hiddenBuiltins, 4);

// ── request boundary and UI wiring ────────────────────────────────────────
const active = extract('_getActiveModel');
const row = extract('_buildModelRowV2');
const manager = extract('_appendModelCustomSection');
const filter = extract('_attachModelFilter');

t('request path applies built-in overrides',
  /_MODEL_STORE\.applyOverrides\(builtins\)/.test(active), true);
t('request path excludes hidden compiled models',
  /!_MODEL_STORE\.isHiddenBuiltin\(m\.id\)/.test(active), true);
t('request path includes reader-added custom models',
  /_MODEL_STORE\.listCustom\(\)/.test(active), true);

t('every editable row gets the fire remove action',
  /ai-assistant-panel-model-remove-btn/.test(row) &&
  /_setActionContent\(removeBtn,\s*'\\ud83d\\udd25',\s*'Delete'\)/.test(row), true);
t('custom row removal deletes the local model',
  /if \(m\._isCustom\)[\s\S]*?_MODEL_STORE\.removeModel\(m\.id\)/.test(row), true);
t('compiled row removal uses a tombstone',
  /else \{[\s\S]*?_MODEL_STORE\.hideBuiltin\(m\.id\)/.test(row), true);
t('removing active row resolves a fallback immediately',
  /if \(wasActive\)[\s\S]*?_setActiveModelId\(fallbackId\)/.test(row), true);

t('Revert control is created', /ai-assistant-panel-custom-revert-btn/.test(manager), true);
t('Revert calls the single aggregate reset', /_MODEL_STORE\.resetToCompiled\(\)/.test(manager), true);
t('Revert clears the tab selection before compiled default resolution',
  /sessionStorage\.removeItem\(_PANEL_MODEL_KEY\)[\s\S]*?_getActiveModelId\(compiled\)/.test(manager), true);
t('Revert does not reset Effort or Thinking registries',
  /resetEffortLevels|resetThinking|_clearRuntimeEffortOverride/.test(manager), false);

t('filter snapshot excludes tombstoned rows',
  /data-model-removed'[\s\S]*?!== 'true'/.test(filter), true);
t('live filter reindex also excludes tombstoned rows',
  (filter.match(/data-model-removed/g) || []).length >= 2, true);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
