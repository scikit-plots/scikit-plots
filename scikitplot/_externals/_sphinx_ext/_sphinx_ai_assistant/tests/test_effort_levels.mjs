// Contract harness for the effort level registry and its surfacing.
//
//   node tests/test_effort_levels.mjs _static/ai-assistant.js
//
// Covers three things that are easy to break silently:
//   1. the level set, its order, and the default;
//   2. resolution of an unknown/absent stored id (which previously left the
//      segmented control with no radio checked and a blank description);
//   3. that both model buttons surface the effort through ONE shared path
//      rather than two copies that can drift.
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

function extractArray(name) {
  const i = src.indexOf('var ' + name + ' = [');
  if (i < 0) throw new Error('not found: ' + name);
  const start = src.indexOf('[', i);
  let depth = 0;
  for (let j = start; j < src.length; j++) {
    if (src[j] === '[') depth++;
    else if (src[j] === ']') { depth--; if (depth === 0) return src.slice(start, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

function extractObject(name) {
  const i = src.indexOf('var ' + name + ' = {');
  if (i < 0) throw new Error('not found: ' + name);
  const start = src.indexOf('{', i);
  let depth = 0;
  for (let j = start; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') { depth--; if (depth === 0) return src.slice(start, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

// Read the SHIPPED values, never a re-implementation of them.
const _EFFORT_LEVELS = (0, eval)('(' + extractArray('_EFFORT_LEVELS') + ')');
const defMatch = src.match(/var _EFFORT_DEFAULT = ('[^']*');/);
if (!defMatch) throw new Error('_EFFORT_DEFAULT not found');
const _EFFORT_DEFAULT = (0, eval)('(' + defMatch[1] + ')');

globalThis._EFFORT_LEVELS = _EFFORT_LEVELS;
globalThis._EFFORT_DEFAULT = _EFFORT_DEFAULT;
const _effortById = (0, eval)('(' + extract('_effortById') + ')');
globalThis._effortById = _effortById;

const _EFFORT_KEY = 'ai-assistant-effort-level';
globalThis._EFFORT_KEY = _EFFORT_KEY;
let store = {};
const okStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
};
globalThis.sessionStorage = okStorage;

const _getEffortLevel = (0, eval)('(' + extract('_getEffortLevel') + ')');
const _setEffortLevel = (0, eval)('(' + extract('_setEffortLevel') + ')');
globalThis._getEffortLevel = _getEffortLevel;
// The accessible name now consults reasoning support, which depends on the
// active model. Declared-supported by default here so these cases exercise the
// level-naming path; the Default path is covered in test_reasoning_support.mjs.
globalThis._getActiveModel = () => ({ provider: 'openai', reasoning: true });
globalThis._cfg = () => ({ panelReasoning: true });
globalThis._safeInt = (v, min, max, fb) => {
  const n = parseInt(v, 10);
  return (isFinite(n) && n >= min && n <= max) ? n : fb;
};
globalThis._THINKING_BUDGET_MIN = 500;
globalThis._THINKING_BUDGET_MAX = 16000;
{
  const i = src.indexOf('var _REASONING_WIRE_DEFAULTS = {');
  const start = src.indexOf('{', i);
  let depth = 0, end = start;
  for (let j = start; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') { depth--; if (depth === 0) { end = j; break; } }
  }
  globalThis._REASONING_WIRE_DEFAULTS = (0, eval)('(' + src.slice(start, end + 1) + ')');
}
globalThis._effortMapCoversCurrentScale =
  (0, eval)('(' + extract('_effortMapCoversCurrentScale') + ')');
globalThis._reasoningSupport = (0, eval)('(' + extract('_reasoningSupport') + ')');

const _modelBtnAccessibleLabel = (0, eval)('(' + extract('_modelBtnAccessibleLabel') + ')');

let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) { pass++; }
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
};
const reset = () => { store = {}; globalThis.sessionStorage = okStorage; };

// ── the level set ──────────────────────────────────────────────────────────
t('five levels', _EFFORT_LEVELS.length, 5);
t('order is low→max',
  _EFFORT_LEVELS.map((e) => e.id).join(','), 'low,medium,high,extra,max');
t('Extra sits between High and Max',
  _EFFORT_LEVELS.map((e) => e.id).indexOf('extra'), 3);
t('ids are unique',
  new Set(_EFFORT_LEVELS.map((e) => e.id)).size, _EFFORT_LEVELS.length);
{
  let bad = 0;
  for (const e of _EFFORT_LEVELS) {
    if (!e.id || !e.label || !e.hint || !e.desc) bad++;
    if (e.desc && !/[.!]$/.test(e.desc)) bad++;      // full sentences
  }
  t('every level is complete', bad, 0);
  t('labels are unique',
    new Set(_EFFORT_LEVELS.map((e) => e.label)).size, _EFFORT_LEVELS.length);
  t('hints are unique',
    new Set(_EFFORT_LEVELS.map((e) => e.hint)).size, _EFFORT_LEVELS.length);
}

// ── the default ────────────────────────────────────────────────────────────
t('default is high', _EFFORT_DEFAULT, 'high');
t('default names a real level',
  _EFFORT_LEVELS.some((e) => e.id === _EFFORT_DEFAULT), true);

reset();
t('unset storage resolves to the default', _getEffortLevel(), _EFFORT_DEFAULT);

// A stored choice is the reader's, and the new default must not override it.
reset();
store[_EFFORT_KEY] = 'medium';
t('a stored choice survives the default change', _getEffortLevel(), 'medium');
reset();
store[_EFFORT_KEY] = 'low';
t('a stored low survives', _getEffortLevel(), 'low');

// ── unknown ids can never reach the UI ─────────────────────────────────────
reset();
store[_EFFORT_KEY] = 'ludicrous';
t('unknown stored id resolves to the default', _getEffortLevel(), _EFFORT_DEFAULT);
reset();
store[_EFFORT_KEY] = '';
t('empty stored id resolves to the default', _getEffortLevel(), _EFFORT_DEFAULT);

t('resolver never returns null for junk', !!_effortById('nope'), true);
t('resolver falls back to the default entry', _effortById('nope').id, _EFFORT_DEFAULT);
t('resolver returns the asked-for level', _effortById('extra').id, 'extra');
t('resolver handles null', _effortById(null).id, _EFFORT_DEFAULT);
t('resolver handles undefined', _effortById(undefined).id, _EFFORT_DEFAULT);

// Storage denial must not throw or yield an unusable id.
reset();
globalThis.sessionStorage = {
  getItem() { throw new Error('denied'); },
  setItem() { throw new Error('denied'); },
};
t('blocked storage resolves to the default', _getEffortLevel(), _EFFORT_DEFAULT);
_setEffortLevel('max');
t('blocked write does not throw', true, true);

// ── the setter refuses ids this build cannot render ────────────────────────
reset();
_setEffortLevel('extra');
t('setter persists a known id', store[_EFFORT_KEY], 'extra');
_setEffortLevel('ludicrous');
t('setter refuses an unknown id', store[_EFFORT_KEY], 'extra');
_setEffortLevel('');
t('setter refuses an empty id', store[_EFFORT_KEY], 'extra');

// ── the segmented control sizes itself from the registry ───────────────────
t('no hard-coded column count in CSS-facing JS',
  src.includes("setProperty('--ai-effort-count'"), true);
t('column count is derived, not literal',
  /--ai-effort-count',\s*\n?\s*String\(_EFFORT_LEVELS\.length\)/.test(src), true);

// ── both model buttons surface effort through ONE path ─────────────────────
t('chip helper defined once',
  (src.match(/function _attachEffortChip\(/g) || []).length, 1);
t('both surfaces attach a chip',
  (src.match(/_attachEffortChip\(/g) || []).length, 3);   // 1 def + 2 uses

t('accessible name computed once',
  (src.match(/function _modelBtnAccessibleLabel\(/g) || []).length, 1);
t('aria sync defined once',
  (src.match(/function _syncModelBtnAria\(/g) || []).length, 1);
// 1 definition + 6 call sites: the chip helper's effort AND model listeners,
// plus init and live-sync on each of the two model buttons. Counting the
// definition in keeps the assertion honest about what the regex matches.
//
// The model listener is not optional padding: reasoning support is a property
// of the active model, so switching models can change the button's effort word
// between a level and "Default" without the effort level itself changing.
t('every model-button state change syncs aria',
  (src.match(/_syncModelBtnAria\(/g) || []).length, 7);

// The old per-surface string must be gone from both builders.
//
// Scoped to the "— current:" form on purpose: the model SHEET carries a plain
// static aria-label of "Model Configuration", which is correct and unrelated.
// A broader regex flagged it, and a check that fires on correct code teaches
// the next reader to delete the check.
t('no surface hand-builds the model aria-label',
  (src.match(/'Model Configuration \\\\u2014 current: '/g) || []).length, 0);

// ── the accessible name names both halves ──────────────────────────────────
reset();
store[_EFFORT_KEY] = 'extra';
{
  const label = _modelBtnAccessibleLabel('GPT-5');
  t('aria names the model', label.includes('GPT-5'), true);
  t('aria names the effort', label.includes('Extra'), true);
  t('aria reads as one phrase', /current: GPT-5, effort: Extra/.test(label), true);
}
reset();
{
  const label = _modelBtnAccessibleLabel('Claude');
  t('aria uses the default effort when unset',
    label.includes(_effortById(_EFFORT_DEFAULT).label), true);
  t('explicit effort argument wins',
    _modelBtnAccessibleLabel('Claude', 'low').includes('Low'), true);
  t('unknown explicit effort falls back',
    _modelBtnAccessibleLabel('Claude', 'nope')
      .includes(_effortById(_EFFORT_DEFAULT).label), true);
}

// ── chip contract ──────────────────────────────────────────────────────────
{
  const helper = extract('_attachEffortChip');
  t('chip is decorative for AT', /aria-hidden/.test(helper), true);
  t('chip listens for effort changes',
    /ai-assistant-effort-change/.test(helper), true);
  t('chip updates the host aria too', /_syncModelBtnAria\(/.test(helper), true);

  const sync = extract('_syncEffortChip');
  t('chip text is the level label', /ef\.label/.test(sync), true);
  t('chip carries the id for styling', /dataset\.effort/.test(sync), true);
  t('chip resolves through the registry', /_effortById\(/.test(sync), true);
}

// ── customizable effort editor + provider-style presets ───────────────────
{
  const presets = (0, eval)('(' + extractObject('_EFFORT_PRESETS') + ')');
  t('Claude preset has five levels', presets.claude.length, 5);
  t('Claude preset matches the shipped labels',
    presets.claude.map((e) => e.label).join('|'), 'Low|Medium|High|Extra|Max');
  t('OpenAI preset has four levels', presets.openai.length, 4);
  t('OpenAI preset matches requested labels',
    presets.openai.map((e) => e.label).join('|'), 'Instant|Medium|High|Pro');
  t('OpenAI preset carries explicit stable ids',
    presets.openai.map((e) => e.id).join(','), 'instant,medium,high,pro');

  globalThis._EFFORT_PRESETS = presets;
  const slug = (0, eval)('(' + extract('_slugifyEffortId') + ')');
  t('slug helper normalizes visible text', slug('Very High!', Object.create(null)), 'very_high');
  const claimed = Object.create(null); claimed.high = true;
  t('slug helper avoids collisions', slug('High', claimed), 'high_2');

  const editor = extract('_buildEffortEditorSection');
  t('editor is a dedicated sheet section', /_buildSheetSection\(\s*'Customize effort'/.test(editor), true);
  t('editor starts collapsed', /section\.hidden = true/.test(editor), true);
  t('editor exposes Claude preset', /_loadPreset\('claude'\)/.test(editor), true);
  t('editor exposes OpenAI preset', /_loadPreset\('openai'\)/.test(editor), true);
  t('preset click does not directly apply', /Presets do[\s\S]*not apply until you save/.test(editor), true);
  t('editor has a dashed-add DOM hook', /ai-assistant-panel-effort-editor-add/.test(editor), true);
  t('editor has per-level remove controls', /ai-assistant-panel-effort-editor-remove/.test(editor), true);
  t('editor enforces minimum two levels', /draft\.length <= 2/.test(editor), true);
  t('editor enforces maximum eight levels', /draft\.length >= 8/.test(editor), true);
  t('Save uses the shared apply path', /_applyRuntimeEffortLevels\(validation\.levels/.test(editor), true);
  t('new-row ids auto-follow labels only while automatic', /if \(lvl\._autoId\)/.test(editor), true);
  t('existing stable ids are not auto-renamed', /Existing ids are stable mapping keys/.test(editor), true);

  const append = extract('_appendModelSheetSections');
  t('effort section exposes Customize toggle', /Customize effort buttons/.test(append), true);
  t('editor section is appended directly after effort',
    /sheet\.appendChild\(effortSection\);\s*sheet\.appendChild\(effortEditor\.section\);/.test(append), true);
  t('sheet refresh preserves dirty editor drafts', /syncFromRegistry\(false\)/.test(append), true);

  const publicApi = src.slice(src.indexOf('ns.setEffortLevels = function'),
    src.indexOf('ns.getEffortLevels = function'));
  t('public API shares the same apply helper', /_applyRuntimeEffortLevels\(/.test(publicApi), true);
  const apply = extract('_applyRuntimeEffortLevels');
  t('scale apply refreshes sheet', /_refreshEffortSheetUI\(\)/.test(apply), true);
  t('scale apply announces chips', /_announceEffortScaleChange\(activeId\)/.test(apply), true);
}

// ── section notes ──────────────────────────────────────────────────────────
{
  const noteMatch = src.match(/var _EFFORT_NOTE =\s*([\s\S]*?);\n/);
  const thinkMatch = src.match(/var _THINKING_NOTE =\s*([\s\S]*?);\n/);
  t('effort note exists', !!noteMatch, true);
  t('thinking note exists', !!thinkMatch, true);

  const EFFORT_NOTE = (0, eval)('(' + noteMatch[1] + ')');
  // Expose it before evaluating the second note: if one note is ever aliased
  // to the other, the expression must EVALUATE so the "the two notes differ"
  // assertion below can report it. Without this the harness threw a
  // ReferenceError and every later case went unrun (Rule 2).
  globalThis._EFFORT_NOTE = EFFORT_NOTE;
  const THINKING_NOTE = (0, eval)('(' + thinkMatch[1] + ')');

  // The note must state the BENEFIT and BOTH costs — a note that mentions only
  // one side of a trade-off is worse than none, because it reads as advice.
  t('note states the benefit', /more thorough/i.test(EFFORT_NOTE), true);
  t('note states the time cost', /longer/i.test(EFFORT_NOTE), true);
  t('note states the usage cost', /usage limit/i.test(EFFORT_NOTE), true);
  // Scope: the control writes to sessionStorage, so the note must not imply
  // the choice is per-message or permanent.
  t('note states the scope', /session/i.test(EFFORT_NOTE), true);
  t('note is a complete sentence set', /\.$/.test(EFFORT_NOTE.trim()), true);
  t('note is not a fragment', EFFORT_NOTE.split('.').length - 1 >= 2, true);
  t('note stays short', EFFORT_NOTE.length < 260, true);

  t('thinking note names the behaviour', /step by step/i.test(THINKING_NOTE), true);
  t('thinking note states a cost', /seconds|tokens/i.test(THINKING_NOTE), true);
  // The two controls sit together and are easy to read as one dial.
  t('thinking note relates itself to effort',
    /effort/i.test(THINKING_NOTE), true);
  t('thinking note stays short', THINKING_NOTE.length < 300, true);

  // Notes must be distinct — a copy-paste would make one of them wrong.
  t('the two notes differ', EFFORT_NOTE === THINKING_NOTE, false);
}

// ── the note is built by the shared section helper, not per caller ─────────
{
  const builder = extract('_buildSheetSection');
  t('section builder accepts a note',
    /function _buildSheetSection\(label, note, noteId\)/.test(builder), true);
  t('section builder emits the note wrapper',
    /ai-assistant-panel-sheet-section-note'/.test(builder), true);
  t('section builder emits the note paragraph',
    /ai-assistant-panel-sheet-section-note-text/.test(builder), true);
  // No note argument -> no empty wrapper.
  t('note is conditional',
    /if \(typeof note === 'string' && note\)/.test(builder), true);
  t('note id is optional', /if \(noteId\)/.test(builder), true);

  // Neither section may hand-roll its own note markup.
  t('note markup exists in exactly one place',
    (src.match(/ai-assistant-panel-sheet-section-note'/g) || []).length, 1);
}

// ── the notes are announced, not merely displayed ──────────────────────────
{
  t('effort control points at its note',
    src.includes("effortSeg.setAttribute('aria-describedby', 'ai-assistant-panel-effort-note')"),
    true);
  t('thinking control points at its note',
    /aria-describedby',\s*\n?\s*'ai-assistant-panel-thinking-note'/.test(src), true);
  // Every referenced id must actually be produced, and every consumer must
  // use the same literal. Three occurrences now: passed to the section
  // builder, referenced by aria-describedby, and looked up by the sink that
  // re-writes the note when the active model changes. Asserting a minimum
  // rather than an exact count would let a typo'd fourth copy slip in.
  for (const id of ['ai-assistant-panel-effort-note',
                    'ai-assistant-panel-thinking-note']) {
    t('id ' + id + ' is used consistently',
      (src.match(new RegExp("'" + id + "'", 'g')) || []).length, 3);
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
