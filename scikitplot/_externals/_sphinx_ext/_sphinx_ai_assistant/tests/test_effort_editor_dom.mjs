// Execution harness for the effort-level editor DOM.
//
//   node tests/test_effort_editor_dom.mjs _static/ai-assistant.js
//
// Source-text assertions can prove the editor code exists, but not that the
// actual builder can construct, add/remove rows, preserve stable ids, and keep
// preset clicks draft-only. This fake DOM runs the shipped builder directly.
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
  throw new Error('unbalanced array: ' + name);
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
  throw new Error('unbalanced object: ' + name);
}

function makeNode(tag) {
  const node = {
    tagName: String(tag).toUpperCase(),
    className: '', id: '', type: '', value: '', textContent: '', innerHTML: '',
    title: '', placeholder: '', maxLength: 0, hidden: false, disabled: false,
    autocomplete: '', spellcheck: true,
    style: {}, dataset: {}, attrs: {}, children: [], listeners: {}, parentNode: null,
    get firstChild() { return this.children[0] || null; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    removeAttribute(k) { delete this.attrs[k]; },
    appendChild(c) {
      if (!c || !c.tagName) throw new TypeError('appendChild received a non-node');
      c.parentNode = node; node.children.push(c); return c;
    },
    removeChild(c) {
      const i = node.children.indexOf(c); if (i >= 0) node.children.splice(i, 1); return c;
    },
    addEventListener(evt, fn) { (node.listeners[evt] = node.listeners[evt] || []).push(fn); },
    emit(evt) { for (const fn of (node.listeners[evt] || [])) fn({ target: node, preventDefault() {} }); },
    click() { if (!node.disabled) node.emit('click'); },
    focus() { node._focused = true; },
    querySelector(sel) { return findAll(node, sel)[0] || null; },
    querySelectorAll(sel) { return findAll(node, sel); },
  };
  node.classList = {
    add(...cs) {
      const set = new Set(node.className.split(/\s+/).filter(Boolean));
      cs.forEach((c) => set.add(c)); node.className = [...set].join(' ');
    },
    remove(...cs) {
      const drop = new Set(cs);
      node.className = node.className.split(/\s+/).filter((c) => c && !drop.has(c)).join(' ');
    },
    contains(c) { return node.className.split(/\s+/).includes(c); },
    toggle(c, force) {
      const has = this.contains(c);
      const on = force === undefined ? !has : !!force;
      if (on) this.add(c); else this.remove(c);
      return on;
    },
  };
  return node;
}
function matches(node, sel) {
  if (!node || !node.tagName) return false;
  if (sel.startsWith('.')) return node.className.split(/\s+/).includes(sel.slice(1));
  if (sel.startsWith('#')) return node.id === sel.slice(1);
  return node.tagName === sel.toUpperCase();
}
function findAll(root, sel) {
  const out = [];
  for (const c of root.children) {
    if (matches(c, sel)) out.push(c);
    out.push(...findAll(c, sel));
  }
  return out;
}

const documentRoot = makeNode('div');
const docListeners = {};
globalThis.document = {
  createElement: makeNode,
  getElementById(id) { return findAll(documentRoot, '#' + id)[0] || null; },
  addEventListener(evt, fn) { (docListeners[evt] = docListeners[evt] || []).push(fn); },
  dispatchEvent(evt) { for (const fn of (docListeners[evt.type] || [])) fn(evt); return true; },
  body: documentRoot,
};
globalThis.CustomEvent = function (type, init) { return { type, detail: (init || {}).detail || {} }; };

globalThis.ICONS = { plus: '<svg></svg>', trash: '<svg></svg>', editAns: '<svg></svg>' };
globalThis._EFFORT_LEVELS = (0, eval)('(' + extractArray('_EFFORT_LEVELS') + ')');
globalThis._EFFORT_PRESETS = (0, eval)('(' + extractObject('_EFFORT_PRESETS') + ')');
globalThis._EFFORT_DEFAULT = 'high';
globalThis._cfg = () => ({});
globalThis._getActiveModel = () => ({ provider: 'anthropic', reasoning: true });
globalThis._reasoningSupport = () => ({ effort: true });
globalThis._slugifyEffortId = (0, eval)('(' + extract('_slugifyEffortId') + ')');
globalThis._validateEffortLevelsArray = (0, eval)('(' + extract('_validateEffortLevelsArray') + ')');
globalThis._buildSheetSection = (0, eval)('(' + extract('_buildSheetSection') + ')');

let applyCalls = [];
globalThis._applyRuntimeEffortLevels = (levels, defaultId) => {
  applyCalls.push({ levels: levels.map((x) => ({ ...x })), defaultId });
  globalThis._EFFORT_LEVELS.splice(0, globalThis._EFFORT_LEVELS.length, ...levels.map((x) => ({ ...x })));
  return { ok: true, defaultId, activeId: defaultId };
};

const _buildEffortEditorSection = (0, eval)('(' + extract('_buildEffortEditorSection') + ')');

let pass = 0, fail = 0;
function t(name, got, want) {
  if (got === want) pass++;
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
}

let editor;
try {
  editor = _buildEffortEditorSection();
  documentRoot.appendChild(editor.section);
  t('editor builds without throwing', true, true);
} catch (err) {
  fail++;
  console.log('  FAIL editor builds without throwing\n       ' + err.stack);
}

if (editor) {
  t('editor is a sheet section', editor.section.classList.contains('ai-assistant-panel-sheet-section'), true);
  t('editor starts hidden', editor.section.hidden, true);
  t('initial draft renders five rows', findAll(editor.section, '.ai-assistant-panel-effort-editor-row').length, 5);

  editor.setOpen(true);
  t('setOpen reveals editor', editor.section.hidden, false);

  const presets = findAll(editor.section, '.ai-assistant-panel-effort-editor-preset');
  const openai = presets.find((b) => b.dataset.preset === 'openai');
  const claude = presets.find((b) => b.dataset.preset === 'claude');
  t('both preset buttons exist', !!openai && !!claude, true);
  t('Claude is suggested for anthropic active model', claude.dataset.suggested, 'true');

  openai.click();
  t('OpenAI preset changes draft to four rows', findAll(editor.section, '.ai-assistant-panel-effort-editor-row').length, 4);
  t('preset is draft-only', applyCalls.length, 0);

  // Stable id: changing visible copy must not silently rewrite an existing map key.
  let labelInputs = findAll(editor.section, '.ai-assistant-panel-effort-editor-label-input');
  let idInputs = findAll(editor.section, '.ai-assistant-panel-effort-editor-id-input');
  const oldId = idInputs[0].value;
  labelInputs[0].value = 'Fast';
  labelInputs[0].emit('input');
  t('editing existing label preserves stable id', idInputs[0].value, oldId);

  // Dashed + adds a blank custom row whose id follows its label until manually edited.
  findAll(editor.section, '.ai-assistant-panel-effort-editor-add')[0].click();
  t('add button increases count', findAll(editor.section, '.ai-assistant-panel-effort-editor-row').length, 5);
  labelInputs = findAll(editor.section, '.ai-assistant-panel-effort-editor-label-input');
  idInputs = findAll(editor.section, '.ai-assistant-panel-effort-editor-id-input');
  const last = labelInputs.length - 1;
  labelInputs[last].value = 'Ultra Deep';
  labelInputs[last].emit('input');
  t('new row auto-generates id from label', idInputs[last].value, 'ultra_deep');

  // Remove returns to four.
  findAll(editor.section, '.ai-assistant-panel-effort-editor-remove')[last].click();
  t('remove button decreases count', findAll(editor.section, '.ai-assistant-panel-effort-editor-row').length, 4);

  // Invalid IDs are rejected before the shared apply path.
  idInputs = findAll(editor.section, '.ai-assistant-panel-effort-editor-id-input');
  idInputs[0].value = 'Bad ID';
  idInputs[0].emit('input');
  findAll(editor.section, '.ai-assistant-panel-effort-editor-save')[0].click();
  t('invalid id does not apply', applyCalls.length, 0);
  t('invalid id surfaces an error',
    findAll(editor.section, '.ai-assistant-panel-effort-editor-status--error').length, 1);

  // Repair and save; now exactly one apply call is allowed.
  idInputs[0].value = 'instant';
  idInputs[0].emit('input');
  findAll(editor.section, '.ai-assistant-panel-effort-editor-save')[0].click();
  t('Save calls shared apply once', applyCalls.length, 1);
  t('saved draft has four levels', applyCalls[0].levels.length, 4);
  t('saved visible edit is preserved', applyCalls[0].levels[0].label, 'Fast');

  // Min-count guard is real, not copy only.
  openai.click();
  let removes = findAll(editor.section, '.ai-assistant-panel-effort-editor-remove');
  removes[0].click();
  removes = findAll(editor.section, '.ai-assistant-panel-effort-editor-remove');
  removes[0].click();
  t('two levels remain after two removes', findAll(editor.section, '.ai-assistant-panel-effort-editor-row').length, 2);
  removes = findAll(editor.section, '.ai-assistant-panel-effort-editor-remove');
  t('remove buttons disable at minimum', removes.every((b) => b.disabled), true);

  findAll(editor.section, '.ai-assistant-panel-effort-editor-cancel')[0].click();
  t('Cancel closes editor', editor.section.hidden, true);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
