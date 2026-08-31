// EXECUTION smoke test for _appendModelCustomSection.
//
//   node tests/test_custom_section_dom.mjs _static/ai-assistant.js
//
// Every other harness in this directory asserts on SOURCE TEXT. That catches
// a great deal and it caught nothing at all when `var cancelBtn` was declared
// below the line that appended it: `var` hoists the binding but not the
// assignment, so the name existed, held `undefined`, and
// `appendChild(undefined)` threw at runtime. `node --check` passes such a
// file. Every regex assertion passes too, because the text is all present --
// just in the wrong order.
//
// The only thing that catches it is running the function. This harness builds
// a fake DOM sufficient for that and executes the real builder, so a
// use-before-assignment, a null dereference, or a listener attached to
// something that does not exist yet becomes a failing test rather than a blank
// panel in a browser.
//
// It is deliberately narrow: it proves the section CONSTRUCTS, not that it
// looks right. Construction is the failure mode that source-text tests are
// structurally blind to.
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

// ── a DOM just real enough to execute against ──────────────────────────────
function makeNode(tag) {
  const node = {
    tagName: String(tag).toUpperCase(),
    className: '', id: '', type: '', value: '', textContent: '',
    placeholder: '', title: '', maxLength: 0,
    disabled: false, checked: false, hidden: false,
    style: {}, dataset: {}, attrs: {}, children: [], listeners: {},
    parentNode: null,
    get firstChild() { return this.children[0] || null; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    removeAttribute(k) { delete this.attrs[k]; },
    appendChild(c) {
      // The assertion this file exists for. A helpful message beats
      // "parameter 1 is not of type 'Node'" from a browser console.
      if (!c || typeof c !== 'object' || !c.tagName) {
        throw new TypeError(
          `appendChild received ${c === undefined ? 'undefined' : JSON.stringify(c)} ` +
          `on <${node.tagName.toLowerCase()}> — an element was used before it was created`
        );
      }
      c.parentNode = node;
      node.children.push(c);
      return c;
    },
    removeChild(c) {
      const i = node.children.indexOf(c);
      if (i >= 0) node.children.splice(i, 1);
      return c;
    },
    replaceChild(c, old) {
      if (!c || !c.tagName) throw new TypeError('replaceChild got a non-node');
      const i = node.children.indexOf(old);
      if (i >= 0) {
        c.parentNode = node;
        old.parentNode = null;
        node.children[i] = c;
      }
      return old;
    },
    insertBefore(c, ref) {
      if (!c || !c.tagName) throw new TypeError('insertBefore got a non-node');
      const i = ref ? node.children.indexOf(ref) : -1;
      if (i >= 0) node.children.splice(i, 0, c); else node.children.push(c);
      return c;
    },
    addEventListener(evt, fn) { (node.listeners[evt] = node.listeners[evt] || []).push(fn); },
    removeEventListener() {},
    scrollIntoView() {},
    focus() {},
    click() { (node.listeners.click || []).forEach((f) => f({ preventDefault() {}, stopPropagation() {} })); },
    contains() { return false; },
    querySelector(sel) { return findOne(node, sel); },
    querySelectorAll(sel) { return findAll(node, sel); },
    classList: {
      add(...cs) {
        const have = node.className.split(/\s+/).filter(Boolean);
        for (const c of cs) if (!have.includes(c)) have.push(c);
        node.className = have.join(' ');
      },
      remove(...cs) {
        node.className = node.className.split(/\s+/)
          .filter((c) => c && !cs.includes(c)).join(' ');
      },
      contains(c) { return node.className.split(/\s+/).includes(c); },
      toggle(c, force) {
        const has = this.contains(c);
        const want = force === undefined ? !has : !!force;
        if (want && !has) this.add(c);
        if (!want && has) this.remove(c);
        return want;
      },
    },
  };
  return node;
}

function matches(n, sel) {
  if (!n || !n.tagName) return false;
  if (sel.startsWith('.')) return (' ' + n.className + ' ').includes(' ' + sel.slice(1) + ' ');
  if (sel.startsWith('#')) return n.id === sel.slice(1);
  if (sel.startsWith('[')) {
    const m = sel.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/);
    if (!m) return false;
    const v = n.getAttribute(m[1]);
    return m[2] === undefined ? v !== null : v === m[2];
  }
  return n.tagName === sel.toUpperCase();
}

function findAll(root, sel) {
  const out = [];
  for (const c of root.children) {
    if (matches(c, sel)) out.push(c);
    out.push(...findAll(c, sel));
  }
  return out;
}
function findOne(root, sel) { return findAll(root, sel)[0] || null; }

const documentEl = makeNode('div');
const documentListeners = {};
globalThis.document = {
  createElement: makeNode,
  createTextNode: (t) => Object.assign(makeNode('#text'), { textContent: String(t) }),
  getElementById: () => null,
  querySelector: (sel) => findOne(documentEl, sel),
  querySelectorAll: (sel) => findAll(documentEl, sel),
  addEventListener(evt, fn) {
    (documentListeners[evt] = documentListeners[evt] || []).push(fn);
  },
  dispatchEvent(ev) {
    for (const fn of (documentListeners[ev.type] || [])) fn(ev);
    return true;
  },
  body: documentEl,
};
globalThis.CustomEvent = function (type, init) { return { type, detail: (init || {}).detail }; };

let ls = {};
globalThis.localStorage = {
  getItem: (k) => (k in ls ? ls[k] : null),
  setItem: (k, v) => { ls[k] = String(v); },
  removeItem: (k) => { delete ls[k]; },
};
globalThis.sessionStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
};
globalThis.window = { localStorage: globalThis.localStorage, getComputedStyle: () => null };
globalThis.ICONS = {
  plus: '<svg></svg>',
  editAns: '<svg></svg>',
  retry: '<svg></svg>',
  close: '<svg></svg>',
};
globalThis._cfg = () => ({ panelApiModels: [
  { id: 'site-model', label: 'Site Model', provider: 'openai', model: 'site/model' }
] });
globalThis._getActiveModelId = (models) => models.length ? models[0].id : null;
globalThis._setActiveModelId = () => {};
globalThis._clearReasoningCircuit = () => {};

// A store stand-in with the surface the section actually touches. Real enough
// to exercise every branch, small enough that a change in the store's API
// makes this fail loudly rather than silently diverge.
const customModels = [];
let overrideIds = ['site-model'];
let hiddenIds = [];
let resetCalls = 0;
globalThis._MODEL_STORE = {
  MAX_CUSTOM: 20,
  countCustom: () => customModels.length,
  listCustom: () => customModels.slice(),
  addModel: (id, m) => {
    customModels.push(Object.assign({ id }, m));
    return { ok: true, id };
  },
  removeModel: (id) => {
    const i = customModels.findIndex((m) => m.id === id);
    if (i >= 0) customModels.splice(i, 1);
    return true;
  },
  isIdAvailable: () => true,
  getOverride: () => null,
  listOverrides: () => overrideIds.slice(),
  listHiddenBuiltins: () => hiddenIds.slice(),
  clearOverride: (id) => { overrideIds = overrideIds.filter((x) => x !== id); return true; },
  setOverride: () => ({ ok: true }),
  resetToCompiled: () => {
    customModels.splice(0);
    overrideIds = [];
    hiddenIds = [];
    resetCalls++;
    return { custom: 0, overrides: 1, hiddenBuiltins: 0 };
  },
  applyOverrides: (a) => a,
};

const _appendModelCustomSection =
  (0, eval)('(' + extract('_appendModelCustomSection') + ')');

let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) { pass++; }
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
};

// ── it must simply RUN ─────────────────────────────────────────────────────
let scrollEl = null;
try {
  scrollEl = makeNode('div');
  const bodyEl = makeNode('div');
  bodyEl._filterReindex = () => {};
  _appendModelCustomSection(scrollEl, bodyEl, 'models', null, '');
  t('the section builds without throwing', true, true);
} catch (err) {
  fail++;
  console.log('  FAIL the section builds without throwing\n       ' + err.message);
}

if (scrollEl) {
  const section = findOne(scrollEl, '.ai-assistant-panel-custom-section');
  t('the section is attached', section !== null, true);

  const editor = findOne(scrollEl, '.ai-assistant-panel-custom-editor');
  const launchAdd = findOne(scrollEl, '.ai-assistant-panel-custom-new-btn');
  const customize = findOne(scrollEl, '.ai-assistant-panel-custom-editor-toggle');
  const revert = findOne(scrollEl, '.ai-assistant-panel-custom-revert-btn');
  t('the shared editor starts collapsed', editor ? editor.hidden : null, true);
  t('the Revert control exists', revert !== null, true);
  t('Revert is placed immediately before Customize',
    revert && customize && revert.parentNode === customize.parentNode
      ? revert.parentNode.children.indexOf(revert) + 1 === revert.parentNode.children.indexOf(customize)
      : false,
    true);
  t('the fast Add model launcher exists', launchAdd !== null, true);
  t('the Effort-style Customize launcher exists', customize !== null, true);
  t('Customize points at the shared editor',
    customize ? customize.getAttribute('aria-controls') : null,
    'ai-assistant-panel-custom-editor');
  const customFieldAdd = findOne(scrollEl, '.ai-assistant-panel-custom-field-add-btn');
  t('working custom-field add control exists', customFieldAdd !== null, true);
  t('custom-field add control starts enabled', customFieldAdd ? customFieldAdd.disabled : null, false);
  if (customFieldAdd) {
    customFieldAdd.click();
    t('custom-field add creates one metadata row',
      findAll(scrollEl, '.ai-assistant-panel-custom-field-row').length, 1);
    t('metadata row has name input',
      findOne(scrollEl, '.ai-assistant-panel-custom-field-name') !== null, true);
    t('metadata row has value input',
      findOne(scrollEl, '.ai-assistant-panel-custom-field-value') !== null, true);
    t('metadata row has display selector',
      findOne(scrollEl, '.ai-assistant-panel-custom-field-display') !== null, true);
    const rmField = findOne(scrollEl, '.ai-assistant-panel-custom-field-remove-btn');
    t('metadata row has remove action', rmField !== null, true);
    if (rmField) rmField.click();
    t('remove action deletes metadata row',
      findAll(scrollEl, '.ai-assistant-panel-custom-field-row').length, 0);
  }

  if (customize && editor) {
    customize.click();
    t('Customize opens the shared editor', editor.hidden, false);
    t('Customize publishes expanded state', customize.getAttribute('aria-expanded'), 'true');
    customize.click();
    t('Customize closes the shared editor', editor.hidden, true);
  }
  if (launchAdd && editor) {
    launchAdd.click();
    t('Add model opens the same editor', editor.hidden, false);

    // The metadata builder must feed the same save path as the core fields.
    const inputs = findAll(scrollEl, '.ai-assistant-panel-custom-input');
    const byPlaceholder = (p) => inputs.find((inp) => inp.placeholder === p);
    const idField = byPlaceholder('my-model-id  (letters, digits, _ -)');
    const labelField = byPlaceholder('Display name');
    const modelField = byPlaceholder('provider/model-name');
    if (idField) idField.value = 'local-model';
    if (labelField) labelField.value = 'Local Model';
    if (modelField) modelField.value = 'local/model';

    const addMeta = findOne(scrollEl, '.ai-assistant-panel-custom-field-add-btn');
    if (addMeta) addMeta.click();
    const metaName = findOne(scrollEl, '.ai-assistant-panel-custom-field-name');
    const metaValue = findOne(scrollEl, '.ai-assistant-panel-custom-field-value');
    const metaDisplay = findOne(scrollEl, '.ai-assistant-panel-custom-field-display');
    if (metaName) metaName.value = 'Context window';
    if (metaValue) metaValue.value = '128K';
    if (metaDisplay) metaDisplay.value = 'badge';

    const saveModel = findOne(scrollEl, '.ai-assistant-panel-custom-add-btn');
    if (saveModel) saveModel.click();
    t('saving a model persists custom metadata through the shared payload',
      customModels.length ? customModels[0].custom_fields[0].value : null, '128K');
    t('saved custom metadata keeps its display mode',
      customModels.length ? customModels[0].custom_fields[0].display : null, 'badge');
    t('successful save clears metadata draft rows',
      findAll(scrollEl, '.ai-assistant-panel-custom-field-row').length, 0);
  }

  // Row-level Edit is the third entry point into the same disclosure.
  if (editor) {
    editor.hidden = true;
    document.dispatchEvent(new CustomEvent('ai-assistant-model-edit', {
      detail: {
        model: { id: 'site-model', label: 'Site Model', provider: 'openai',
                 model: 'site/model', description: '', info_url: '', reasoning: false },
        isCustom: false,
      },
    }));
    t('row Edit opens the same editor', editor.hidden, false);
    const editorAddBtn = findOne(scrollEl, '.ai-assistant-panel-custom-add-btn');
    t('row Edit switches the form to save mode',
      editorAddBtn ? editorAddBtn.textContent : null, 'Save changes');
  }

  if (revert) {
    t('Revert is enabled when local model changes exist', revert.disabled, false);
    revert.click();
    t('Revert calls the store reset exactly once', resetCalls, 1);
    t('Revert disables itself after returning to compiled state', revert.disabled, true);
  }

  // The exact crash: these three must exist and be in the button row.
  const btnRow = findOne(scrollEl, '.ai-assistant-panel-custom-btn-row');
  t('the button row exists', btnRow !== null, true);
  t('add button is present',
    findOne(scrollEl, '.ai-assistant-panel-custom-add-btn') !== null, true);
  t('cancel button is present',
    findOne(scrollEl, '.ai-assistant-panel-custom-cancel-btn') !== null, true);
  t('reset button is present',
    findOne(scrollEl, '.ai-assistant-panel-custom-reset-btn') !== null, true);
  t('all three live in the button row', btnRow ? btnRow.children.length : 0, 3);

  // Every child of every container must be a real node — the general form of
  // the bug, not just the instance that was reported.
  let nonNodes = 0;
  (function walk(n) {
    for (const c of n.children) {
      if (!c || !c.tagName) nonNodes++;
      else walk(c);
    }
  }(scrollEl));
  t('no placeholder or undefined child anywhere', nonNodes, 0);

  // The form fields the editor depends on.
  t('the effort/thinking capability selects are rendered',
    findAll(scrollEl, '.ai-assistant-panel-custom-select').length >= 2, true);
  t('the error region exists',
    findOne(scrollEl, '.ai-assistant-panel-custom-err') !== null, true);

  // Idempotency: a second call must not double-inject.
  const before = scrollEl.children.length;
  _appendModelCustomSection(scrollEl, makeNode('div'), 'models', null, '');
  t('a second call is a no-op', scrollEl.children.length, before);
}

// Construction order is UX-significant: model management must be before
// Effort/Thinking in both the normal and empty-model paths.
const normalCustom = src.indexOf('_appendModelCustomSection(scrollEl, bodyEl, groupName');
const normalReason = src.indexOf('_appendModelSheetSections(scrollEl);', normalCustom - 500);
t('normal sheet puts model management before Effort/Thinking',
  normalCustom >= 0 && normalReason > normalCustom, true);
const emptyCustom = src.indexOf('_appendModelCustomSection(scrollElEmpty, bodyEl');
const emptyReason = src.indexOf('_appendModelSheetSections(scrollElEmpty);', emptyCustom - 500);
t('empty sheet puts model management before Effort/Thinking',
  emptyCustom >= 0 && emptyReason > emptyCustom, true);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
