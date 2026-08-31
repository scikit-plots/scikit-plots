// Behavioural harness for the AI panel trigger-pill visibility logic,
// exercised outside a browser.
//
//   node tests/test_panel_trigger.mjs _static/ai-assistant.js
//
// Mirrors tests/test_copy_mode.mjs: the pure functions are extracted from the
// source text and evaluated against fakes, rather than booting the whole IIFE.
// That keeps the harness dependency-free and makes it a real CI gate on the
// shipped file — not on a copy of it.
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

// ── fakes ──────────────────────────────────────────────────────────────────
let store = {};
const workingStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
};
globalThis.localStorage = workingStorage;

let CFG = {};
globalThis._cfg = () => CFG;
globalThis._PANEL_TRIGGER_KEY = 'ai-assistant-panel-trigger';

// Minimal element stand-ins. Only the surface the code under test touches.
function fakeEl() {
  return {
    attrs: {},
    style: {},
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    removeAttribute(k) { delete this.attrs[k]; },
  };
}

let _aiPanelEl = null;
let _aiTriggerEl = null;
let created = 0;

Object.defineProperty(globalThis, '_aiPanelEl', {
  get: () => _aiPanelEl, set: (v) => { _aiPanelEl = v; },
});
Object.defineProperty(globalThis, '_aiTriggerEl', {
  get: () => _aiTriggerEl, set: (v) => { _aiTriggerEl = v; },
});

globalThis._createTriggerPill = () => { created++; return fakeEl(); };
globalThis.document = { body: { appendChild() {} } };

// ── functions under test ───────────────────────────────────────────────────
const NAMES = [
  '_panelTriggerVisible',
  '_setPanelTriggerPref',
  '_panelState',
  '_panelTriggerState',
  '_ensureTriggerPill',
  '_applyPanelTriggerVisibility',
  '_panelTriggerAccessibleLabel',
  '_syncPanelTriggerUI',
  'setPanelTriggerVisible',
];
const F = {};
for (const n of NAMES) F[n] = (0, eval)('(' + extract(n) + ')');
for (const n of NAMES) globalThis[n] = F[n];

const {
  _panelTriggerVisible, _setPanelTriggerPref, _panelState, _panelTriggerState,
  _applyPanelTriggerVisibility, _panelTriggerAccessibleLabel, setPanelTriggerVisible,
} = F;

// The sync path needs a document even before the live-DOM section below mounts
// one; a null-returning stand-in keeps the earlier pure cases document-free.
globalThis.document.getElementById = () => null;
globalThis.document.querySelector = () => null;

// ── runner ─────────────────────────────────────────────────────────────────
let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) { pass++; }
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
};

function reset(cfg) {
  store = {};
  globalThis.localStorage = workingStorage;
  CFG = cfg || {};
  _aiPanelEl = null;
  _aiTriggerEl = null;
  created = 0;
}

// ── AC2: default resolution is visible ─────────────────────────────────────
reset();
t('default is visible', _panelTriggerVisible(), true);

reset({ panelStartMinimized: false });
t('build default false honoured', _panelTriggerVisible(), false);

reset({ panelStartMinimized: true });
t('build default true honoured', _panelTriggerVisible(), true);

// ── AC3: stored reader preference wins ─────────────────────────────────────
reset({ panelStartMinimized: true });
_setPanelTriggerPref(false);
t('stored hide wins over build show', _panelTriggerVisible(), false);

reset({ panelStartMinimized: false });
_setPanelTriggerPref(true);
t('stored show wins over build hide', _panelTriggerVisible(), true);

// ── AC4: toggle disabled pins the build value ──────────────────────────────
reset({ panelStartMinimized: true, panelTriggerToggle: false });
store[globalThis._PANEL_TRIGGER_KEY] = 'false';
t('toggle off pins build show', _panelTriggerVisible(), true);

reset({ panelStartMinimized: false, panelTriggerToggle: false });
store[globalThis._PANEL_TRIGGER_KEY] = 'true';
t('toggle off pins build hide', _panelTriggerVisible(), false);

// ── AC5: corrupt stored value ignored ──────────────────────────────────────
reset({ panelStartMinimized: true });
store[globalThis._PANEL_TRIGGER_KEY] = 'nonsense';
t('corrupt stored ignored (show)', _panelTriggerVisible(), true);

reset({ panelStartMinimized: false });
store[globalThis._PANEL_TRIGGER_KEY] = '1';
t('corrupt stored ignored (hide)', _panelTriggerVisible(), false);

// ── AC6: storage denial is non-fatal both ways ─────────────────────────────
reset({ panelStartMinimized: true });
globalThis.localStorage = {
  getItem() { throw new Error('denied'); },
  setItem() { throw new Error('denied'); },
};
t('read denial -> build value', _panelTriggerVisible(), true);
_setPanelTriggerPref(false);
t('write denial does not throw', true, true);

// ── _panelState ────────────────────────────────────────────────────────────
reset();
t('no panel -> idle', _panelState(), 'idle');

reset();
_aiPanelEl = fakeEl(); _aiPanelEl.style.display = 'flex';
t('visible panel -> open', _panelState(), 'open');

reset();
_aiPanelEl = fakeEl(); _aiPanelEl.style.display = 'none';
_aiPanelEl.setAttribute('data-minimized', 'true');
t('minimized panel -> minimized', _panelState(), 'minimized');

reset();
_aiPanelEl = fakeEl(); _aiPanelEl.style.display = 'none';
t('closed panel -> idle', _panelState(), 'idle');

// ── AC7: minimize always shows the pill ────────────────────────────────────
reset({ panelStartMinimized: true });
_setPanelTriggerPref(false);
t('minimized overrides hidden pref', _applyPanelTriggerVisibility('minimized'), true);
t('minimized created the pill', created, 1);
// Read through helpers: a regression that leaves the pill uncreated must be
// reported as a failure, not crash the harness before the later cases run.
const pillDisplay = () => (_aiTriggerEl ? _aiTriggerEl.style.display : null);
const pillMinimized = () => (_aiTriggerEl ? _aiTriggerEl.getAttribute('data-minimized') : null);

t('minimized pill display', pillDisplay(), 'flex');
t('minimized pill data-minimized', pillMinimized(), 'true');

// ── AC8: open always hides the pill ────────────────────────────────────────
reset({ panelStartMinimized: true });
_applyPanelTriggerVisibility('idle');
t('idle+show creates pill', created, 1);
_applyPanelTriggerVisibility('open');
t('open hides pill', pillDisplay(), 'none');
t('open clears data-minimized', pillMinimized(), null);
t('open did not create a second pill', created, 1);

// ── idle honours the preference, and costs no DOM when hidden ──────────────
reset({ panelStartMinimized: true });
_setPanelTriggerPref(false);
t('idle+hidden -> not shown', _applyPanelTriggerVisibility('idle'), false);
t('idle+hidden creates no pill', created, 0);

reset({ panelStartMinimized: true });
t('idle+shown -> shown', _applyPanelTriggerVisibility('idle'), true);

// ── idempotent creation (C-4) ──────────────────────────────────────────────
reset({ panelStartMinimized: true });
_applyPanelTriggerVisibility('idle');
_applyPanelTriggerVisibility('idle');
_applyPanelTriggerVisibility('minimized');
t('pill created exactly once', created, 1);

// ── state resolved from the live panel when omitted ────────────────────────
reset({ panelStartMinimized: true });
_setPanelTriggerPref(false);
_aiPanelEl = fakeEl(); _aiPanelEl.style.display = 'none';
_aiPanelEl.setAttribute('data-minimized', 'true');
t('omitted state resolves to minimized', _applyPanelTriggerVisibility(), true);

reset({ panelStartMinimized: true });
_setPanelTriggerPref(false);
t('omitted state resolves to idle', _applyPanelTriggerVisibility(), false);

// ── accessible labels ──────────────────────────────────────────────────────
reset({ panelTriggerLabel: 'Ask AI' });
t('labels differ per state',
  _panelTriggerAccessibleLabel(true) === _panelTriggerAccessibleLabel(false), false);
t('shown label states the action', /Activate to hide/.test(_panelTriggerAccessibleLabel(true)), true);
t('hidden label states the action', /Activate to show/.test(_panelTriggerAccessibleLabel(false)), true);
t('label uses configured pill name', /Ask AI/.test(_panelTriggerAccessibleLabel(true)), true);
reset({ panelTriggerLabel: 'Docs Copilot' });
t('label honours custom pill name', /Docs Copilot/.test(_panelTriggerAccessibleLabel(false)), true);
t('explicit label argument wins', /Override/.test(_panelTriggerAccessibleLabel(true, 'Override')), true);

// ── AC9: rendered switch DOM contract ──────────────────────────────────────
//
// Built against a minimal createElement fake, so the assertions are about the
// markup the shipped code actually emits rather than about a description of it.
function makeNode(tag) {
  const n = {
    tagName: String(tag).toUpperCase(),
    className: '',
    id: '',
    type: '',
    title: '',
    textContent: '',
    dataset: {},
    children: [],
    attrs: {},
    listeners: {},
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    removeAttribute(k) { delete this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    classList: {
      add(...cs) {
        const have = n.className.split(/\s+/).filter(Boolean);
        for (const c of cs) if (!have.includes(c)) have.push(c);
        n.className = have.join(' ');
      },
      remove(...cs) {
        n.className = n.className.split(/\s+/)
          .filter((c) => c && !cs.includes(c)).join(' ');
      },
      contains(c) { return n.className.split(/\s+/).includes(c); },
    },
    addEventListener(evt, fn) { (this.listeners[evt] = this.listeners[evt] || []).push(fn); },
    querySelector(sel) { return find(this, sel); },
  };
  return n;
}

function matches(node, sel) {
  if (sel.startsWith('.')) return (' ' + node.className + ' ').includes(' ' + sel.slice(1) + ' ');
  if (sel.startsWith('#')) return node.id === sel.slice(1);
  return node.tagName === sel.toUpperCase();
}

function find(root, sel) {
  for (const c of root.children) {
    if (matches(c, sel)) return c;
    const hit = find(c, sel);
    if (hit) return hit;
  }
  return null;
}

globalThis.document.createElement = makeNode;
globalThis.getStaticAssetUrl = (f, p) => (p || '_static/') + f;
globalThis.createMenuItem = (0, eval)('(' + extract('createMenuItem') + ')');
const createPanelSection = (0, eval)('(' + extract('createPanelSection') + ')');

reset({ panelStartMinimized: true, panelTitle: 'AI Assistant', panelTriggerLabel: 'Ask AI' });
let section = createPanelSection('_static/');
let sw = find(section, '#ai-assistant-panel-trigger-toggle');

t('section class', section.className, 'ai-assistant-panel-section');
t('section records visible state', section.dataset.panelTrigger, 'visible');
t('section records toggle presence', section.dataset.panelHasToggle, 'true');
t('row class', section.children[0].className, 'ai-assistant-panel-row');
t('action item id preserved', find(section, '#ai-assistant-ai-panel-open') !== null, true);
t('action item is a sibling of the switch', section.children[0].children.length, 2);

t('switch exists', sw !== null, true);
t('switch class', sw.className, 'ai-assistant-panel-mode-switch ai-assistant-mic-popup-toggle');
t('switch is a button', sw.tagName, 'BUTTON');
t('switch type', sw.type, 'button');
t('switch role', sw.getAttribute('role'), 'menuitemcheckbox');
t('switch checked when visible', sw.getAttribute('aria-checked'), 'true');
t('switch labelled', /Activate to hide/.test(sw.getAttribute('aria-label')), true);
t('switch title mirrors label', sw.title, sw.getAttribute('aria-label'));

const track = find(sw, '.ai-assistant-panel-toggle-track');
t('track present', track !== null, true);
t('track reuses the mic primitive',
  track.className, 'ai-assistant-mic-toggle-track ai-assistant-panel-toggle-track');
t('track hidden from AT', track.getAttribute('aria-hidden'), 'true');
t('thumb inside track',
  track.children[0].className, 'ai-assistant-mic-toggle-thumb ai-assistant-panel-toggle-thumb');
t('state text present', find(sw, '.ai-assistant-panel-toggle-text').textContent, 'Shown');
t('switch has a click handler', (sw.listeners.click || []).length, 1);

// Hidden preference is reflected in the initial render.
reset({ panelStartMinimized: true });
_setPanelTriggerPref(false);
section = createPanelSection('_static/');
sw = find(section, '#ai-assistant-panel-trigger-toggle');
t('hidden pref renders unchecked', sw.getAttribute('aria-checked'), 'false');
t('hidden pref state text', find(sw, '.ai-assistant-panel-toggle-text').textContent, 'Hidden');
t('hidden pref section dataset', section.dataset.panelTrigger, 'hidden');

// AC4 at the DOM level: no switch at all when the site pinned the value.
reset({ panelStartMinimized: true, panelTriggerToggle: false });
section = createPanelSection('_static/');
t('pinned build value renders no switch',
  find(section, '#ai-assistant-panel-trigger-toggle'), null);
t('pinned section records absence', section.dataset.panelHasToggle, 'false');
t('pinned row keeps the plain menu item',
  find(section, '#ai-assistant-ai-panel-open') !== null, true);

// ── GAP: switch must never disagree with the pill ──────────────────────────
//
// Reported scenario: switch OFF, open the panel, then minimize. The pill is
// forced on screen (I3), so a switch still reading "Hidden" would be a visible
// lie. These cases walk the full lifecycle with a LIVE switch in the DOM and
// assert pill and switch agree at every step.
{
  globalThis.document.createElement = makeNode;
  globalThis.getStaticAssetUrl = (f, p) => (p || '_static/') + f;
  globalThis.createMenuItem = (0, eval)('(' + extract('createMenuItem') + ')');
  const build = (0, eval)('(' + extract('createPanelSection') + ')');

  // A live document stand-in: the sync path looks the switch up by id and the
  // section up by class, exactly as it does in a browser.
  let live = null;
  globalThis.document.getElementById = (id) => (live ? find(live, '#' + id) : null);
  globalThis.document.querySelector = (sel) => {
    if (!live) return null;
    return matches(live, sel) ? live : find(live, sel);
  };

  const mount = () => { live = build('_static/'); return live; };
  // What createAIAssistantUI() does at page load, after the dropdown is built.
  const initIdle = () => _applyPanelTriggerVisibility('idle');
  // "Is the pill on screen?" — a pill that was never created is not on screen,
  // which is the zero-DOM-cost hidden case, not a missing assertion target.
  const pillShown = () => !!_aiTriggerEl && _aiTriggerEl.style.display === 'flex';
  const swEl = () => find(live, '#ai-assistant-panel-trigger-toggle');
  const readSwitch = () => {
    const s = swEl();
    return {
      checked: s.getAttribute('aria-checked'),
      text: find(s, '.ai-assistant-panel-toggle-text').textContent,
      locked: s.disabled === true,
      ariaDisabled: s.getAttribute('aria-disabled'),
    };
  };
  const openPanel = () => {
    _aiPanelEl = fakeEl();
    _aiPanelEl.style.display = 'flex';
    _applyPanelTriggerVisibility('open');
  };
  const minimizePanel = () => {
    _aiPanelEl.style.display = 'none';
    _aiPanelEl.setAttribute('data-minimized', 'true');
    _applyPanelTriggerVisibility('minimized');
  };
  const closePanel = () => {
    _aiPanelEl.removeAttribute('data-minimized');
    _applyPanelTriggerVisibility('idle');
  };

  // ---- switch OFF, then open → minimize → close ---------------------------
  reset({ panelStartMinimized: true });
  _setPanelTriggerPref(false);
  mount();
  t('OFF: initial switch unchecked', readSwitch().checked, 'false');
  initIdle();
  t('OFF: initial pill not on screen', pillShown(), false);
  t('OFF: hidden costs no DOM', _aiTriggerEl, null);

  openPanel();
  t('OFF+open: pill not on screen', pillShown(), false);
  t('OFF+open: switch still shows the preference', readSwitch().checked, 'false');
  t('OFF+open: switch interactive', readSwitch().locked, false);

  minimizePanel();
  t('OFF+minimized: pill shown', pillDisplay(), 'flex');
  t('OFF+minimized: switch SYNCED to shown', readSwitch().checked, 'true');
  t('OFF+minimized: switch text synced', readSwitch().text, 'Shown');
  t('OFF+minimized: switch locked', readSwitch().locked, true);
  t('OFF+minimized: locked for AT', readSwitch().ariaDisabled, 'true');
  t('OFF+minimized: lock explained',
    /Close the panel to change this/.test(swEl().getAttribute('aria-label')), true);
  t('OFF+minimized: section records lock', live.dataset.panelTriggerLocked, 'true');

  // The lock must refuse a programmatic flip too, not just a pointer click.
  t('OFF+minimized: setter refused', setPanelTriggerVisible(false), false);
  t('OFF+minimized: pill still shown after refusal', pillDisplay(), 'flex');

  closePanel();
  t('OFF+closed: pill hidden again', pillDisplay(), 'none');
  t('OFF+closed: switch back to the preference', readSwitch().checked, 'false');
  t('OFF+closed: switch text restored', readSwitch().text, 'Hidden');
  t('OFF+closed: switch interactive again', readSwitch().locked, false);
  t('OFF+closed: aria-disabled cleared', readSwitch().ariaDisabled, null);
  t('OFF+closed: preference survived the round trip', _panelTriggerVisible(), false);

  // ---- switch ON, same walk ----------------------------------------------
  reset({ panelStartMinimized: true });
  mount();
  initIdle();
  t('ON: initial switch checked', readSwitch().checked, 'true');
  t('ON: initial pill shown', pillShown(), true);

  openPanel();
  t('ON+open: pill hidden by the panel', pillDisplay(), 'none');
  t('ON+open: switch still reports the preference', readSwitch().checked, 'true');

  minimizePanel();
  t('ON+minimized: pill shown', pillDisplay(), 'flex');
  t('ON+minimized: switch checked', readSwitch().checked, 'true');
  // Locked even without divergence: un-checking here would strand the transcript.
  t('ON+minimized: switch locked', readSwitch().locked, true);

  closePanel();
  t('ON+closed: pill returns', pillDisplay(), 'flex');
  t('ON+closed: switch checked', readSwitch().checked, 'true');
  t('ON+closed: switch interactive', readSwitch().locked, false);

  // ---- reader flips the switch while the panel is open --------------------
  reset({ panelStartMinimized: true });
  mount();
  initIdle();
  openPanel();
  t('open: flip accepted', setPanelTriggerVisible(false), true);
  t('open: switch follows the new preference', readSwitch().checked, 'false');
  t('open: pill stays hidden while open', pillDisplay(), 'none');
  closePanel();
  t('open-flip takes effect on close', pillShown(), false);
  t('open-flip switch consistent after close', readSwitch().checked, 'false');

  // ---- invariant sweep: pill and switch agree unless the panel is open ----
  reset({ panelStartMinimized: true });
  mount();
  let drift = 0;
  for (const pref of [true, false]) {
    _setPanelTriggerPref(pref);
    for (const state of ['idle', 'minimized']) {
      _aiPanelEl = fakeEl();
      if (state === 'minimized') {
        _aiPanelEl.style.display = 'none';
        _aiPanelEl.setAttribute('data-minimized', 'true');
      } else {
        _aiPanelEl = null;
      }
      const shown = _applyPanelTriggerVisibility(state);
      if ((readSwitch().checked === 'true') !== shown) drift++;
    }
  }
  t('no pill/switch drift across states', drift, 0);

  // ---- dropdown built while already minimized renders locked --------------
  reset({ panelStartMinimized: true });
  _setPanelTriggerPref(false);
  _aiPanelEl = fakeEl();
  _aiPanelEl.style.display = 'none';
  _aiPanelEl.setAttribute('data-minimized', 'true');
  mount();
  t('late-built switch renders synced', readSwitch().checked, 'true');
  t('late-built switch renders locked', readSwitch().locked, true);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
