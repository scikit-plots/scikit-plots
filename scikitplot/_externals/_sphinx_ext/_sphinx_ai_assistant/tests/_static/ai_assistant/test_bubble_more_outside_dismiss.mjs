// Run 91 regression: answer-level More dismisses on any outside pointer/tap,
// including mobile taps that do not cause focusout. One delegated controller
// owns all bubbles; nested More -> Home state closes atomically with the root.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
function t(name, got, want=true) {
  if (got === want) pass++;
  else { fail++; console.log(`FAIL ${name}\n  got: ${JSON.stringify(got)}\n want: ${JSON.stringify(want)}`); }
}
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

const closeFn = extract('_closeBubbleMoreWrapper');
const ensureFn = extract('_ensureBubbleMoreOutsideDismissal');
const moreFn = extract('_buildBubbleMore');

t('singleton outside-dismiss flag exists', src.includes('var _bubbleMoreOutsideDismissBound = false'));
t('each builder only ensures shared controller', moreFn.includes('_ensureBubbleMoreOutsideDismissal();'));
t('builder does not register per-answer document pointer listener', !moreFn.includes("document.addEventListener('pointerdown'"));
t('outside controller queries only open root menus', ensureFn.includes('.ai-assistant-panel-bubble-action-more-menu[data-open="true"]'));
t('modern path uses pointerdown', ensureFn.includes("document.addEventListener('pointerdown', _dismissBubbleMoreOnOutsidePointer, true)"));
t('outside pointer listener is capture phase', /pointerdown'[\s\S]*?_dismissBubbleMoreOnOutsidePointer, true\)/.test(ensureFn));
t('inside wrapper is exempt from dismissal', ensureFn.includes('wrapper.contains(e.target)'));
t('old WebKit touch fallback exists', ensureFn.includes("document.addEventListener('touchstart', _dismissBubbleMoreOnOutsidePointer, true)"));
t('non-PointerEvent mouse fallback exists', ensureFn.includes("document.addEventListener('mousedown', _dismissBubbleMoreOnOutsidePointer, true)"));
t('Escape dismissal is shared', ensureFn.includes("document.addEventListener('keydown'"));
t('Escape restores trigger focus without scroll', ensureFn.includes('rootToggle.focus({ preventScroll: true })'));

t('central close resets root data-open', closeFn.includes("rootMenu.setAttribute('data-open', 'false')"));
t('central close resets root aria-expanded', closeFn.includes("rootToggle.setAttribute('aria-expanded', 'false')"));
t('central close resets nested disclosure aria', closeFn.includes("secondaryToggle.setAttribute('aria-expanded', 'false')"));
t('central close resets nested disclosure data-open', closeFn.includes("secondaryBody.setAttribute('data-open', 'false')"));
t('central close hides nested disclosure', closeFn.includes('secondaryBody.hidden = true'));
t('toggle close uses central close helper', moreFn.includes('_closeBubbleMoreWrapper(wrapper, false);'));
t('focusout uses same central close helper', /focusout[\s\S]*?_closeBubbleMoreWrapper\(wrapper, false\)/.test(moreFn));

function makeRuntime(pointerEvents=true) {
  const listeners = {};
  const document = {
    activeElement: null,
    addEventListener(type, fn, capture) {
      (listeners[type] ||= []).push({fn, capture});
    },
    querySelectorAll() { return []; },
  };
  const window = pointerEvents ? {PointerEvent: function PointerEvent(){}} : {};
  const build = new Function('document', 'window', `
    var _bubbleMoreOutsideDismissBound = false;
    ${closeFn}
    ${ensureFn}
    return {
      ensure: _ensureBubbleMoreOutsideDismissal,
      close: _closeBubbleMoreWrapper
    };
  `);
  return { ...build(document, window), document, window, listeners };
}

function fixture() {
  const state = {rootOpen: 'true', rootExpanded: 'true', secondaryExpanded: 'true', secondaryOpen: 'true', hidden: false, focused: 0};
  const inside = {};
  const rootMenu = {
    parentElement: null,
    setAttribute(k,v) { if (k === 'data-open') state.rootOpen = v; },
  };
  const rootToggle = {
    setAttribute(k,v) { if (k === 'aria-expanded') state.rootExpanded = v; },
    focus() { state.focused++; },
  };
  const secondaryToggle = {
    setAttribute(k,v) { if (k === 'aria-expanded') state.secondaryExpanded = v; },
  };
  const secondaryBody = {
    setAttribute(k,v) { if (k === 'data-open') state.secondaryOpen = v; },
    get hidden() { return state.hidden; },
    set hidden(v) { state.hidden = v; },
  };
  const wrapper = {
    contains(target) { return target === inside; },
    querySelector(sel) {
      if (sel === '.ai-assistant-panel-bubble-action-more-menu') return rootMenu;
      if (sel === '.ai-assistant-panel-bubble-action--more-toggle') return rootToggle;
      if (sel === '.ai-assistant-panel-bubble-action--secondary-toggle') return secondaryToggle;
      if (sel === '.ai-assistant-panel-bubble-action-more-secondary') return secondaryBody;
      return null;
    },
  };
  rootMenu.parentElement = wrapper;
  return {state, inside, rootMenu, rootToggle, wrapper};
}

// Modern PointerEvent path: one listener no matter how many bubbles call ensure.
{
  const rt = makeRuntime(true);
  rt.ensure(); rt.ensure(); rt.ensure();
  t('singleton registers one pointerdown controller', (rt.listeners.pointerdown || []).length, 1);
  t('singleton registers one Escape controller', (rt.listeners.keydown || []).length, 1);
  t('modern path does not duplicate touch fallback', (rt.listeners.touchstart || []).length, 0);
  const f = fixture();
  rt.document.querySelectorAll = () => [f.rootMenu];
  rt.listeners.pointerdown[0].fn({target: {}});
  t('outside pointer closes root menu at runtime', f.state.rootOpen, 'false');
  t('outside pointer collapses root trigger at runtime', f.state.rootExpanded, 'false');
  t('outside pointer collapses nested More at runtime', f.state.secondaryExpanded, 'false');
  t('outside pointer hides nested body at runtime', f.state.hidden, true);

  // Reset and prove a tap inside the owning wrapper does not self-dismiss.
  Object.assign(f.state, {rootOpen:'true', rootExpanded:'true', secondaryExpanded:'true', secondaryOpen:'true', hidden:false});
  rt.listeners.pointerdown[0].fn({target: f.inside});
  t('inside pointer keeps owning menu open', f.state.rootOpen, 'true');
  t('inside pointer keeps nested disclosure intact', f.state.secondaryOpen, 'true');

  // Escape closes and restores focus to the owner.
  rt.document.activeElement = f.inside;
  rt.listeners.keydown[0].fn({key:'Escape'});
  t('Escape closes root menu at runtime', f.state.rootOpen, 'false');
  t('Escape restores owner trigger focus', f.state.focused, 1);
}

// Two open menus: tapping inside B is outside A, so A closes while B survives.
{
  const rt = makeRuntime(true); rt.ensure();
  const a = fixture(), b = fixture();
  rt.document.querySelectorAll = () => [a.rootMenu, b.rootMenu];
  rt.listeners.pointerdown[0].fn({target: b.inside});
  t('tap another More closes previously open menu', a.state.rootOpen, 'false');
  t('tap another More preserves target menu until its click handler runs', b.state.rootOpen, 'true');
}

// Legacy no-PointerEvent path keeps immediate touch dismissal.
{
  const rt = makeRuntime(false); rt.ensure();
  t('legacy path registers touchstart capture', (rt.listeners.touchstart || []).length, 1);
  t('legacy path registers mousedown capture', (rt.listeners.mousedown || []).length, 1);
  t('legacy path skips pointerdown listener', (rt.listeners.pointerdown || []).length, 0);
  const f = fixture();
  rt.document.querySelectorAll = () => [f.rootMenu];
  rt.listeners.touchstart[0].fn({target:{}});
  t('legacy touchstart closes root menu', f.state.rootOpen, 'false');
}

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
