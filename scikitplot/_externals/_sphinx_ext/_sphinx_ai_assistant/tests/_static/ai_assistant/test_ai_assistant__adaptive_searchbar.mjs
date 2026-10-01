// Behavioural/static contract for the host-aware standalone AI search bar.
//
//   node test_ai_assistant__adaptive_searchbar.mjs ai-assistant.js ai-assistant.css
//
// The test extracts the shipped pure presentation functions and evaluates them
// against dependency-free fakes.  It also asserts the DOM/CSS accessibility
// contract for the icon-only rail state.
import fs from 'node:fs';
import path from 'node:path';

const jsPath = process.argv[2];
const cssPath = process.argv[3] || path.join(path.dirname(jsPath), 'ai-assistant.css');
const src = fs.readFileSync(jsPath, 'utf8');
const css = fs.readFileSync(cssPath, 'utf8');

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
function t(name, got, want) {
  if (got === want) pass++;
  else {
    fail++;
    console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`);
  }
}

globalThis._SEARCH_BAR_ICON_ONLY_MAX_PX = 96;
let CFG = {};
globalThis._cfg = () => CFG;

const hostCollapsed = (0, eval)('(' + extract('_searchBarHostIsCollapsed') + ')');
globalThis._searchBarHostIsCollapsed = hostCollapsed;
const syncPresentation = (0, eval)('(' + extract('_syncSearchBarPresentation') + ')');

function fakeHost(width, { matches = false, closest = false, invalid = false } = {}) {
  return {
    clientWidth: width,
    getBoundingClientRect() { return { width }; },
    matches() { if (invalid) throw new Error('bad selector'); return matches; },
    closest() { if (invalid) throw new Error('bad selector'); return closest ? this : null; },
  };
}

function fakeBar() {
  const classes = new Set();
  const attrs = {};
  const launchAttrs = {};
  const launcher = { setAttribute(k, v) { launchAttrs[k] = String(v); } };
  return {
    classes,
    attrs,
    launchAttrs,
    classList: {
      toggle(k, on) { if (on) classes.add(k); else classes.delete(k); },
    },
    setAttribute(k, v) { attrs[k] = String(v); },
    querySelector(sel) { return sel === '.ai-assistant-searchbar-launcher' ? launcher : null; },
  };
}

// Explicit theme state outranks width.
t('custom host selector collapses wide host',
  hostCollapsed(fakeHost(260, { matches: true }), { searchBarCollapsedSelector: '.rail.is-collapsed' }), true);
t('custom ancestor selector collapses wide host',
  hostCollapsed(fakeHost(260, { closest: true }), { searchBarCollapsedSelector: '.shell.is-collapsed' }), true);

// Width is the portable fallback and hidden/zero-width hosts do not become a
// visible icon state merely because they are not measurable.
t('64px rail collapses by width', hostCollapsed(fakeHost(64), {}), true);
t('96px rail boundary collapses', hostCollapsed(fakeHost(96), {}), true);
t('97px host stays input', hostCollapsed(fakeHost(97), {}), false);
t('wide host stays input', hostCollapsed(fakeHost(280), {}), false);
t('zero-width hidden host does not force icon', hostCollapsed(fakeHost(0), {}), false);
t('invalid custom selector fails over to width',
  hostCollapsed(fakeHost(64, { invalid: true }), { searchBarCollapsedSelector: '[' }), true);

// Presentation is one state machine over the same DOM.
let bar = fakeBar();
t('wide default presentation', syncPresentation(bar, fakeHost(280), {
  searchBarAdaptive: true, searchBarMini: false,
}), 'full');
t('wide default has no icon class', bar.classes.has('ai-assistant-searchbar--icon-only'), false);

bar = fakeBar();
t('wide mini preference remains compact input', syncPresentation(bar, fakeHost(280), {
  searchBarAdaptive: true, searchBarMini: true,
}), 'mini');

bar = fakeBar();
t('adaptive narrow host becomes icon', syncPresentation(bar, fakeHost(64), {
  searchBarAdaptive: true, searchBarMini: false,
}), 'icon');
t('icon class applied', bar.classes.has('ai-assistant-searchbar--icon-only'), true);
t('icon launcher gets typing-specific label', bar.launchAttrs['aria-label'], 'Open AI Assistant and start typing');

bar = fakeBar();
t('adaptive off pins base full input', syncPresentation(bar, fakeHost(64), {
  searchBarAdaptive: false, searchBarMini: false,
}), 'full');
t('adaptive off does not add icon class', bar.classes.has('ai-assistant-searchbar--icon-only'), false);

// Shipped DOM/behaviour contract.
const build = extract('_buildSearchBar');
const open = extract('_openSearchPanelForTyping');
const bind = extract('_bindAdaptiveSearchBar');
t('leading icon is a real button',
  /document\.createElement\('button'\)/.test(build) && /ai-assistant-searchbar-launcher/.test(build), true);
t('launcher click reuses the shared go path',
  /launcher\.addEventListener\('click',[\s\S]*?_go\(\)/.test(build), true);
t('icon-only launcher never auto-submits a hidden draft',
  /contains\('ai-assistant-searchbar--icon-only'\)[\s\S]*?_openSearchPanelForTyping\(\);[\s\S]*?return;[\s\S]*?_go\(\)/.test(build), true);
t('empty launcher path opens panel rather than doing nothing',
  /var panelInput = _openSearchPanelForTyping\(\);[\s\S]*?if \(!q\) return/.test(build), true);
t('panel launcher focuses composer',
  /getElementById\('ai-assistant-panel-input'\)/.test(open) && /panelInput\.focus/.test(open), true);
t('input is a native search field and remains Enter-submit capable', /inp\.type = 'search'/.test(build) && /e\.key === 'Enter'/.test(build), true);
t('input is bounded', /inp\.maxLength = 4096/.test(build), true);
t('adaptive controller observes actual host size',
  /new ResizeObserver\(sync\)/.test(bind) && /observe\(host\)/.test(bind), true);
t('adaptive controller observes host/ancestor collapse attributes',
  /new MutationObserver\(sync\)/.test(bind) && /while \(observeNode && observeNode\.nodeType === 1\)/.test(bind) && /attributeFilter: \['class', 'style', 'open', 'aria-expanded'\]/.test(bind), true);

// CSS geometry/accessibility contract.
t('normal bar keeps 40px minimum touch row', /\.ai-assistant-searchbar \{[\s\S]*?min-height:\s*2\.5rem;/s.test(css), true);
t('icon-only rail is exactly 40px square',
  /\.ai-assistant-searchbar--icon-only,[\s\S]*?width:\s*2\.5rem;[\s\S]*?height:\s*2\.5rem;/s.test(css), true);
t('icon-only rail hides input and shortcut hint',
  /\.ai-assistant-searchbar--icon-only input,[\s\S]*?ai-assistant-searchbar-kbd-hint[\s\S]*?display:\s*none;/s.test(css), true);
t('launcher has visible keyboard focus',
  /\.ai-assistant-searchbar \.ai-assistant-searchbar-launcher:focus-visible[\s\S]*?outline:/s.test(css), true);
t('icon-only launcher owns the full 40px rail target',
  /\.ai-assistant-searchbar--icon-only \.ai-assistant-searchbar-launcher[\s\S]*?width:\s*2\.5rem;[\s\S]*?height:\s*2\.5rem/s.test(css), true);

console.log(`\n${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
