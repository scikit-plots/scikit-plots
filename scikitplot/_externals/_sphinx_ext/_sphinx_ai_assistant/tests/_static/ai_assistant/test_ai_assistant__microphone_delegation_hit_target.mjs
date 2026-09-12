// Run 75 regression: the dynamically rebuilt microphone list must use one
// stable delegated interaction layer and the informational capability text must
// never intercept the last radio row.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let pass = 0, fail = 0;
function t(name, got, want) {
  if (got === want) pass++;
  else { fail++; console.log(`FAIL ${name}\n  got: ${JSON.stringify(got)}\n want: ${JSON.stringify(want)}`); }
}
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

t('list binds stable interaction controller', src.includes('_bindMicDeviceListInteractions(devList);'), true);
t('delegation uses closest row selector', src.includes("element.closest('.ai-assistant-mic-device-item')"), true);
t('delegation verifies containment', src.includes('return item && listEl.contains(item) ? item : null;'), true);
const refreshBody = extract('_refreshMicDeviceList');
t('rows no longer get per-refresh click listener', refreshBody.includes("item.addEventListener('click'"), false);
t('rows no longer get per-refresh keydown listener', refreshBody.includes("item.addEventListener('keydown'"), false);
t('radiogroup keeps aria checked state', src.includes("item.setAttribute('aria-checked', dev.deviceId === effectiveId ? 'true' : 'false')"), true);
t('central sync keeps aria checked state', src.includes("items[i].setAttribute('aria-checked', active ? 'true' : 'false')"), true);
t('scrollbar gutter stable', /\.ai-assistant-mic-device-list\s*\{[\s\S]*?scrollbar-gutter:\s*stable;/.test(css), true);
t('capability status cannot hit-test', /\.ai-assistant-mic-device-capability\s*\{[\s\S]*?pointer-events:\s*none;/.test(css), true);
t('device list paints above capability status', /\.ai-assistant-mic-device-list\s*\{[\s\S]*?z-index:\s*1;/.test(css), true);

const listeners = Object.create(null);
const attrs = new Map();
const items = [];
const list = {
  getAttribute(k) { return attrs.has(k) ? attrs.get(k) : null; },
  setAttribute(k, v) { attrs.set(k, String(v)); },
  addEventListener(type, fn) { listeners[type] = fn; },
  contains(el) { return items.includes(el); },
  querySelectorAll(sel) { return sel === '.ai-assistant-mic-device-item' ? items : []; },
};
function makeItem(id) {
  return {
    nodeType: 1,
    id,
    focused: false,
    closest(sel) { return sel === '.ai-assistant-mic-device-item' ? this : null; },
    getAttribute(k) { return k === 'data-device-id' ? id : null; },
    focus() { this.focused = true; },
  };
}
const first = makeItem('default');
const second = makeItem('communications');
items.push(first, second);
const nested = { nodeType: 1, closest(sel) { return sel === '.ai-assistant-mic-device-item' ? second : null; } };
let selected = null;
globalThis._selectMicDevice = id => { selected = id; };
globalThis._finishMicDeviceListPointerInteraction = () => {};
const bind = (0, eval)('(' + extract('_bindMicDeviceListInteractions') + ')');
bind(list);
t('binder marks list once', attrs.get('data-interactions-bound'), 'true');
t('click listener registered once', typeof listeners.click, 'function');
t('keydown listener registered once', typeof listeners.keydown, 'function');

let prevented = false, stopped = false;
listeners.click({
  target: nested,
  preventDefault() { prevented = true; },
  stopPropagation() { stopped = true; },
});
t('nested child click selects enclosing device', selected, 'communications');
t('delegated click prevents button default', prevented, true);
t('delegated click stops ancestor dismissal', stopped, true);
t('delegated click focuses selected row', second.focused, true);

const clickBefore = listeners.click;
bind(list);
t('rebinding same list is idempotent', listeners.click === clickBefore, true);

selected = null;
first.focused = false;
second.focused = false;
let keyPrevented = false;
listeners.keydown({
  target: first,
  key: 'ArrowRight',
  preventDefault() { keyPrevented = true; },
});
t('ArrowRight selects next radio', selected, 'communications');
t('ArrowRight focuses next radio', second.focused, true);
t('ArrowRight prevents page movement', keyPrevented, true);

selected = null;
listeners.click({
  target: { nodeType: 1, closest() { return null; } },
  preventDefault() {}, stopPropagation() {},
});
t('non-device click ignored', selected, null);

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
