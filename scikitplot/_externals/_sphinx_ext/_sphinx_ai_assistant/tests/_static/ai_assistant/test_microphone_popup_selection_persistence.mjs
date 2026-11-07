// Run 74 regression: microphone routing rows must remain clickable inside the
// explicitly pinned popup and their selection must survive popup refreshes.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
function t(name, got, want) {
  if (got === want) pass++;
  else { fail++; console.log(`FAIL ${name}\n  got: ${JSON.stringify(got)}\n want: ${JSON.stringify(want)}`); }
}

// Root-cause guard: a button-opened popup is explicitly owned by the user and
// focusout must not synchronously revoke data-pinned before click dispatch.
t('explicit pin state exists', src.includes('var _micPopupExplicitPinned = false;'), true);
t('pointer interaction promotes explicit pin', src.includes('_setMicPopupPinned(true, true);'), true);
t('focusout preserves explicit pin', src.includes('if (_micPopupExplicitPinned) return;'), true);
t('focusout is deferred', src.includes('_micPopupFocusCloseTimer = setTimeout(function ()'), true);
t('focusout no longer directly writes pinned false',
  /micPopup\.addEventListener\('focusout',[\s\S]{0,1400}micPopup\.setAttribute\('data-pinned', 'false'\)/.test(src), false);

// Device-row interaction must commit through the canonical selection function.
// Run 75 delegates from the stable radiogroup because enumeration refreshes
// replace the row buttons; closest() handles nested label/SVG click targets.
t('stable list interaction binder exists', src.includes('function _bindMicDeviceListInteractions(listEl)'), true);
t('device list uses delegated closest matching', src.includes("element.closest('.ai-assistant-mic-device-item')"), true);
t('delegated click stops propagation', /listEl\.addEventListener\('click',[\s\S]{0,700}e\.stopPropagation\(\)/.test(src), true);
t('delegated click commits selected id', /listEl\.addEventListener\('click',[\s\S]{0,900}_selectMicDevice\(item\.getAttribute\('data-device-id'\)/.test(src), true);
t('delegated click focuses chosen row', /listEl\.addEventListener\('click',[\s\S]{0,1100}item\.focus/.test(src), true);

// Selection still persists as a preference and refresh reads _micDeviceId,
// rather than overwriting it with browser default.
t('non-default selection stored in sessionStorage',
  src.includes("sessionStorage.setItem('ai-assistant-mic-device-id', _micDeviceId)"), true);
t('refresh derives checked row from selected preference', src.includes("var effectiveId = _micDeviceId || 'default';"), true);
t('refresh does not reset missing selection to default',
  /if \(!selectedExists[^}]+\)[\s\S]{0,500}_setMicDevice\(['\"]default/.test(src), false);


// Execute the real setter with storage/release stubs to guard the other half
// of the reported symptom: reopening must not collapse a committed route back
// to the default route.
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
let stored = new Map();
globalThis._micDeviceId = '';
globalThis._releaseMicPinTrack = () => {};
globalThis._releaseMicWarmStream = () => {};
globalThis._syncMicDeviceUI = () => {};
globalThis.sessionStorage = {
  setItem(k, v) { stored.set(k, String(v)); },
  removeItem(k) { stored.delete(k); },
};
const setDevice = (0, eval)('(' + extract('_setMicDevice') + ')');
setDevice('communications');
t('communications remains selected preference', globalThis._micDeviceId, 'communications');
t('communications persists across popup refresh', stored.get('ai-assistant-mic-device-id'), 'communications');
setDevice('physical-abc');
t('physical id remains selected preference', globalThis._micDeviceId, 'physical-abc');
t('physical id persists across popup refresh', stored.get('ai-assistant-mic-device-id'), 'physical-abc');
setDevice('default');
t('default normalizes to empty system route', globalThis._micDeviceId, '');
t('default removes explicit device persistence', stored.has('ai-assistant-mic-device-id'), false);

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
