// Regression harness: microphone rows must select immediately.
//
// Run 58 coupled radio selection to a temporary getUserMedia verification
// promise. On real browsers/pseudo devices that promise can fail or resolve
// differently, leaving the visible radio stuck on "Default" even though the
// user clicked another row. Selection is a routing preference; capture
// verification belongs to recording time.
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

let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) pass++;
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
};

const body = extract('_selectMicDevice');
t('selection no longer performs capture verification', body.includes('getUserMedia('), false);
t('selection uses canonical setter', body.includes('_setMicDevice(requestedId)'), true);
t('selection explains deferred verification', /verified when recording starts/.test(body), true);

// Execute the actual function with tiny dependency stubs. The critical
// guarantee is synchronous commit before this call returns.
globalThis._isListening = false;
globalThis._speechStartPending = false;
let selected = null;
let notice = null;
globalThis._setMicDevice = (id) => { selected = id; };
globalThis.showNotification = (msg) => { notice = msg; };
const select = (0, eval)('(' + body + ')');

select('communications');
t('communications pseudo-device commits immediately', selected, 'communications');
t('selection status is immediate', /Microphone selected/.test(notice || ''), true);

selected = null;
select('physical-device-123');
t('physical device commits immediately', selected, 'physical-device-123');

selected = null;
select('default');
t('default route commits immediately', selected, 'default');

// Recording must still verify the selected route and fail closed. The picker
// fix must never reintroduce silent fallback to another physical microphone.
const toggle = extract('_toggleSpeechRecognition');
t('recording uses selected-device constraints',
  toggle.includes('getUserMedia(_micConstraintsForDevice(_micDeviceId))'), true);
t('recording reports unavailable selection',
  toggle.includes('Selected microphone is unavailable'), true);
t('recording does not advertise fallback', /using browser default/i.test(toggle), false);

// UI state remains ordinary radio semantics and the whole selected row receives
// a persistent visual treatment, not only the tiny check icon.
t('rows retain role radio', src.includes("item.setAttribute('role', 'radio')"), true);
t('checked state is synchronized centrally',
  src.includes("items[i].setAttribute('aria-checked', active ? 'true' : 'false')"), true);

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
