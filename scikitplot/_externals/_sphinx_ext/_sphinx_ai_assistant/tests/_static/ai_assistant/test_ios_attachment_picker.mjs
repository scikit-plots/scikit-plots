import fs from 'node:fs';
import assert from 'node:assert/strict';

const js = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let passed = 0;
function ok(cond, msg) { assert.ok(cond, msg); passed++; }

ok(js.includes("attachInput.className = 'ai-assistant-panel-file-input'"), 'native file input gets dedicated visually-hidden class');
ok(!js.includes('attachInput.hidden = true;'), 'file input no longer uses hidden/display-none semantics');
ok(js.includes("typeof attachInput.showPicker === 'function'"), 'native showPicker is preferred when available');
ok(js.includes('attachInput.showPicker();'), 'showPicker is invoked under the activation path');
ok(js.includes('attachInput.click();'), 'synthetic click remains compatibility fallback');

const changeStart = js.indexOf("attachInput.addEventListener('change', function () {");
const changeEnd = js.indexOf("        });", changeStart);
const changeBlock = js.slice(changeStart, changeEnd + 10);
const snapshotPos = changeBlock.indexOf('Array.prototype.slice.call(attachInput.files || [])');
const clearPos = changeBlock.indexOf("attachInput.value = '';");
const queuePos = changeBlock.indexOf('_queueComposerFiles(files)');
ok(changeStart >= 0, 'file-input change handler exists');
ok(snapshotPos >= 0, 'change handler snapshots FileList to a plain array');
ok(clearPos > snapshotPos, 'FileList snapshot happens before resetting input value');
ok(queuePos > clearPos, 'staging receives the stable snapshot after reset');
ok(changeBlock.includes('if (files.length) _queueComposerFiles(files);'), 'cancelled picker does not create an empty staging job');

ok(css.includes('.ai-assistant-panel-file-input {'), 'visually-hidden native input CSS exists');
ok(css.includes('clip-path: inset(50%) !important;') && css.includes('opacity: 0 !important;'), 'native input stays rendered but visually clipped');

// Model the WebKit failure mode: clearing input.value empties the live FileList.
const fileA = {name: 'iphone-photo.heic'};
const fileB = {name: 'notes.txt'};
let live = [fileA, fileB];
const fakeInput = {
  get files() { return live; },
  set value(v) { if (v === '') live = []; },
};
const stable = Array.prototype.slice.call(fakeInput.files || []);
fakeInput.value = '';
ok(stable.length === 2 && fakeInput.files.length === 0, 'snapshot survives WebKit-style live FileList invalidation');

console.log(`Run 76 iOS attachment picker: ${passed}/${passed} passed`);
