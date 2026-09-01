import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let pass=0, fail=0;
function ok(v,n){ if(v){pass++;}else{fail++;console.error('FAIL '+n);} }
function section(a,b){ const i=src.indexOf(a), j=src.indexOf(b,i+1); return i>=0&&j>i ? src.slice(i,j) : ''; }
const helper=section('function _wirePayloadFormatTabs', "var contributionInspectFormat = 'jsonl'");
const render=section('function _renderPayloadCode', 'var sheet = document.createElement');
const sheet=section('function _buildDatasetContributionSheet()', 'function _buildKeyboardShortcutsSheet()');

ok(helper.includes("tablist.setAttribute('role', 'tablist')"), 'shared switcher owns tablist semantics');
ok(helper.includes("tab.setAttribute('role', 'tab')"), 'format buttons are tabs');
ok(helper.includes("setAttribute('aria-controls'"), 'tabs reference their panels');
ok(helper.includes("setAttribute('role', 'tabpanel')"), 'format views are tabpanels');
ok(helper.includes("setAttribute('aria-labelledby'"), 'panels reference their tabs');
ok(helper.includes("key === 'ArrowRight'") && helper.includes("key === 'ArrowLeft'"), 'arrow keys navigate tabs');
ok(helper.includes("key === 'Home'") && helper.includes("key === 'End'"), 'home/end navigate tabs');
ok(sheet.includes('savedStructureBtn.tabIndex = jsonOn ? 0 : -1'), 'contribution uses roving tabindex');
ok(sheet.includes('feedbackSavedStructureBtn.tabIndex = jsonOn ? 0 : -1'), 'feedback uses roving tabindex');
ok(!sheet.includes("setAttribute('aria-pressed', jsonOn"), 'role=tab no longer mixes aria-pressed state');

ok(sheet.includes("'⎘ Copy JSON'") && sheet.includes("'⎘ Copy JSONL'"), 'copy button names active format');
ok(sheet.includes("'↓ Download JSON'") && sheet.includes("'↓ Download JSONL'"), 'download button names active format');
ok(sheet.includes("toUpperCase() + ' · ' + _formatByteSize"), 'size badge identifies active format');
ok(sheet.includes("setAttribute('aria-live', 'polite')"), 'format-size changes are announced politely');

ok(render.includes("pre.textContent = ''"), 'syntax renderer clears with textContent');
ok(render.includes('span.textContent = value'), 'syntax tokens use inert textContent');
ok(!render.includes('innerHTML'), 'syntax renderer never injects HTML');
ok(render.includes("ai-assistant-panel-json-token--"), 'syntax renderer emits semantic token classes');
ok(sheet.includes('_renderPayloadCode(feedbackPreview'), 'feedback JSONL uses safe syntax renderer');
ok(sheet.includes('_renderPayloadCode(feedbackSavedStructurePreview'), 'feedback JSON uses safe syntax renderer');
ok(sheet.includes('_renderPayloadCode(preview'), 'contribution JSONL uses safe syntax renderer');
ok(sheet.includes('_renderPayloadCode(savedStructurePreview'), 'contribution JSON uses safe syntax renderer');
ok(sheet.includes('_renderPayloadCode(telemetrySavedPreview'), 'telemetry JSONL uses safe syntax renderer');

ok(!src.includes("getElementById('ai-assistant-feedback-persist-toggle')"), 'obsolete Endpoint telemetry DOM hook removed');

console.log(`${pass} passed, ${fail} failed`);
if(fail) process.exit(1);
