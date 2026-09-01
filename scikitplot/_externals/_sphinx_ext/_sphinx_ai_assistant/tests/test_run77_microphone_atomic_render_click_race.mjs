// Run 77 / B96 regression: microphone refresh must never destroy a pressed
// radio row between pointerdown and click, and normal device sets must not be
// clipped by the old 130px list cap.
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

const refresh = extract('_refreshMicDeviceList');
const bind = extract('_bindMicDeviceListInteractions');
const finish = extract('_finishMicDeviceListPointerInteraction');

t('maintenance warning documents destructive-clear bug', src.includes('DO NOT reintroduce a clear-then-async-rebuild pattern here.'), true);
t('maintenance warning documents pointerdown-to-click loss', src.includes('mouseup/click no longer targets the same element'), true);
t('refresh body contains no destructive innerHTML clear', refresh.includes("listEl.innerHTML = ''"), false);
t('refresh generation guards stale async results', refresh.includes('generation !== _micDeviceListRenderGeneration'), true);
t('refresh builds replacement off DOM', refresh.includes('document.createDocumentFragment()'), true);
t('refresh commits atomically', refresh.includes('listEl.replaceChildren(fragment)'), true);
t('existing rows stay mounted while enumeration is pending', refresh.includes('if (!listEl.children.length)'), true);
t('refresh queues during active pointer press', refresh.includes('_micDeviceListRefreshQueued = true'), true);
t('completed enumeration defers commit during active press', refresh.includes('_micDeviceListDeferredCommit = commitRender'), true);
t('radiogroup marks pointerdown transaction', bind.includes("addEventListener('pointerdown'"), true);
t('click finishes pointer transaction only after selection', bind.indexOf('_selectMicDevice(') < bind.indexOf('_finishMicDeviceListPointerInteraction('), true);
t('pointerup fallback is deferred to next task', bind.includes("addEventListener('pointerup'") && bind.includes('setTimeout(function ()'), true);
t('queued refresh discards older deferred render', finish.includes('_micDeviceListDeferredCommit = null'), true);
t('old 130px cap removed', /max-height:\s*130px/.test(css), false);
t('normal list is content-sized', /\.ai-assistant-mic-device-list\s*\{[\s\S]*?height:\s*auto;/.test(css), true);
t('long lists use viewport-aware cap', /max-height:\s*min\(24rem,\s*46vh\);/.test(css), true);
t('list-level layout transition disabled', /\.ai-assistant-mic-device-list\s*\{[\s\S]*?transition:\s*none;/.test(css), true);
t('coarse pointer rows meet 44px target', /@media\s*\(pointer:\s*coarse\)[\s\S]*?\.ai-assistant-mic-device-item\s*\{[\s\S]*?min-height:\s*44px;/.test(css), true);

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
