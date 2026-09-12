// Run 83 regression: footer responsiveness belongs to the resizable panel,
// not the browser viewport. Decorative waveform capacity must yield before
// stable attach/model/mic/send controls overflow.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
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

const panel = extract('createAIPanel');
const fit = extract('_syncFooterActionFit');
const count = extract('_computeSoundbarBarCount');
const viz = extract('_startVizLoops');

t('panel has a fit-based footer controller', panel.includes('function _syncFooterActionFit(panelWidth)'));
t('shared panel ResizeObserver drives footer fit', panel.includes('_syncFooterActionFit(w);'));
t('footer fit is also established after DOM connection', panel.includes('document.body.appendChild(panel);') && panel.includes('_syncFooterActionFit(panel.getBoundingClientRect'));
t('fit decision is based on footer client width, not viewport width', fit.includes('footerActions.clientWidth') && !fit.includes('window.innerWidth'));
t('wide picker width is cached before contraction to prevent oscillation', fit.includes('_footerWidePickerWidth') && fit.includes('!wasCompact'));
t('fit controller reserves hidden mic reveal width', fit.includes('micRevealW') && fit.includes('_footerElementWidth(micExpandBtn)'));
t('soundbar contributes zero required control width', fit.includes('The soundbar intentionally contributes zero required width'));
t('fit controller has hysteresis for manual edge dragging', fit.includes('_FOOTER_FIT_HYSTERESIS_PX'));
t('compact state is stored on the panel', fit.includes("panel.setAttribute('data-footer-compact'"));
t('desktop compact state mirrors mobile picker contraction', /data-footer-compact="true"\][\s\S]*?ai-assistant-panel-inline-model-picker[\s\S]*?min-width:\s*unset;/.test(css));
t('desktop compact state exposes model icon', /data-footer-compact="true"\][\s\S]*?ai-assistant-panel-inline-picker-icon\s*\{[\s\S]*?display:\s*inline-flex;/.test(css));
t('desktop compact state hides verbose picker pieces', /data-footer-compact="true"\][\s\S]*?ai-assistant-panel-inline-picker-dot,[\s\S]*?ai-assistant-panel-inline-picker-label,[\s\S]*?ai-assistant-panel-inline-picker-chev[\s\S]*?display:\s*none;/.test(css));
t('legacy mobile media fallback remains available', css.includes('@media (max-width: 575px)'));
t('right footer cluster explicitly allows flex shrink', /\.ai-assistant-panel-footer-actions-right\s*\{[\s\S]*?min-width:\s*0;[\s\S]*?max-width:\s*100%;/.test(css));
t('decorative soundbar is shrinkable and zero-min-width', /\.ai-assistant-footer-soundbar\s*\{[\s\S]*?flex:\s*0 1 auto;[\s\S]*?min-width:\s*0;[\s\S]*?overflow:\s*hidden;/.test(css));
t('soundbar count resolves panel width first', count.includes("document.getElementById('ai-assistant-panel')") && count.includes('getBoundingClientRect'));
t('soundbar count accepts explicit panel width', count.includes('Number(panelWidth)'));
t('soundbar tiers no longer key primarily from viewport breakpoints', count.includes('if (w < 330) return 8;') && count.includes('if (w < 450) return 12;') && count.includes('if (w < 720) return 16;'));
t('viewport is only a no-panel fallback', count.includes('window.innerWidth') && count.indexOf("document.getElementById('ai-assistant-panel')") < count.indexOf('window.innerWidth'));
t('recording loop re-evaluates capacity after live panel resize', viz.includes('var _nextSbCount = _computeSoundbarBarCount();'));
t('recording loop rebuilds bars when tier changes', viz.includes('_rebuildSoundbarBars(soundbarEl, _sbCount)'));
t('recording loop resizes ring buffer with bar count', viz.includes('_soundbarHeights.slice(-_sbCount)') && viz.includes('_soundbarHeights.unshift(_SOUNDBAR_MIN_H)'));

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
