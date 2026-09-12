// Run 84 regression: mobile/tablet composer must be first-class, panel-bounded,
// touch-discoverable, safe-area aware, and consistent with Run 83 panel-width fit.
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

// Accessibility/discoverability: no aria-hidden ancestor around a focusable mic options trigger.
t('mic expand wrapper is not aria-hidden', !panel.includes("micExpandWrapper.setAttribute('aria-hidden', 'true')"));
t('mic options trigger remains labelled', panel.includes("micExpandBtn.setAttribute('aria-label', 'Microphone options')"));
t('coarse/no-hover devices expose mic options without hover', /@media \(hover: none\), \(pointer: coarse\)[\s\S]*?\.ai-assistant-mic-expand-wrapper,[\s\S]*?width:\s*2\.5rem;[\s\S]*?\.ai-assistant-mic-expand-btn\s*\{[\s\S]*?opacity:\s*1;/.test(css));
t('touch path disables desktop-only invisible hover bridge', /@media \(hover: none\), \(pointer: coarse\)[\s\S]*?\.ai-assistant-mic-wrapper::before\s*\{[\s\S]*?display:\s*none;/.test(css));

// Dense touch target policy and shortcut de-cluttering.
t('footer touch controls receive 40px minimum targets', /@media \(hover: none\), \(pointer: coarse\)[\s\S]*?\.ai-assistant-panel-footer-btn,[\s\S]*?min-width:\s*2\.5rem;[\s\S]*?min-height:\s*2\.5rem;/.test(css));
t('attachment and hold rows keep 44px touch rows', /\.ai-assistant-mic-popup-row--hold,[\s\S]*?\.ai-assistant-panel-attach-menu-item[\s\S]*?min-height:\s*44px;/.test(css));
t('touch attachment menu hides keyboard-only shortcut chrome', /\.ai-assistant-panel-attach-menu-shortcut\s*\{[\s\S]*?display:\s*none;/.test(css));

// Safe-area support belongs to the composer itself.
t('footer respects left safe area', css.includes('padding-left: max(0.75rem, env(safe-area-inset-left, 0px));'));
t('footer respects right safe area', css.includes('padding-right: max(0.75rem, env(safe-area-inset-right, 0px));'));
t('footer respects bottom home-indicator safe area', css.includes('padding-bottom: max(0.625rem, env(safe-area-inset-bottom, 0px));'));

// One responsive authority: model representation follows panel fit, not device class.
const mobileStart = css.indexOf('@media (max-width: 575px) {');
const mobileEnd = mobileStart >= 0 ? css.indexOf('\n}', mobileStart) : -1;
const mobile575 = mobileStart >= 0 && mobileEnd >= 0 ? css.slice(mobileStart, mobileEnd + 2) : '';
t('575px viewport rule no longer forces composer model picker compact', !mobile575.includes('.ai-assistant-panel-inline-model-picker'));
t('panel compact state remains canonical model-picker contraction', /data-footer-compact="true"\][\s\S]*?ai-assistant-panel-inline-model-picker/.test(css));
t('no-ResizeObserver fallback listens to window resize', panel.includes("window.addEventListener('resize', _fallbackResponsiveSync"));
t('fallback also follows visualViewport resize for mobile keyboard/orientation', panel.includes("window.visualViewport.addEventListener('resize', _fallbackResponsiveSync"));
t('fallback still delegates to panel-fit controller', panel.includes('_syncFooterActionFit(w);') && panel.includes('_syncSheetHeaderOverflow(w);'));

// Attachment menu uses real anchor geometry rather than body-width-only sizing.
t('attach positioning measures wrap geometry', panel.includes('wrapRect = attachWrap.getBoundingClientRect'));
t('attach positioning derives right-side budget from anchored left', panel.includes('rightBudget') && panel.includes('bodyRect.right - edgeInset - (wrapRect.left + localLeft)'));
t('attach menu writes bounded local left', panel.includes("attachMenu.style.left = localLeft + 'px'"));
t('attach fallback clears dynamic left', panel.includes("attachMenu.style.left = ''"));

// Microphone popup is panel-body bounded in both dimensions.
t('mic popup has shared fit hook', panel.includes('var _syncMicPopupFit = function () {}'));
t('mic popup fit uses panel body geometry', panel.includes("document.getElementById('ai-assistant-panel-body')") && panel.includes('boundaryWidth'));
t('mic popup target width is capped at 330', panel.includes('var popupWidth = Math.min(330, Math.floor(boundaryWidth));'));
t('mic popup right edge uses free space through send side', panel.includes('wrapperRect.right - targetRight'));
t('mic popup fit applies vertical capacity', panel.includes("micPopup.style.maxHeight = Math.max(96, availableHeight) + 'px'"));
t('opening mic popup synchronizes fit first', panel.includes('if (open) _syncMicPopupFit();'));
t('panel ResizeObserver path also refreshes popup fit', fit.includes('_syncMicPopupFit();'));
t('popup CSS has no fixed 330px min-width', !css.includes('min-width: 330px'));
t('popup CSS has defensive viewport max and scrolling', /\.ai-assistant-mic-popup\s*\{[\s\S]*?max-width:\s*calc\(100vw - 1rem\);[\s\S]*?overflow-y:\s*auto;[\s\S]*?overscroll-behavior:\s*contain;/.test(css));

// Compact VU keeps all semantic zones instead of clipping the red end.
t('VU strip can shrink as a flex child', /\.ai-assistant-mic-level-bars\s*\{[\s\S]*?width:\s*100%;[\s\S]*?min-width:\s*0;/.test(css));
t('compact popup tightens VU gaps instead of reducing bars', /\.ai-assistant-mic-popup\[data-compact="true"\] \.ai-assistant-mic-level-bars\s*\{[\s\S]*?gap:\s*0\.5px;/.test(css));
t('JS marks compact popup by measured width', panel.includes("micPopup.setAttribute('data-compact', popupWidth < 315 ? 'true' : 'false')"));

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
