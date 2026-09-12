// Run 90 regression: Detailed feedback reveal must retire the floating popup and
// bring a useful leading slice of the inline form into the PANEL body viewport.
// It must never scroll the host page or blindly center a tall form.
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

const revealSrc = extract('_revealDetailedFeedbackInPanelBody');
const fbkSrc = extract('_buildFbkFloat');
const blockSrc = extract('_buildFeedbackBlock');

t('reveal helper is panel-body scoped', revealSrc.includes("document.getElementById('ai-assistant-panel-body')"));
t('reveal helper does not use window/page scroll', !revealSrc.includes('window.scrollTo') && !revealSrc.includes('scrollIntoView'));
t('reveal waits two animation frames for display/layout settlement', /requestAnimationFrame\(function \(\) \{[\s\S]*?requestAnimationFrame\(_afterLayout\)/.test(revealSrc));
t('reveal reserves answer context with bounded useful slice', revealSrc.includes('usableHeight * 0.55') && revealSrc.includes('Math.min(220') && revealSrc.includes('Math.max(96'));
t('reveal clamps target to body scroll range', revealSrc.includes('body.scrollHeight - body.clientHeight') && revealSrc.includes('Math.min(maxScroll'));
t('reveal honors reduced motion', revealSrc.includes("prefers-reduced-motion: reduce") && revealSrc.includes("behavior: reduced ? 'auto' : 'smooth'"));
t('reveal focuses first form control without causing a second scroll', revealSrc.includes("'.ai-assistant-panel-feedback-btn, .ai-assistant-panel-feedback-text, '") && revealSrc.includes('focus({ preventScroll: true })'));

t('Detailed feedback computes explicit opening state', fbkSrc.includes("var opening = !fbBlock.classList.contains('ai-assistant-panel-feedback--revealed')"));
t('opening Detailed feedback closes floating feedback popup', /if \(opening\) \{[\s\S]*?_dismissFbkPopup\(\);/.test(fbkSrc));
t('opening Detailed feedback invokes reveal helper', /if \(opening\) \{[\s\S]*?_revealDetailedFeedbackInPanelBody\(fbBlock, true\)/.test(fbkSrc));
t('Hide path keeps popup available and repositions it', /else \{[\s\S]*?_schedulePinnedFeedbackPopupPosition\(\);/.test(fbkSrc));
t('feedback block exposes group semantics', blockSrc.includes("wrap.setAttribute('role', 'group')"));
t('feedback block is labelled by its visible question', blockSrc.includes("wrap.setAttribute('aria-labelledby', q.id)") && blockSrc.includes("q.id = 'ai-assistant-panel-feedback-question-' + answerIndex"));

// Runtime geometry probe for the helper itself.
const reveal = eval('(' + revealSrc + ')');
function runCase({ bodyRect, blockRect, scrollTop=0, scrollHeight=1000, clientHeight=300, reduced=false }) {
  const calls = [];
  let focused = 0;
  const first = { focus(opts) { focused++; calls.push(['focus', opts && opts.preventScroll]); } };
  const block = {
    getBoundingClientRect() { return blockRect; },
    querySelector() { return first; }
  };
  const body = {
    scrollTop,
    scrollHeight,
    clientHeight,
    contains(x) { return x === block; },
    getBoundingClientRect() { return bodyRect; },
    scrollTo(opts) { calls.push(['scrollTo', opts.top, opts.behavior]); this.scrollTop = opts.top; }
  };
  const raf = [];
  globalThis.document = { getElementById(id) { return id === 'ai-assistant-panel-body' ? body : null; } };
  globalThis.window = { matchMedia() { return { matches: reduced }; } };
  globalThis.requestAnimationFrame = (fn) => { raf.push(fn); return raf.length; };
  reveal(block, true);
  while (raf.length) raf.shift()();
  return { calls, body, focused };
}

const already = runCase({
  bodyRect: { top: 0, bottom: 300, height: 300 },
  blockRect: { top: 145, bottom: 245, height: 100 },
  scrollTop: 80
});
t('already-useful form position does not scroll', !already.calls.some(c => c[0] === 'scrollTo'));
t('already-useful form still receives logical focus', already.focused === 1);

const below = runCase({
  bodyRect: { top: 0, bottom: 300, height: 300 },
  blockRect: { top: 280, bottom: 580, height: 300 },
  scrollTop: 100
});
const belowScroll = below.calls.find(c => c[0] === 'scrollTo');
t('below-viewport form scrolls panel body', !!belowScroll);
t('below-viewport reveal scrolls downward but not to transcript bottom', belowScroll && belowScroll[1] > 100 && belowScroll[1] < 700);
t('normal reveal uses smooth panel-local motion', belowScroll && belowScroll[2] === 'smooth');

const reducedCase = runCase({
  bodyRect: { top: 0, bottom: 240, height: 240 },
  blockRect: { top: 225, bottom: 525, height: 300 },
  scrollTop: 50,
  reduced: true
});
const reducedScroll = reducedCase.calls.find(c => c[0] === 'scrollTo');
t('reduced-motion reveal still makes form visible', !!reducedScroll);
t('reduced-motion reveal disables smooth animation', reducedScroll && reducedScroll[2] === 'auto');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
