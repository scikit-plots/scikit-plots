import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) {
  if (cond) passed++;
  else { failed++; console.error(`FAIL ${name}`); }
}

const start = src.indexOf('function _positionAnchoredPopupWithinPanelBody(popup, options)');
const end = src.indexOf('function _positionPinnedFeedbackPopups()', start);
ok(start >= 0 && end > start, 'shared panel-body positioning helper exists');
const helper = src.slice(start, end);

function runCase({kind='feedback', bodyRect, anchorRect, naturalWidth, naturalHeight}) {
  naturalWidth ??= kind === 'more' ? 160 : 230;
  naturalHeight ??= kind === 'more' ? 120 : 180;
  const style = {};
  const attrs = kind === 'more' ? {'data-open': 'true'} : {'data-pinned': 'true'};
  const wrapperClass = kind === 'more'
    ? '.ai-assistant-panel-bubble-action-more'
    : '.ai-assistant-fbk-float-wrapper';
  const body = { getBoundingClientRect: () => ({...bodyRect}) };
  const wrapper = {
    offsetWidth: anchorRect.width,
    offsetHeight: anchorRect.height,
    getBoundingClientRect: () => ({...anchorRect})
  };
  const popup = {
    style,
    getAttribute: (k) => attrs[k] ?? null,
    setAttribute: (k,v) => { attrs[k] = String(v); },
    closest: (sel) => sel === wrapperClass ? wrapper : null,
    getBoundingClientRect: () => {
      const minW = Number.parseFloat(style.minWidth) || 0;
      const maxW = Number.parseFloat(style.maxWidth) || naturalWidth;
      const maxH = Number.parseFloat(style.maxHeight) || naturalHeight;
      return {
        width: Math.max(minW, Math.min(naturalWidth, maxW)),
        height: Math.min(naturalHeight, maxH)
      };
    }
  };
  const context = {
    document: { getElementById: (id) => id === 'ai-assistant-panel-body' ? body : null },
    isFinite
  };
  vm.createContext(context);
  vm.runInContext(`${helper}; this.positionFeedback = _positionFbkPopupWithinPanelBody; this.positionMore = _positionBubbleMoreMenuWithinPanelBody;`, context);
  (kind === 'more' ? context.positionMore : context.positionFeedback)(popup);
  const width = Math.max(
    Number.parseFloat(style.minWidth) || 0,
    Math.min(naturalWidth, Number.parseFloat(style.maxWidth) || naturalWidth)
  );
  const height = Math.min(naturalHeight, Number.parseFloat(style.maxHeight) || naturalHeight);
  const left = anchorRect.left + (Number.parseFloat(style.left) || 0);
  const top = anchorRect.top + (Number.parseFloat(style.top) || 0);
  return {style, attrs, left, top, width, height};
}

const body = {left: 100, top: 100, right: 500, bottom: 500, width: 400, height: 400};
let r = runCase({bodyRect: body, anchorRect: {left: 390, top: 430, right: 470, bottom: 460, width: 80, height: 30}});
ok(r.attrs['data-placement'] === 'top', 'feedback prefers top when top fully fits');

r = runCase({bodyRect: body, anchorRect: {left: 390, top: 115, right: 470, bottom: 145, width: 80, height: 30}});
ok(r.attrs['data-placement'] === 'bottom', 'feedback flips below near panel-body top edge');

r = runCase({bodyRect: {left: 0, top: 0, right: 700, bottom: 240, width: 700, height: 240},
  anchorRect: {left: 250, top: 105, right: 300, bottom: 135, width: 50, height: 30}, naturalHeight: 190});
ok(r.attrs['data-placement'] === 'right', 'feedback uses right side when vertical space is tight');

r = runCase({bodyRect: {left: 0, top: 0, right: 700, bottom: 240, width: 700, height: 240},
  anchorRect: {left: 500, top: 105, right: 550, bottom: 135, width: 50, height: 30}, naturalHeight: 190});
ok(r.attrs['data-placement'] === 'left', 'feedback uses left side when right side cannot fit');

const tiny = {left: 10, top: 20, right: 180, bottom: 130, width: 170, height: 110};
r = runCase({bodyRect: tiny, anchorRect: {left: 80, top: 65, right: 110, bottom: 90, width: 30, height: 25}, naturalWidth: 230, naturalHeight: 220});
ok(r.left >= tiny.left + 8 - 0.01, 'feedback clamps left edge inside panel body');
ok(r.top >= tiny.top + 8 - 0.01, 'feedback clamps top edge inside panel body');
ok(r.left + r.width <= tiny.right - 8 + 0.01, 'feedback clamps right edge inside panel body');
ok(r.top + r.height <= tiny.bottom - 8 + 0.01, 'feedback clamps bottom edge inside panel body');
ok(r.style.maxHeight === '94px', 'feedback caps height to panel-body inner height');
ok(r.attrs['data-boundary-positioned'] === 'true', 'feedback marks adaptive boundary positioning');

// The answer-bubble More menu uses the exact same boundary/side-selection core.
r = runCase({kind: 'more', bodyRect: body,
  anchorRect: {left: 150, top: 430, right: 205, bottom: 460, width: 55, height: 30}});
ok(r.attrs['data-placement'] === 'top', 'More menu prefers top when room exists');
ok(Math.abs(r.left - 150) < 0.01, 'More menu preserves historical left/start alignment');

r = runCase({kind: 'more', bodyRect: body,
  anchorRect: {left: 150, top: 112, right: 205, bottom: 142, width: 55, height: 30}});
ok(r.attrs['data-placement'] === 'bottom', 'More menu flips below near panel-body top edge');

r = runCase({kind: 'more', bodyRect: {left: 0, top: 0, right: 520, bottom: 170, width: 520, height: 170},
  anchorRect: {left: 180, top: 72, right: 230, bottom: 102, width: 50, height: 30}, naturalHeight: 145});
ok(r.attrs['data-placement'] === 'right', 'More menu can move right in a short panel');

r = runCase({kind: 'more', bodyRect: tiny,
  anchorRect: {left: 80, top: 65, right: 110, bottom: 90, width: 30, height: 25}, naturalWidth: 210, naturalHeight: 220});
ok(r.left >= tiny.left + 8 - 0.01 && r.left + r.width <= tiny.right - 8 + 0.01,
  'More menu clamps horizontally inside panel body');
ok(r.top >= tiny.top + 8 - 0.01 && r.top + r.height <= tiny.bottom - 8 + 0.01,
  'More menu clamps vertically inside panel body');
ok(r.style.maxHeight === '94px', 'More menu owns bounded internal height');
ok(r.attrs['data-boundary-positioned'] === 'true', 'More menu marks adaptive boundary positioning');

const fbkBuilder = src.slice(src.indexOf('function _buildFbkFloat('), src.indexOf('function _buildFeedbackBlock(', src.indexOf('function _buildFbkFloat(')));
ok(fbkBuilder.includes('_ensureFeedbackPopupBoundaryObservers();'), 'feedback open path installs shared boundary observers');
ok(fbkBuilder.includes('_positionFbkPopupWithinPanelBody(popup);'), 'feedback open path positions immediately');

const moreBuilder = src.slice(src.indexOf('function _buildBubbleMore('), src.indexOf('// ── Page-help prompt', src.indexOf('function _buildBubbleMore(')));
ok(moreBuilder.includes('_ensureFeedbackPopupBoundaryObservers();'), 'More open path reuses shared boundary observers');
ok(moreBuilder.includes('_positionBubbleMoreMenuWithinPanelBody(menu);'), 'More open path positions immediately');

ok(src.includes("'.ai-assistant-panel-bubble-action-more-menu[data-open=\"true\"]'"),
  'shared coordinator discovers all open More menus');
ok(src.includes("body.addEventListener('scroll', _schedulePinnedFeedbackPopupPosition, { passive: true })"),
  'panel-body scroll repositions both popup families');
ok(src.includes("window.addEventListener('resize', _schedulePinnedFeedbackPopupPosition, { passive: true })"),
  'window resize repositions both popup families');
ok(src.includes('new ResizeObserver(_schedulePinnedFeedbackPopupPosition)'),
  'panel-body resize observer repositions both popup families');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
