// Run 173 T47 - the preview is a window: movable, resizable, minimisable.
//
// The header carried `data-drag-handle="true"` with no behaviour behind it
// anywhere in the file. This drives the geometry model directly, because the
// property that matters -- a window can never be moved somewhere it cannot be
// moved back from -- is arithmetic, not markup.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
function extract(n){const st=src.indexOf('function '+n+'(');if(st<0)throw new Error('missing '+n);
 let d=0,b=false,q='',e=false,l=false,bl=false,re=false,cls=false,prev='';
 for(let i=st;i<src.length;i++){const c=src[i],x=src[i+1];
  if(l){if(c==='\n')l=false;continue;} if(bl){if(c==='*'&&x==='/'){bl=false;i++;}continue;}
  if(re){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c==='['){cls=true;continue;}if(c===']'){cls=false;continue;}if(c==='/'&&!cls)re=false;continue;}
  if(q){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c===q)q='';continue;}
  if(c==='/'&&x==='/'){l=true;i++;continue;} if(c==='/'&&x==='*'){bl=true;i++;continue;}
  if(c==='/'){ if(/[(,=:[!&|?{};+\-*%^~<>]/.test(prev)||/\breturn$/.test(src.slice(Math.max(0,i-6),i))){re=true;cls=false;} continue; }
  if(c==='"'||c==="'"||c==='`'){q=c;continue;}
  if(c==='{'){d++;b=true;} else if(c==='}'){ if(--d===0&&b) return src.slice(st,i+1); }
  if(!/\s/.test(c)) prev=c;
 } throw new Error('unterminated '+n);}
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};

const clamp = new Function('window','document','_PREVIEW_MIN_W','_PREVIEW_MIN_H','_PREVIEW_KEEP_VISIBLE',
  extract('_previewViewport') + '\n' + extract('_previewClamp') + '\nreturn _previewClamp;')(
  { innerWidth: 1000, innerHeight: 800 }, { documentElement: {} }, 320, 180, 64);

// ── A window must always be recoverable ──────────────────────────────────
const farLeft = clamp({ left: -5000, top: -5000, width: 600, height: 400 });
ok(farLeft.left + farLeft.width >= 64,'dragged far left, part of the window stays on screen');
ok(farLeft.top >= 0,'it can never be dragged above the top edge, where the header would be unreachable');
const farRight = clamp({ left: 99999, top: 99999, width: 600, height: 400 });
ok(farRight.left <= 1000 - 64,'dragged far right, at least a grabbable strip remains');
ok(farRight.top <= 800 - 64,'and it cannot be dropped below the bottom edge');

// ── Size bounds ──────────────────────────────────────────────────────────
const tiny = clamp({ left: 10, top: 10, width: 1, height: 1 });
ok(tiny.width >= 320 && tiny.height >= 180,'a window cannot be resized smaller than its controls');
const huge = clamp({ left: 0, top: 0, width: 99999, height: 99999 });
ok(huge.width <= 1000 && huge.height <= 800,'nor larger than the viewport');
const normal = clamp({ left: 100, top: 120, width: 640, height: 420 });
ok(normal.left === 100 && normal.top === 120 && normal.width === 640 && normal.height === 420,'an ordinary geometry passes through untouched');

// ── The centring transform must be dropped when placing explicitly ───────
const apply = extract('_previewApplyGeom');
ok(apply.includes("dialog.style.transform = 'none';"),'explicit placement drops the centring transform');
ok(apply.includes("dialog.style.transform = '';"),'and restores it when geometry is cleared');
ok(apply.includes('_previewClamp(geom)'),'every write goes through the clamp, not just drags');

// ── Modes ────────────────────────────────────────────────────────────────
const setMode = extract('_previewSetMode');
ok(setMode.includes('_previewWindow.restore = _previewWindow.geom || _previewReadGeom(dialog);'),'maximising remembers where the window was');
ok(/mode === 'maximized' \? 'Restore preview size' : 'Maximise preview'/.test(setMode),'the control says which way it will go');
ok(setMode.includes("dialog.setAttribute('data-window-mode', mode)"),'the mode is exposed for styling');
ok(extract('_previewResetMode').includes("_previewWindow.mode = 'normal';"),'reopening never restores a minimised window as a title bar');

// ── Drag mechanics ───────────────────────────────────────────────────────
const drag = extract('_previewBindDrag');
// The dblclick handler contains the same substring, so a bare search passed
// while the pointerdown guard had been removed. Both guards are named.
ok(drag.includes("if (ev.button !== 0 || (ev.target && ev.target.closest &&\n                    ev.target.closest('button'))) return;"),'pressing a header button does not start a drag');
ok((drag.match(/ev\.target\.closest\('button'\)/g) || []).length === 2,'both the drag and the double-click guard exclude buttons');
ok(drag.includes('setPointerCapture'),'the drag survives the pointer leaving the header');
ok(drag.includes('releasePointerCapture'),'and the capture is released');
ok(drag.includes("header.addEventListener('pointercancel', end);"),'a cancelled pointer ends the drag rather than sticking');
ok(drag.includes("header.addEventListener('dblclick'"),'double-clicking the title bar toggles maximise');
ok(src.includes("if (_previewWindow.geom) _previewApplyGeom(dialog, _previewWindow.geom);"),'a viewport resize re-clamps, so a shrunk window stays reachable');

// ── File actions menu ────────────────────────────────────────────────────
//
// The chevron acts on the file; the three controls beside it act on the window
// showing it. Two groups, because merging them would make Close read as a peer
// of Download.
ok(src.includes("_buildOverflowMenu('File actions', function () {"),'the file menu is built from the shared menu, not a fourth copy');
ok(src.includes("'ai-assistant-panel-attachment-preview-menu-btn', ICONS.chevronDown)"),'it uses a chevron rather than the overflow glyph');
ok(extract('_buildOverflowMenu').includes('btn.innerHTML = iconSvg || ICONS.overflowV;'),'the icon is optional, so the three existing callers keep their glyph');
// Items are resolved per open, so a menu built before a file was loaded still
// acts on the file currently shown.
ok(/_buildOverflowMenu\('File actions', function \(\) \{\s*var it = _attachmentPreviewState\.item;/.test(src),'its items read the item at open time, not at build time');
ok(src.includes('if (!it) return [];'),'with no item open the menu offers nothing rather than acting on null');
ok(src.includes("_artifactNameSlugPreservingExtension(it.name || '')"),'the download filename passes the same sanitiser as every other download');
ok(src.indexOf('controls.appendChild(docMenu);') < src.indexOf('controls.appendChild(minBtn);'),'the file menu precedes the window controls');
// Rotation is driven from aria-expanded, so the glyph cannot disagree with the
// state assistive technology is told.
ok(/\.ai-assistant-panel-attachment-preview-menu-btn\[aria-expanded="true"\] svg[\s\S]{0,80}?rotate\(180deg\)/.test(css),'the chevron rotates from its announced state');
ok(/prefers-reduced-motion: reduce[\s\S]{0,200}?preview-menu-btn svg[\s\S]{0,60}?transition:\s*none/.test(css),'the rotation is opt-out');
// Asserted against the single touch block rather than a distance-limited
// lookahead: the menu button now shares the selector list with the window
// controls, so the earlier window-count regex no longer spans it.
const touchBlock = (css.match(/@media \(max-width: 640px\), \(pointer: coarse\) \{[\s\S]*?\n\}/) || [''])[0];
ok(/preview-menu-btn/.test(touchBlock) && /2\.5rem/.test(touchBlock),'it grows to a fingertip target on touch');
ok((css.match(/@media \(max-width: 640px\), \(pointer: coarse\)/g) || []).length === 1,'there is one touch block for the preview, not two read together');
ok(/\[data-bs-theme="dark"\] \.ai-assistant-panel-attachment-preview-menu-btn/.test(css),'and states its own dark-theme colours');

// ── Title bar order ──────────────────────────────────────────────────────
//
// Minimise and maximise were appended at their construction site, which ran
// before the heading was added, so the bar rendered "minimise maximise title
// close": the window controls split around the thing they act on, and Close
// separated from its two peers.
const headingAt = src.indexOf('header.appendChild(heading);');
const controlsAt = src.indexOf('header.appendChild(controls);');
ok(headingAt > 0 && controlsAt > headingAt,'the title comes before the window controls');
ok(!/header\.appendChild\(minBtn\)|header\.appendChild\(maxBtn\)|header\.appendChild\(close\)/.test(src),'no control is appended to the header directly, where order depends on construction site');
const order = ['controls.appendChild(minBtn);','controls.appendChild(maxBtn);','controls.appendChild(close);']
  .map(function (frag) { return src.indexOf(frag); });
ok(order.every(function (i) { return i > 0; }),'all three controls live in the group');
ok(order[0] < order[1] && order[1] < order[2],'minimise, maximise, close, in that order');
// A long filename must never displace Close.
ok(/\.ai-assistant-panel-attachment-preview-heading \{[^}]*min-width:\s*0/.test(css),'the heading can shrink');
ok(/\.ai-assistant-panel-attachment-preview-title,[\s\S]{0,140}?text-overflow:\s*ellipsis/.test(css),'and truncates instead of pushing the controls off the edge');
ok(/\.ai-assistant-panel-attachment-preview-controls \{[^}]*margin-inline-start:\s*auto/.test(css),'the group sits at the trailing edge');

// ── Presentation ─────────────────────────────────────────────────────────
ok(/\.ai-assistant-panel-attachment-preview \{[^}]*resize:\s*both/.test(css),'the window resizes from its corner');
ok(/touch-action:\s*none/.test(css),'a touch drag moves the window instead of scrolling the page');
ok(/\[data-window-mode="minimized"\][^{]*\{[^}]*resize:\s*none/.test(css),'a minimised window cannot be resized into an unusable strip');
ok(/forced-colors: active[\s\S]{0,300}?preview-window-btn[\s\S]{0,80}?ButtonText/.test(css),'the controls survive forced-colours mode');

// ── The minimum size must not exceed the display ─────────────────────────
//
// Written as `max(320, min(width, viewport))` the floor wins on a 280px screen
// and the window is sized wider than the display it is on: the minimum meant
// to keep the controls usable pushes them off the edge instead.
const clampAt = (vw, vh) => new Function('window','document','_PREVIEW_MIN_W','_PREVIEW_MIN_H','_PREVIEW_KEEP_VISIBLE',
  extract('_previewViewport') + '\n' + extract('_previewClamp') + '\nreturn _previewClamp;')(
  { innerWidth: vw, innerHeight: vh }, { documentElement: {} }, 320, 180, 64);
[[280, 400], [320, 568], [200, 300]].forEach(function (v) {
  const g = clampAt(v[0], v[1])({ left: 0, top: 0, width: 200, height: 200 });
  ok(g.width <= v[0] && g.height <= v[1],
     'at ' + v[0] + 'x' + v[1] + ' the window never exceeds the viewport');
});
ok(clampAt(1000, 800)({ left: 0, top: 0, width: 10, height: 10 }).width === 320,
   'the minimum still applies when the viewport can hold it');

// ── Touch and small screens ──────────────────────────────────────────────
// A movable, corner-resizable window is a pointer idea: no hover reveals the
// resize corner, the corner is smaller than a fingertip, and a finger-dragged
// window is easy to strand.
const touch = (css.match(/@media \(max-width: 640px\), \(pointer: coarse\) \{[\s\S]*?\n\}/) || [''])[0];
ok(touch.length > 0,'there is a touch and small-screen mode');
ok(/pointer: coarse/.test(css),'keyed on input type too, so a touch laptop is covered at any width');
ok(/resize:\s*none/.test(touch),'resizing is off where the corner cannot be hit');
ok(/touch-action:\s*auto/.test(touch),'the header scrolls the page again instead of dragging');
ok(/width:\s*100%\s*!important/.test(touch) && /height:\s*100%\s*!important/.test(touch),'the preview fills the screen rather than floating');
ok(/2\.5rem/.test(touch),'controls grow to a fingertip target');
ok(/data-window-mode="minimized"/.test(touch),'minimise still works there, docked to the bottom');

// ── Both themes ──────────────────────────────────────────────────────────
// The hover mix falls back to #fff when a site defines no surface token, which
// on a dark page is a near-white wash under a light glyph.
ok(/\[data-bs-theme="dark"\] \.ai-assistant-panel-attachment-preview-window-btn/.test(css),'the dark theme states its own control colours');
ok(/rgba\(255, 255, 255, 0\.12\)/.test(css),'its hover does not depend on a token the host may not set');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
