// Run 173 T8 - working files on the wire, and stale-response protection.
//
// A file continued across turns needs its revision and digest, not just its
// bytes. Without that binding a slow answer built from r3 commits as r5 over
// the reader's own r4, and the file they keep is one neither party authored.
import fs from 'node:fs';
import crypto from 'node:crypto';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
function extract(n){const st=src.indexOf('function '+n+'(');if(st<0)throw new Error('missing '+n);
 let d=0,b=false,q='',e=false,l=false,bl=false;
 for(let i=st;i<src.length;i++){const c=src[i],x=src[i+1];
  if(l){if(c==='\n')l=false;continue;} if(bl){if(c==='*'&&x==='/'){bl=false;i++;}continue;}
  if(q){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c===q)q='';continue;}
  if(c==='/'&&x==='/'){l=true;i++;continue;} if(c==='/'&&x==='*'){bl=true;i++;continue;}
  if(c==='"'||c==="'"||c==='`'){q=c;continue;} if(c==='{'){d++;b=true;}else if(c==='}'&&--d===0&&b)return src.slice(st,i+1);}
 throw new Error('unterminated '+n);}
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};

const ledger = Object.create(null);
const api = new Function('_generatedArtifactLedger','_generatedArtifactIsAvailable',
  [extract('_workingFileCapsParse'), extract('_artifactContentRevision'), extract('_workingFileBindingIsCurrent'), extract('_turnWorkingFileBinding'),
   'return {caps:_workingFileCapsParse, current:_workingFileBindingIsCurrent, binding:_turnWorkingFileBinding};'].join('\n'))(
  ledger, e => !!e && typeof e.content === 'string');

// ── Capability negotiation ────────────────────────────────────────────────
ok(api.caps({max_files:4,max_file_chars:48000,max_total_chars:96000,digest:'sha256'}) !== null,'published bounds are accepted');
ok(api.caps({max_files:4,max_file_chars:48000,max_total_chars:96000,digest:'md5'}) === null,'an unknown digest yields no working files');
ok(api.caps({max_files:0,max_file_chars:1,max_total_chars:1}) === null,'zero bounds are rejected');
ok(api.caps(null) === null,'absent capability yields no working files');
const clamped = api.caps({max_files:9999,max_file_chars:9e9,max_total_chars:9e9,digest:'sha256'});
ok(clamped.maxFiles <= 8 && clamped.maxFileChars <= 200000,'server bounds are clamped, never expanded');

// ── Staleness ─────────────────────────────────────────────────────────────
ledger['docs/index.rst'] = { key:'docs/index.rst', path:'docs/index.rst', revision:4, content:'now' };
ok(api.current({key:'docs/index.rst',revision:4}) === true,'an answer built from the current revision still owns the file');
ok(api.current({key:'docs/index.rst',revision:3}) === false,'an answer built from a superseded revision does not');
ok(api.current({key:'docs/absent.rst',revision:3}) === true,'a file no longer tracked does not block its own answer');
ok(api.current(null) === true,'an unbound answer is never treated as stale');
ok(api.binding({workingFileBindings:[{path:'a'},{path:'b'}]},'b').path === 'b','the binding for a path is found by path');
ok(api.binding({},'a') === null,'a turn with no bindings yields none');

// ── Digest agreement with the server field ────────────────────────────────
const digest = crypto.createHash('sha256').update('x=1\n').digest('hex');
ok(/^[0-9a-f]{64}$/.test(digest),'sha256 hex is the shape the server validates');
ok(src.includes("crypto.subtle.digest('SHA-256'"),'the browser digest is SHA-256');
ok(src.includes("if (!digest) { skipped++; continue; }"),'a file that cannot be digested is not sent unbound');

// ── Contract wiring ───────────────────────────────────────────────────────
ok(src.includes('bodyObj.working_files = wfPlan.files.map'),'working files ride the validated envelope');
ok(src.includes('var wfPlan = await _workingFilesForRequest(proxyCaps && proxyCaps.workingFiles);'),'selection uses the negotiated bounds');
ok(src.includes('_workingFileContinuations[entry.key] = Date.now();'),'only explicitly continued files become eligible');
ok(src.includes('Object.keys(_workingFileContinuations)'),'selection reads the explicit opt-in registry, not the whole ledger');
ok(src.includes('_markStaleAnswerCandidate(pre, path, binding, current);') && src.includes('return;'),'a stale answer is surfaced rather than registered');
// The guard is only a guard if control actually leaves before registration.
// Asserting "no _registerGeneratedArtifact nearby" is not that assertion --
// the very next statement after the guarded block IS the registration, which
// is correct. Assert the exit instead.
// The guard condition itself must be present: asserting only that a stale
// branch EXISTS passes even when the branch is unreachable.
ok(src.includes('if (binding && !_workingFileBindingIsCurrent(binding)) {'),
   'registration is gated on the binding still being current');
const staleAt = src.indexOf('_markStaleAnswerCandidate(pre, path, binding, current);');
const regAt = src.indexOf('_registerGeneratedArtifact({', staleAt);
const between = src.slice(staleAt, regAt);
ok(staleAt > 0 && regAt > staleAt,'the guard precedes registration in the same function');
ok(/\breturn;/.test(between),'control returns before registration when the answer is stale');
ok(between.indexOf('return;') < between.indexOf('}'),'the return is inside the guarded block, not after it');
ok(css.includes('.ai-md-stale-candidate'),'the stale notice is styled');
ok(src.includes("note.setAttribute('role', 'status')"),'the stale notice is announced to assistive technology');

// ── Continuation is intent, not a second copy of the bytes ───────────────
//
// Continue used to stage a composer attachment AND register a working file, so
// on an endpoint that accepts working_files the same bytes travelled twice --
// once unbound. That doubled the token cost of every continued file and gave
// the model two copies to reconcile.
const cont = extract('_generatedArtifactContinueEditing');
// Deliberately reversed from R173T26. Staging only on endpoints without
// working-file support created two authorities for one intent: on those
// endpoints the chip was the truth, on the others the registry was, and no
// chip existed at all -- files travelled with nothing in the composer to show
// for it, and removing a chip left the registry set.
//
// The chip is now the visible truth everywhere; the duplicate transmission it
// could cause is suppressed at request build, where the transport is chosen.
ok(cont.includes("sourceKind: 'working-file'"),'a continued file is always staged so the reader can see and remove it');
ok(!cont.includes('if (!_lastWorkingFileCaps) {'),'staging is no longer conditional on the endpoint');
const sync = extract('_syncContinuationForRemovedItem');
ok(sync.includes("item.sourceKind !== 'working-file'"),'only working-file chips clear a continuation');
ok(sync.includes('delete _workingFileContinuations[keys[i]];')&&sync.includes('_refreshContinuationTray();'),'removing the chip clears the registry and refreshes the tray');
ok(extract('_removeComposerResourceItem').includes('_syncContinuationForRemovedItem(item);'),'every removal path runs the sync, not just the menu');

// Double-send prevention, where the transport is actually known.
ok(src.includes('bodyObj.resources = bodyObj.resources.filter(function (res) {'),'a file carried as a working file is dropped from the resource descriptors');
ok(src.includes('requestResources = requestResources.filter(function (row) {'),'and from the multipart body, which is built from a different array');
ok(src.includes("' omitted (carried as bound working files instead)'"),'the activity receipt says what was suppressed and why');
ok(cont.includes('if (_workingFileContinuations[entry.key]) return true;'),'queuing the same file twice is a no-op');
ok(src.includes('_lastWorkingFileCaps = (proxyCaps && proxyCaps.workingFiles) || null;'),'the negotiated caps are remembered for the next Continue click');

// Dropping, one file and all of them.
const stop = extract('_generatedArtifactStopContinuing');
ok(stop.includes('delete _workingFileContinuations[key];'),'a queued file can be dropped individually');
const clear = extract('_generatedArtifactClearContinuations');
ok(clear.includes('delete _workingFileContinuations[key];')&&clear.includes('_continuationCount()'),'the whole queue can be cleared at once');
ok(src.includes("? { label: 'Stop continuing'"),'the menu item toggles rather than offering a one-way action');

// Batch queuing respects the endpoint's own limits and says what did not fit.
const all = extract('_generatedArtifactContinueAll');
ok(all.includes('_lastWorkingFileCaps || _WORKING_FILE_FALLBACK'),'batch queuing uses the negotiated limits');
ok(all.includes('_continuationCount() >= limits.maxFiles')&&all.includes('entry.content.length > limits.maxFileChars')&&all.includes('chars + entry.content.length > limits.maxTotalChars'),'every published bound is applied, not just the file count');
ok(all.includes('left out (unavailable, or beyond this endpoint'),'what did not fit is named rather than silently dropped');
ok(all.includes("{ quiet: true }"),'batch queuing does not fire one notification per file');

// The queue is visible and manageable.
ok(src.includes("tray.className = 'ai-assistant-panel-changed-files-tray'")&&src.includes("tray.setAttribute('role', 'status')"),'the queue is shown as a status line');
ok(src.includes('_generatedArtifactClearContinuations();'),'the tray offers the action that is awkward through per-file menus');
const tray_css = fs.readFileSync(process.argv[3], 'utf8');
ok(/\.ai-assistant-panel-changed-files-tray\s*\{/.test(tray_css),'the tray is styled');
ok(/forced-colors: active[\s\S]{0,300}?ai-assistant-panel-changed-files-continue-all[\s\S]{0,80}?LinkText/.test(tray_css),'the batch control survives forced-colours mode');

// The composer instruction must never overwrite what the reader typed.
const prime = extract('_primeComposerForContinuation');
ok(prime.includes("if (!String(input.value || '').trim()) {"),'a composer the reader has typed into is left alone');
ok(prime.includes("' attached files. Return each complete updated file.'"),'the instruction matches the number of files attached');

// ── Re-adding a dropped file, and queuing more than one ──────────────────
//
// Two reported failures, one shape each.
//
// A menu built from a static array froze the Continue/Stop label at row-build
// time, so after dropping a file the item still read "Stop continuing" and the
// next click stopped an already-stopped continuation. The file looked
// impossible to re-add.
const overflow = extract('_buildOverflowMenu');
ok(overflow.includes("var resolveItems = (typeof items === 'function')"),'a menu may be built from a function so its items can depend on state');
ok(overflow.includes('resolveItems().forEach(function (item) {'),'items are resolved when the menu opens, not when the row is built');
ok(extract('_buildFileOverflow').includes('return _fileOverflowItems(key);'),'the file menu resolves its canonical items per open');
ok(extract('_fileOverflowItems').includes('_workingFileContinuations[key]'),'the canonical list still resolves Continue/Stop from live state');

// Stop must undo everything Continue did. Deleting only the registry key left
// the bytes staged, so the dropped file still travelled and re-adding staged a
// second copy.
const stopFn = extract('_generatedArtifactStopContinuing');
// With the guard, not just the call: a call left inside `if (false)` still
// contains the substring, which is how the first version of this assertion
// passed while the mutant survived.
ok(stopFn.includes('if (entry) _unstageContinuationAttachment(entry.path);'),'stopping also unstages any attachment the fallback created');
ok(extract('_unstageContinuationAttachment').includes('_removeComposerResourceItem(item);'),'unstaging goes through the composer pipeline that owns the item');
ok(extract('_generatedArtifactClearContinuations').includes('_unstageContinuationAttachment(e.path);'),'clearing the queue unstages every one of them');

// The fallback cap silently dropped every file but one from Continue-all.
ok(/_WORKING_FILE_FALLBACK = \{ maxFiles: 4,/.test(src),'the pre-discovery cap mirrors the server default instead of guessing one file');
ok(/maxFileChars: 48000, maxTotalChars: 96000/.test(src),'the character bounds mirror it too');

// The tray reported whatever count it had when the section rendered.
// The tray refresh moved into the shared surface refresh; the entry point is
// now a coalescing wrapper, so the contract is asserted where it lives.
const trayFn = extract('_refreshContinuationSurfaces');
ok(trayFn.includes('_continuationCount()')&&trayFn.includes('tray.hidden'),'the tray is refreshed rather than left at its render-time count');
// Four surfaces answer "what travels with my next message". Each mutation used
// to refresh whichever ones its author remembered, which left the registry
// emptied and every chip still on screen.
ok(trayFn.includes('_renderComposerAttachments();'),'the composer chips are redrawn');
ok(trayFn.includes('_renderAttachmentManagerList();'),'the attachment manager is redrawn');
ok(trayFn.includes('_applyContinueAllLabel'),'the bulk control is relabelled');
const entry = extract('_refreshContinuationTray');
ok(entry.includes('if (_continuationRefreshDepth > 0) return;'),'nested refreshes coalesce instead of redrawing once per removed file');
ok(entry.includes('_continuationRefreshDepth--;')&&entry.includes('finally'),'the depth is released even when a redraw throws');
ok(extract('_generatedArtifactClearContinuations').includes('_continuationRefreshDepth++;'),'a bulk clear draws once, from the final state');
['_generatedArtifactStopContinuing','_generatedArtifactClearContinuations','_generatedArtifactContinueEditing','_generatedArtifactContinueAll'].forEach(function (fn) {
  ok(extract(fn).includes('_refreshContinuationTray()'), fn + ' refreshes the tray');
});

// ── The bulk control says what the next click does ───────────────────────
//
// Attaching several files was one click; detaching them meant dismantling the
// queue one menu at a time, so the reader who wanted none of it had the most
// work to do.
const label = new Function('_continuationCount', extract('_applyContinueAllLabel') + '\nreturn _applyContinueAllLabel;');
function fakeBtn(total){
  const attrs = { 'data-ai-continue-all-total': String(total) };
  return { textContent: '', title: '',
           getAttribute: k => attrs[k], setAttribute: (k, v) => { attrs[k] = v; },
           attrs };
}
let b = fakeBtn(2);
label(() => 0)(b);
ok(b.textContent === 'Continue editing all 2 files','an empty queue offers to attach every file');
ok(b.attrs['aria-label'] === 'Attach all 2 presented files to your next message','and says so in its accessible name');
label(() => 2)(b);
ok(b.textContent === 'Remove all 2 attached files','a non-empty queue offers to detach them');
ok(b.attrs['aria-label'] === 'Remove all 2 attached files from your next message','and says so in its accessible name');
label(() => 1)(b);
ok(b.textContent === 'Remove all 1 attached file','the label is singular for one file');
ok(!/Continue/.test(b.textContent),'the two directions never read the same');
label(() => 0)(b);
ok(b.textContent === 'Continue editing all 2 files','clearing the queue restores the attach label');

// Built once, so it must be refreshed rather than trusted.
ok(src.includes("var buttons = document.querySelectorAll('.ai-assistant-panel-changed-files-continue-all');")&&src.includes('Array.prototype.forEach.call(buttons, _applyContinueAllLabel);'),'every bulk control is relabelled from the one refresh point');
ok(src.includes('if (_continuationCount()) _generatedArtifactClearContinuations();')&&src.includes('else _generatedArtifactContinueAll();'),'the click follows the live count, not the label it was built with');

// Two buttons for one action is the duplication this section keeps removing.
ok(!src.includes("clear.className = 'ai-assistant-panel-changed-files-tray-clear'"),'the tray no longer carries a second clear control');
ok(src.includes('tray.appendChild(trayText);')&&!src.includes('tray.appendChild(clear);'),'the tray is status only');

// ── The menu stays inside the panel body ─────────────────────────────────
//
// It was placed by CSS alone -- inset-inline-end: 0; top: 100% -- which is
// right only when there happens to be room below and to the left. A file row
// near the bottom of the body, or a narrow panel, pushed the menu past the
// edge where it was clipped, and Save as / Patch / Continue became
// unreachable.
// Anchored to the trigger, not clamped inside the panel body.
//
// The body-scoped routine measured against `.ai-assistant-panel-body`, and a
// trigger in the footer sits outside it: no side fitted, so the fallback
// clamped the menu into the body's own box and it appeared in the middle of
// the panel, unattached to the button that opened it.
const place = extract('_positionMenuNearTrigger');
ok(src.includes('_positionMenuNearTrigger(menu, btn);'),'the menu is placed against its own trigger');
ok(place.includes("menu.style.position = 'fixed';"),'placement is viewport-fixed');
// The triggers live in four different subtrees with different overflow and
// transform ancestors; R173T58 is the cost of assuming a containing block.
ok(place.includes('btn.getBoundingClientRect()'),'coordinates come from the trigger it must sit against');
ok(place.includes('var placeAbove = (m.height > below) && (above > below);'),'it opens upward only when there is more room there');
ok(place.includes("menu.style.maxHeight = space + 'px';"),'and bounds itself to that side, so a long list scrolls rather than overflowing');
ok(place.includes('var left = t.right - m.width;'),'aligned to the trigger trailing edge');
// Clamped against the bounds, not the raw viewport, since R173T64.
ok(/left = Math\.min\(\s*Math\.max\(bounds\.left \+ margin, left\)/.test(place),'then clamped, so a wide menu slides along the bounds instead of hanging off them');
ok(place.includes("menu.setAttribute('data-placement'"),'the chosen side is exposed for styling');

// ── The containing block is not always the viewport ──────────────────────
//
// R173T62 assumed `position: fixed` resolves against the viewport. It does not
// when an ancestor has a `transform`, and the panel has one on its open state.
// Viewport coordinates were being written into a different coordinate space,
// so the menu landed at an offset that grew with the panel's position.
//
// Driven, not read: the arithmetic is the whole fix.
ok(place.includes('var origin = { left: m.left, top: m.top };'),'the menu measures its own origin while pinned at (0,0)');
ok(place.includes('Math.round(left - origin.left)') && place.includes("- origin.top) + 'px'"),'and converts viewport coordinates into that space');
function placeWith(originX, originY) {
  const style = {};
  const fakeMenu = {
    style: style, setAttribute() {},
    getBoundingClientRect: () => ({ left: originX, top: originY, width: 220, height: 300 })
  };
  const fakeBtn = { getBoundingClientRect: () => ({ left: 700, right: 760, top: 600, bottom: 630 }) };
  new Function('window','document', place + '\nreturn _positionMenuNearTrigger;')(
    { innerWidth: 1000, innerHeight: 911 }, { documentElement: {} })(fakeMenu, fakeBtn);
  return { left: parseFloat(style.left), top: parseFloat(style.top) };
}
const plain = placeWith(0, 0);
const shifted = placeWith(120, 80);
ok(plain.left - shifted.left === 120 && plain.top - shifted.top === 80,'a shifted containing block shifts the written coordinates by exactly that much');
ok(plain.left === shifted.left + 120 && plain.top === shifted.top + 80,'so the menu lands in the same place on screen either way');
ok(shifted.top + 80 + 300 <= 911 && shifted.left + 120 + 220 <= 1000,'and inside the viewport, which is what was clipping it');

// ── Bounded by the panel, not only by the screen ─────────────────────────
//
// A narrow panel inside a wide window has plenty of viewport beside it, so a
// viewport clamp let the menu spill onto the page: a list belonging to a
// control in the panel, drawn over the documentation behind it.
ok(place.includes("btn.closest('.ai-assistant-panel')"),'the menu finds the panel it belongs to');
ok(place.includes('bounds.right = Math.min(bounds.right, pr.right);'),'and bounds itself by the intersection of panel and viewport');
ok(place.includes('if (pr.width > 0 && pr.height > 0)'),'a collapsed or hidden panel does not clamp everything into a point');
ok(!/Math\.max\(margin, left\)/.test(place) && !/vw - m\.width - margin/.test(place),'no clamp still reads the raw viewport');

function placeInPanel(panelRect, btnRect, menuSize) {
  const style = {};
  const panel = { getBoundingClientRect: () => panelRect };
  const fakeMenu = { style: style, setAttribute() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: menuSize.w, height: menuSize.h }) };
  const fakeBtn = { closest: () => panel, getBoundingClientRect: () => btnRect };
  new Function('window','document', place + '\nreturn _positionMenuNearTrigger;')(
    { innerWidth: 1600, innerHeight: 911 }, { documentElement: {} })(fakeMenu, fakeBtn);
  return { left: parseFloat(style.left), top: parseFloat(style.top),
           maxHeight: style.maxHeight ? parseFloat(style.maxHeight) : menuSize.h };
}
// A narrow panel down the right of a wide window, trigger near its bottom.
const narrow = { left: 1200, top: 60, right: 1580, bottom: 880, width: 380, height: 820 };
const r = placeInPanel(narrow, { left: 1480, right: 1540, top: 800, bottom: 830 }, { w: 260, h: 400 });
ok(r.left >= narrow.left && r.left + 260 <= narrow.right,'the menu stays within the panel horizontally');
ok(r.top >= narrow.top && r.top + Math.min(400, r.maxHeight) <= narrow.bottom,'and within it vertically, opening upward from a footer trigger');

// A panel shorter than its content: the viewport half of the intersection is
// what keeps the menu on screen.
const tall = { left: 1200, top: -200, right: 1580, bottom: 1400, width: 380, height: 1600 };
const r2 = placeInPanel(tall, { left: 1480, right: 1540, top: 300, bottom: 330 }, { w: 260, h: 400 });
ok(r2.top >= 0 && r2.top + Math.min(400, r2.maxHeight) <= 911,'a panel extending past the screen is still clipped by the screen');

// ── Width: as wide as the longest row, no wider ──────────────────────────
//
// A fixed width padded short menus and truncated the long labels in the model
// list, where the label is the information. `max-content` sizes to the longest
// row -- and needs a ceiling, or one long label makes the menu wider than the
// panel it belongs to.
const menuW = (css.replace(/\/\*[\s\S]*?\*\//g, '')
  .match(/^\.ai-assistant-panel-changed-file-menu \{[^}]*\}/gm) || []).join('\n');
ok(/width:\s*max-content/.test(menuW),'the menu sizes to its longest row');
ok(place.includes("menu.style.maxWidth = Math.max(160,"),'the ceiling comes from the placement bounds, not a guessed constant');
ok(place.includes('(bounds.right - bounds.left) - margin * 2'),'so it is the panel width the reader actually has');
// Constrained before measuring, or every later calculation uses a box the menu
// will never have.
ok(place.indexOf('menu.style.maxWidth = Math.max(160,') < place.indexOf('var m = menu.getBoundingClientRect();'),'the width is constrained before the box is measured');
ok(place.includes("menu.style.maxWidth = '';"),'and reset first, so a reopen is not bounded by the last panel size');
// With max-content, a row that sets its own width decides the menu's.
ok(/\.ai-assistant-panel-changed-file-menu-item \{[^}]*width:\s*100%/.test(css),'rows fill the menu rather than sizing it');
ok(/\.ai-assistant-panel-changed-file-menu-hint \{[^}]*white-space:\s*normal/.test(css),'the secondary hint wraps instead of making the menu wide');

const wide = placeInPanel(narrow, { left: 1480, right: 1540, top: 300, bottom: 330 }, { w: 900, h: 200 });
ok(wide.left >= narrow.left,'an over-wide menu is pinned inside the panel rather than hanging off it');

// ── The composer is a floor, not empty space ─────────────────────────────
//
// The bubble action menu has always been bounded this way -- it clamps within
// `.ai-assistant-panel-body`, which ends where the footer begins -- and it
// reads better for it. A menu drawn over the composer looks like it belongs to
// the composer, and hides the draft about to be sent.
ok(place.includes("panel.querySelector('.ai-assistant-panel-footer')"),'the routine finds the composer');
// The floor now also yields when it would squeeze the menu: keeping clear of
// the composer is a readability preference, showing every row is not.
ok(place.includes('if (fr.height > 0 && t.bottom <= fr.top &&'),'the floor applies only to triggers above the composer');
ok(place.includes('(fr.top - bounds.top) >= _MENU_MIN_USABLE_H'),'and only while it leaves a usable menu');
ok(/_MENU_MIN_USABLE_H = 200/.test(src),'with the threshold named rather than inlined');
// Driven: a tall panel keeps the floor, a short one drops it.
const tallPanel = { left: 1200, top: 60, right: 1580, bottom: 880 };
function withFooter(panelRect, footerTop, btnRect) {
  const style = {};
  const footer = { getBoundingClientRect: () => ({ top: footerTop, bottom: panelRect.bottom, height: panelRect.bottom - footerTop }) };
  const panel = { getBoundingClientRect: () => panelRect, querySelector: () => footer };
  const fakeMenu = { style: style, setAttribute() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 240, height: 400 }) };
  const fakeBtn = { closest: () => panel, getBoundingClientRect: () => btnRect };
  new Function('window','document', place + '\nreturn _positionMenuNearTrigger;')(
    { innerWidth: 1600, innerHeight: 911 }, { documentElement: {} })(fakeMenu, fakeBtn);
  return style.maxHeight ? parseFloat(style.maxHeight) : 400;
}
const roomy = withFooter(tallPanel, 700, { left: 1300, right: 1360, top: 300, bottom: 330 });
ok(roomy <= 700 - 60,'with room, the menu still stops above the composer');
const cramped = withFooter({ left: 1200, top: 400, right: 1580, bottom: 700 }, 520,
                           { left: 1300, right: 1360, top: 430, bottom: 450 });
ok(cramped > 520 - 400,'on a short panel the floor is dropped rather than cutting rows off');

// The bubble menu is bounded by the panel, not the transcript.
ok(src.includes("boundarySelector: '.ai-assistant-panel',"),'the bubble menu is bounded by the panel');
ok(src.includes("? document.querySelector(opts.boundarySelector)"),'the shared routine honours a caller-supplied boundary');
ok(src.includes("document.getElementById('ai-assistant-panel-body');"),'and still defaults to the transcript for callers that want it');
ok(place.includes('bounds.bottom = Math.min(bounds.bottom, fr.top);'),'the floor narrows the bounds rather than replacing them');

function placeWithFooter(btnRect) {
  const style = {};
  const footer = { getBoundingClientRect: () => ({ top: 700, bottom: 880, height: 180 }) };
  const panel = { getBoundingClientRect: () => narrow, querySelector: () => footer };
  const fakeMenu = { style: style, setAttribute() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 260, height: 300 }) };
  const fakeBtn = { closest: () => panel, getBoundingClientRect: () => btnRect };
  new Function('window','document', place + '\nreturn _positionMenuNearTrigger;')(
    { innerWidth: 1600, innerHeight: 911 }, { documentElement: {} })(fakeMenu, fakeBtn);
  return { top: parseFloat(style.top), maxH: style.maxHeight ? parseFloat(style.maxHeight) : 300 };
}
const fromTranscript = placeWithFooter({ left: 1300, right: 1360, top: 300, bottom: 330 });
ok(fromTranscript.top + Math.min(300, fromTranscript.maxH) <= 700,'a menu opened from the transcript never covers the composer');
// The picker's own trigger lives in the footer; a bound above its own button
// would leave that menu nowhere to go.
const fromFooter = placeWithFooter({ left: 1480, right: 1540, top: 800, bottom: 830 });
ok(!isNaN(fromFooter.top),'a menu whose trigger is inside the footer is still placed');
ok(fromFooter.top >= narrow.top,'and still inside the panel');

// ── Each menu is bounded by the surface it is in ─────────────────────────
//
// A menu opened from the preview dialog was bounded by the panel. The dialog
// is a floating window of its own, often wider and placed elsewhere, so the
// menu was clamped to a box its trigger was not in -- the reported
// `max-width: 1193px` on a preview menu is a ceiling from the wrong surface.
ok(place.includes("btn.closest('.ai-assistant-panel-attachment-preview') ||"),'the preview dialog is checked first');
ok(place.includes("btn.closest('.ai-assistant-panel')"),'with the panel as the fallback');
ok(place.indexOf("btn.closest('.ai-assistant-panel-attachment-preview')") < place.indexOf("btn.closest('.ai-assistant-panel')"),'nearest surface wins, so a transcript menu still picks the panel');

// ── One builder, two class names, on purpose ─────────────────────────────
//
// The legacy name is after one of four callers, so a rule written for "the
// file menu" silently restyled the preview and the model picker too.
ok(src.includes("menu.className = 'ai-assistant-menu ai-assistant-panel-changed-file-menu';"),'menus carry a neutral class alongside the legacy one');
ok(/\.ai-assistant-menu\[data-placement="above"\]/.test(css),'new shared rules use the neutral name');
ok(/\.ai-assistant-panel-attachment-preview \.ai-assistant-menu/.test(css),'surface overrides are scoped by the caller, not by the shared class');
// The legacy name is retained, not replaced: removing it would break anything
// selecting on the shipped DOM.
ok(/\.ai-assistant-panel-changed-file-menu/.test(css),'the legacy name still has its rules');

// ── The trigger closes what it opened ────────────────────────────────────
//
// The outside-click handler runs on `document` in the CAPTURE phase, so it sees
// a click before the trigger's own handler. The trigger holds an `<svg>`, so a
// click lands on the glyph: `e.target !== btn` was true, the menu closed here,
// and the trigger's handler then found nothing open and reopened it. The menu
// could be opened and never closed from its own button.
const menuSrc = extract('_buildOverflowMenu');
ok(menuSrc.includes('!menu.contains(e.target) && !btn.contains(e.target)'),'a click anywhere inside the trigger counts as clicking the trigger');
// Comments stripped: the rule's own comment quotes the old identity test while
// explaining why it is gone, and matching that sentence reported correct code
// as broken. Same trap as R173T39, in JavaScript this time.
const menuCode = menuSrc.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
ok(!/e\.target !== btn/.test(menuCode),'no identity test remains, which the glyph would defeat');
ok(menuSrc.includes("document.addEventListener('click', rec.onDocClick, true);"),'the handler is still capture-phase, so the fix is in the test, not the ordering');
ok(menuSrc.includes('var wasOpen = _fileMenuOpen && _fileMenuOpen.btn === btn;'),'and the trigger still toggles rather than always opening');

// Driven: the capture handler must not fire for the glyph, and must for
// anything outside.
const glyphNode = {};
const triggerNode = { contains: function (n) { return n === triggerNode || n === glyphNode; } };
let closes = 0;
const onDocClick = new Function('menu','btn','_closeFileMenu',
  'return function (e) { if (!menu.contains(e.target) && !btn.contains(e.target)) _closeFileMenu(); };')(
  { contains: function () { return false; } }, triggerNode, function () { closes++; });
onDocClick({ target: glyphNode });
ok(closes === 0,'clicking the glyph inside the trigger does not close it first');
onDocClick({ target: {} });
ok(closes === 1,'clicking outside still closes it');
// Measured after insertion, or the box is sized from nothing.
ok(src.indexOf('_positionMenuNearTrigger(menu, btn);') > src.indexOf('btn.parentNode.appendChild(menu);'),'placement runs after insertion, when the box can be measured');
ok(src.includes("rec.onReflow = function () { _positionMenuNearTrigger(menu, btn); };"),'the menu is repositioned on reflow');
ok(/@media \(pointer: coarse\)[\s\S]{0,200}?changed-file-menu[\s\S]{0,60}?min\(60vh,\s*24rem\)/.test(css),'a coarse pointer gets more of the screen for the same list');
ok(src.includes("menu.setAttribute('data-open', 'true');"),'the menu marks itself open so the routine will place it');
// Two call sites: the one after insertion and the one inside the reflow
// handler. An indexOf comparison found whichever came first in the file and
// passed even with the insertion-time call deleted -- the reflow call is later
// than appendChild too.
const placeCalls = (src.match(/_positionMenuNearTrigger\(menu, btn\);/g) || []).length;
ok(placeCalls === 2,'the menu is placed once on open and once per reflow');

// A menu anchored to a row inside a scrolling body must follow that row.
ok(src.includes("scroller.addEventListener('scroll', rec.onReflow, true);")&&src.includes("window.addEventListener('resize', rec.onReflow);"),'it follows panel scrolling and window resizing');
ok(extract('_closeFileMenu').includes("rec.scroller.removeEventListener('scroll', rec.onReflow, true);")&&extract('_closeFileMenu').includes("window.removeEventListener('resize', rec.onReflow);"),'both reflow listeners are removed on close');

// Three popups, one placement implementation.
ok((src.match(/function _positionAnchoredPopupWithinPanelBody/g) || []).length === 1,'there is exactly one placement implementation');
const menu_css = fs.readFileSync(process.argv[3], 'utf8');
// The exact bound, not just "some vh": a mutation that inflates it to 500vh
// leaves a menu taller than any screen while still matching a loose pattern.
ok(/^\.ai-assistant-panel-changed-file-menu \{[^}]*max-height:\s*min\(50vh,\s*20rem\)/m.test(menu_css),'a menu is bounded to half the screen, capped at 20rem');
ok(/@media \(pointer: coarse\)[\s\S]{0,200}?changed-file-menu[\s\S]{0,60}?min\(60vh,\s*24rem\)/.test(menu_css),'a coarse pointer gets more of the screen for the same list');
ok(/\.ai-assistant-panel-changed-file-menu\s*\{[^}]*overscroll-behavior:\s*contain/.test(menu_css),'scrolling the menu does not scroll the transcript behind it');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
