// Run 172 — observable activity + latest-revision generated-file preview.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const cssSrc = fs.readFileSync(process.argv[3], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function extract(name) {
  const start = src.indexOf('function ' + name + '('); if (start < 0) throw new Error('missing ' + name);
  let depth=0,began=false,quote='',esc=false,line=false,block=false;
  for (let i=start;i<src.length;i++) { const c=src[i],n=src[i+1];
    if(line){if(c==='\n')line=false;continue;} if(block){if(c==='*'&&n==='/'){block=false;i++;}continue;}
    if(quote){if(esc){esc=false;continue;}if(c==='\\'){esc=true;continue;}if(c===quote)quote='';continue;}
    if(c==='/'&&n==='/'){line=true;i++;continue;} if(c==='/'&&n==='*'){block=true;i++;continue;}
    if(c==='"'||c==="'"||c==='`'){quote=c;continue;} if(c==='{'){depth++;began=true;} else if(c==='}'&&--depth===0&&began)return src.slice(start,i+1);
  } throw new Error('unterminated '+name);
}
const start=extract('_startTurnActivity'), finish=extract('_activityFinish'), completed=extract('_activityResponseCompleted');
const wire=extract('_activityIngestWireEvent'), metadata=extract('_activityIngestResponseMetadata');
const artifact=extract('_registerGeneratedArtifact'), bind=extract('_generatedArtifactBindLatest');
const openLatest=extract('_generatedArtifactOpenLatest'), downloadLatest=extract('_generatedArtifactDownloadLatest');
const sync=extract('_syncExplicitCodeArtifacts'), changed=extract('_appendChangedFileSummary');
const fence=extract('_parseCodeFenceInfo'), safePathSrc=extract('_generatedArtifactSafePath');
const submit=extract('handleAIPanelSubmit'), clear=extract('clearConversation');
const stopActive=extract('_stopActivePanelResponse'), ensureActive=extract('_panelTurnEnsureActive');
const apiCall=extract('_panelApiCall'), streamCall=extract('_panelApiCallStreaming');
const delay=extract('_panelTurnDelay'), sessionBudget=extract('_generatedArtifactEnsureSessionBudget');
const retentionUnavailable=extract('_generatedArtifactMakeRetentionUnavailable');
const fetchFallback=extract('_fetchWithReasoningFallback');

ok(src.includes('_TURN_ACTIVITY_MAX_STEPS = 48')&&src.includes('_TURN_ACTIVITY_LABEL_MAX_CHARS = 180')&&src.includes('_TURN_ACTIVITY_DETAIL_MAX_CHARS = 2000'),'activity metadata has explicit bounded limits');
ok(src.includes('_TURN_ACTIVITY_FILE_MAX_BYTES = 256 * 1024')&&src.includes('_TURN_ACTIVITY_FILE_TOTAL_MAX_BYTES = 1024 * 1024')&&src.includes('_TURN_ACTIVITY_FILE_SESSION_TOTAL_MAX_BYTES = 8 * 1024 * 1024')&&src.includes('_TURN_ACTIVITY_FILE_MAX_COUNT = 24'),'file previews have per-file/per-turn/session/count bounds');
ok(start.includes('Hidden model reasoning is never displayed.'),'surface distinguishes public activity from hidden reasoning');
ok(start.includes("stop.textContent = 'Stop'")&&start.includes('_stopActivePanelResponse()')&&stopActive.includes('controller.abort()')&&stopActive.includes("reader.cancel('AI_REQUEST_CANCELLED')"),'visible stop tears down fetch and stream reader');
ok(stopActive.includes('_panelUnlockComposerAfterCancel()'),'Stop immediately unlocks composer instead of waiting on transport teardown');
ok(finish.includes('panelActivityAutoCollapse')&&finish.includes('_activitySetOpen(st, false)'),'completed activity auto-collapses');
ok(completed.includes("st.state !== 'running'"),'late completion cannot overwrite stopped state');
ok(start.includes("panelActivityTimeline === false")&&start.includes("_activeTurnActivity = st")&&submit.includes("!turnActivity.root"),'timeline-off mode retains headless turn state and legacy typing fallback');
ok(src.includes('_panelActiveRequestToken')&&ensureActive.includes('_panelActiveRequestToken !== token')&&submit.includes('requestToken = { id: ++_panelRequestTokenSeq'),'turn ownership is independent of AbortController and visualization');
ok(delay.includes('_panelActiveRequestToken !== requestToken')&&delay.includes("activity.state === 'cancelled'"),'local slow work observes turn-owned cancellation');
ok(apiCall.includes('requestController ? requestController.signal : undefined')&&!apiCall.includes('_fetchAbortController ? _fetchAbortController.signal'),'non-stream request uses captured controller, not replaceable global controller');
ok(streamCall.includes('_panelActiveStreamOwner = { reader: reader, activity: activity }')&&streamCall.includes('_panelTurnEnsureActive(activity, requestController, requestToken)'),'stream reader is turn-owned and guarded before continuation');
ok(streamCall.includes('streamBubble.parentNode.removeChild(streamBubble)')&&streamCall.includes("throw new Error('AI_STREAM_BROKEN_PIPE')"),'broken-pipe/reader fallback paths remove orphan provisional bubbles');
ok(fetchFallback.includes('shouldContinue')&&fetchFallback.includes('ensureContinuation()'),'reasoning fallback cannot launch retry after turn cancellation');
ok(clear.includes('_panelActiveRequestToken.cancelled = true')&&clear.includes("reader.cancel('AI_CONVERSATION_CLEARED')"),'conversation clear invalidates token and active reader');

ok(wire.includes("kind === 'thinking'")&&wire.includes("kind === 'reasoning'")&&wire.includes("kind === 'chain_of_thought'")&&wire.includes("kind = 'summary'"),'reasoning-shaped kinds normalize to public summary');
ok(wire.includes('_redactSecrets(String(payload.label')&&wire.includes('_redactSecrets(String(payload.detail'),'public activity is secret-redacted');
ok(wire.includes('Sensitive token pattern redacted from activity metadata'),'redaction warning is bounded');
ok(!wire.includes('payload.chain_of_thought')&&!wire.includes('payload.thinking')&&!wire.includes('payload.reasoning'),'raw hidden reasoning fields are never rendered');
ok(metadata.includes('data.activity')&&metadata.includes('data.artifacts')&&metadata.includes('_TURN_ACTIVITY_MAX_STEPS')&&metadata.includes('_TURN_ACTIVITY_FILE_MAX_COUNT'),'buffered sidecars are bounded');
ok(src.includes("sseEventType === 'activity' || sseEventType === 'assistant.activity'")&&src.includes("sseEventType === 'artifact' || sseEventType === 'assistant.artifact'"),'SSE supports public activity/artifact event forms');
ok(src.includes('_activityIngestWireEvent(activity, publicEvent)')&&src.includes('_activityIngestArtifactEvent(activity, publicEvent)'),'SSE events feed bounded registries');

ok(artifact.includes('old ? old.revision + 1 : 1'),'changed content increments revision');
ok(artifact.includes('diff: _diffLineStat(old && old.content, content)')&&artifact.includes('baseRevision: old ? _artifactContentRevision(old) : 0'),'each revision records its own +/- stat and the content revision it came from');
// The split is the point: a content change advances contentRevision, while a
// preview eviction advances only the ledger event counter.
ok(artifact.includes('contentRevision: old ? _artifactContentRevision(old) + 1 : 1'),'a content change advances the content revision');
ok(src.includes('contentRevision: _artifactContentRevision(old),'),'losing a preview does not advance the content revision');
ok(src.includes('_artifactContentRevision(entry) === Number(binding.revision)'),'staleness is judged on content, not on ledger events');
ok(src.includes("(typeof entry.contentRevision === 'number')"),'records written before the split still resolve a revision');
ok(artifact.includes('_generatedArtifactIsAvailable(old)')&&artifact.includes('old.content === content')&&artifact.includes('return old'),'identical currently-available content reuses revision');
ok(artifact.includes('_generatedArtifactLedger[path] = entry'),'ledger is keyed by canonical path');
ok(bind.includes('_generatedArtifactOpenLatest(key, el)')&&bind.includes('_generatedArtifactRefs[key]'),'historical controls bind by key');
ok(openLatest.includes('var entry = _generatedArtifactLedger[key]'),'preview resolves latest at click');
ok(downloadLatest.includes('var entry = _generatedArtifactLedger[key]'),'download resolves latest at click');
// "Changed files" claimed more than happened -- nothing outside the browser
// changed. The latest-revision guarantee is unchanged and still asserted.
ok(changed.includes("'Presented ' + combined.length")&&changed.includes('links open the latest revision'),'answer gets a presented-files section that resolves the latest revision');
ok(changed.includes("hint.textContent = 'drafts, not applied"),'the draft status rides with the count, not a tooltip');
ok(!changed.includes("'Changed files'"),'no surface still claims files were changed');
ok(src.includes('function _collapseArtifactPreBlocks(root)'),'complete files collapse to an in-place preview');
// `sync` is the per-chunk path: it runs on every streamed chunk while a fence
// is still growing. Collapsing there would wrap a partial file and keep
// re-wrapping it. The finalization path is where this belongs.
ok(src.includes('_collapseArtifactPreBlocks(root);'),'complete files are collapsed somewhere');
ok(sync.indexOf('_collapseArtifactPreBlocks') === -1,'collapse never runs on the per-chunk streaming path');
ok(src.includes('if (lines < _FILE_PREVIEW_COLLAPSE_MIN_LINES) return;'),'short files stay open rather than costing a click for nothing');
ok(src.includes("if (!path) return;"),'anonymous snippets are not collapsed -- they are usually the answer itself');
// The original wrap is moved, never re-created: copy, download and the
// artifact path all keep working because they are the same element.
// The gutter moved into a shared builder used by the inline sheet and the
// preview overlay, so the contract is asserted on the builder.
const sheetFn = extract('_buildLineNumberedSheet');
ok(sheetFn.includes('sheet.appendChild(pre);')&&src.includes('body.appendChild(sheet);'),'the sheet holds the original pre, so copy and download are untouched');
ok(sheetFn.includes('var parent = pre.parentNode;')&&sheetFn.includes('parent.insertBefore(sheet, next);'),'the sheet replaces the pre in place, keeping its position among siblings');
ok(src.includes("_buildLineNumberedSheet(wrap, text, 'ai-md-file-sheet')"),'the inline file view uses the shared builder');
ok(src.includes("_buildLineNumberedSheet(pre, item.previewText,"),'the preview overlay uses it too, so every preview is numbered');
ok((src.match(/gutter\.className = 'ai-md-file-gutter'/g) || []).length === 1,'there is exactly one gutter implementation');
// Numbers must never enter the code, in either surface.
ok(sheetFn.includes("gutter.setAttribute('aria-hidden', 'true')"),'the gutter is out of the accessibility tree');
ok(!/pre\.textContent\s*=\s*[^;]*numbers/.test(src),'no code path writes a line number into the pre');
ok(/\.ai-assistant-panel-attachment-preview-sheet \.ai-md-file-gutter\s*\{[^}]*line-height:\s*inherit/.test(cssSrc),'the overlay gutter inherits the code line box, so numbers cannot drift');

// ── Every code block is numbered, not only collapsed files ───────────────
//
// The collapse pass numbers the complete files it collapses. Short files and
// every snippet that never declared a path were left unnumbered, so a reader
// could cite a line only in the blocks large enough to have been collapsed.
const numFn = extract('_numberRemainingCodeBlocks');
ok(src.includes('_numberRemainingCodeBlocks(root);'),'the remaining blocks are numbered too');
ok(numFn.includes("_buildLineNumberedSheet(wrap, code.textContent || '',"),'they use the same builder, so a line number means the same thing everywhere');
// Two independent guards against nesting one sheet inside another.
ok(numFn.includes("wrap.getAttribute('data-ai-line-numbered') === 'true'"),'numbering twice is a no-op');
ok(numFn.includes("wrap.parentNode.classList.contains('ai-md-file-sheet')"),'a block already sheeted by the collapse pass is skipped');
// Same timing rule as the collapse: the per-chunk path would wrap a fence that
// is still arriving and re-wrap it on every chunk.
ok(sync.indexOf('_numberRemainingCodeBlocks') === -1,'numbering never runs on the per-chunk streaming path');
ok(src.indexOf('_numberRemainingCodeBlocks(root);') > src.indexOf('_collapseArtifactPreBlocks(root);'),'numbering runs after the collapse, so collapsed files are not double-wrapped');
ok(/\.ai-md-snippet-sheet[^{]*\{[^}]*border-radius/.test(cssSrc),'a snippet sheet keeps the code block radius rather than the file sheet squared edge');
ok(/\.ai-md-snippet-sheet \.ai-md-file-gutter[\s\S]{0,120}?line-height:\s*inherit/.test(cssSrc),'its gutter shares the code line box');
ok(/\.ai-md-snippet-sheet \{[^}]*max-height:\s*none[^}]*overflow:\s*visible[^}]*overscroll-behavior:\s*auto/.test(cssSrc),'inline answer snippets yield vertical wheel and touch scrolling to the conversation');
ok(/\.ai-md-snippet-sheet \.ai-md-pre \{[^}]*overscroll-behavior-y:\s*auto[^}]*touch-action:\s*pan-x pan-y pinch-zoom/.test(cssSrc),'snippet code owns horizontal pan without trapping vertical touch gestures');

// ── One scroller per sheet, and no wrapping where numbers are shown ──────
//
// Two faults from putting a gutter beside a block that already scrolled.
// The <pre> had max-height + overflow:auto, so it scrolled in a fixed box
// while the gutter, having neither, rendered every line of the file: a
// 5543-line preview stretched the dialog to the length of the file, and the
// numbers slid out of step the moment either scrolled.
const sheetCss = cssSrc.replace(/\/\*[\s\S]*?\*\//g, '');
const sheetRule = (sheetCss.match(/\.ai-md-file-sheet \{[^}]*\}/) || [''])[0];
ok(/overflow-y:\s*auto/.test(sheetRule) && /max-height/.test(sheetRule),'the sheet is the scroll container, not the code');
const codeRule = (sheetCss.match(/\.ai-md-file-sheet \.ai-md-pre,[\s\S]*?\{[^}]*\}/) || [''])[0];
// `hidden`, not `visible`: CSS promotes a visible axis to `auto` when the other
// is not visible, so `overflow-y: visible` beside `overflow-x: auto` made the
// block a scroll container on both axes -- silently, and only for files with
// long lines. It then swallowed the wheel while having nothing to scroll.
ok(/max-height:\s*none/.test(codeRule) && /overflow-y:\s*hidden/.test(codeRule),'the code no longer scrolls itself, so gutter and code move together');
ok(!/overflow-y:\s*visible/.test(codeRule),'no visible axis remains to be promoted beside the horizontal scroll');
ok(/overflow-x:\s*auto/.test(codeRule),'long lines still scroll horizontally');
// The overlay body is the scroller there, so the sheet must not claim the wheel.
ok(/\.ai-assistant-panel-attachment-preview-body > \.ai-md-file-sheet \{[^}]*overscroll-behavior:\s*auto/.test(cssSrc),'a sheet that does not scroll releases the wheel to the page');
// The sheets that DO own a scrollbar keep containing it.
const ownScroll = (cssSrc.replace(/\/\*[\s\S]*?\*\//g, '').match(/^\.ai-md-file-sheet \{[^}]*\}/m) || [''])[0];
ok(/overscroll-behavior:\s*contain/.test(ownScroll),'a sheet with its own scrollbar still contains the gesture');
// A logical line wrapping to three visual rows fills three line boxes in the
// code and one in the gutter, so every later number is wrong.
ok(/white-space:\s*pre;/.test(codeRule) && !/pre-wrap/.test(codeRule),'numbered code does not wrap');
ok(/overflow-x:\s*auto/.test(codeRule),'long lines scroll horizontally instead');
ok(/overflow-wrap:\s*normal/.test(codeRule) && /word-break:\s*normal/.test(codeRule),'inherited wrapping rules are overridden too, not just white-space');
// The overlay body is already a scroller; a sheet scrolling inside it would
// give two vertical scrollbars for one document.
ok(/\.ai-assistant-panel-attachment-preview-body > \.ai-md-file-sheet \{[^}]*max-height:\s*none/.test(sheetCss),'inside the preview body the sheet defers to the body scroller');
// One rule per selector: two blocks for .ai-md-file-sheet meant the cascade
// had to be read to know what the sheet does, and an assertion could match the
// wrong one -- which is how this very check first passed against a rule with
// no overflow in it.
ok((sheetCss.match(/^\.ai-md-file-sheet \{/gm) || []).length === 1,'the sheet is defined once, not by two rules read together');

// Line numbers must never enter the code. Prefixing each line is the usual
// shortcut and it poisons every copy, download and patch taken from the block.
ok(src.includes("gutter.className = 'ai-md-file-gutter'")&&src.includes('gutter.textContent = numbers.join'),'line numbers live in their own element');
ok(src.includes("gutter.setAttribute('aria-hidden', 'true')"),'the gutter is out of the accessibility tree');
ok(/\.ai-md-file-gutter\s*\{[^}]*user-select:\s*none/.test(cssSrc),'the gutter cannot be drag-selected into a copy');
ok(/\.ai-md-file-gutter\s*\{[^}]*pointer-events:\s*none/.test(cssSrc),'the gutter is not a click target');
ok(!/code[^\n]*textContent\s*=[^\n]*ln\b/.test(src),'no code path writes a line number into the code element');
const gutterRule = (cssSrc.match(/\.ai-md-file-gutter\s*\{[^}]*\}/) || [''])[0];
const preRule = (cssSrc.match(/\.ai-md-file-sheet \.ai-md-pre\s*\{[^}]*\}/) || [''])[0];
const lh = r => (r.match(/line-height:\s*([\d.]+)/) || [])[1];
const fs2 = r => (r.match(/font-size:\s*([\d.]+rem)/) || [])[1];
ok(lh(gutterRule) && lh(gutterRule) === lh(preRule),'gutter and code share a line height, or the numbers drift');
ok(fs2(gutterRule) && fs2(gutterRule) === fs2(preRule),'gutter and code share a font size');
// Presentation contract, asserted against the paired stylesheet rather than
// trusting the class names to exist.
ok(/\.ai-md-file-disclosure-head\s*\{/.test(cssSrc),'the disclosure row is styled');
ok(/\[aria-expanded="true"\][^{]*\.ai-md-file-disclosure-caret\s*\{[^}]*rotate\(90deg\)/.test(cssSrc),'the caret reflects aria state, not a separate class');
ok(/prefers-reduced-motion: reduce/.test(cssSrc),'caret motion is opt-out');
ok(/\.ai-md-file-disclosure-body\[hidden\]\s*\{\s*display:\s*none/.test(cssSrc),'hidden bodies are actually hidden');
ok(/\.ai-assistant-panel-changed-files-list\[hidden\]/.test(cssSrc),'the presented-files list collapses too');
ok(src.includes("head.setAttribute('aria-label',"),'the row carries one full sentence for assistive tech');
// Superseded by the footer assertion below: the strip now holds both bulk
// controls, so hiding the footer covers what this used to check on its own.
ok(src.includes('if (footerRef) footerRef.hidden = open;'),'collapsing the summary hides its bulk controls with it');
ok(src.includes("wrap.getAttribute('data-ai-file-disclosure') === 'true'"),'collapsing twice is a no-op');
ok(src.includes("head.setAttribute('aria-controls', body.id)")&&src.includes("head.setAttribute('aria-expanded', 'false')"),'the disclosure is a real, announced control');
// The label now names the count; the contract it guards -- resolving each
// file from the ledger at click time -- is unchanged and still asserted.
// The label moved into the shared icon-button decorator; the contract it
// guards -- resolving each file from the ledger at click time, and naming the
// count -- is unchanged.
ok(changed.includes("'Download all ' + combined.length + ' files'")&&changed.includes('var entry = _generatedArtifactLedger[key]'),'bulk download re-resolves latest files and names the count');
ok(changed.includes('_decorateIconButton(all, ICONS.exportTxt,'),'the bulk download carries the download glyph');
// Patch export is git's format, so it carries git's mark rather than the
// generic download arrow -- the two footer actions produce different kinds of
// artifact and should not look interchangeable.
ok(changed.includes('_decorateIconButton(series, ICONS.gitMark,'),'the patch export carries the git logomark');
ok(/gitMark:'<svg[^']*fill="currentColor"/.test(src),'the git mark inherits colour rather than shipping fixed-colour variants');
ok(!/gitMark:[^\n]*#f03c2e|gitMark:[^\n]*#100f0d|gitMark:[^\n]*fill="#fff"/.test(src),'no theme-specific copy of the mark is shipped');

// Icon-only below a narrow surface width, measured on the panel rather than
// the viewport: the panel is resizable, maximizable and embeddable, so the two
// widths are different numbers.
ok(/container-name:\s*ai-artifact-surface/.test(cssSrc),'the artifact surfaces are query containers');
ok(/@container ai-artifact-surface \(max-width: 26rem\)/.test(cssSrc),'per-file labels collapse before a full-width mobile row becomes crowded');
ok(/@container ai-artifact-surface \(max-width: 22rem\)/.test(cssSrc),'bulk footer labels stay readable until the tighter threshold');
const perFileIconOnly = (cssSrc.match(/@container ai-artifact-surface \(max-width: 26rem\) \{[\s\S]*?\n\}/) || [''])[0];
const bulkIconOnly = (cssSrc.match(/@container ai-artifact-surface \(max-width: 22rem\) \{[\s\S]*?\n\}/) || [''])[0];
['ai-md-artifact-download-label','ai-assistant-panel-changed-file-download']
  .forEach(function (cls) { ok(perFileIconOnly.includes(cls), cls + ' collapses early to protect the filename'); });
['ai-assistant-panel-changed-files-download-all','ai-assistant-panel-changed-files-series']
  .forEach(function (cls) { ok(bulkIconOnly.includes(cls), cls + ' keeps text until the tighter bulk threshold'); });
ok(!perFileIconOnly.includes('ai-assistant-panel-changed-files-download-all') &&
   !perFileIconOnly.includes('ai-assistant-panel-changed-files-series'),
   'per-file compaction does not prematurely hide descriptive bulk-action labels');
// Comments stripped first. The rule's own comment explains why display:none
// is not used, and matching that sentence reported correct CSS as broken.
const perFileIconOnlyCode = perFileIconOnly.replace(/\/\*[\s\S]*?\*\//g, '');
const bulkIconOnlyCode = bulkIconOnly.replace(/\/\*[\s\S]*?\*\//g, '');
ok(/clip-path:\s*inset\(50%\)/.test(perFileIconOnlyCode)&&!/display:\s*none/.test(perFileIconOnlyCode),'per-file label is clipped, not removed, so the hit area and title survive');
ok(/clip-path:\s*inset\(50%\)/.test(bulkIconOnlyCode)&&!/display:\s*none/.test(bulkIconOnlyCode),'bulk labels use the same accessible clipping contract');
ok(/min-width:\s*2\.25rem/.test(perFileIconOnly),'an icon-only per-file button keeps a usable target size');
// Safe only because the accessible name never depended on the visible label.
ok(src.includes("dlBtn.setAttribute('aria-label', 'Download ' + filename)"),'the snippet download names its file regardless of width');
ok(src.includes("download.setAttribute('aria-label', 'Download latest ' + entry.path"),'the file download names its file regardless of width');
const deco = extract('_decorateIconButton');
ok(deco.includes("glyph.setAttribute('aria-hidden', 'true')"),'the glyph is decoration: removing it changes nothing announced');
ok(deco.includes('glyph.innerHTML = iconSvg;'),'the glyph comes from an ICONS constant, never user or model content');
ok(deco.includes("btn.textContent = '';"),'the button is cleared first, so decorating twice cannot duplicate the label');
ok((src.match(/allBtn\.innerHTML = ICONS/g) || []).length === 0,'the snippet all-button uses the shared decorator rather than its own two lines');
ok(/\.ai-md-artifact-btn-icon svg \{[^}]*width:\s*\.85em/.test(cssSrc),'the glyph scales with its button text rather than a fixed pixel size');
// Card layout: Preview and Download own the primary row; Patch and Continue
// drop to a quieter second line rather than competing for the same weight.
// The primary row is now preview | download | save as. Patch and Continue
// still sit on the secondary line, asserted below.
// Three controls: preview fills the row, then download, then one ⋮ menu.
// Save-as, patch and continue moved into that menu -- asserted below, so the
// capability is still guarded, just at its new home.
// Presented files now use the same segmented control as the snippet cards:
// preview and download joined, then the overflow menu beside them.
ok(changed.includes('primary.appendChild(_buildArtifactSegmentGroup(entry.path, preview, download));')&&changed.includes('primary.appendChild(_buildFileOverflow(key, entry));'),'a presented file is one segmented control plus its overflow menu');
ok(src.includes('function _buildArtifactSegmentGroup(ariaLabel, primary, secondary)'),'both artifact surfaces share one segmented-control builder');
ok(/\.ai-md-artifact-group\s*>\s*\.ai-md-artifact-card[\s\S]{0,220}?border:\s*0/.test(cssSrc),'all preview segments shed their own chrome through the shared group rule');
ok(/\.ai-assistant-panel-changed-file-primary\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\) auto/.test(cssSrc),'the group takes the room the overflow menu leaves');
ok(!changed.includes("className = 'ai-assistant-panel-changed-file-saveas'"),'save-as is no longer a fifth button on the card');
ok((cssSrc.match(/\.ai-assistant-panel-changed-file-primary\s*\{/g)||[]).length===1,'Presented files have one primary-row layout authority');
ok(!/\.ai-assistant-panel-changed-file-primary\s*\{[^}]*grid-template-columns:[^;}]*auto auto/.test(cssSrc),'obsolete three-column Presented-file grids are gone');
ok(changed.includes("download.className = 'ai-md-artifact-download-label ai-assistant-panel-changed-file-download';"),'Presented-file Download reuses the same base visual class as normal artifacts');
ok(/@media \(max-width: 420px\)[\s\S]*?\.ai-assistant-panel-changed-file-primary\s*\{[^}]*minmax\(0, 1fr\)/.test(cssSrc),'a narrow panel wraps the actions instead of clipping the filename');

// Save-as is a download alias, never a rename: the ledger key is the file's
// identity and the revision chain, diff base and patch headers all hang off it.
ok(src.includes('function _generatedArtifactSaveAs(key)'),'files can be saved under a chosen name');
ok(!/_generatedArtifactSaveAs[\s\S]{0,900}?_generatedArtifactLedger\[[^\]]+\]\s*=/.test(src),'save-as never rewrites the ledger key');
ok(src.includes('_artifactNameSlugPreservingExtension(raw)'),'a reader-supplied filename is sanitized like any other');
ok(src.includes("text.split(/[\\\\/]/).pop()"),'a typed directory is stripped rather than silently ignored');
ok(src.includes('_generatedArtifactIsAvailable(entry)'),'an unavailable revision cannot be saved');
// The menu mechanics now live in one shared builder used by both the snippet
// card and the tracked-file card; `_buildFileOverflow` is the thin wrapper that
// supplies this surface's items. Assert each where it lives.
const menu = extract('_buildOverflowMenu');
// ── Menu items carry a glyph ─────────────────────────────────────────────
//
// Decoration in the strict sense: aria-hidden, with the accessible name coming
// from the label text, so removing an icon changes nothing announced.
ok(menu.includes("iconWrap.setAttribute('aria-hidden', 'true');"),'the glyph is out of the accessibility tree');
ok(menu.includes('if (item.icon) iconWrap.innerHTML = item.icon;'),'it is optional, and only ever an ICONS constant');
ok(menu.includes("text.className = 'ai-assistant-panel-changed-file-menu-text'"),'label and hint sit in their own stack beside it');
// The gutter is reserved either way: a list where some labels are indented and
// others are not is harder to scan than one with no icons at all.
ok(/\.ai-assistant-panel-changed-file-menu-item \{[^}]*grid-template-columns:\s*1rem minmax\(0, 1fr\)/.test(cssSrc),'the icon column is reserved whether or not an item has a glyph');
ok(/\.ai-assistant-panel-changed-file-menu-icon \{[^}]*align-items:\s*center/.test(cssSrc),'the glyph is centred in its own column');
ok(/\.ai-assistant-panel-changed-file-menu-icon \{[^}]*margin-top/.test(cssSrc),'and aligned to the first line, not the middle of a two-line item');
ok(/forced-colors: active[\s\S]{0,200}?menu-icon[\s\S]{0,60}?ButtonText/.test(cssSrc),'and survives forced-colours mode');
// Each glyph matches what its action produces.
ok(/'Download patch'[^}]*icon: ICONS\.gitMark/.test(src),'the patch export carries the git mark');
ok(/'Save as\\u2026'[^}]*icon: ICONS\.exportTxt/.test(src),'save-as carries the download arrow');
ok(/'Open in a sheet'[^}]*icon: ICONS\.terms/.test(src),'opening a sheet carries the document glyph');
const fileItems = extract('_fileOverflowItems');
const fileMenu = extract('_buildFileOverflow');
ok(fileItems.includes('_generatedArtifactDownloadPatch(key)')&&fileItems.includes('_generatedArtifactContinueEditing(key)'),'patch and continue live in the canonical tracked-file action list, still one click away');
ok(fileItems.includes('_generatedArtifactSaveAs(key)')&&fileItems.includes('_generatedArtifactOpenSheet(key)'),'save-as and open-in-a-sheet are canonical tracked-file actions');
ok(fileMenu.includes('return _fileOverflowItems(key);'),'the Presented-file wrapper delegates to the canonical action list');
ok(menu.includes("menu.setAttribute('role', 'menu')")&&menu.includes("row.setAttribute('role', 'menuitem')"),'the menu is announced as a menu');
ok(menu.includes("btn.setAttribute('aria-haspopup', 'menu')")&&menu.includes("btn.setAttribute('aria-expanded', 'true')"),'the trigger reports its popup and its state');
ok(menu.includes("if (e.key !== 'Escape') return;")&&menu.includes('btn.focus();'),'Escape closes the menu and returns focus to the trigger');
// Both listeners, by name. Asserting that the string appears at all passed
// while one of the two removals had been deleted -- the mutant found it.
const closer = extract('_closeFileMenu');
ok(closer.includes("document.removeEventListener('click', rec.onDocClick, true);"),'closing the menu removes its click listener');
ok(closer.includes("document.removeEventListener('keydown', rec.onKeyDown, true);"),'closing the menu removes its keydown listener');
ok(menu.includes("document.addEventListener('click', rec.onDocClick, true);")&&menu.includes("document.addEventListener('keydown', rec.onKeyDown, true);"),'both listeners are registered when the menu opens');
ok(extract('_closeFileMenu').includes('_fileMenuOpen = null;'),'only one file menu can be open at a time');
// One builder, two surfaces: the snippet card must not grow a second menu with
// its own (and inevitably weaker) keyboard handling.
ok(src.includes("'ai-md-artifact-overflow');"),'the snippet card uses the shared menu builder');
ok(!src.includes("className = 'ai-md-artifact-promote'"),'the snippet card no longer carries a second full-width button');
ok((src.match(/document\.addEventListener\('keydown', rec\.onKeyDown, true\);/g) || []).length === 1,'there is exactly one menu keyboard implementation');
const sheet = extract('_generatedArtifactOpenSheet');
ok(sheet.includes('_generatedArtifactOpenLatest(key, null, { sheet: true })'),'the sheet is the same viewer asked to show everything');
ok(sheet.includes('_generatedArtifactIsAvailable(entry)'),'an unavailable revision cannot be opened as a sheet');
ok(changed.includes("badge.textContent = (ext || 'file').slice(0, 6).toUpperCase();"),'each card carries a type badge that survives path truncation');
// The bulk pair is now the wide version of a file row: Download all is the big
// segment, Download patch series the narrow one beside it.
ok(changed.includes("_buildArtifactSegmentGroup(\n                'All ' + combined.length + ' presented files', all, series)"),'bulk actions are one segmented control');
// One file: a single action, spanning the section and wearing the group's
// chrome, so the footer does not change kind when a second file arrives.
ok(changed.includes("series.classList.add('ai-assistant-panel-changed-files-solo');")&&changed.includes('footer.appendChild(series);'),'a single-file section offers patch export as one full-width control');
ok(/\.ai-assistant-panel-changed-files-solo\s*\{[^}]*flex:\s*1 1 100%/.test(cssSrc),'the solo footer control spans the section');
ok(/\.ai-assistant-panel-changed-files-solo\s*\{[^}]*border:\s*1px/.test(cssSrc),'the solo control wears the group chrome rather than sitting bare');
// Wording follows the count: a "series" of one makes a reader look for the
// other files.
ok(changed.includes("many ? 'Download patch series' : 'Download patch'"),'a one-file export is called a patch, not a series');
ok(changed.includes("'Download ' + _generatedArtifactLedger[combined[0]].path + ' as a git patch'"),'the one-file accessible name says which file');
ok(changed.includes("'Download all ' + combined.length + ' tracked files as one git patch series'"),'the many-file accessible name says how many');
ok(!changed.includes("'Download every tracked file as one git patch series'"),'the count-blind label is gone');
// The preview segment is built from the snippet card's own classes, not a
// parallel class tree styled to match -- the two had already diverged.
ok(changed.includes("preview.className = 'ai-md-artifact-card ai-assistant-panel-changed-file-preview';"),'a presented file uses the snippet card markup');
ok(changed.includes("icon.className = 'ai-md-artifact-icon'")&&changed.includes("copy.className = 'ai-md-artifact-info'")&&changed.includes("name.className = 'ai-md-artifact-name'"),'its parts use the shared card classes');
ok(!changed.includes("open.textContent = 'Preview'"),'the redundant trailing Preview word is gone; the card is the affordance');
ok(changed.includes("preview.setAttribute('aria-label', 'Preview ' + entry.path)"),'the card still says what it does, in its accessible name');
ok(changed.includes('typeLine.appendChild(badge);')&&changed.includes('typeLine.appendChild(meta);'),'badge and live state share the type line');
ok(/\.ai-assistant-panel-changed-file-preview \.ai-assistant-panel-changed-file-meta:empty[\s\S]{0,120}?display:\s*none/.test(cssSrc),'an empty state line leaves no gap');
ok(/\.ai-assistant-panel-changed-files-footer\s*>\s*\.ai-md-artifact-group\s*\{[^}]*flex:\s*1 1 100%/.test(cssSrc),'the bulk control spans the footer');
ok(/\.ai-md-artifact-group\s*>\s*\.ai-assistant-panel-changed-files-download-all[\s\S]{0,200}?border:\s*0/.test(cssSrc),'the bulk segments shed their own chrome inside the group');
ok(changed.includes('if (footerRef) footerRef.hidden = open;'),'collapsing the summary hides the whole footer, not just one control');
ok(changed.includes("id: 'presented-files', kind: 'file', state: 'done'"),'the presentation is reported in the activity timeline as a file event');
ok(changed.includes("' r' + _artifactContentRevision(e)"),'the timeline names each file at its content revision');
ok(/\.ai-assistant-panel-changed-files-download-all\s*\{[^}]*flex:\s*1 1 100%/.test(cssSrc),'download-all spans the strip so it reads as covering every card');
ok(!/\.ai-assistant-panel-changed-file-secondary\b/.test(cssSrc)&&!changed.includes('ai-assistant-panel-changed-file-secondary'),'the obsolete secondary action row is removed rather than hidden');
ok(changed.includes('_attachmentPathAlias(entry.path)')&&changed.includes('aliasCollision')&&changed.includes('portable filesystem'),'bulk download fails closed on portable path collisions');
ok(sessionBudget.includes('_generatedArtifactMakeRetentionUnavailable')&&sessionBudget.includes('Released older file preview'),'session pressure evicts oldest retained previews instead of silently exceeding memory');
ok(retentionUnavailable.includes("entry.state = 'unavailable'")&&retentionUnavailable.includes('entry.content = null')&&!retentionUnavailable.includes('entry.revision + 1'),'local retention eviction invalidates bytes without inventing a file revision');
ok(sync.includes(".ai-md-pre[data-artifact-path]")&&sync.includes('_registerGeneratedArtifact'),'annotated fences feed ledger');
ok(src.includes('_syncExplicitCodeArtifacts(streamBubble, activity)'),'stream registers completed file fences before finish');
ok(src.includes('data-artifact-path=')&&src.includes('_parseCodeFenceInfo(info ||'),'markdown preserves safe file metadata');
ok(fence.includes('(?:file|filename|path)=')&&fence.includes('_generatedArtifactSafePath'),'fence aliases pass through safe path validation');
ok(submit.includes('_startTurnActivity(body')&&submit.includes('_activityFinish(turnActivity'),'submit owns one activity per turn');
ok(clear.includes('_generatedArtifactLedger = Object.create(null)')&&clear.includes('_generatedArtifactRefs = Object.create(null)'),'clear resets file revision authority');
ok(src.includes('browser preview/download entry; it does not mean the file was applied to a repository.'),'preview is never represented as write authority');

ok(css.includes('.ai-assistant-panel-activity')&&css.includes('.ai-assistant-panel-activity-stop')&&css.includes('.ai-assistant-panel-activity-step'),'activity has dedicated styles');
ok(css.includes('.ai-assistant-panel-changed-files')&&css.includes('.ai-assistant-panel-changed-file-preview'),'changed files have dedicated styles');
ok(css.includes('@media (prefers-reduced-motion: reduce)')&&css.includes('animation: none'),'reduced motion is respected');
ok(css.includes('@media (max-width: 560px)')&&css.includes('.ai-assistant-panel-changed-file {'),'mobile layout exists');
ok(css.includes('[data-artifact-unavailable]'),'unavailable latest-state controls have visible warning styling');

const safePathFactory = new Function(`
  var _ATTACHMENT_IMPORT_MAX_PATH_CHARS = 1024;
  ${extract('_attachmentSafeRelativePath')}
  ${safePathSrc}
  return _generatedArtifactSafePath;
`);
const safePath=safePathFactory();
ok(safePath('src/example.py')==='src/example.py'&&safePath('a/b/config.toml')==='a/b/config.toml','normal relative paths accepted');
ok(safePath('../secret')===''&&safePath('/etc/passwd')===''&&safePath('C:/x')===''&&safePath('a\\b')==='','traversal/absolute/drive/backslash rejected');
ok(safePath('a/./b')===''&&safePath('a//b')===''&&safePath('a/\u202esecret')==='','dot/empty/bidi aliases rejected');
ok(fence.includes('file|filename|path')||fence.includes('(?:file|filename|path)'),'fence parser recognizes file metadata aliases');

const stepSrc=extract('_activityAddStep');
ok(stepSrc.includes('else if (detail)')&&stepSrc.includes("ai-assistant-panel-activity-step-detail"),'running step can gain detail when a later update completes it');
ok(src.includes("reviewed.action === 'cancel' || opConversationId !== boundConversationId || opConversationId !== _getConversationId()"),'share review conversation-race guard remains intact');
ok(src.includes('localActiveModel, turnActivity')&&src.includes("'assistant', undefined, { activity: activity }"),'browser-local replies remain attached to the turn activity surface');

// Run-time latest-state safety: execute the real registration/state functions
// with only DOM-free dependencies stubbed.
const latestFactory = new Function(`
  var _generatedArtifactLedger = Object.create(null);
  var _generatedArtifactRefs = Object.create(null);
  var _TURN_ACTIVITY_FILE_MAX_BYTES = 262144;
  var _TURN_ACTIVITY_FILE_TOTAL_MAX_BYTES = 1048576;
  var _TURN_ACTIVITY_FILE_SESSION_TOTAL_MAX_BYTES = 8 * 1024 * 1024;
  var _TURN_ACTIVITY_FILE_MAX_COUNT = 24;
  var _DIFF_STAT_MAX_LINES = 20000;
  var _DIFF_STAT_LCS_BUDGET = 4000000;
  function _cfg(){ return {panelGeneratedFilePreview:true}; }
  ${extract('_artifactContentRevision')}
  ${extract('_generatedArtifactEntryBytes')}
  ${extract('_diffStatSplitLines')}
  ${extract('_diffStatMultiset')}
  ${extract('_diffStatLcs')}
  ${extract('_diffLineStat')}
  function _generatedArtifactSafePath(v){
    if (typeof v !== 'string') return '';
    var p=v.trim(); if(!p || p[0]==='/' || p.includes('\\\\') || /^[A-Za-z]:/.test(p)) return '';
    var a=p.split('/'); if(a.some(x=>!x||x==='.'||x==='..')) return ''; return p;
  }
  function _activityBoundedText(v,n){ return String(v == null ? '' : v).slice(0,n); }
  function _utf8ByteLength(v){ return Buffer.byteLength(String(v),'utf8'); }
  function _activityAddStep(){}
  function _activityAddArtifactStep(){}
  function _generatedArtifactRefreshRefs(){}
  ${extract('_generatedArtifactIsAvailable')}
  ${extract('_generatedArtifactSessionBytes')}
  ${retentionUnavailable}
  ${sessionBudget}
  ${extract('_generatedArtifactPublishState')}
  ${extract('_registerGeneratedArtifact')}
  return {ledger:_generatedArtifactLedger, register:_registerGeneratedArtifact};
`);
function turnState(){ return {fileBytesByKey:Object.create(null),filePreviewBytes:0,changedFileKeys:Object.create(null),fileRevisionCount:0}; }
const rt=latestFactory(), st=turnState();
let e=rt.register({path:'src/a.py',content:'one',mediaType:'text/plain',source:'endpoint'},st);
ok(e&&e.revision===1&&e.state==='available'&&e.content==='one','r1 is retained as available latest revision');
e=rt.register({path:'src/a.py',content:'x'.repeat(262145),source:'endpoint'},st);
ok(e&&e.revision===2&&e.state==='unavailable'&&e.content===null,'oversized r2 invalidates stale r1');
e=rt.register({path:'src/a.py',content:'three',source:'endpoint'},st);
ok(e&&e.revision===3&&e.state==='available'&&e.content==='three','later valid r3 restores previewability');
e=rt.register({path:'src/a.py',operation:'remove',source:'endpoint'},st);
ok(e&&e.revision===4&&e.state==='removed'&&e.content===null,'remove creates authoritative latest tombstone');
e=rt.register({path:'src/a.py',content:'five',source:'endpoint'},st);
ok(e&&e.revision===5&&e.state==='available','path becomes available again after tombstone');
const dup=rt.register({path:'src/a.py',content:'five',source:'endpoint'},st);
ok(dup.revision===5,'exact currently-available duplicate does not invent revision');
const empty=rt.register({path:'src/empty.txt',content:'',source:'endpoint'},st);
ok(empty&&empty.state==='available'&&empty.content==='','empty complete file remains a valid latest revision');
const missing=rt.register({path:'src/a.py',source:'endpoint'},st);
ok(missing&&missing.revision===6&&missing.state==='unavailable'&&missing.content===null,'missing newer content invalidates prior retained bytes');
const rt2=latestFactory(), st2=turnState();
let agg=rt2.register({path:'src/a.txt',content:'old'},st2);
st2.filePreviewBytes=1048576; st2.fileBytesByKey['src/a.txt']=3;
agg=rt2.register({path:'src/a.txt',content:'1234'},st2);
ok(agg&&agg.revision===2&&agg.state==='unavailable'&&agg.content===null,'aggregate-budget rejection invalidates stale existing revision');

const rt3=latestFactory(), st3=turnState();
for (let i=0;i<32;i++) rt3.register({path:`old/${i}.txt`,content:'z'.repeat(262144),source:'endpoint'},turnState());
const beforeOldRevision=rt3.ledger['old/0.txt'].revision;
const newest=rt3.register({path:'newest.txt',content:'n'.repeat(1024),source:'endpoint'},st3);
ok(newest&&newest.state==='available','session budget admits newest preview by releasing older retained bytes');
ok(rt3.ledger['old/0.txt'].state==='unavailable'&&rt3.ledger['old/0.txt'].content===null&&rt3.ledger['old/0.txt'].revision===beforeOldRevision,'session eviction makes old preview unavailable without inventing revision');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
