// Run 106 compatibility regression, superseded by Run 109 semantics:
// every visible PAGE / MD / FILE card is one-turn transport state. Saved pins
// and current-page defaults are source conveniences only; they do not silently
// authorize reuse on later questions.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('missing ' + name);
  let d = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { d++; started = true; }
    else if (src[j] === '}') { d--; if (started && d === 0) return src.slice(i, j + 1); }
  }
  throw new Error('unbalanced ' + name);
}

const meta = extract('_attachmentItemMeta');
const sanitize = extract('_sanitizeTurnAttachmentSummaries');
const replayNames = extract('_replayAttachmentNames');
const liveView = extract('_turnResourceLiveView');
const snapshot = extract('_composerTurnAttachmentSnapshot');
const clear = extract('_clearComposerAttachments');
const shelf = extract('_contextShelfItems');
const render = extract('_renderBubble');
const append = extract('_appendPanelMessage');
const record = extract('_recordMessage');
const load = extract('_loadTranscript');
const submit = extract('handleAIPanelSubmit');
const effective = extract('_prepareComposerEffectiveAttachmentPlan');
const attachmentContext = extract('_prepareComposerAttachmentPlan');
const privacyPages = extract('_privacyPrepareDocumentationContext');

ok(meta.includes("parts.push('Next message')") && meta.includes("parts.push('One turn')"), 'all staged cards expose next-message one-turn semantics');
ok(meta.includes("Current · Auto-selected") && meta.includes("Pinned page · saved source"), 'page source kind remains visible without implying transport persistence');
ok(src.includes("Pages and files staged for the next message only"), 'composer tray accessibility names one-turn staging explicitly');
ok(shelf.includes('_pageContextConsumed(currentUrl)'), 'current page disappears after one-turn consumption');
ok(shelf.includes('_pageContextConsumed(sourceUrl)'), 'saved pinned pages also disappear after one-turn consumption');
ok(privacyPages.includes('_pageContextConsumed(currentUrl)') && privacyPages.includes('_pageContextConsumed(sourceUrl)'), 'request preparation cannot silently reuse consumed pages');

ok(src.includes('var _TURN_RESOURCE_LIVE_MAX_ITEMS = 512') && src.includes('var _TURN_RESOURCE_PERSIST_MAX_ITEMS = 24'), 'Run111 separates live resource completeness from compact persistence');
ok(sanitize.includes("'page'") && sanitize.includes("'audio'") && sanitize.includes("'video'") && sanitize.includes("'data'"), 'turn metadata sanitizer accepts page plus first-class multimodal provenance explicitly');
ok(sanitize.includes('_attachmentSafeName'), 'turn metadata sanitizes names');
ok(!sanitize.includes('previewText =') && !sanitize.includes('item.file ='), 'persisted metadata sanitizer does not retain preview bodies or File objects');
ok(liveView.includes('summary.turnScoped = true'), 'live turn preview capability is marked session-only');
ok(liveView.includes('summary.file = item.file'), 'live local preview may retain a File reference only in memory');
ok(snapshot.includes("status: item.contextRole === 'current' ? 'Current page · used once' : 'Pinned page · used once'"), 'page provenance records one-shot use');
ok(snapshot.includes("'Included once · bounded excerpt'") && snapshot.includes("'Local only · not sent'"), 'file provenance distinguishes sent bounded excerpts versus local-only');
ok(snapshot.includes("status: included ? 'Reused once' : 'Not sent · shared context budget'"), 'explicit Retry/Edit file replay remains visible');
ok(replayNames.includes('/^Attachment:\\s+(.+)$/'), 'replay filenames are reconstructed from canonical attachment headers');

ok(submit.includes('var turnAttachments = _composerTurnAttachmentSnapshot(attachmentPlan, preparedPageContext);'), 'submit snapshots pages and files after privacy preparation');
ok(submit.includes('_consumePreparedPageContexts(preparedPageContext, false);'), 'submit consumes every participating page source exactly once');
ok(submit.includes('_clearComposerAttachments();'), 'submit consumes uploaded/replay attachment state');
ok(submit.includes("_appendPanelMessage(questionText, 'user', requestQuestion"), 'visible user question stays separate from hidden canonical envelopes');
ok(submit.includes('attachments: turnAttachments'), 'user turn receives immutable resource provenance');
ok(clear.includes('_composerAttachments = []'), 'clear removes staged file state');
ok(clear.includes("_composerReplayAttachmentContext = ''"), 'clear consumes explicit replay context');

ok(record.includes('entry.resources = resourceManifest'), 'transcript stores canonical live resource manifest');
ok(load.includes('_sanitizeTurnResourceManifest') && load.includes('_TURN_RESOURCE_PERSIST_MAX_ITEMS'), 'persisted transcript revalidates compact resource manifests');
// The replay meta object gained a restored-activity field; the two contracts
// this assertion guards -- legacy `attachments` fallback and memory-only
// runtime -- are unchanged and asserted individually rather than as one literal.
ok(src.includes('resources: m.resources || m.attachments,'), 'transcript replay keeps the legacy metadata fallback');
ok(src.includes('resourceRuntime: m.resourceRuntime || null,'), 'transcript replay keeps preview capability memory-only');
ok(src.includes('restoredActivity: m.activity || null'), 'transcript replay carries the persisted activity summary');
ok(append.includes('turnMeta'), 'live message append uses same turn metadata path');

ok(render.includes("files.className = 'ai-assistant-panel-attachments ai-assistant-panel-user-turn-attachments'"), 'user turn reuses composer attachment-strip classes');
ok(render.includes("files.setAttribute('aria-label', 'Files and pages used for this question')"), 'historical resource strip is accessible');
ok(render.includes("preview.className = 'ai-assistant-panel-attachment-card'"), 'historical resources reuse the same preview-card anatomy');
ok(render.includes('_openAttachmentPreview(item, preview)'), 'historical cards use the same preview dialog');
ok(render.includes("questionText.className = 'ai-assistant-panel-user-turn-question'"), 'question has dedicated section below resources');
ok(render.includes("questionToggle.textContent = 'Show more'"), 'long question remains expandable');
ok(render.includes("questionToggle.textContent = next ? 'Show less' : 'Show more'"), 'long question disclosure is reversible');

ok(css.includes('.ai-assistant-panel-user-turn-attachments'), 'turn resource strip has focused layout styling');
ok(css.includes('.ai-assistant-panel-user-turn-question[data-collapsed="true"]'), 'long question clamp remains styled');
ok(css.includes('-webkit-line-clamp: 7'), 'collapsed question has bounded visible slice');
ok(css.includes('@media (max-width: 480px)') && css.includes('width: 94%'), 'resource-bearing user turn adapts on narrow panels');

ok(attachmentContext.includes("item.kind === 'text'") && attachmentContext.includes('_readAttachmentText'), 'local-only binary/image bytes remain excluded while text is read lazily');
ok(effective.includes('_prepareComposerAttachmentPlan(snapshotItems)') && effective.includes('_mergeAttachmentContexts(plan.text, _composerReplayAttachmentContext)'), 'only fresh files plus explicit Retry/Edit file replay can enter file context');
ok(src.includes('_splitQuestionWithAttachments(canonicalQuestion)'), 'Retry/Edit still use canonical one-turn file context explicitly');

// Execute the resource snapshot helpers together so page/file status cannot
// drift even when source-level strings still look correct.
const factory = new Function(`
  var _TURN_RESOURCE_LIVE_MAX_ITEMS = 512;
  var _ATTACHMENT_MAX_PREVIEW_BYTES = 512 * 1024;
  function _attachmentSafeName(v) { return String(v || 'file').replace(/[\\u0000-\\u001f\\u007f]/g, ' ').trim().slice(0,240) || 'file'; }
  function _attachmentExtension(v) { var n=_attachmentSafeName(v); var d=n.lastIndexOf('.'); return d>0 ? n.slice(d+1).replace(/[^A-Za-z0-9+-]/g,'').slice(0,10).toUpperCase() || 'FILE' : 'FILE'; }
  function _attachmentItemBadge(item) { if (item && item.kind === 'page') return item.contextRole === 'current' ? 'PAGE' : 'MD'; return _attachmentExtension(item && item.name); }
  function _normalizeContextPageUrl(v) { return String(v || ''); }
  function _attachmentLineCount(v) { return String(v || '').split(/\\r?\\n/).length; }
  var textItem = {name:'api.md',kind:'text',previewText:'hello',size:5,lineCount:1};
  var imageItem = {name:'diagram.png',kind:'image',size:1200,lineCount:0};
  var _composerAttachments = [textItem, imageItem];
  var _composerReplayAttachmentContext = 'Attachment: old.txt (text/plain)\\nlegacy';
  ${sanitize}
  ${liveView}
  ${replayNames}
  ${snapshot}
  return { sanitize:_sanitizeTurnAttachmentSummaries, replay:_replayAttachmentNames, snapshot:_composerTurnAttachmentSnapshot, items:_composerAttachments };
`);
const rt = factory();
const pages = {sources:[
  {name:'Introduction',kind:'page',contextRole:'current',sourceUrl:'https://x/intro',text:'intro',previewText:'intro',lineCount:1},
  {name:'API Reference',kind:'page',contextRole:'pinned',sourceUrl:'https://x/api',text:'api',previewText:'api',lineCount:1}
]};
const plan = {text:'Attachment: api.md (text/markdown)\nhello\n\nAttachment: old.txt (text/plain)\nlegacy', included:[{item:rt.items[0],boundedExcerpt:false}], budgetExcluded:[]};
const snap = rt.snapshot(plan, pages);
ok(snap.length === 5, 'runtime snapshot includes current page, pinned page, sent text, local-only image, and replay');
ok(snap[0].kind === 'page' && snap[0].status === 'Current page · used once', 'runtime current PAGE is one-shot');
ok(snap[1].kind === 'page' && snap[1].status === 'Pinned page · used once', 'runtime pinned MD is one-shot');
ok(snap[2].name === 'api.md' && snap[2].included === true && snap[2].status === 'Included once', 'runtime fresh text is marked Included once');
ok(snap[3].name === 'diagram.png' && snap[3].localOnly === true && snap[3].status.includes('not sent'), 'runtime image remains visibly local-only');
ok(snap[4].name === 'old.txt' && snap[4].replay === true && snap[4].status === 'Reused once', 'runtime Retry/Edit file context is marked Reused once');
ok(rt.sanitize(snap).every(x => !('previewText' in x) && !('file' in x)), 'persisted summaries strip live preview bodies and File objects');
ok(rt.sanitize(Array.from({length:600}, (_,i)=>({name:'f'+i+'.txt'}))).length === 512, 'runtime metadata sanitizer enforces independent live-resource bound');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
