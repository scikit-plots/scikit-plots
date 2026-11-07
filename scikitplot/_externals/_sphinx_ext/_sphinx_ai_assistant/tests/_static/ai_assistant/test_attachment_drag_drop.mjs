// Run 62 regression: attachment picker/Alt+U/OS drag-drop are three entry
// surfaces into one bounded local staging pipeline. File drag handling must not
// intercept non-file drags or AltGr text entry.
import fs from 'node:fs';

const js = fs.readFileSync(process.argv[2], 'utf8');
const cssPath = process.argv[3] || process.argv[2].replace(/ai-assistant\.js$/, 'ai-assistant.css');
const css = fs.readFileSync(cssPath, 'utf8');
let pass = 0, fail = 0;
function t(name, got, want=true) {
  if (got === want) pass++;
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
}

t('menu advertises Alt+U', js.includes("uploadItem.setAttribute('aria-keyshortcuts', 'Alt+U')"));
t('visible keycap says Alt U', js.includes('<kbd>Alt</kbd><kbd>U</kbd>'));
t('Alt+U is panel scoped', /panel\.addEventListener\('keydown',[\s\S]*?e\.altKey && !e\.ctrlKey[\s\S]*?_openAttachmentPicker\(\)/.test(js));
t('physical KeyU handles Option dead-key layouts', js.includes("e.code === 'KeyU'"));
t('AltGr is excluded by ctrl guard', js.includes('e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey'));
t('shortcut sheet documents attachment chord', js.includes("shortcutRow('Add files or photos', ['Alt', 'U']"));
t('real file-drag detector exists', js.includes('function _attachmentDragHasFiles(eventOrTransfer)'));
t('non-file drags are not blindly accepted', /_attachmentDragHasFiles[\s\S]*String\(types\[i\]\) === 'Files'/.test(js));
t('drop extractor retains directories for bounded inventory rather than flattening them into files', js.includes('entry && entry.isDirectory') && js.includes('directoryEntries.push(entry)'));
t('drop uses same staging queue as picker', /panel\.addEventListener\('drop',[\s\S]*?_queueComposerFiles\(dropped\.files\)/.test(js));
t('file picker uses same staging queue', /attachInput\.addEventListener\('change',[\s\S]*?_queueComposerFiles\(files\)/.test(js));
t('drop overlay has canonical id', js.includes("attachmentDropOverlay.id = 'ai-assistant-panel-attachment-drop-overlay'"));
t('overlay positions from live panel/body rects', js.includes('br.left - pr.left') && js.includes('br.top - pr.top') && js.includes('br.width') && js.includes('br.height'));
t('overlay closes attachment menu during drag', /dragenter[\s\S]*?_closeAttachMenu\(false\)/.test(js));
t('dragover remains available for metadata-only ZIP/folder inventory when composer is full', js.includes("if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy'") && js.includes('folders/ZIPs may still be inventoried safely'));
t('full composer overlay distinguishes direct staging from ZIP/folder inventory', js.includes("attachmentDropOverlay.setAttribute('data-full', full ? 'true' : 'false')") && js.includes('Composer full · ZIP/folder inventory still available') && js.includes('Direct files need a free slot'));
t('drag depth prevents child churn flicker', js.includes('var attachmentDragDepth = 0') && js.includes('attachmentDragDepth++') && js.includes('attachmentDragDepth--'));
t('drop state is reset on blur/dragend', js.includes("document.addEventListener('dragend', _hideAttachmentDropOverlay, true)") && js.includes("window.addEventListener('blur', _hideAttachmentDropOverlay)"));
t('drop overlay resize tracks panel body', js.includes('attachmentDropResizeObserver.observe(body)'));
t('drop overlay is pointer inert', /\.ai-assistant-panel-attachment-drop-overlay\s*\{[\s\S]*pointer-events:\s*none;/.test(css));
t('drop overlay is panel absolute', /\.ai-assistant-panel-attachment-drop-overlay\s*\{[\s\S]*position:\s*absolute;/.test(css));
t('drop card uses dashed affordance', /\.ai-assistant-panel-attachment-drop-card\s*\{[\s\S]*border:\s*2px dashed/.test(css));
t('reduced-motion-safe animation gate', css.includes('@media (prefers-reduced-motion: no-preference)'));
t('all file entry points serialize through one queue', js.includes('var _attachmentStageQueue = Promise.resolve()') && js.includes('function _queueComposerFiles(fileList)'));
t('clear invalidates in-flight file reads', js.includes('_attachmentStageGeneration++') && js.includes('expectedGeneration !== _attachmentStageGeneration'));
t('queue snapshots ephemeral FileList objects', js.includes('var files = Array.prototype.slice.call(fileList || [])'));
t('drop UI resets when panel minimizes/closes', /function minimizeAIPanel\(\)[\s\S]*?_resetAttachmentDropUi/.test(js) && /function closeAIPanel\(\)[\s\S]*?_resetAttachmentDropUi/.test(js));
t('items-null fallback reaches DataTransfer.files', js.includes('Some engines expose DataTransfer.items') && js.includes('var files = dataTransfer.files'));
t('menu teaches drag-and-drop', js.includes('Choose files or drag them into the assistant.'));

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
