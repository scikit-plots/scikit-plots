// Run 61/64 regression: staged attachments render as rich cards and open a
// draggable, viewport-level local preview. Preview/download never widens
// the text-only model transport authority.
import fs from 'node:fs';

const js = fs.readFileSync(process.argv[2], 'utf8');
const cssPath = process.argv[3] || process.argv[2].replace(/ai-assistant\.js$/, 'ai-assistant.css');
const css = fs.readFileSync(cssPath, 'utf8');
let pass = 0, fail = 0;
function t(name, got, want=true) {
  if (got === want) pass++;
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
}

t('composer uses attachment tiles', js.includes("tile.className = 'ai-assistant-panel-attachment-tile'"));
t('each tile owns a capability-truth preview/details button', js.includes("preview.className = 'ai-assistant-panel-attachment-card'") && js.includes("_attachmentInteractionAria(item)"));
t('image thumbnails use local object URLs', js.includes('_attachmentEnsureObjectUrl(item)') && js.includes("img.className = 'ai-assistant-panel-attachment-thumb-image'"));
t('remove revokes image object URL', js.includes('_attachmentRevokeObjectUrl(item)') && js.includes('_composerAttachments.splice(composerIndex, 1)'));
t('clear revokes every staged object URL', js.includes('_composerAttachments.forEach(_attachmentRevokeObjectUrl)'));
t('ipynb is a recognized text-like extension', /jsonl\|ipynb\|/.test(js));
t('local preview has larger cap than outbound text', js.includes('var _ATTACHMENT_MAX_PREVIEW_BYTES = 512 * 1024') && js.includes('var _ATTACHMENT_MAX_READ_BYTES = 256 * 1024'));
t('large text is staged metadata-first instead of rejected', js.includes("kind: 'text', modality: 'text'") && js.includes('Read on Send') && !js.includes("is too large. Text attachments are limited to"));
t('arbitrary files are first-class binary resources instead of silently local-only', js.includes("kind: 'file', modality: 'binary'") && js.includes('rawEligible: true'));
t('preview is a dialog', js.includes("dialog.setAttribute('role', 'dialog')"));
t('preview is viewport-level and independent of panel body', js.includes('document.body.appendChild(layer)') && !js.includes("st.layer.style.width = Math.max(1, br.width) + 'px'"));
t('preview advertises pinned open state', js.includes("st.layer.setAttribute('data-pinned', 'true')"));
t('dialog is draggable by header', js.includes("header.setAttribute('data-drag-handle', 'true')") && js.includes("if (moved < 3) return"));
t('drag is clamped to visual viewport', js.includes('function _clampAttachmentPreviewDialog()') && js.includes('window.visualViewport') && js.includes('width - st.dialog.offsetWidth'));
t('text preview uses textContent not innerHTML', js.includes('pre.textContent = item.previewText'));
t('oversized/non-previewable content offers local download', js.includes("dl.className = 'ai-assistant-panel-attachment-preview-download'") && js.includes('_downloadAttachmentItem(item)'));
t('download is a local blob URL', js.includes('URL.createObjectURL(item.file)') && js.includes('a.download = _attachmentSafeName(item.name)'));
t('image preview remains bounded', js.includes('var _ATTACHMENT_IMAGE_PREVIEW_MAX_BYTES = 12 * 1024 * 1024'));
t('safe text is lazily eligible for outbound attachment context', /_prepareComposerAttachmentPlan[\s\S]*item\.kind === 'text'[\s\S]*_readAttachmentText/.test(js));
t('large text preview explains separate shared send budget', js.includes('Showing only the bounded preview prefix. Send uses a separate shared 48k context budget.'));
t('preview closes on panel minimize', /function minimizeAIPanel\(\)[\s\S]*?_closeAttachmentPreview\(false\)/.test(js));
t('preview closes on panel close', /function closeAIPanel\(\)[\s\S]*?_closeAttachmentPreview\(false\)/.test(js));
t('cards are 120px claude-like tiles', /\.ai-assistant-panel-attachment-tile\s*\{[\s\S]*width:\s*120px;[\s\S]*height:\s*120px;/.test(css));
t('attachment tray is horizontal and bounded', /\.ai-assistant-panel-attachments\s*\{[\s\S]*flex-wrap:\s*nowrap;[\s\S]*overflow-x:\s*auto;/.test(css));
t('preview layer is fixed over viewport', /\.ai-assistant-panel-attachment-preview-layer\s*\{[\s\S]*position:\s*fixed;[\s\S]*inset:\s*0;/.test(css));
t('preview code wraps long text', /\.ai-assistant-panel-attachment-preview-code\s*\{[\s\S]*white-space:\s*pre-wrap;[\s\S]*overflow-wrap:\s*anywhere;/.test(css));
t('image preview uses object contain', /\.ai-assistant-panel-attachment-preview-image\s*\{[\s\S]*object-fit:\s*contain;/.test(css));

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
