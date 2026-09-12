// Run 64 regression: attachment preview is a true viewport-level modal.
import fs from 'node:fs';
const js = fs.readFileSync(process.argv[2], 'utf8');
const cssPath = process.argv[3] || process.argv[2].replace(/ai-assistant\.js$/, 'ai-assistant.css');
const css = fs.readFileSync(cssPath, 'utf8');
let pass=0, fail=0;
function t(name, got, want=true){ if(got===want) pass++; else { fail++; console.log(`  FAIL ${name}`); } }
t('layer mounts under document.body', js.includes('document.body.appendChild(layer)'));
t('layer no longer mounts under panel', !js.includes('panel.appendChild(layer)'));
t('dialog declares modal semantics', js.includes("dialog.setAttribute('aria-modal', 'true')"));
t('preview layer is fixed full viewport', /\.ai-assistant-panel-attachment-preview-layer\s*\{[\s\S]*position:\s*fixed;[\s\S]*inset:\s*0;/.test(css));
t('preview is above assistant stacking contexts', /\.ai-assistant-panel-attachment-preview-layer\s*\{[\s\S]*z-index:\s*2147483647;/.test(css));
t('dialog uses viewport width not panel percent', css.includes('width: min(720px, calc(100vw - 1rem'));
t('dialog uses dynamic viewport height', css.includes('max-height: calc(100dvh - 1rem'));
t('mobile breakpoint expands preview', css.includes('@media (max-width: 640px)') && css.includes('calc(100vw - 0.75rem'));
t('safe-area insets are respected', css.includes('env(safe-area-inset-left') && css.includes('env(safe-area-inset-bottom'));
t('drag clamp uses visual viewport', js.includes('var vv = window.visualViewport;') && js.includes('vv && vv.width') && js.includes('vv.offsetLeft') && js.includes('vv.offsetTop'));
t('visual viewport resize reclamps', js.includes("window.visualViewport.addEventListener('resize', _positionAttachmentPreviewLayer"));
t('visual viewport scroll reclamps', js.includes("window.visualViewport.addEventListener('scroll', _positionAttachmentPreviewLayer"));
t('orientation change reclamps', js.includes("window.addEventListener('orientationchange', _positionAttachmentPreviewLayer"));
t('focus trap handles Tab', js.includes("if (e.key === 'Tab')") && js.includes('var focusable = Array.prototype.slice.call(dialog.querySelectorAll'));
t('Escape still closes and restores focus', js.includes("if (e.key === 'Escape')") && js.includes('_closeAttachmentPreview(true)'));
t('background scroll is contained', /\.ai-assistant-panel-attachment-preview-layer\s*\{[\s\S]*overscroll-behavior:\s*contain;/.test(css));
t('preview body restores touch scrolling', /\.ai-assistant-panel-attachment-preview-body\s*\{[\s\S]*touch-action:\s*pan-x pan-y;/.test(css));
const posFn = (js.match(/function _positionAttachmentPreviewLayer\(\) \{[\s\S]*?\n    \}/) || [''])[0];
t('legacy panel-body positioning removed from preview positioning', !posFn.includes('ai-assistant-panel-body') && !posFn.includes('getBoundingClientRect'));
console.log(`${pass} passed, ${fail} failed`); if(fail) process.exit(1);
