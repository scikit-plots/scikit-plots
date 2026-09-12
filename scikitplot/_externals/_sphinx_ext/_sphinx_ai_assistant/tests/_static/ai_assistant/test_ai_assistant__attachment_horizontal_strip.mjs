// Run 66 regression: many staged attachments remain in one horizontal strip
// with discoverable overflow and wheel/touch ergonomics. No attachment data
// or transport authority changes belong to this run.
import fs from 'node:fs';

const js = fs.readFileSync(process.argv[2], 'utf8');
const cssPath = process.argv[3] || process.argv[2].replace(/ai-assistant\.js$/, 'ai-assistant.css');
const css = fs.readFileSync(cssPath, 'utf8');
let pass = 0, fail = 0;
function t(name, got, want=true) {
  if (got === want) pass++;
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
}

t('attachment tray is one horizontal row', /\.ai-assistant-panel-attachments\s*\{[\s\S]*flex-direction:\s*row;[\s\S]*flex-wrap:\s*nowrap;/.test(css));
t('tray is width bounded and horizontally scrollable', /\.ai-assistant-panel-attachments\s*\{[\s\S]*width:\s*100%;[\s\S]*min-width:\s*0;[\s\S]*overflow-x:\s*auto;[\s\S]*overflow-y:\s*hidden;/.test(css));
t('cards remain fixed 120px tiles', /\.ai-assistant-panel-attachment-tile\s*\{[\s\S]*width:\s*120px;[\s\S]*height:\s*120px;[\s\S]*flex:\s*0\s+0\s+120px;/.test(css));
t('composer can shrink around overflowing strip', /\.ai-assistant-panel-input-group\s*\{[\s\S]*min-width:\s*0;[\s\S]*max-width:\s*100%;/.test(css));
t('touch momentum scrolling supported', css.includes('-webkit-overflow-scrolling: touch') && css.includes('touch-action: pan-x pan-y pinch-zoom'));
t('horizontal overscroll is contained', css.includes('overscroll-behavior-inline: contain'));
t('thin horizontal scrollbar is styled', css.includes('.ai-assistant-panel-attachments::-webkit-scrollbar') && css.includes('height: 6px'));
t('overflow state helper exists', js.includes('function _updateAttachmentTrayOverflow(tray)'));
t('left overflow affordance is tracked', js.includes("tray.toggleAttribute('data-overflow-start'"));
t('right overflow affordance is tracked', js.includes("tray.toggleAttribute('data-overflow-end'"));
t('edge shadows are conditional', css.includes('[data-overflow-start][data-overflow-end]') && css.includes('inset -14px 0 12px -14px'));
t('wheel translation is explicitly non-passive', js.includes("tray.addEventListener('wheel', function (e)") && js.includes('{ passive: false }'));
t('native horizontal wheel/trackpad remains native', js.includes('if (Math.abs(dx) >= Math.abs(dy) || Math.abs(dy) < 1) return;'));
t('browser zoom gestures are not intercepted', js.includes('if (!e || e.ctrlKey || e.metaKey) return;'));
t('vertical wheel only captured when horizontal position changes', js.includes('if (Math.abs(next - before) < 0.5) return;') && js.includes('e.preventDefault();'));
t('page scroll remains available at horizontal edges', js.includes('Math.abs(next - before) < 0.5'));
t('tray responds to resize', js.includes("typeof ResizeObserver === 'function'") && js.includes('_aiAttachmentResizeObserver'));
t('new attachment auto-reveals end of strip', js.includes('if (currentCount > previousCount)') && js.includes("tray.scrollTo({ left: max"));
t('reduced motion disables smooth reveal', js.includes("prefers-reduced-motion: reduce") && js.includes("behavior: reduced ? 'auto' : 'smooth'"));
t('removal preserves nearest scroll position', js.includes('tray.scrollLeft = Math.min(previousScrollLeft, max)'));
t('empty tray clears overflow state', js.includes("tray.removeAttribute('data-overflow-start')") && js.includes("tray.removeAttribute('data-overflow-end')"));
t('overflow accessibility hint is dynamic', js.includes('Context and attached files. Scroll horizontally for more items.'));

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
