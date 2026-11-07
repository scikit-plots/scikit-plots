import fs from 'node:fs';
const js = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let pass=0, fail=0;
function t(name, got, want=true){ if(got===want){pass++;console.log('ok - '+name);} else {fail++;console.error('not ok - '+name);} }

t('linkMode remains a native 16x16 filled icon', /linkMode: '<svg viewBox="0 0 16 16" fill="currentColor">/.test(js));
t('syncRetry remains a native 16x16 filled icon', /syncRetry: '<svg viewBox="0 0 16 16" fill="currentColor"/.test(js));

t('generic bubble SVG rule no longer forces stroke paint', !/\.ai-assistant-panel-bubble-action svg\s*\{[^}]*stroke:\s*currentColor;[^}]*fill:\s*none;/s.test(css));
t('filled bubble icons force fill currentColor', /\.ai-assistant-panel-bubble-action svg\[fill="currentColor"\]\s*\{[^}]*fill:\s*currentColor !important;[^}]*stroke:\s*none !important;/s.test(css));
t('retry native 16px grid renders at 16 CSS px', /\.ai-assistant-panel-bubble-action--retry svg\[viewBox="0 0 16 16"\]\s*\{[^}]*width:\s*16px !important;[^}]*height:\s*16px !important;/s.test(css));

t('generic panel icon rule no longer forces stroke paint', !/\.ai-assistant-panel-icon-btn svg\s*\{[^}]*stroke:\s*currentColor;[^}]*fill:\s*none;/s.test(css));
t('filled panel icons force fill currentColor', /\.ai-assistant-panel-icon-btn svg\[fill="currentColor"\]\s*\{[^}]*fill:\s*currentColor !important;[^}]*stroke:\s*none !important;/s.test(css));

t('link-mode export trigger renders filled 16px icon 1:1', /\.ai-assistant-export-trigger > span:first-child svg\[fill="currentColor"\]\s*\{[^}]*width:\s*16px !important;[^}]*height:\s*16px !important;[^}]*fill:\s*currentColor !important;[^}]*stroke:\s*none !important;/s.test(css));
t('share-sheet trigger renders filled 16px icon 1:1', /\.ai-assistant-share-export-trigger-icon svg\[fill="currentColor"\]\s*\{[^}]*width:\s*16px !important;[^}]*height:\s*16px !important;[^}]*fill:\s*currentColor !important;[^}]*stroke:\s*none !important;/s.test(css));
t('link-mode menu control keeps filled paint', /\.ai-assistant-export-menu-mode-icon svg\[fill="currentColor"\]\s*\{[^}]*width:\s*16px !important;[^}]*height:\s*16px !important;[^}]*fill:\s*currentColor !important;[^}]*stroke:\s*none !important;/s.test(css));
t('share-sheet link-mode control keeps filled paint', /\.ai-assistant-share-export-mode-icon svg\[fill="currentColor"\]\s*\{[^}]*width:\s*16px !important;[^}]*height:\s*16px !important;[^}]*fill:\s*currentColor !important;[^}]*stroke:\s*none !important;/s.test(css));

t('stroke-authored panel icons retain stroke paint', /\.ai-assistant-panel-icon-btn svg\[fill="none"\]\s*\{[^}]*fill:\s*none !important;[^}]*stroke:\s*currentColor !important;/s.test(css));
t('stroke-authored bubble icons retain stroke paint', /\.ai-assistant-panel-bubble-action svg\[fill="none"\]\s*\{[^}]*fill:\s*none !important;[^}]*stroke:\s*currentColor !important;/s.test(css));

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
