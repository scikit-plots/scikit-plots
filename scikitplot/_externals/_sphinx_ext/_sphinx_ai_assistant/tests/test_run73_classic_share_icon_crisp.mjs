import fs from 'node:fs';
const js = fs.readFileSync(new URL('../_static/ai-assistant.js', import.meta.url), 'utf8');
const css = fs.readFileSync(new URL('../_static/ai-assistant.css', import.meta.url), 'utf8');
const svg = fs.readFileSync(new URL('../_static/share-link.svg', import.meta.url), 'utf8');
let n=0; function t(name, ok){n++; if(!ok) throw new Error('FAIL '+name); console.log('ok '+n+' - '+name)}
t('classic share path restored in JS', /linkMode:[^\n]*M3\.75 6\.5a\.25\.25/.test(js));
t('classic share path restored in static svg', svg.includes('M3.75 6.5a.25.25'));
t('link mode remains fill-authored', /linkMode: '<svg viewBox="0 0 16 16" fill="currentColor">/.test(js));
t('trigger filled icon keeps native 16px width', /\.ai-assistant-export-trigger > span:first-child svg\[fill="currentColor"\][\s\S]*?width: 16px !important/.test(css));
t('trigger filled icon keeps native 16px height', /\.ai-assistant-export-trigger > span:first-child svg\[fill="currentColor"\][\s\S]*?height: 16px !important/.test(css));
t('trigger filled icon disables stroke', /\.ai-assistant-export-trigger > span:first-child svg\[fill="currentColor"\][\s\S]*?stroke: none !important/.test(css));
t('menu filled icon keeps 16px grid', /\.ai-assistant-export-menu-mode-icon svg\[fill="currentColor"\][\s\S]*?width: 16px !important/.test(css));
t('tray upload geometry remains a separate upload icon', /upload:[^\n]*M2\.75 14A1\.75 1\.75/.test(js));
console.log(`1..${n}`);
