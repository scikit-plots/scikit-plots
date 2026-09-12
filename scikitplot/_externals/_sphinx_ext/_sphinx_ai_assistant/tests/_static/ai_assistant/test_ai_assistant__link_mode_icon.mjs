import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let n=0; function t(name, ok){n++; if(!ok){throw new Error('FAIL '+name)} console.log('ok '+n+' - '+name)}
const desired = 'linkMode: \'<svg viewBox="0 0 16 16" fill="currentColor"><path d="M3.75 6.5a.25.25';
t('linkMode uses classic Octicon share geometry', src.includes(desired));
t('tray upload geometry is not reused for linkMode', !/linkMode:[^\n]*M2\.75 14A1\.75 1\.75/.test(src));
t('mode resolver still selects semantic linkMode registry entry', /return linkMode \? ICONS\.linkMode : ICONS\.exportTxt/.test(src));
t('link icon stays inline/currentColor', /linkMode: '<svg viewBox="0 0 16 16" fill="currentColor">/.test(src));
t('no external icon dependency added', !/linkMode:[^\n]*(?:https?:|sprites-core)/.test(src));
console.log(`1..${n}`);
