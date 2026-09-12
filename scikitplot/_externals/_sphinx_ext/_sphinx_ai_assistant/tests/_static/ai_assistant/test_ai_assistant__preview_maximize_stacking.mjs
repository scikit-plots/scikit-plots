import fs from 'node:fs';

const css = fs.readFileSync(process.argv[3], 'utf8');
let pass = 0;
let fail = 0;
function t(name, ok) { if (ok) { pass++; console.log(`ok - ${name}`); } else { fail++; console.error(`not ok - ${name}`); } }
const max = css.match(/\.ai-assistant-panel\[data-maximized="true"\]\s*\{[\s\S]*?z-index:\s*(\d+)\s*;/);
const preview = css.match(/\.ai-assistant-panel-attachment-preview-layer\s*\{[\s\S]*?z-index:\s*(\d+)\s*;/);
t('maximized panel has explicit bounded z-index', !!max);
t('viewport preview has explicit z-index', !!preview);
t('preview is strictly above maximized panel', !!max && !!preview && Number(preview[1]) > Number(max[1]));
t('preview owns CSS integer ceiling', !!preview && Number(preview[1]) === 2147483647);
t('maximized panel remains far above ordinary theme stacking contexts', !!max && Number(max[1]) > 1_000_000);
console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
