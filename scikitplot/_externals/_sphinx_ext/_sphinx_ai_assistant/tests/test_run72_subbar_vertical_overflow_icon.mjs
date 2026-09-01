// Run 72 — subbar overflow uses shared vertical three-dot SVG.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let pass=0, fail=0;
function t(name, got, want=true){ if(got===want) pass++; else {fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`);} }

t('shared horizontal overflow icon remains available', /overflowH:\s*'<svg[^']*<circle cx="5" cy="12" r="1\.5"\/><circle cx="12" cy="12" r="1\.5"\/><circle cx="19" cy="12" r="1\.5"\/>/.test(src));
t('shared vertical overflow icon exists', /overflowV:\s*'<svg[^']*<circle cx="12" cy="5" r="1\.5"\/><circle cx="12" cy="12" r="1\.5"\/><circle cx="12" cy="19" r="1\.5"\/>/.test(src));
t('subbar overflow uses vertical icon', /rightOverflowBtn\.innerHTML = ICONS\.overflowV;/.test(src));
t('subbar overflow no longer uses horizontal icon', !/rightOverflowBtn\.innerHTML = ICONS\.overflowH;/.test(src));
t('overflow button accessible label unchanged', /rightOverflowBtn\.setAttribute\('aria-label', 'More options'\)/.test(src));
t('overflow button remains menu trigger', /rightOverflowBtn\.setAttribute\('aria-haspopup', 'menu'\)/.test(src));
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
