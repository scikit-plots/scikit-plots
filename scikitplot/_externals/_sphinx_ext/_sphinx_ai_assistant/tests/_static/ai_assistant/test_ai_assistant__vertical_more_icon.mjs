// Run 67 — More disclosure uses a vertical ellipsis glyph.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let pass=0, fail=0;
function t(name, got, want=true){ if(got===want) pass++; else {fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`);} }
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth=0, started=false;
  for (let j=i;j<src.length;j++) {
    if(src[j]==='{'){depth++;started=true;}
    else if(src[j]==='}'){depth--;if(started&&depth===0)return src.slice(i,j+1);}
  }
  throw new Error('unbalanced: ' + name);
}
const menu = extract('_buildHamburgerMenu');
t('More icon uses vertical ellipsis U+22EE', /moreIcon\.textContent = '⋮';/.test(menu));
t('More icon no longer uses horizontal ellipsis', !/moreIcon\.textContent = '…';/.test(menu));
t('More icon remains decorative', /moreIcon\.setAttribute\('aria-hidden', 'true'\)/.test(menu));
t('More keeps the same icon class', /moreIcon\.className = 'ai-assistant-panel-hamburger-item-icon'/.test(menu));
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
