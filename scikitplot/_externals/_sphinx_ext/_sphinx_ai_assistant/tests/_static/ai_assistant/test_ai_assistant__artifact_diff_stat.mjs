// Run 173 T4 - deterministic +added / -removed revision statistics.
//
// Guards two things a screenshot cannot: that the counts are correct for
// the ordinary edit shapes, and that colour is never the sole carrier of
// the meaning (sign characters in the text, aria-label on the wrapper).
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
function extract(n){const st=src.indexOf('function '+n+'(');if(st<0)throw new Error('missing '+n);
 let d=0,b=false,q='',e=false,l=false,bl=false;
 for(let i=st;i<src.length;i++){const c=src[i],x=src[i+1];
  if(l){if(c==='\n')l=false;continue;} if(bl){if(c==='*'&&x==='/'){bl=false;i++;}continue;}
  if(q){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c===q)q='';continue;}
  if(c==='/'&&x==='/'){l=true;i++;continue;} if(c==='/'&&x==='*'){bl=true;i++;continue;}
  if(c==='"'||c==="'"||c==='`'){q=c;continue;} if(c==='{'){d++;b=true;}else if(c==='}'&&--d===0&&b)return src.slice(st,i+1);}
 throw new Error('unterminated '+n);}
const diff = new Function('_DIFF_STAT_MAX_LINES','_DIFF_STAT_LCS_BUDGET',
  [extract('_diffStatSplitLines'),extract('_diffStatMultiset'),extract('_diffStatLcs'),
   extract('_diffLineStat'),'return _diffLineStat;'].join('\n'))(20000, 4000000);
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};
const eq=(r,a,d)=>r.added===a&&r.removed===d;
ok(eq(diff(null,'a\nb\nc\n'),3,0),'new file counts every line as added');
ok(eq(diff('a\nb\nc\n',null),0,3),'emptied file counts every line as removed');
ok(eq(diff('a\nb\nc\n','a\nb\nc\n'),0,0),'identical content reports no change');
ok(eq(diff('a\nb\nc\n','a\nb\nc\nd\n'),1,0),'pure append is +1 / -0');
ok(eq(diff('a\nb\nc\n','a\nc\n'),0,1),'pure deletion is +0 / -1');
ok(eq(diff('a\nb\nc\n','a\nB\nc\n'),1,1),'edit-in-place is +1 / -1');
ok(eq(diff('a\nb\n','x\ny\nz\n'),3,2),'full rewrite counts both sides');
ok(eq(diff('a\nb\nc','a\nb\nc\n'),0,0),'trailing newline alone is not a change');
ok(eq(diff('a\r\nb\r\n','a\nb\n'),0,0),'CRLF and LF compare equal');
ok(diff('a\nb\nc\n','a\nB\nc\n').exact===true,'small diffs report exact');
const big = Array.from({length:3000},(_,i)=>'line '+i).join('\n');
const big2 = big.replace(/line 1500/,'CHANGED');
const r = diff(big,big2);
ok(eq(r,1,1) && r.exact===true,'3000-line file with one edit stays exact via prefix/suffix trim');
const t0=Date.now(); diff(big, Array.from({length:3000},(_,i)=>'x '+(i*7%3000)).join('\n'));
ok(Date.now()-t0 < 5000,'worst-case 3000x3000 completes inside the budget');
// Presentation contract: the paired stylesheet must define both tones for
// light and dark, and must survive forced-colours mode.
const css = fs.readFileSync(process.argv[3], 'utf8');
ok(css.includes('--ai-diff-add-rgb') && css.includes('--ai-diff-del-rgb'),'diff tones are design tokens, not literals');
ok(css.includes('[data-bs-theme="dark"]') && /--ai-diff-add-rgb:\s*45, 212, 191/.test(css),'dark theme redefines the addition tone');
ok(css.includes('forced-colors: active'),'forced-colours mode is handled');
ok(/\.ai-assistant-panel-diff-stat-add\s*\{[^}]*color:\s*rgb\(var\(--ai-diff-add-rgb\)\)/.test(css),'addition class consumes its token');
ok(src.includes("plus.setAttribute('aria-hidden', 'true')") && src.includes("minus.setAttribute('aria-hidden', 'true')"),'coloured numbers are hidden from assistive tech');
ok(src.includes("wrap.setAttribute('aria-label', label)") && src.includes("' lines added'") && src.includes("' lines removed'"),'wrapper carries one full accessible sentence');
ok(src.includes("plus.textContent = '+' + added") && src.includes("minus.textContent = '\\u2212' + removed"),'sign characters carry meaning without colour');
ok(src.includes('diff: _diffLineStat(old && old.content, content)'),'stat is computed once at registration, not by retaining old bytes');
// The stat and the patch must never disagree about what changed: both run the
// same prefix/suffix trim, so a divergence means one of them regressed.
const dsrc = src;
ok(dsrc.includes('function _diffOps(before, after)') && dsrc.includes('_diffStatSplitLines(beforeText)'),'diff stat and unified diff share one line splitter');
// Patch export moved from a card button into the ⋮ menu; the capability is
// unchanged and is asserted where it now lives.
ok(dsrc.includes("{ label: 'Download patch'")&&dsrc.includes('_generatedArtifactDownloadPatch(key);'),'patch export is reachable from every file row');
ok(/\.ai-assistant-panel-changed-file-menu-item\s*\{/.test(css),'patch export inherits the styled overflow-menu item contract');
ok(/\.ai-assistant-panel-changed-file-primary\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\) auto/.test(css),'the Presented row is one artifact group plus one overflow trigger');
ok(!/\.ai-assistant-panel-changed-file-primary\s*\{[^}]*auto auto/.test(css),'no obsolete direct patch/download column survives');
// ── Artifact accent ────────────────────────────────────────────────────────
// Four controls, one idea -- "take this file". They were drifting: the snippet
// download label carried the theme accent, the presented-file and bulk
// downloads inherited body text, and the section heading was plain bold.
const accentRule = (css.match(/button\.ai-md-artifact-download-label,[\s\S]*?\}/) || [''])[0];
ok(/--ai-artifact-accent:\s*var\(--pst-color-primary/.test(css),'the accent is a token over the theme colour, not a literal');
ok(/\[data-bs-theme="dark"\]\s*\{[^}]*--ai-artifact-accent/.test(css),'the dark theme carries its own accent rather than reusing the light one');
['button.ai-md-artifact-download-label',
 '.ai-assistant-panel-changed-file-download',
 '.ai-assistant-panel-changed-files-download-all',
 '.ai-assistant-panel-changed-files-head strong'].forEach(function (sel) {
  ok(accentRule.includes(sel), 'accent covers ' + sel);
});
ok(accentRule.includes('color: var(--ai-artifact-accent)'),'they all read the same token');
ok(/:hover,[\s\S]{0,400}?color:\s*var\(--ai-artifact-accent\)/.test(css),'hover keeps the accent rather than dropping to body text');
ok(/forced-colors: active[\s\S]{0,500}?color:\s*LinkText/.test(css),'forced-colours modes get a system colour, not a discarded hue');
// The regression this rule fixes: a later `color: inherit` silently overrode
// the accent on the one control that already had it.
ok(!/button\.ai-md-artifact-download-label\s*\{[^}]*color:\s*inherit/.test(css),'no later rule resets the download label to inherited text');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
