// Run 173 T16 - transcript scale, measured rather than assumed.
//
// Virtualising the transcript was deferred three times "pending a measured
// rendering problem". Deferring because measurement says it is unnecessary and
// deferring because it looks expensive are indistinguishable from outside, so
// this supplies the evidence and keeps supplying it.
//
// Assertions are on ELEMENT COUNT and PERSISTED SIZE, both deterministic.
// Timing is printed for information only: a timing assertion in CI measures the
// runner's load as much as the code's cost, and a gate that fails for unrelated
// reasons is a gate people learn to ignore.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
function extract(n){
  const st = src.indexOf('function '+n+'('); if (st<0) throw new Error('missing '+n);
  let d=0,b=false,q='',e=false,l=false,bl=false,re=false,cls=false,prev='';
  for(let i=st;i<src.length;i++){const c=src[i],x=src[i+1];
    if(l){if(c==='\n')l=false;continue;} if(bl){if(c==='*'&&x==='/'){bl=false;i++;}continue;}
    if(re){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c==='['){cls=true;continue;}if(c===']'){cls=false;continue;}if(c==='/'&&!cls)re=false;continue;}
    if(q){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c===q)q='';continue;}
    if(c==='/'&&x==='/'){l=true;i++;continue;} if(c==='/'&&x==='*'){bl=true;i++;continue;}
    if(c==='/'){ if(/[(,=:[!&|?{};+\-*%^~<>]/.test(prev)||/\breturn$/.test(src.slice(Math.max(0,i-6),i))){re=true;cls=false;} continue; }
    if(c==='"'||c==="'"||c==='`'){q=c;continue;}
    if(c==='{'){d++;b=true;} else if(c==='}'){ if(--d===0&&b) return src.slice(st,i+1); }
    if(!/\s/.test(c)) prev=c;
  } throw new Error('unterminated '+n);
}
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};
const fenceInfo = info => {
  const m = /(?:^|\s)(?:file|filename|path)=([^\s]+)/i.exec(info || '');
  return { lang: (info || '').split(/\s+/)[0] || '', path: m ? m[1] : '' };
};
const safePath = v => String(v).includes('..') ? '' : String(v);
const md = new Function('_escapeHtml','_parseCodeFenceInfo','_generatedArtifactSafePath','_highlightCode','_langLabel',
  extract('_mdToHtml') + '\nreturn _mdToHtml;')(
  s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),
  fenceInfo, safePath, c => c, l => l);
const elide = new Function('_parseCodeFenceInfo','_generatedArtifactSafePath','_RAW_FILE_MARKER_PREFIX','_RAW_FILE_MARKER_SUFFIX','_RAW_ELIDE_MIN_CHARS',
  extract('_elideFileBodiesForRaw') + '\nreturn _elideFileBodiesForRaw;')(
  fenceInfo, safePath, '\u27e6ai-assistant:file-body ', '\u27e7', 2000);

// Read the ceilings from the source so this cannot drift from what it defends.
const maxTurns = Number((src.match(/_TRANSCRIPT_MAX_TURNS_DEFAULT\s*=\s*(\d+)/) || [])[1]);
const maxStorage = Number((src.match(/_TRANSCRIPT_RESTORE_MAX_STORAGE_CHARS\s*=\s*(\d+)/) || [])[1]);
ok(maxTurns === 200, 'the transcript ceiling is the value this harness sizes against');
ok(maxStorage === 2000000, 'the storage ceiling is the value this harness sizes against');

const prose = 'This paragraph explains the change in reasonable detail, with a `code span` and a [link](https://example.invalid/docs).\n\n';
const bigFile = Array.from({length:600},(_,i)=>'.. line '+i+' of the reStructuredText source').join('\n')+'\n';
const turn = withFile => prose.repeat(3) + (withFile ? '```rst file=docs/index.rst\n'+bigFile+'```\n\n' : '') + prose;
const elementsIn = html => (html.match(/<[a-zA-Z]/g) || []).length;
function measure(label, fileEvery) {
  const turns = Array.from({length:maxTurns},(_,i)=> turn(fileEvery && i % fileEvery === 0));
  const t0 = performance.now();
  let els = 0;
  for (const t of turns) els += elementsIn(md(t));
  console.error(`  ${label.padEnd(22)} elements=${els}  mdToHtml=${(performance.now()-t0).toFixed(0)}ms`);
  return els;
}
const proseOnly = measure('prose only', 0);
const quarter   = measure('file every 4th turn', 4);
const everyTurn = measure('file every turn', 1);

// A fenced file is ONE <pre><code> however many lines it holds. That is why
// element count barely moves as file content grows, and why virtualising would
// buy little: the DOM node count is not where the size goes.
ok(proseOnly < 4000, 'a full prose transcript stays well under a DOM-weight concern');
ok(everyTurn < 5000, 'the worst case -- every turn carrying a 600-line file -- stays under it too');
ok(everyTurn - proseOnly < proseOnly, 'file content adds elements sublinearly: a fence is one block, not one node per line');
ok(quarter > proseOnly && everyTurn > quarter, 'the measurement still responds to load, so a regression would show');

// Persisted size is where file bodies did hurt, and where T14 acted.
const heavy = Array.from({length:maxTurns},(_,i)=> turn(i % 4 === 0));
const before = JSON.stringify(heavy).length;
const after  = JSON.stringify(heavy.map(elide)).length;
console.error(`  persisted data-raw     before=${(before/1e6).toFixed(2)}MB after=${(after/1e6).toFixed(2)}MB (-${(100*(1-after/before)).toFixed(1)}%)`);
ok(before > after * 5, 'eliding file bodies cuts persisted transcript size by most of it');
ok(after < maxStorage / 4, 'an elided worst case leaves the storage ceiling real headroom');
ok(before < maxStorage, 'even un-elided this shape fits, so the win is headroom rather than a fix for breakage');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
