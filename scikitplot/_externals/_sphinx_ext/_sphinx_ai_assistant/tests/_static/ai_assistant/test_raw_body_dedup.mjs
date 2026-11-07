// Run 173 T14 - data-raw holds the answer, not a second copy of every file.
//
// `data-raw` preserves answer markdown for copy, share and export. For a
// snippet answer that is right: the fenced code IS the answer. For a
// file-editing answer a 600-line file was held three times -- rendered <pre>,
// data-raw, and the artifact ledger -- and the data-raw copy is the one also
// serialized into session storage, competing with the persistence budget for
// bytes nobody reads.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
// Regex-aware extractor. The naive version used by the older harnesses treats
// the quotes and backticks INSIDE a regex literal as string delimiters, which
// silently truncates any function containing one -- and both functions under
// test here match fenced code, so both are full of them.
function extract(n){
  const st = src.indexOf('function '+n+'(');
  if (st < 0) throw new Error('missing '+n);
  let depth=0, began=false, quote='', esc=false, line=false, block=false, regex=false, cls=false;
  let prev='';
  for (let i=st; i<src.length; i++){
    const c = src[i], x = src[i+1];
    if (line){ if (c === '\n') line=false; continue; }
    if (block){ if (c === '*' && x === '/'){ block=false; i++; } continue; }
    if (regex){
      if (esc){ esc=false; continue; }
      if (c === '\\'){ esc=true; continue; }
      if (c === '['){ cls=true; continue; }
      if (c === ']'){ cls=false; continue; }
      if (c === '/' && !cls){ regex=false; }
      continue;
    }
    if (quote){
      if (esc){ esc=false; continue; }
      if (c === '\\'){ esc=true; continue; }
      if (c === quote) quote='';
      continue;
    }
    if (c === '/' && x === '/'){ line=true; i++; continue; }
    if (c === '/' && x === '*'){ block=true; i++; continue; }
    if (c === '/'){
      // A slash after an operator or opening bracket starts a regex; after a
      // value it is division. This distinction is all the extractor needs.
      if (/[(,=:[!&|?{};+\-*%^~<>]/.test(prev) || /\breturn$/.test(src.slice(Math.max(0,i-6), i))){
        regex = true; cls = false; continue;
      }
      continue;
    }
    if (c === '"' || c === "'" || c === '`'){ quote=c; continue; }
    if (c === '{'){ depth++; began=true; }
    else if (c === '}'){ if (--depth === 0 && began) return src.slice(st, i+1); }
    if (!/\s/.test(c)) prev = c;
  }
  throw new Error('unterminated '+n);
}
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};

const ledger = Object.create(null);
const api = new Function('_generatedArtifactLedger','_RAW_FILE_MARKER_PREFIX','_RAW_FILE_MARKER_SUFFIX','_RAW_ELIDE_MIN_CHARS','_generatedArtifactSafePath',
  [extract('_parseCodeFenceInfo'), extract('_elideFileBodiesForRaw'), extract('_bubbleRawText'),
   'return {elide:_elideFileBodiesForRaw, hydrate:_bubbleRawText};'].join('\n'))(
  ledger, '\u27e6ai-assistant:file-body ', '\u27e7', 2000,
  v => (String(v).includes('..') || String(v).startsWith('/')) ? '' : String(v));

const bigBody = Array.from({length:400},(_,i)=>'line '+i).join('\n') + '\n';
const answer = 'Here is the updated guide.\n\n```rst file=docs/index.rst\n' + bigBody + '```\n\nThat covers it.';
const elided = api.elide(answer);

ok(elided.length < answer.length / 4,'a large file body is elided from data-raw');
ok(elided.includes('\u27e6ai-assistant:file-body docs/index.rst\u27e7'),'the marker names the path it stands for');
ok(elided.includes('Here is the updated guide.') && elided.includes('That covers it.'),'the prose around the file is untouched');
ok(elided.includes('```rst file=docs/index.rst'),'the fence and its info string survive, so the shape is still markdown');

// Round-trip through the ledger must be exact: this is what copy and export use.
ledger['docs/index.rst'] = { content: bigBody };
ok(api.hydrate({ getAttribute: () => elided }, '') === answer,'rehydrating from the ledger reproduces the answer byte for byte');

// Ledger gone: fall back to the rendered pre still in the bubble.
//
// The stub models the DOM as data, because the accessor now compares the
// artifact path against each block's data-artifact-path attribute instead of
// interpolating it into a CSS selector.  A selector built by concatenation
// cannot be escaped correctly -- a trailing backslash in the path escapes the
// escape and re-opens the quoted attribute value -- and a malformed selector
// throws SyntaxError, aborting the export rather than falling back.
delete ledger['docs/index.rst'];
const preFor = (path, body) => ({
  getAttribute: name => name === 'data-artifact-path' ? path : null,
  querySelector: () => ({ textContent: body })
});
const bubbleFor = (raw, ...blocks) => ({
  getAttribute: () => raw,
  querySelectorAll: () => blocks
});
const bubbleWith = (...blocks) => bubbleFor(elided, ...blocks);
ok(api.hydrate(bubbleWith(preFor('docs/index.rst', bigBody)), '') === answer,'a missing ledger entry falls back to the rendered block');

// Exact match, never a prefix or a selector-shaped coincidence: a block for a
// different path must not satisfy the marker.
ok(api.hydrate(bubbleWith(preFor('docs/index.rst.bak', bigBody)), '').includes('\u27e6ai-assistant:file-body'),'a block for a different path is not accepted for this marker');
ok(api.hydrate(bubbleWith(preFor('other.rst', 'wrong'), preFor('docs/index.rst', bigBody)), '') === answer,'the matching block is found among several rendered blocks');

// A path carrying selector metacharacters resolves by value, so no escaping of
// any kind is involved and the previous quote-only escape cannot be bypassed.
const oddPath = 'docs/we"ird\\.rst';
const oddAnswer = 'x\n\n```rst file=' + oddPath + '\n' + bigBody + '```\n';
const oddElided = api.elide(oddAnswer);
ok(oddElided.includes('\u27e6ai-assistant:file-body ' + oddPath + '\u27e7'),'a path with quote and backslash still produces a marker');
ok(api.hydrate(bubbleFor(oddElided, preFor(oddPath, bigBody)), '') === oddAnswer,'a path with quote and backslash rehydrates by exact value');

// Neither available: leave the marker visible rather than shipping an empty file.
ok(api.hydrate({ getAttribute: () => elided, querySelectorAll: () => [] }, '').includes('\u27e6ai-assistant:file-body'),'an unresolvable marker stays visible rather than exporting an empty file');

// Snippets and small files are left exactly as they are.
const snippet = 'Try this:\n\n```python\nprint(1)\n```\n';
ok(api.elide(snippet) === snippet,'a short snippet answer is untouched');
const anon = 'x\n\n```python\n' + bigBody + '```\n';
ok(api.elide(anon) === anon,'an anonymous block has no ledger entry, so it is never elided');
const smallFile = 'x\n\n```rst file=a.rst\nshort\n```\n' + 'padding '.repeat(400);
ok(api.elide(smallFile).includes('short'),'a small file body costs less than its marker and is kept');
ok(api.hydrate(null, 'plain answer') === 'plain answer','a bubble without data-raw falls back to the answer text');
ok(api.hydrate({ getAttribute: () => 'no markers here' }, '') === 'no markers here','markerless raw text is returned unchanged');

// Wiring: every consumer goes through the accessor, no raw reads remain.
ok(src.includes('copyToClipboard(_bubbleRawText(bubbleEl, text), false, onSuccess);'),'copy rehydrates');
ok(src.includes('var raw    = _bubbleRawText(bubbleEl, answerText);'),'share rehydrates');
ok((src.match(/getAttribute\('data-raw'\)/g) || []).length === 1,'exactly one place reads data-raw directly, and it is the accessor');
ok(src.includes("bubble.setAttribute('data-raw', _elideFileBodiesForRaw(text));"),'the rendered bubble stores elided raw');
ok((src.match(/setAttribute\('data-raw', _elideFileBodiesForRaw/g) || []).length === 3,'every data-raw writer elides, including both streaming paths');

// A body with no trailing newline must round-trip too: the marker stands
// exactly where the body stood, so neither shape may gain or lose a line.
ledger['a.rst'] = { content: 'x'.repeat(2100) };
const noNl = 'p\n\n```rst file=a.rst\n' + 'x'.repeat(2100) + '```\n';
ok(api.hydrate({ getAttribute: () => api.elide(noNl) }, '') === noNl,'a body without a trailing newline round-trips exactly');
const cssSrc = fs.readFileSync(process.argv[3], 'utf8');
// The secondary disclosure row was replaced by a single ⋮ menu; extra actions
// are still hidden until asked for, now by not being rendered at all.
ok(/\.ai-assistant-panel-changed-file-menu\s*\{[^}]*position:\s*absolute/.test(cssSrc),'the overflow menu is styled as a popup, not an inline row');
ok(!/\.ai-assistant-panel-changed-file-secondary\b/.test(src),'the old secondary action row is gone');
ok(src.includes("btn.setAttribute('aria-haspopup', 'menu')"),'the extra actions are behind an announced menu trigger');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
