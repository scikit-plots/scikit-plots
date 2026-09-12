// Run 173 T9 - retry says what it does, and revisions count content.
//
// "Resend this question as-is" was true when a request was the question and
// nothing else. Once history and working files travel with it, identical text
// produces a different request, and the old label promises a replay this panel
// does not retain.
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
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};

const ledger = Object.create(null);
let transcript = [];
const receipts = [];
const api = new Function('_TURN_CONTEXT_RECEIPTS','_TURN_CONTEXT_RECEIPT_MAX','_generatedArtifactLedger','_transcriptRef',
  ['var _transcript = _transcriptRef.v;',
   extract('_artifactContentRevision'), extract('_recordTurnContextReceipt'),
   extract('_turnContextReceiptFor'), extract('_turnContextDrift'),
   'return {record:_recordTurnContextReceipt, drift:function(q){_transcript=_transcriptRef.v; return _turnContextDrift(q);}, find:_turnContextReceiptFor};'
  ].join('\n'))(receipts, 32, ledger, { get v(){ return transcript; } });

// ── The label must not claim an exact replay ──────────────────────────────
// Scoped to user-facing strings: the explanatory comments in the source may
// legitimately quote the old label while explaining why it was wrong.
const uiStrings = (src.match(/(?:setAttribute\('(?:aria-label|title)',\s*|\.title\s*=\s*)'[^']*'/g) || []).join('\n');
ok(!/as-is/.test(uiStrings),'no user-facing label still claims an as-is replay');
ok(!/(aria-label|title)[^\n]*Retry\s*\u2014\s*re-send the same question/.test(uiStrings),'the assistant-side retry label was corrected too');
ok(uiStrings.includes('Ask this question again with the current context'),'both retry controls name the current context');
ok(src.includes("'Ask this question again with the current context'"),'the control states what it actually does');
ok(src.includes('var drift = _turnContextDrift(replay.question);'),'drift is computed before re-asking');

// ── Content revision, not the ledger event counter ────────────────────────
const cr = new Function(extract('_artifactContentRevision')+'\nreturn _artifactContentRevision;')();
ok(cr({revision:5, contentRevision:2}) === 2,'content revision wins when present');
ok(cr({revision:5}) === 5,'records predating the split fall back to the event counter');
ok(cr(null) === 0,'a missing entry resolves to zero, not NaN');

// ── Drift reporting ───────────────────────────────────────────────────────
ledger['docs/index.rst'] = { key:'docs/index.rst', path:'docs/index.rst', revision:9, contentRevision:3, content:'x' };
transcript = [1,2,3,4];
api.record({ at:1, question:'update the guide', transcriptLength:4,
             workingFiles:[{key:'docs/index.rst', path:'docs/index.rst', revision:3}] });
ok(api.drift('update the guide') === '','unchanged context reports no drift');
ok(api.drift('a different question') === '','a question with no receipt makes no claim');

ledger['docs/index.rst'].contentRevision = 4;
let d = api.drift('update the guide');
ok(d.includes('docs/index.rst is now r4 (was r3)'),'a moved file revision is named with both numbers');

transcript = [1,2,3,4,5,6];
d = api.drift('update the guide');
ok(d.includes('2 later turns now exist'),'added turns are reported');
ok(d.includes(';'),'several changes are reported together, not just the first');

delete ledger['docs/index.rst'];
ok(api.drift('update the guide').includes('no longer tracked'),'a dropped file is reported rather than silently ignored');

// ── The receipt holds no bytes ────────────────────────────────────────────
const held = JSON.stringify(receipts);
ok(!held.includes('content'),'receipts record identifiers and counts, never file or page bytes');
ok(src.includes('while (_TURN_CONTEXT_RECEIPTS.length > _TURN_CONTEXT_RECEIPT_MAX)'),'receipts are bounded');
for (let i=0;i<40;i++) api.record({at:i, question:'q'+i, transcriptLength:0, workingFiles:[]});
ok(receipts.length <= 32,'the receipt log cannot grow without bound');
ok(api.find('q39') !== null && api.find('q0') === null,'eviction drops the oldest receipts first');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
