// Run 173 T10 - persistence tells the truth about itself.
//
// `_ssSet` swallowed every failure. Defensible for a cache, indefensible for
// the transcript: quota exhaustion, private browsing and disabled storage all
// fail there, and the panel kept showing "Remember conversation" switched on
// while nothing survived a reload. A switch describing a capability the browser
// is refusing is worse than no switch, because the reader stops taking notes.
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

// ── _ssSet must report, not swallow ───────────────────────────────────────
function buildSsSet(throwing){
  const store = new Map();
  const sessionStorage = { setItem(k,v){ if(throwing) throw new Error('QuotaExceededError'); store.set(k,v); } };
  return { fn: new Function('sessionStorage', extract('_ssSet')+'\nreturn _ssSet;')(sessionStorage), store };
}
const good = buildSsSet(false), bad = buildSsSet(true);
ok(good.fn('k','v') === true,'a successful write reports true');
ok(good.store.get('k') === 'v','a successful write actually stores');
ok(bad.fn('k','v') === false,'a quota failure reports false instead of being swallowed');
ok(!/catch \(_\) \{ \/\* ignore \*\/ \}\s*\n\s*\}\s*\n\s*function _ssDel/.test(src),'the silent-ignore form is gone from _ssSet');

// ── Notification fires once per condition, not per keystroke ──────────────
const notes = [];
const health = new Function('showNotification',
  ['var _persistenceUnavailableNotified = false;',
   extract('_persistenceReportHealthy'), extract('_persistenceReportUnavailable'),
   'return {up:_persistenceReportHealthy, down:_persistenceReportUnavailable, count:function(){return _persistenceUnavailableNotified;}};'
  ].join('\n'))((m,isErr)=>notes.push({m,isErr}));
health.down('quota'); health.down('quota'); health.down('serialization');
ok(notes.length === 1,'the unavailable warning is reported once, not per save');
ok(notes[0].isErr === true,'it is surfaced as a problem, not a passing remark');
ok(/will not\s+survive a reload/.test(notes[0].m),'it names the actual consequence');
// Precise: the message may DESCRIBE storage as disabled -- that is the actual
// condition -- but must not INSTRUCT the reader to go change something, since
// the panel cannot know which browser control, if any, would help.
ok(!/(enable|turn on|go to|check your|open your)\s+(it|settings|storage|preferences)/i.test(notes[0].m),
   'it describes the condition without instructing the reader to change a setting');
health.up();
health.down('quota');
ok(notes.length === 2,'recovering and failing again reports again');

// ── Shortfall detection ───────────────────────────────────────────────────
const marker = new Map();
const shortfall = new Function('_ssGet','_TRANSCRIPT_COUNT_KEY', extract('_persistenceRestoredShortfall')+'\nreturn _persistenceRestoredShortfall;').bind(null)(
  k => marker.has(k) ? marker.get(k) : null, 'ai-assistant-transcript-count');
ok(shortfall(5) === 0,'no marker makes no claim');
marker.set('ai-assistant-transcript-count','12');
ok(shortfall(12) === 0,'a complete restore reports no shortfall');
ok(shortfall(9) === 3,'a truncated restore reports exactly how many turns were dropped');
ok(shortfall(14) === 0,'a longer restore never reports a negative shortfall');
marker.set('ai-assistant-transcript-count','not-a-number');
ok(shortfall(3) === 0,'a corrupt marker is ignored rather than trusted');
marker.set('ai-assistant-transcript-count','0');
ok(shortfall(0) === 0,'a zero marker makes no claim');

// ── Wiring ────────────────────────────────────────────────────────────────
ok(src.includes('var stored = _ssSet(_TRANSCRIPT_KEY, JSON.stringify(persisted));'),'the transcript save checks its own result');
ok(src.includes("_persistenceReportUnavailable('serialization');"),'a serialization failure is distinguished from a quota failure');
ok(src.includes('_ssSet(_TRANSCRIPT_COUNT_KEY, String(persisted.length));'),'the marker is written only after a successful save');
ok(src.includes('var missing = _persistenceRestoredShortfall(restored.length);'),'restore compares against the marker');
ok((src.match(/_ssDel\(_TRANSCRIPT_COUNT_KEY\)/g) || []).length >= 4,'the marker is pruned everywhere the transcript is');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
