// Run 173 T15 - the activity timeline survives a reload.
//
// The timeline is how a reader checks what a turn actually did. On reload it
// vanished, so a remembered conversation came back as answers with no account
// of how they were produced -- the part a sceptical reader most wants to
// re-read.
//
// What persists is a bounded SUMMARY, not the live state: that object owns
// budgets, cancellation and byte accounting, none of which mean anything once
// the page is gone.
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

const api = new Function('_ACTIVITY_PERSIST_MAX_STEPS','_ACTIVITY_PERSIST_LABEL_CHARS','_ACTIVITY_PERSIST_DETAIL_CHARS','_ACTIVITY_STEP_KINDS','_ACTIVITY_STEP_STATES','_activityBoundedText',
  [extract('_activityPersistSummary'), extract('_activityRestoreSummary'),
   'return {save:_activityPersistSummary, load:_activityRestoreSummary};'].join('\n'))(
  12, 120, 240, ['status','command','file','verify','note'], ['done','running','error','skipped'],
  (v, max) => String(v == null ? '' : v).replace(/\s+/g,' ').trim().slice(0, max));

// Minimal DOM stand-in: rows exactly as the live renderer builds them.
// Selector-accurate on purpose. A stub that answers every selector with
// whichever node it has handy will happily return a detail node for a
// diff-stat query, and the harness then tests a DOM that cannot exist.
function row(kind, state, label, detail, stat){
  const nodes = {
    '.ai-assistant-panel-activity-step-label': { textContent: label },
    '.ai-assistant-panel-activity-step-detail': detail ? { textContent: detail } : null,
    '.ai-assistant-panel-diff-stat': stat
      ? { getAttribute: a => a === 'aria-label' ? stat : null, textContent: stat }
      : null
  };
  return {
    getAttribute: a => a === 'data-kind' ? kind : a === 'data-state' ? state : null,
    querySelector: sel => Object.prototype.hasOwnProperty.call(nodes, sel) ? nodes[sel] : null
  };
}
const st = (rows) => ({ list: { querySelectorAll: () => rows } });

// ── Capture ───────────────────────────────────────────────────────────────
const saved = api.save(st([
  row('status','done','Prepared request context','6 recent turns · 1 working file'),
  row('file','done','Presented 2 files','docs/index.rst r3 · README.md r1')
]));
ok(saved && saved.length === 2,'finished steps are captured');
ok(saved[0].kind === 'status' && saved[1].kind === 'file','each step keeps its kind');
ok(saved[1].detail.includes('docs/index.rst r3'),'details are kept, not just labels');
ok(api.save(st([])) === null,'an empty timeline persists nothing rather than an empty shell');
ok(api.save(null) === null,'a turn with no timeline persists nothing');
ok(api.save({ list: null }) === null,'a timeline that was never rendered persists nothing');
ok(api.save(st([row('status','done','','x')])) === null,'a step with no label is dropped, not persisted blank');

// ── Bounds: this shares the transcript's storage budget ───────────────────
const many = api.save(st(Array.from({length:40},(_,i)=>row('status','done','step '+i,'d'))));
ok(many.length === 12,'the step count is capped');
const long = api.save(st([row('status','done','L'.repeat(500),'D'.repeat(900))]));
ok(long[0].label.length <= 120 && long[0].detail.length <= 240,'label and detail are bounded');

// ── Restore re-validates: session storage is same-origin, not trustworthy ──
ok(api.load(saved).length === 2,'a well-formed summary round-trips');
ok(api.load(null) === null && api.load('nope') === null,'non-array input yields nothing');
const tampered = api.load([{ kind:'<script>', state:'pwned', label:'Verified by admin' }]);
ok(tampered[0].kind === 'status' && tampered[0].state === 'done','unknown kind and state fall back to the safe defaults');
ok(api.load([{ label: 123 }]) === null,'a non-string label is rejected');
ok(api.load([{ kind:'file', state:'done', label:'ok', detail:{evil:1} }])[0].detail === undefined,'a non-string detail is dropped');
ok(api.load(Array.from({length:40},()=>({kind:'status',state:'done',label:'x'}))).length === 12,'a tampered oversized record is still capped');
ok(api.load([{ kind:'status', state:'done', label:'   ' }]) === null,'a whitespace-only label is rejected');

// ── The restored timeline is read-only ────────────────────────────────────
const restored = extract('_renderRestoredActivity');
ok(restored.includes("root.setAttribute('data-state', 'done')"),'a restored section is marked done');
ok(restored.includes("root.setAttribute('data-restored', 'true')"),'a restored section is distinguishable from a live one');
ok(!restored.includes('activity-stop'),'no Stop button: a control that cannot act invites a click that does nothing');
ok(/panelActivityTimeline\s*===\s*false\)\s*return null/.test(restored),'the timeline setting suppresses a restored section too');
ok(restored.includes('Restored from this browser session'),'the reader is told these are restored, not live, details');
// Deliberately reversed: after a reload the reader has lost every other cue
// about what happened, so making them click twice to see the list is the wrong
// default. What stays collapsed is each step's DETAIL.
ok(restored.includes("panel.hidden = false"),'a restored timeline shows its step list without a click');
ok(restored.includes("detailEl.hidden = true;"),'each step detail starts collapsed');
ok(restored.includes("stepBtn.setAttribute('aria-expanded', 'false')")&&restored.includes("stepBtn.setAttribute('aria-controls', stepId)"),'each step with a detail is its own announced disclosure');
ok(restored.includes("summaryBits.push(fileSteps"),'the collapsed summary names files, not just a step count');
ok(restored.includes("step.kind === 'command'")&&restored.includes("activity-step-detail--code"),'a command detail is rendered as code, not prose');

// The wiring that R173T15 shipped without: a caller must actually supply the
// activity object, or the persistence stores nothing at all.
ok(src.includes("_recordMessage('assistant', accumulated || '(no response)', _streamModelInfo,\n            null, { activity: activity });"),'the streamed assistant turn passes its activity to _recordMessage');

// ── Wiring ────────────────────────────────────────────────────────────────
ok(src.includes('if (activitySummary) entry.activity = activitySummary;'),'the summary is recorded with the assistant turn');
ok(src.includes('_activityRestoreSummary(e.activity)'),'the persisted record is re-validated on load');
ok(src.includes('restoredActivity: m.activity || null'),'replay hands the summary to the renderer');
ok(src.includes('if (restoredEl) body.appendChild(restoredEl);'),'the restored section is rendered above its answer');

// ── End to end: record -> persist -> restore ──────────────────────────────
//
// R173T15 asserted every piece of this chain by reading the source and shipped
// a feature that stored nothing, because no assertion ever ran _recordMessage
// with an activity object. This drives the real function.
const transcript = [];
const record = new Function('_transcript','_cfg','_activityPersistSummary','_TRANSCRIPT_RESTORE_MAX_TEXT_CHARS','_buildTurnResourceRuntime','_sanitizeTurnResourceManifest','_TURN_RESOURCE_PERSIST_MAX_ITEMS','_TURN_RESOURCE_LIVE_MAX_ITEMS','_saveTranscript',
  extract('_recordMessage') + '\nreturn _recordMessage;')(
  transcript,
  () => ({ panelMaxTranscriptTurns: 200 }),
  api.save, 100000, () => null, () => ({ totalCount: 0, items: [] }), 8, 8, () => {});

const liveActivity = st([
  row('status','done','Prepared request context','3 recent turns (~6.4k characters)'),
  row('file','done','Presented 1 file','docs/index.rst r1')
]);
record('assistant', 'Here is the file.', null, null, { activity: liveActivity });
const stored = transcript[transcript.length - 1];
ok(stored && Array.isArray(stored.activity),'recording an assistant turn with an activity object persists its summary');
ok(stored.activity.length === 2 && stored.activity[1].kind === 'file','the persisted summary carries every step and its kind');

record('assistant', 'No timeline here.', null, null, null);
ok(transcript[transcript.length - 1].activity === undefined,'a turn recorded without an activity object stores no summary');

record('user', 'a question', null, null, { activity: liveActivity });
ok(transcript[transcript.length - 1].activity === undefined,'a user turn never carries an activity summary');

// And the restore side accepts exactly what the record side produced.
ok(api.load(stored.activity).length === 2,'what the recorder stored is what the restorer accepts');

// The live timeline row must carry the counts too, from the same builder the
// file card uses, so the two can never disagree about the numbers.
const fileStep = extract('_activityAddArtifactStep');
ok(fileStep.includes('var stat = _diffStatElement(entry);')&&fileStep.includes('if (stat) button.appendChild(stat);'),'a live file row carries its +/- counts');

// ── Diff counts survive into a restored timeline ─────────────────────────
// The stat element cannot be serialized, but its aria-label is the same
// sentence a screen reader already gets, so it folds into the label.
const withStat = api.save(st([
  row('file','done','Updated file preview: docs/index.rst','Latest r2', '84 lines added, 85 lines removed')
]));
ok(withStat[0].label.includes('84 lines added'),'a restored file row still reports how much changed');
ok(withStat[0].label.startsWith('Updated file preview: docs/index.rst'),'the path stays first, so the row is still scannable');
const noStat = api.save(st([row('file','done','Presented 1 file','docs/index.rst r1')]));
ok(!/lines added/.test(noStat[0].label),'a row with no stat gains no invented numbers');
const twice = api.save(st([row('file','done','Updated x — 3 lines added, 0 lines removed','d','3 lines added, 0 lines removed')]));
ok((twice[0].label.match(/3 lines added/g) || []).length === 1,'a label that already carries the stat is not given it twice');
ok(api.save(st([row('file','done','L'.repeat(200),'d','9 lines added, 1 lines removed')]))[0].label.length <= 120,'the folded label still respects the persisted bound');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
