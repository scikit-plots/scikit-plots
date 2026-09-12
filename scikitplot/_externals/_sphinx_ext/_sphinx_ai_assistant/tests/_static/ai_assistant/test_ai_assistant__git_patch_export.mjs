// Run 173 T5 - git-compatible patch export.
//
// The panel emits git's interchange format rather than running git: a patch a
// reader applies in their own repository, with their own identity, so the
// server never holds their document content.
//
// The value of that only survives if the patches actually apply, so this
// harness shells out to real `git am` when git is present. Structural
// assertions alone would have passed the first implementation, whose hunks
// overlapped and which git rejected outright.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const src = fs.readFileSync(process.argv[2], 'utf8');
function extract(n){const st=src.indexOf('function '+n+'(');if(st<0)throw new Error('missing '+n);
 let d=0,b=false,q='',e=false,l=false,bl=false;
 for(let i=st;i<src.length;i++){const c=src[i],x=src[i+1];
  if(l){if(c==='\n')l=false;continue;} if(bl){if(c==='*'&&x==='/'){bl=false;i++;}continue;}
  if(q){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c===q)q='';continue;}
  if(c==='/'&&x==='/'){l=true;i++;continue;} if(c==='/'&&x==='*'){bl=true;i++;continue;}
  if(c==='"'||c==="'"||c==='`'){q=c;continue;} if(c==='{'){d++;b=true;}else if(c==='}'&&--d===0&&b)return src.slice(st,i+1);}
 throw new Error('unterminated '+n);}

const api = new Function('_DIFF_STAT_MAX_LINES','_DIFF_STAT_LCS_BUDGET','_DIFF_HUNK_LCS_BUDGET','_DIFF_HUNK_CONTEXT','_ARTIFACT_NAME_MAX_CHARS','_ARTIFACT_NAME_RESERVED',
  [extract('_diffStatSplitLines'),extract('_diffStatMultiset'),extract('_diffStatLcs'),extract('_diffLineStat'),
   extract('_diffBacktrack'),extract('_diffOps'),extract('_diffUnified'),
   extract('_patchSafeSubjectText'),extract('_artifactContentRevision'), extract('_gitPatchText'),
   extract('_artifactNameSlug'),extract('_gitPatchFilename'),
   'return {unified:_diffUnified, patch:_gitPatchText, name:_gitPatchFilename, subject:_patchSafeSubjectText};'].join('\n'))(
  20000, 4000000, 1440000, 3, 48, /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])$/i);

let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};

// ── Hunks must never overlap ───────────────────────────────────────────────
function hunkRanges(body){
  return [...body.matchAll(/@@ -(\d+),(\d+) \+(\d+),(\d+) @@/g)]
    .map(m => ({oldStart:+m[1], oldCount:+m[2], newStart:+m[3], newCount:+m[4]}));
}
const before='Install\n=======\n\nUse pip.\n\nNotes\n-----\n\nOld note.\n';
const after ='Install\n=======\n\nUse pip or conda.\n\nNotes\n-----\n\nOld note.\nNew note.\n';
const u = api.unified(before, after);
const ranges = hunkRanges(u.body);
let overlap=false;
for(let i=1;i<ranges.length;i++){
  if(ranges[i].oldStart < ranges[i-1].oldStart + ranges[i-1].oldCount) overlap=true;
  if(ranges[i].newStart < ranges[i-1].newStart + ranges[i-1].newCount) overlap=true;
}
ok(!overlap,'hunks never overlap on either side');
ok(u.added===2 && u.removed===1,'unified diff counts match the edit');

// Widely separated changes must produce more than one hunk.
const wide = Array.from({length:60},(_,i)=>'line '+i).join('\n')+'\n';
const wide2 = wide.replace('line 2\n','CHANGED 2\n').replace('line 55\n','CHANGED 55\n');
const uw = api.unified(wide, wide2);
const rw = hunkRanges(uw.body);
ok(rw.length===2,'distant changes split into separate hunks');
let overlap2=false;
for(let i=1;i<rw.length;i++) if(rw[i].oldStart < rw[i-1].oldStart+rw[i-1].oldCount) overlap2=true;
ok(!overlap2,'multi-hunk output stays non-overlapping');

// Adjacent changes within 2x context must merge into one hunk.
const near2 = wide.replace('line 2\n','CHANGED 2\n').replace('line 5\n','CHANGED 5\n');
ok(hunkRanges(api.unified(wide, near2).body).length===1,'nearby changes merge into one hunk');

// ── Patch envelope ─────────────────────────────────────────────────────────
const entry={path:'docs/index.rst',content:after,revision:2,base:{revision:1,content:before}};
const patch=api.patch(entry);
ok(patch.startsWith('From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001'),'mailbox magic line present');
ok(patch.includes('\n---\ndiff --git a/docs/index.rst b/docs/index.rst'),'diffstat separator precedes the diff');
ok(patch.endsWith('-- \n2.0.0\n'),'mailbox signature terminator present');
ok(!/^index [0-9a-f]+\.\.[0-9a-f]+/m.test(patch),'no fabricated blob index line');
ok(patch.includes('was NOT applied to any repository'),'patch states it was never applied');
ok(api.name(entry)==='0002-docs-index-rst.patch','patch filename is ordered and portable');
const fresh={path:'examples/annoy_cli.py',content:'import sys\nprint(sys.argv)\n',revision:1,base:null};
ok(api.patch(fresh).includes('new file mode 100644'),'absent base emits a new-file patch');
ok(api.subject('drop\nthe\nnewlines\u0000')==='drop the newlines','subject cannot inject mailbox headers');

// ── Real git application ───────────────────────────────────────────────────
let gitAvailable=true;
try { execFileSync('git',['--version'],{stdio:'ignore'}); } catch { gitAvailable=false; }
if(gitAvailable){
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'ai-patch-'));
  const g=(...a)=>execFileSync('git',a,{cwd:dir,stdio:'pipe'});
  g('init','-q'); g('config','user.email','t@example.invalid'); g('config','user.name','T');
  fs.mkdirSync(path.join(dir,'docs'),{recursive:true});
  fs.writeFileSync(path.join(dir,'docs/index.rst'),before);
  g('add','-A'); g('commit','-qm','base');
  fs.writeFileSync(path.join(dir,'u.patch'),patch);
  fs.writeFileSync(path.join(dir,'n.patch'),api.patch(fresh));
  let applied=true;
  try { g('am','u.patch'); } catch(e){ applied=false; console.error(String(e.stderr||e)); }
  ok(applied,'git am applies the update patch');
  ok(fs.readFileSync(path.join(dir,'docs/index.rst'),'utf8')===after,'applied content matches the revision exactly');
  let applied2=true;
  try { g('am','n.patch'); } catch(e){ applied2=false; console.error(String(e.stderr||e)); }
  ok(applied2,'git am applies the new-file patch');
  ok(fs.existsSync(path.join(dir,'examples/annoy_cli.py')),'new file lands at its declared path');
  fs.rmSync(dir,{recursive:true,force:true});
} else {
  console.error('note: git unavailable, structural assertions only');
}

// ── Multi-file series ──────────────────────────────────────────────────────
// A concatenated mailbox is what git itself emits for a range, so a multi-file
// turn must apply as an ordered set of commits rather than as several
// downloads the reader has to sequence by hand.
if (gitAvailable) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ai-series-'));
  const g = (...a) => execFileSync('git', a, { cwd: dir, stdio: 'pipe' });
  g('init','-q'); g('config','user.email','t@example.invalid'); g('config','user.name','T');
  fs.mkdirSync(path.join(dir,'docs'),{recursive:true});
  fs.writeFileSync(path.join(dir,'docs/index.rst'), before);
  g('add','-A'); g('commit','-qm','base');

  const entries = [
    { path:'docs/index.rst', content:after, revision:2, base:{revision:1, content:before} },
    { path:'examples/annoy_cli.py', content:'import sys\nprint(sys.argv)\n', revision:1, base:null },
    { path:'README.md', content:'# Project\n\nSee docs/index.rst.\n', revision:1, base:null }
  ];
  const mailbox = entries
    .map(e => api.patch(e, { subject: (e.base ? 'Update ' : 'Add ') + e.path }))
    .join('\n');
  fs.writeFileSync(path.join(dir,'series.patch'), mailbox);

  let seriesOk = true;
  try { g('am','series.patch'); } catch (e) { seriesOk = false; console.error(String(e.stderr || e)); }
  ok(seriesOk,'git am applies a concatenated multi-file series');
  ok(fs.readFileSync(path.join(dir,'docs/index.rst'),'utf8') === after,'series leaves the updated file byte-exact');
  ok(fs.existsSync(path.join(dir,'examples/annoy_cli.py')) && fs.existsSync(path.join(dir,'README.md')),'every file in the series lands');
  const log = execFileSync('git',['log','--oneline'],{cwd:dir,encoding:'utf8'}).trim().split('\n');
  ok(log.length === 4,'each file in the series becomes its own commit');
  fs.rmSync(dir,{recursive:true,force:true});
}

// ── Snippet promotion and continuation contracts ──────────────────────────
ok(src.includes('function _promoteSnippetToFile(root, code, lang, suggested)'),'snippets can be promoted to tracked files');
ok(src.includes('_generatedArtifactSafePath(String(raw).trim())'),'a reader-supplied path is validated by the existing safe-path authority');
ok(src.includes('_stageComposerFiles([{'),'continuation reuses the composer attachment pipeline rather than a private channel');
ok(src.includes("sourceKind: 'working-file'"),'continued files are identifiable as working files at the staging boundary');
ok(src.includes("relativePath: entry.path"),'continuation preserves the directory, not just the basename');
ok(!/_promoteSnippetToFile[\s\S]{0,400}new File\(\[/.test(src) || src.includes('_generatedArtifactIsAvailable(entry)'),'continuation refuses unavailable revisions');

// ── Retention budget has exactly one owner ────────────────────────────────
ok(src.includes('total += _utf8ByteLength(entry.base.content)'),'retained base bytes are charged to the session budget');
ok(src.includes('entry.base = null;'),'eviction releases the retained base with the content');
ok(src.includes('_utf8ByteLength(old.content) <= _TURN_ACTIVITY_FILE_MAX_BYTES'),'base retention respects the per-file ceiling');

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
