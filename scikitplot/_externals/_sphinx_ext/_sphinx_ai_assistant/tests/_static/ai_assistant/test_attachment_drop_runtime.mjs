// Run 62 behavior-level tests for file-drag classification/extraction.
import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(process.argv[2], 'utf8');
function extract(name, nextName) {
  const start = src.indexOf(`    function ${name}(`);
  if (start < 0) throw new Error(`missing ${name}`);
  const end = src.indexOf(`    function ${nextName}(`, start + 1);
  if (end < 0) throw new Error(`missing next function ${nextName}`);
  return src.slice(start, end);
}
const code = [
  extract('_attachmentDragHasFiles', '_attachmentFilesFromDrop'),
  extract('_attachmentFilesFromDrop', '_attachmentLineCount'),
].join('\n');
const ctx = {};
vm.createContext(ctx);
vm.runInContext(code + '\nthis.hasFiles=_attachmentDragHasFiles; this.fromDrop=_attachmentFilesFromDrop;', ctx);

let pass=0, fail=0;
function t(name, got, want=true) {
  if (JSON.stringify(got) === JSON.stringify(want)) pass++;
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
}

t('Files type recognized', ctx.hasFiles({ types: ['text/plain', 'Files'] }), true);
t('file item recognized without types', ctx.hasFiles({ types: [], items: [{kind:'file'}] }), true);
t('plain text drag ignored', ctx.hasFiles({ types: ['text/plain'], items: [{kind:'string'}], files: [] }), false);
t('null transfer ignored', ctx.hasFiles(null), false);

const f1={name:'a.txt', size:1}, f2={name:'b.md', size:2};
const dt={items:[
  {kind:'file', webkitGetAsEntry:()=>({isDirectory:false}), getAsFile:()=>f1},
  {kind:'file', webkitGetAsEntry:()=>({isDirectory:true}), getAsFile:()=>null},
  {kind:'string', getAsFile:()=>f2},
  {kind:'file', getAsFile:()=>f2},
]};
const out=ctx.fromDrop(dt);
t('extracts ordinary files in order', out.files.map(f=>f.name), ['a.txt','b.md']);
t('skips directories', out.skippedDirectories, 1);

const fallback=ctx.fromDrop({items:[], files:[f2,f1]});
t('falls back to FileList order', fallback.files.map(f=>f.name), ['b.md','a.txt']);
t('fallback has no fake directory count', fallback.skippedDirectories, 0);

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
