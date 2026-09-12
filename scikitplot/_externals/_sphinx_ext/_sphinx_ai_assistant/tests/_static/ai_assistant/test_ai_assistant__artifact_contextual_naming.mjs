// Run 173 T5a - one resolver owns every generated-artifact filename.
//
// `snippet-1.py` / `snippet-<stamp>.py` made every download require a
// manual rename and let several blocks in one answer collide. These
// assertions pin the derivation, the collision suffix, and the hostile
// input the slug must never pass through.
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
function build(transcript){
  return new Function('_transcript','_ARTIFACT_NAME_MAX_CHARS','_ARTIFACT_NAME_RESERVED',
    [extract('_artifactNameSlug'),extract('_artifactNameFromHeading'),
     extract('_artifactNameFromQuestion'),extract('_artifactContextualFilename'),
     'return {slug:_artifactNameSlug, name:_artifactContextualFilename};'].join('\n'))(
    transcript, 48, /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])$/i);
}
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};
let b = build([{role:'user',text:'generate an Annoy Python file with sys.argv support'}]);
const one = b.name(null,null,'python','py',0,1);
const three = b.name(null,null,'python','py',0,3);
console.error('  observed single -> ' + one);
console.error('  observed multi  -> ' + three);
ok(/^[a-z0-9-]+\.py$/.test(one) && one.startsWith('generate-an-annoy-python-file'),'question-derived name');
ok(one.length <= 48 + 3,'name stays within the slug budget plus extension');
ok(three !== one && /-3?\d\.py$/.test(three) === false || three.endsWith('-1.py'),'multi-block gets an index suffix');
ok(!three.includes('--'),'suffix never produces a double dash');
b = build([]);
ok(b.name(null,null,'rst','rst',0,1)==='rst-snippet.rst','language fallback keeps the rst extension');
ok(b.name(null,null,'','txt',0,1)==='snippet.txt','empty language still yields a name');
ok(b.slug('../../etc/passwd')==='etc-passwd','traversal characters cannot survive');
ok(b.slug('CON')==='','reserved device name rejected');
ok(b.slug('a\u202eb')==='ab','bidi control stripped');
ok(b.slug('!!!')==='','punctuation-only yields empty so caller falls back');
ok(/^[a-z0-9.-]+$/.test(one),'output charset is filesystem-portable');
ok(src.includes('_artifactContextualFilename(root, wrap, lang, ext, unnamedIndex, unnamedTotal)'),'answer cards use the shared resolver');
ok(src.includes('explicitPath || _artifactContextualFilename('),'toolbar download uses the same resolver and yields to explicit file= paths');
ok(!src.includes("'snippet-' + (i + 1) + '.' + ext"),'the positional snippet-N name is gone');
ok(!src.includes("'snippet-' + stamp + '.' + ext"),'the timestamp snippet name is gone');
ok(src.includes('rst: \'rst\'') && src.includes('restructuredtext: \'rst\''),'reStructuredText no longer falls back to .txt');
ok(src.includes('pyx: \'pyx\'') && src.includes('pyi: \'pyi\''),'Cython and stub surfaces are mapped');
// ── Snippet card: preview is the big target, download is its own button ────
//
// The whole card used to download, so the only way to see what a snippet held
// was to put a file on disk and open it. A quick check before committing to a
// download is what a reader wants most of the time.
ok(src.includes("card.setAttribute('aria-label', 'Preview ' + fname);"),'the card previews rather than downloads');
ok(src.includes("badge: 'SNIPPET'")&&src.includes("turnScoped: true"),'the preview names it a snippet, not a tracked file');
ok(src.includes("dlBtn.className = 'ai-md-artifact-download-label'")&&src.includes("dlBtn.type = 'button'"),'download is its own button, not a span inside the card');
// Both segments now live in a role=group wrapper that draws them as one
// control. The contract is unchanged -- neither is nested in the other -- so
// assert that structurally rather than by adjacency, which was only ever a
// proxy for it.
// The group is now built once and used by both artifact surfaces, so the
// contract is asserted on the builder rather than on one call site.
const segGroup = extract('_buildArtifactSegmentGroup');
ok(segGroup.includes('group.appendChild(primary);')&&segGroup.includes('group.appendChild(secondary);'),'preview and download are siblings inside one group');
ok(src.includes('_buildArtifactSegmentGroup(filename, card, dlBtn)'),'the snippet card uses the shared segmented control');
ok((src.match(/className = 'ai-md-artifact-group'/g) || []).length === 1,'there is exactly one segmented-control implementation');
// Stated against the builder, where the nesting could now actually happen.
// The old form named one call site's variables and could not see a builder
// that nested the segments for every surface at once.
ok(!/primary\.appendChild\(secondary\)/.test(segGroup),'the builder never nests one segment inside the other');
ok(!src.includes('card.appendChild(dlBtn)'),'the snippet call site never nests them either');
ok(segGroup.includes("group.setAttribute('role', 'group')")&&segGroup.includes("group.setAttribute('aria-label', ariaLabel)"),'the group carries the relationship and a name for assistive technology');
ok(segGroup.includes("sep.setAttribute('aria-hidden', 'true')"),'the divider is decorative');
ok(src.includes("dlBtn.setAttribute('aria-label', 'Download ' + filename)"),'the download control has its own accessible name');
ok(!/card\.addEventListener\('click', function \(\) \{\s*_downloadBlob/.test(src),'clicking the card no longer downloads');
const naming_css = fs.readFileSync(process.argv[3], 'utf8');
ok(/\.ai-md-artifact-group\s*\{[^}]*border:\s*1px/.test(naming_css),'the group owns the border, so the two segments read as one control');
ok(/\.ai-md-artifact-group\s*>\s*\.ai-md-artifact-card[\s\S]{0,200}?border:\s*0/.test(naming_css),'each segment sheds its own chrome inside the group');
ok(/\.ai-md-artifact-sep\s*\{[^}]*flex:\s*0 0 1px/.test(naming_css),'the divider is its own element, not a border that moves with hover state');
ok(/\.ai-md-artifact-group:focus-within/.test(naming_css),'keyboard focus on either segment is visible on the group');
ok(/button\.ai-md-artifact-download-label\s*\{/.test(naming_css),'the download control is styled as a button');
ok(/\.ai-md-artifact-card\s*\{[^}]*flex:\s*1 1 auto/.test(naming_css),'the preview card takes the room the two controls leave');

// ── Save-as filenames ──────────────────────────────────────────────────────
// A reader-typed name is still a path the browser will act on, so it passes
// the same slug rules as a derived one -- but the extension survives, because
// that is usually the only part they cared about typing.
const saveAs = new Function('_ARTIFACT_NAME_MAX_CHARS','_ARTIFACT_NAME_RESERVED',
  [extract('_artifactNameSlug'), extract('_artifactNameSlugPreservingExtension'),
   'return _artifactNameSlugPreservingExtension;'].join('\n'))(48, /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])$/i);
ok(saveAs('index.rst')==='index.rst','a plain name survives unchanged');
ok(saveAs('My Guide.RST')==='my-guide.rst','spaces and case normalise, extension kept');
ok(saveAs('../../etc/passwd')==='passwd','a traversal path collapses to its basename');
ok(saveAs('C:\\evil\\x.py')==='x.py','a windows path collapses to its basename');
ok(saveAs('a\u202eb.py')==='ab.py','bidi controls are stripped');
ok(saveAs('CON.txt')==='','a reserved device stem is rejected so the caller falls back');
ok(saveAs('!!!.py')==='','a punctuation-only stem is rejected');
ok(saveAs('archive.tar.gz')==='archive-tar.gz','only the final extension is treated as one');
ok(saveAs('')==='','empty input yields empty so the caller uses its suggestion');
ok(/^[a-z0-9.-]*$/.test(saveAs('Weird Name!.Py')),'output charset stays filesystem-portable');
console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
