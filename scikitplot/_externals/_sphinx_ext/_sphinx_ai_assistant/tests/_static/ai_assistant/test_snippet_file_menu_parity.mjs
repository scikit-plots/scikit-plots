// R173T97 — answer snippets and Presented files share one menu grammar.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let pass=0, fail=0; const ok=(c,m)=>c?pass++:(fail++,console.error('FAIL '+m));
function extract(n){const st=src.indexOf('function '+n+'(');if(st<0)throw new Error('missing '+n);let d=0,b=false,q='',e=false,l=false,bl=false;for(let i=st;i<src.length;i++){const c=src[i],x=src[i+1];if(l){if(c==='\n')l=false;continue;}if(bl){if(c==='*'&&x==='/'){bl=false;i++;}continue;}if(q){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c===q)q='';continue;}if(c==='/'&&x==='/'){l=true;i++;continue;}if(c==='/'&&x==='*'){bl=true;i++;continue;}if(c==='"'||c==="'"||c==='`'){q=c;continue;}if(c==='{'){d++;b=true;}else if(c==='}'&&--d===0&&b)return src.slice(st,i+1);}throw new Error('unterminated '+n);}
const snippet=extract('_buildSnippetOverflow');
const tracked=extract('_fileOverflowItems');
const promote=extract('_promoteSnippetToFile');
const wrapper=extract('_buildFileOverflow');
ok(src.includes("var snippetMenu = _buildSnippetOverflow(\n                root, card, codeEl ? codeEl.textContent : '', lang, filename, typeLabel);"),'snippet cards route through the shared capability wrapper');

// Same workflow shape: inspect, save/export, lifecycle action, continue.
ok(snippet.includes("label: 'Open in a sheet'") && tracked.includes("label: 'Open in a sheet'"),'both menus begin with Open in a sheet');
ok(snippet.includes("label: 'Save as\\u2026'") && tracked.includes("label: 'Save as\\u2026'"),'both menus use the same Save as wording');
ok(snippet.includes("label: 'Continue editing'") && tracked.includes("label: 'Continue editing'"),'both menus expose Continue editing');
ok(snippet.includes("label: 'Track as file\\u2026'") && tracked.includes("label: 'Download patch'"),'the third slot reflects capability: track first, patch once tracked');

// An anonymous snippet must not claim git capabilities before it has identity.
ok(!snippet.includes("label: 'Download patch'"),'an untracked snippet does not offer a fake patch action');
ok(snippet.includes("hint: 'Add revisions, diffs and patch export'"),'Track as file explains what identity unlocks');
ok(snippet.includes("hint: 'Track it, then attach to your next message'"),'Continue editing explains the one-time tracking step');

// Menu graduation is stateful rather than a second visual implementation.
ok(snippet.includes('if (entry) return _fileOverflowItems(entry.key);'),'after promotion the snippet trigger becomes the canonical tracked-file menu');
ok(wrapper.includes('return _fileOverflowItems(key);'),'Presented files use that same canonical tracked-file action list');
ok((src.match(/function _buildOverflowMenu\(/g)||[]).length===1,'there is exactly one overflow-menu mechanics implementation');

// Open-in-sheet is the same preview system, not a parallel viewer.
ok(snippet.includes('sheet: true') && snippet.includes('_openAttachmentPreview(snippetPreviewItem(), card)'),'snippet Open in a sheet uses the attachment preview in sheet mode');

// Continue needs stable identity; promotion returns it, then the standard continuation path is used.
ok(snippet.includes('var tracked = trackSnippet();') && snippet.includes('_generatedArtifactContinueEditing(tracked.key);'),'snippet Continue editing tracks then uses the canonical continuation pipeline');
ok(promote.includes('return existing;'),'reusing an already-identical tracked path still returns its identity');
ok(promote.includes('return entry;'),'new promotion returns the registered tracked revision');

// Labels from the old two-row vocabulary are gone from the snippet menu.
ok(!snippet.includes('Save as a tracked file') && !snippet.includes('Download as'),'the old two-row snippet vocabulary is retired');

console.log(`${pass} passed, ${fail} failed`); if(fail) process.exit(1);
