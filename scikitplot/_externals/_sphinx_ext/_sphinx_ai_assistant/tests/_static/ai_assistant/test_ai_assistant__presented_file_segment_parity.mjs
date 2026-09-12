// T94 — Presented files must reuse the normal artifact segmented-control layout.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let passed=0, failed=0;
function ok(cond, name){ if(cond) passed++; else { failed++; console.error('FAIL '+name); } }
function extract(name){
  const start=src.indexOf('function '+name+'('); if(start<0) throw new Error('missing '+name);
  let d=0,b=false,q='',esc=false,line=false,block=false;
  for(let i=start;i<src.length;i++){const c=src[i],n=src[i+1];
    if(line){if(c==='\n')line=false;continue;} if(block){if(c==='*'&&n==='/'){block=false;i++;}continue;}
    if(q){if(esc){esc=false;continue;}if(c==='\\'){esc=true;continue;}if(c===q)q='';continue;}
    if(c==='/'&&n==='/'){line=true;i++;continue;} if(c==='/'&&n==='*'){block=true;i++;continue;}
    if(c==='"'||c==="'"||c==='`'){q=c;continue;} if(c==='{'){d++;b=true;} else if(c==='}'&&--d===0&&b)return src.slice(start,i+1);
  } throw new Error('unterminated '+name);
}
const group=extract('_buildArtifactSegmentGroup');
const changed=extract('_appendChangedFileSummary');

// DOM order is authoritative: primary content, separator, then Download.
const p=group.indexOf('group.appendChild(primary)');
const s=group.indexOf("sep.className = 'ai-md-artifact-sep'");
const a=group.indexOf('group.appendChild(sep)');
const d=group.indexOf('group.appendChild(secondary)');
ok(p>=0&&s>p&&a>s&&d>a,'shared builder orders preview → separator → download');
ok(group.includes("group.setAttribute('role', 'group')"),'segmented control keeps a named accessibility group');

// Presented files must opt into the exact same visual primitives as normal artifacts.
ok(changed.includes("preview.className = 'ai-md-artifact-card ai-assistant-panel-changed-file-preview';"),'Presented preview carries the base artifact-card class');
ok(changed.includes("download.className = 'ai-md-artifact-download-label ai-assistant-panel-changed-file-download';"),'Presented Download carries the base download-label class');
ok(changed.includes('primary.appendChild(_buildArtifactSegmentGroup(entry.path, preview, download));'),'Presented row uses the shared segment builder');
ok(changed.includes('primary.appendChild(_buildFileOverflow(key, entry));'),'overflow remains outside the segmented artifact control');
ok(changed.includes("download.title = 'Download latest ' + entry.path + ' under its own name';"),'Presented Download has the same tooltip discoverability as a normal artifact');

// One row authority only: group fills, overflow follows. Historical 3/4-column layouts must stay gone.
const primaryRules=css.match(/\.ai-assistant-panel-changed-file-primary\s*\{[^}]*\}/g)||[];
ok(primaryRules.length===1,'Presented-file primary row has exactly one CSS rule');
ok(/display:\s*grid/.test(primaryRules[0]),'Presented-file primary row is a grid');
ok(/grid-template-columns:\s*minmax\(0, 1fr\) auto/.test(primaryRules[0]),'primary row is exactly [artifact group] [overflow]');
ok(!/auto\s+auto/.test(primaryRules[0]),'no stale third primary-row column remains');
ok(!/\.ai-assistant-panel-changed-file-(?:saveas|secondary|more|patch|continue)\b/.test(css),'dead pre-overflow Presented-file controls no longer own CSS');

// Shared segmented geometry owns both normal and Presented artifacts.
const cardRule=(css.match(/\.ai-md-artifact-group\s*>\s*\.ai-md-artifact-card\s*\{[^}]*\}/)||[''])[0];
const dlRule=(css.match(/\.ai-md-artifact-group\s*>\s*button\.ai-md-artifact-download-label\s*\{[^}]*\}/g)||[]).join('\n');
ok(/flex:\s*1 1 auto/.test(cardRule)&&/min-width:\s*0/.test(cardRule),'preview segment owns remaining width and may shrink');
ok(/width:\s*auto/.test(cardRule),'group cancels the historical full-width card constraint');
ok(/flex:\s*0 0 auto/.test(dlRule),'Download remains a stable trailing segment');
ok(/\.ai-md-artifact-sep\s*\{[^}]*flex:\s*0 0 1px/.test(css),'separator is a one-pixel middle segment');
ok(!/\.ai-md-artifact-group\s*>\s*\.ai-assistant-panel-changed-file-preview\s*\{/.test(css),'Presented preview has no parallel geometry override');
ok(!/\.ai-md-artifact-group\s*>\s*\.ai-assistant-panel-changed-file-download\s*\{/.test(css),'Presented Download has no parallel geometry override');

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
