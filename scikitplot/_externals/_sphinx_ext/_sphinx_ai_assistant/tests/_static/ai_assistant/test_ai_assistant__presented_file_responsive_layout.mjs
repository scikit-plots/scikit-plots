// T95 — Presented-file segmented geometry must survive 560px/mobile widths.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let passed=0, failed=0;
function ok(cond, name){ if(cond) passed++; else { failed++; console.error('FAIL '+name); } }
function rule(sel){
  const esc=sel.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const re=new RegExp(esc+'\\s*\\{([^}]*)\\}','g');
  return [...css.matchAll(re)].map(m=>m[1]).join('\n');
}
function blockAfter(anchor){
  const start=css.indexOf(anchor); if(start<0) return '';
  let open=css.indexOf('{',start), d=0;
  for(let i=open;i<css.length;i++){
    if(css[i]==='{') d++;
    else if(css[i]==='}' && --d===0) return css.slice(start,i+1);
  }
  return '';
}

// Root cause guard: the <=560px rule must never make the in-group Download full width.
const viewport560=blockAfter('@media (max-width: 560px)');
ok(viewport560.includes('.ai-assistant-panel-activity'),'560px viewport rule may still tune unrelated activity margins');
ok(!viewport560.includes('ai-assistant-panel-changed-file-download'),'560px viewport rule does not own Presented Download geometry');
ok(!/\.ai-assistant-panel-changed-file-download\s*\{[^}]*width:\s*100%/s.test(css),'Presented Download is never forced to width:100%');

// Responsive Presented-files heading follows the component width, not viewport width.
const container35=blockAfter('@container ai-artifact-surface (max-width: 35rem)');
ok(container35.includes('.ai-assistant-panel-changed-files-head'),'Presented heading has a component-width responsive rule');
ok(/flex-direction:\s*column/.test(container35),'narrow Presented heading stacks vertically');
ok(/text-align:\s*left/.test(container35),'narrow Presented heading secondary copy remains readable');
ok(!container35.includes('changed-file-download'),'35rem heading rule does not mutate segmented Download geometry');

// Every grid/flex ancestor in the card path is allowed to shrink on small devices.
ok(/min-width:\s*0/.test(rule('.ai-assistant-panel-changed-files-list')),'Presented list can shrink below content min-width');
ok(/min-width:\s*0/.test(rule('.ai-assistant-panel-changed-file')),'Presented card can shrink below content min-width');
const primary=rule('.ai-assistant-panel-changed-file-primary');
ok(/grid-template-columns:\s*minmax\(0, 1fr\) auto/.test(primary),'primary stays [artifact group] [overflow]');
ok(/min-width:\s*0/.test(primary),'primary grid may shrink on phone widths');

// The shared group owns the internal order and sizing at every width.
const groupRule=rule('.ai-md-artifact-group');
ok(/display:\s*flex/.test(groupRule)&&/min-width:\s*0/.test(groupRule),'shared artifact group is shrinkable flex');
const previewRule=rule('.ai-md-artifact-group > .ai-md-artifact-card');
ok(/flex:\s*1 1 auto/.test(previewRule)&&/min-width:\s*0/.test(previewRule),'Preview owns flexible filename space');
const dlRule=rule('.ai-md-artifact-group > button.ai-md-artifact-download-label');
ok(/flex:\s*0 0 auto/.test(dlRule),'Download remains a fixed trailing segment');
ok(!/width:\s*100%/.test(dlRule),'shared Download segment never claims the whole group');

// Wider phones compact only the visible per-file label; smaller widths keep bulk policy separate.
const perFile=blockAfter('@container ai-artifact-surface (max-width: 26rem)');
ok(perFile.includes('.ai-assistant-panel-changed-file-download .ai-md-artifact-btn-label'),'per-file Download label compacts by component width');
ok(/min-width:\s*2\.25rem/.test(perFile)&&/justify-content:\s*center/.test(perFile),'compact Download keeps a usable icon hit target');
const bulk=blockAfter('@container ai-artifact-surface (max-width: 22rem)');
ok(!bulk.includes('.ai-assistant-panel-changed-file-download .ai-md-artifact-btn-label'),'bulk 22rem stage does not re-own per-file Download');
ok(bulk.includes('.ai-assistant-panel-changed-files-download-all'),'bulk labels retain their independent tighter threshold');

// Runtime structure still uses the shared segmented builder rather than a mobile alternate DOM.
ok(src.includes("download.className = 'ai-md-artifact-download-label ai-assistant-panel-changed-file-download';"),'Presented Download uses the shared base visual class');
ok(src.includes('primary.appendChild(_buildArtifactSegmentGroup(entry.path, preview, download));'),'Presented card has one shared segmented DOM at all widths');

console.log(`${passed} passed, ${failed} failed`);
if(failed) process.exit(1);
