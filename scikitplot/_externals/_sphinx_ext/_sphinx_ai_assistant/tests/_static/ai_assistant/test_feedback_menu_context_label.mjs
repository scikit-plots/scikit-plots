import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) return '';
  let d=0,s=false;
  for(let j=i;j<src.length;j++){
    if(src[j]==='{'){d++;s=true;} else if(src[j]==='}'){d--; if(s&&d===0)return src.slice(i,j+1);}
  }
  return '';
}
const popup = extract('_buildFbkFloat');
const bubbleMore = extract('_buildBubbleMore');
ok(popup.includes("label: 'Feedback center\\u2026'"), 'compact feedback menu names the full workspace clearly');
ok(popup.includes("ariaLabel: 'Open feedback center for privacy, review, and submission status'"), 'accessible name describes privacy, review, and submission destination');
ok(!popup.includes('Manage feedback & sharing'), 'old mixed feedback/sharing label is removed');
ok(popup.includes("label: 'Detailed feedback'"), 'detailed feedback remains a distinct immediate action');
ok(popup.includes("label: 'Contribute this Q&A\\u2026'"), 'dataset contribution remains a separate explicit action');
ok(bubbleMore.includes("secondaryLabel.textContent = 'More'") && bubbleMore.includes("homeMenuLbl.textContent = 'Home'"), 'low-frequency destinations use progressive disclosure in canonical answer More');
console.log(`passed=${passed} failed=${failed}`);
if (failed) process.exit(1);
