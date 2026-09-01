import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
ok(src.includes("feedbackCenterLabel.textContent = 'Manage feedback & sharing…';"), 'compact feedback menu uses action-oriented feedback/sharing label');
ok(src.includes("feedbackCenterRow.setAttribute('aria-label', 'Manage feedback privacy, sharing, and review status');"), 'accessible name describes privacy, sharing, and review destination');
ok(!src.includes("feedbackCenterLabel.textContent = 'Feedback & review...';"), 'ambiguous legacy label is removed');
ok(src.includes("formLabel.textContent = 'Detailed feedback \\u2193';"), 'detailed feedback remains a distinct note/form action');
ok(src.includes("contributeLabel.textContent = 'Contribute this Q&A\\u2026';"), 'dataset contribution remains a separate action');
console.log(`passed=${passed} failed=${failed}`);
if (failed) process.exit(1);
