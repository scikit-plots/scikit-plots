// V98 regression: Detailed feedback owns a dedicated additive icon while the
// existing generic discussion icon stays available to the Feedback workspace.
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const root = process.cwd();
const jsPath = path.join(root, 'docs/source/scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_static/ai-assistant.js');
const src = fs.readFileSync(jsPath, 'utf8');
let passed = 0, failures = 0;
function ok(value, message) {
  if (!value) { console.error('FAIL:', message); failures += 1; }
  else { passed += 1; console.log('ok:', message); }
}

ok(src.includes("feedbackDetail: '<svg width=\"20\" height=\"20\" viewBox=\"0 0 20 20\" fill=\"currentColor\""),
   'ICONS exposes the new 20x20 detailed-feedback glyph');
ok(src.includes('icon: ICONS.feedbackDetail || ICONS.commentDiscussion || ICONS.chat,'),
   'Detailed feedback prefers the dedicated glyph with backward-compatible fallbacks');
ok(src.includes("_workspaceButton('feedback', 'Feedback', ICONS.commentDiscussion);"),
   'Feedback workspace keeps the existing generic discussion glyph');
ok(src.includes('commentDiscussion:'),
   'existing commentDiscussion icon remains in the registry');
ok(src.includes('dedicated detailed-feedback glyph'),
   'developer comment explains the semantic separation and compatibility intent');

console.log(`${passed} passed, ${failures} failed`);
if (failures) process.exit(1);
