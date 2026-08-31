import fs from 'node:fs';
import path from 'node:path';

const jsPath = process.argv[2];
const src = fs.readFileSync(jsPath, 'utf8');
const css = fs.readFileSync(path.join(path.dirname(jsPath), 'ai-assistant.css'), 'utf8');
let pass = 0, fail = 0;
const ok = (cond, name) => {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
};

ok(/className = 'ai-assistant-panel-ep-import-form'/.test(src), 'import uses dedicated form surface');
ok(/className = 'ai-assistant-panel-ep-import-editor'/.test(src), 'import uses bounded editor shell');
ok(/Profiles JSON/.test(src), 'editor has a visible field label');
ok(/max 128 KB/.test(src), 'editor communicates payload cap');
ok(/importTA\.maxLength\s*=\s*131072/.test(src), 'textarea enforces 128 KiB cap');
ok(/_IMPORT_MAX_CHARS\s*=\s*131072/.test(src), 'runtime validates import size before parse');
ok(/aria-describedby', 'ai-assistant-panel-ep-import-status'/.test(src), 'textarea is linked to validation status');
ok(/aria-live', 'polite'/.test(src), 'validation status is announced accessibly');
ok(/importBtn\.disabled\s*=\s*true/.test(src), 'import starts disabled');
ok(/Valid · .*profile/.test(src), 'valid JSON reports ready profile count');
ok(/Invalid JSON · line/.test(src), 'invalid JSON reports sanitized location');
ok(!/Invalid JSON:' \+ _e\.message/.test(src), 'raw parser error message is not echoed');
ok(/Ctrl\/⌘ \+ Enter/.test(src), 'keyboard import shortcut is discoverable');
ok(/ev\.ctrlKey \|\| ev\.metaKey/.test(src) && /ev\.key === 'Enter'/.test(src), 'keyboard import shortcut is implemented');
ok(/ai-assistant-panel-ep-import-clear-btn/.test(src), 'import surface has a clear action');
ok(/skipped\+\+/.test(src), 'invalid entries are counted rather than echoed');
ok(!/errors\.push\(_ky \+ ': ' \+ _res\.error\)/.test(src), 'profile keys and validation errors are not echoed');

ok(/\.ai-assistant-panel-ep-import-ta\s*\{[\s\S]*max-height:\s*14rem/.test(css), 'textarea height is bounded on desktop');
ok(/\.ai-assistant-panel-ep-import-editor:focus-within/.test(css), 'editor has coherent focus state');
ok(/\.ai-assistant-panel-ep-import-status--error/.test(css), 'error status has dedicated styling');
ok(/\.ai-assistant-panel-ep-import-actions\s*\{[\s\S]*justify-content:\s*space-between/.test(css), 'actions have stable footer layout');
ok(/@media \(max-width: 500px\)[\s\S]*\.ai-assistant-panel-ep-import-actions\s*\{[\s\S]*flex-direction:\s*column/.test(css), 'narrow panels stack import actions');
ok(/@media \(max-width: 500px\)[\s\S]*\.ai-assistant-panel-ep-import-ta\s*\{[\s\S]*max-height:\s*11rem/.test(css), 'narrow textarea remains bounded');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
