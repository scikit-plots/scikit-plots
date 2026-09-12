// R173T93 — inline answer snippets yield vertical scroll to the conversation.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const cssRaw = fs.readFileSync(process.argv[3], 'utf8');
const css = cssRaw.replace(/\/\*[\s\S]*?\*\//g, '');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function rules(selector) {
  const esc = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(esc + '\\s*\\{([^}]*)\\}', 'g');
  let out = '', m;
  while ((m = re.exec(css)) !== null) out += '\n' + m[1];
  return out;
}

const sheet = rules('.ai-md-file-sheet');
const snippet = rules('.ai-md-snippet-sheet');
const snippetPre = rules('.ai-md-snippet-sheet .ai-md-pre');
const numberedCode = (css.match(/\.ai-md-file-sheet \.ai-md-pre,\s*\.ai-md-file-sheet \.ai-assistant-panel-attachment-preview-code\s*\{([^}]*)\}/) || ['', ''])[1];

ok(!!sheet, 'generic numbered-sheet rule exists');
ok(/overflow-y:\s*auto/.test(sheet), 'real file sheets explicitly own bounded vertical scrolling');
ok(/overflow-x:\s*hidden/.test(sheet), 'real file sheets keep horizontal ownership out of the outer sheet');
ok(/overscroll-behavior:\s*contain/.test(sheet), 'only a sheet that owns vertical scrolling contains its overscroll');
ok(!/(^|;)\s*overflow\s*:/.test(sheet), 'generic sheet has no overflow shorthand that can silently erase axis ownership');

ok(!!snippet, 'inline answer snippet rule exists');
ok(/max-height:\s*none/.test(snippet), 'inline snippets are not capped into a nested vertical viewport');
ok(/overflow:\s*visible/.test(snippet), 'inline snippet wrapper is not a scroll container');
ok(/overscroll-behavior:\s*auto/.test(snippet), 'inline snippet releases wheel/touch overscroll to the conversation');
ok(!/overscroll-behavior:\s*contain/.test(snippet), 'inline snippet never traps vertical chaining');
ok(/clip-path:\s*inset\(0 round 8px\)/.test(snippet), 'rounded visual crop does not rely on overflow trapping');

ok(!!numberedCode, 'numbered code-cell rule exists');
ok(/overflow-y:\s*hidden/.test(numberedCode), 'code cell never becomes a nested vertical scroller');
ok(/overflow-x:\s*auto/.test(numberedCode), 'long code lines remain horizontally scrollable');
ok(/white-space:\s*pre/.test(numberedCode), 'line-numbered code remains unwrapped');

ok(!!snippetPre, 'snippet code-cell interaction rule exists');
ok(/overscroll-behavior-y:\s*auto/.test(snippetPre), 'vertical overscroll from the horizontal code scroller chains outward');
ok(/touch-action:\s*pan-x pan-y pinch-zoom/.test(snippetPre), 'touch/tablet code permits horizontal pan and vertical conversation pan');
ok(!/touch-action:\s*none/.test(snippetPre), 'snippet code never captures all touch gestures');

ok(!/\.ai-md-(?:snippet-sheet|pre)[\s\S]{0,120}addEventListener\(['"]wheel/.test(src), 'no custom wheel interceptor steals native scroll chaining from snippets');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
