// Contract harness for the hamburger menu's shortcut accelerators.
//
//   node tests/test_menu_shortcuts.mjs _static/ai-assistant.js
//
// The failure this guards against is quiet: a menu that PRINTS a key on every
// row and does not respond to it is worse than one that prints nothing, because
// the reader stops trusting the other hints too. So the registry, the keycaps,
// and the keydown handler are all checked against one source of truth.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');

function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { depth++; started = true; }
    else if (src[j] === '}') { depth--; if (started && depth === 0) return src.slice(i, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

function extractArray(name) {
  const i = src.indexOf('var ' + name + ' = [');
  if (i < 0) throw new Error('not found: ' + name);
  const start = src.indexOf('[', i);
  let depth = 0;
  for (let j = start; j < src.length; j++) {
    if (src[j] === '[') depth++;
    else if (src[j] === ']') { depth--; if (depth === 0) return src.slice(start, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

const _MENU_ITEMS = (0, eval)('(' + extractArray('_MENU_ITEMS') + ')');

let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) { pass++; }
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
};

// ── the registry ───────────────────────────────────────────────────────────
t('every requested item is present', _MENU_ITEMS.length >= 9, true);
t('registry order matches primary then More reading order',
  _MENU_ITEMS.map((m) => m.key).join(''), 'MCSDLUPTK');
t('primary order is configuration then conversation',
  _MENU_ITEMS.filter((m) => m.group === 'primary').map((m) => m.key).join(''), 'MCSD');
t('More order is links, policy group, shortcuts',
  _MENU_ITEMS.filter((m) => m.group === 'more').map((m) => m.key).join(''), 'LUPTK');

{
  // A duplicate letter leaves one item permanently unreachable by keyboard,
  // while still printing its cap — the exact silent failure this file exists
  // to prevent.
  const keys = _MENU_ITEMS.map((m) => m.key);
  t('accelerators are unique', new Set(keys).size, keys.length);
  t('accelerators are single characters when present',
    keys.every((k) => typeof k === 'string' && (k.length === 0 || k.length === 1)), true);
  t('accelerators are uppercase',
    keys.every((k) => k === k.toUpperCase()), true);

  // Guessable, not memorised: the key is the first letter of the label
  // wherever the label allows it.
  let mnemonic = 0;
  for (const m of _MENU_ITEMS) {
    if (m.label.toUpperCase().split(/[^A-Z]+/).some((w) => w.startsWith(m.key))) {
      mnemonic++;
    }
  }
  t('every accelerator is a letter from its own label',
    mnemonic, _MENU_ITEMS.length);

  let complete = 0;
  for (const m of _MENU_ITEMS) {
    if (m.icon && m.label && typeof m.key === 'string' && m.hook) complete++;
  }
  t('every entry is complete', complete, _MENU_ITEMS.length);
  t('hooks are unique',
    new Set(_MENU_ITEMS.map((m) => m.hook)).size, _MENU_ITEMS.length);
}

// ── reserved surface shortcuts ─────────────────────────────────────────────
{
  t('E is reserved for Exit rather than a menu destination',
    _MENU_ITEMS.some((m) => m.key === 'E'), false);
  const menu = extract('_buildHamburgerMenu');
  t('Exit is published through the same accelerator map',
    /accelerators\.E = hooks\.onExit/.test(menu), true);
  t('footer renders E as the Exit keycap',
    /kbdExit[\s\S]*?_createShortcutCaps\(\['E'\]\)/.test(menu), true);
}

// ── the requested items, by name ───────────────────────────────────────────
for (const [label, key] of [
  ['Model Configuration', 'M'],
  ['Endpoint Configuration', 'C'],
  ['Share', 'S'],
  ['Project Links', 'L'],
  ['Usage Policy', 'U'],
  ['Privacy & Responsibility', 'P'],
  ['Terms of Service', 'T'],
  ['Keyboard shortcuts', 'K'],
]) {
  const found = _MENU_ITEMS.find((m) => m.label === label);
  t('item exists: ' + label, !!found, true);
  t('accelerator for ' + label, found && found.key, key);
}

// ── the destructive item ───────────────────────────────────────────────────
{
  const del = _MENU_ITEMS.find((m) => m.key === 'D');
  t('clear-conversation item exists', !!del, true);
  t('clear item is marked destructive', del.danger, true);
  // A destructive action reached by a single keypress, with focus placed on
  // the menu automatically, is one stray keystroke away from losing the
  // transcript. It must confirm.
  t('clear item confirms first', typeof del.confirm === 'string', true);
  t('confirmation says what is lost',
    /cannot be recovered/i.test(del.confirm), true);
  t('clear item is wired to clearConversation',
    /onClear:\s*function \(\) \{ clearConversation\(\); \}/.test(src), true);
}

// ── the accelerator actually runs ──────────────────────────────────────────
{
  const menu = extract('_buildHamburgerMenu');
  t('accelerators are collected from the registry',
    /accelerators\[spec\.key\.toUpperCase\(\)\] = activate/.test(menu), true);
  t('the map is published for the panel-level binder',
    /pop\._accelerators = accelerators/.test(menu), true);

  // Click and key must run the SAME function, or the confirmation can be
  // skipped by whichever path forgot it.
  t('click and key share one activation path',
    (menu.match(/function activate\(\)/g) || []).length, 1);
  t('click uses it', /item\.addEventListener\('click', activate\)/.test(menu), true);
  t('the confirmation lives inside the shared path',
    /function activate\(\)[\s\S]*?spec\.confirm/.test(menu), true);

}

// ── accelerators reach every sheet, not just the menu ──────────────────────
//
// The reported defect: a popover-scoped listener only fires while focus is
// inside the popover, so opening a sheet — which moves focus into it — killed
// every accelerator while leaving the keys printed on rows still on screen.
{
  const bind = extract('_attachMenuAccelerators');

  // One listener, at panel level, reached by keydown bubbling from the menu,
  // any sheet, any nested sub-sheet, or the transcript.
  t('the listener is bound to the panel',
    /panel\.addEventListener\('keydown'/.test(bind), true);
  t('the popover no longer binds its own listener',
    /pop\.addEventListener\('keydown'/.test(src), false);
  t('accelerators are bound exactly once',
    (src.match(/function _attachMenuAccelerators\(/g) || []).length, 1);
  t('the binder is called at panel build',
    /_attachMenuAccelerators\(panel, hamburgerMenuEl\)/.test(src), true);

  // NOT document: a bare letter must not be claimed anywhere else on the page.
  t('the listener is not on the document',
    /document\.addEventListener\('keydown'[\s\S]{0,300}_accelerators/.test(src),
    false);

  // Every guard below is load-bearing — see the docstring.
  t('modifier chords are ignored',
    /if \(e\.altKey \|\| e\.ctrlKey \|\| e\.metaKey\) return;/.test(bind), true);
  t('only single characters match', /e\.key\.length !== 1/.test(bind), true);
  t('auto-repeat is ignored', /e\.repeat/.test(bind), true);
  t('IME composition is ignored', /isComposing|229/.test(bind), true);
  t('already-handled events are left alone',
    /e\.defaultPrevented/.test(bind), true);
  t('text entry is skipped', /_isTextEntryTarget\(e\.target\)/.test(bind), true);
}

// ── the text-entry guard is what makes panel-wide letters safe ─────────────
{
  const isText = (0, eval)('(' + extract('_isTextEntryTarget') + ')');
  {
    const m = src.match(/var _NON_TEXT_INPUT_TYPES = \[([\s\S]*?)\];/);
    globalThis._NON_TEXT_INPUT_TYPES = (0, eval)('([' + m[1] + '])');
  }

  // The composer is the reason this guard exists: without it, typing "model"
  // would open four sheets and delete the conversation.
  t('textarea is text entry', isText({ tagName: 'TEXTAREA' }), true);
  t('plain input is text entry', isText({ tagName: 'INPUT' }), true);
  t('select is text entry', isText({ tagName: 'SELECT' }), true);
  t('contenteditable is text entry',
    isText({ tagName: 'DIV', isContentEditable: true }), true);

  for (const type of ['text', 'search', 'url', 'email', 'password',
                      'number', 'tel', 'date']) {
    t('input[type=' + type + '] is text entry',
      isText({ tagName: 'INPUT', type }), true);
  }
  // Controls that are not text entry must NOT block the accelerators, or a
  // reader who has just tabbed to the effort radios loses every shortcut.
  for (const type of ['radio', 'checkbox', 'range', 'button', 'submit',
                      'reset', 'file', 'color']) {
    t('input[type=' + type + '] is not text entry',
      isText({ tagName: 'INPUT', type }), false);
  }
  t('a button is not text entry', isText({ tagName: 'BUTTON' }), false);
  t('a div is not text entry', isText({ tagName: 'DIV' }), false);
  t('null target is not text entry', isText(null), false);
  t('a target with no tagName is not text entry', isText({}), false);

  // Printed keys the menu cannot receive would be a promise it cannot keep.
  t('focus moves into the menu on open',
    /querySelector\('\.ai-assistant-panel-hamburger-item'\)[\s\S]{0,80}focus\(\)/.test(src),
    true);
}

// ── one keycap component for both kinds of shortcut ────────────────────────
{
  t('caps component defined once',
    (src.match(/function _createShortcutCaps\(/g) || []).length, 1);
  // Per-item accelerators AND the multi-key panel shortcut row. Counted from
  // code lines only: a bare regex also matches the identifier where it appears
  // in a comment, which would let a REMOVED call site be masked by prose that
  // still mentions it.
  const codeLines = src.split('\n')
    .filter((l) => !l.trim().startsWith('//') && !l.trim().startsWith('*'))
    .join('\n');
  t('every shortcut surface uses the shared component',
    (codeLines.match(/_createShortcutCaps\(/g) || []).length >= 5, true);

  const caps = extract('_createShortcutCaps');
  t('caps are aria-hidden', /aria-hidden/.test(caps), true);
  t('caps use the shared class', /ai-assistant-kbd-group/.test(caps), true);
  t('each cap is a <kbd>', /createElement\('kbd'\)/.test(caps), true);
  t('cap text goes through the glyph map', /_shortcutGlyph\(/.test(caps), true);

  // The old inline builder, with its '+' separators, must be gone.
  t('no surface hand-builds keycaps',
    /kbdRow\.appendChild\(document\.createTextNode\('\+'\)\)/.test(src), false);
}

// ── glyphs are platform-correct and readable to AT ─────────────────────────
{
  // Node exposes navigator as a getter-only global, so it is redefined rather
  // than assigned. Doing this through defineProperty keeps the harness able to
  // exercise BOTH platform branches — a single-platform run would leave half
  // the glyph map untested.
  const setPlatform = (platform, userAgent) =>
    Object.defineProperty(globalThis, 'navigator', {
      value: { platform, userAgent }, configurable: true, writable: true,
    });

  setPlatform('MacIntel', 'Mac');
  const glyph = (0, eval)('(' + extract('_shortcutGlyph') + ')');
  t('mac shift glyph', glyph('Shift'), '\u21e7');
  t('mac command glyph', glyph('Meta'), '\u2318');
  t('mac option glyph', glyph('Alt'), '\u2325');
  t('mac control glyph', glyph('Ctrl'), '\u2303');

  setPlatform('Win32', 'Windows');
  const glyphWin = (0, eval)('(' + extract('_shortcutGlyph') + ')');
  t('windows spells Alt', glyphWin('Alt'), 'Alt');
  t('windows spells Ctrl', glyphWin('Ctrl'), 'Ctrl');
  // Shift is a universally understood glyph; it does not change.
  t('shift glyph is universal', glyphWin('Shift'), '\u21e7');
  t('letters are uppercased', glyphWin('m'), 'M');
  t('unknown multi-char tokens pass through', glyphWin('F5'), 'F5');
  t('missing token does not throw', glyphWin(undefined), '');

  // Glyphs do not read aloud usefully, so the spoken form must differ.
  const spoken = (0, eval)('(' + extract('_shortcutSpoken') + ')');
  globalThis._shortcutSpoken = spoken;
  t('shift is spelled for AT', spoken('Shift'), 'Shift');
  t('meta is spelled for AT', spoken('Meta'), 'Command');
  t('spoken form is not a glyph', /[\u2300-\u23FF]/.test(spoken('Ctrl')), false);

  const list = (0, eval)('(' + extract('_shortcutSpokenList') + ')');
  t('spoken list joins with spaces', list(['Ctrl', 'Shift', 'M']),
    'Control Shift M');
}

// ── the shortcut reaches assistive technology ──────────────────────────────
{
  const menu = extract('_buildHamburgerMenu');
  t('items declare aria-keyshortcuts',
    /setAttribute\('aria-keyshortcuts', spec\.key\)/.test(menu), true);
  t('items name their shortcut',
    /', shortcut ' \+ _shortcutSpokenList/.test(menu), true);
  t('the panel shortcut row declares its chord too',
    /kbdRow\.setAttribute\('aria-keyshortcuts'/.test(menu), true);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
