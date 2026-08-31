// Contract harness for the shared export-format registry.
//
//   node tests/test_export_formats.mjs _static/ai-assistant.js
//
// Guards the invariant that motivated the refactor: the toolbar dropdown and
// the share sheet's format cards render ONE registry, in ONE order, with each
// format appearing exactly once. Two hand-maintained arrays previously drifted
// (different order, a duplicated TOML preview); this makes that drift a failing
// test rather than a visual defect someone has to notice.
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

// Balanced-bracket extraction of `var NAME = [ ... ];` at module scope.
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

// ── fakes ──────────────────────────────────────────────────────────────────
globalThis.ICONS = {
  exportJson: '<svg data-icon="json"></svg>',
  exportHtml: '<svg data-icon="html"></svg>',
  exportTxt:  '<svg data-icon="txt"></svg>',
};

const stubIconMatch = src.match(/var _EXPORT_STUB_ICON =\s*([\s\S]*?);\n/);
if (!stubIconMatch) throw new Error('_EXPORT_STUB_ICON not found');
globalThis._EXPORT_STUB_ICON = (0, eval)('(' + stubIconMatch[1] + ')');

globalThis._stubFormat = (0, eval)('(' + extract('_stubFormat') + ')');

const _EXPORT_FORMATS      = (0, eval)('(' + extractArray('_EXPORT_FORMATS') + ')');
const _EXPORT_STUB_FORMATS = (0, eval)('(' + extractArray('_EXPORT_STUB_FORMATS') + ')');
globalThis._EXPORT_FORMATS = _EXPORT_FORMATS;
globalThis._EXPORT_STUB_FORMATS = _EXPORT_STUB_FORMATS;

// Evaluate the SHIPPED composition expression, never a re-implementation of it.
// Rebuilding it here would make every ordering and duplication assertion below
// vacuous: the harness would be testing its own arithmetic, and a bookend or a
// reversal in the source would sail straight through.
const cardExpr = src.match(/var _EXPORT_CARD_FORMATS = ([\s\S]*?);\n/);
if (!cardExpr) throw new Error('_EXPORT_CARD_FORMATS not found');
const _EXPORT_CARD_FORMATS = (0, eval)('(' + cardExpr[1] + ')');

const _exportFormatDesc      = (0, eval)('(' + extract('_exportFormatDesc') + ')');
const _exportFormatAccessibleLabel = (0, eval)('(' + extract('_exportFormatAccessibleLabel') + ')');
const _stubFormat            = globalThis._stubFormat;

// ── runner ─────────────────────────────────────────────────────────────────
let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) { pass++; }
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
};

// ── no duplicates — the reported defect ────────────────────────────────────
{
  const keys = _EXPORT_CARD_FORMATS.map((f) => f.fmt);
  t('every fmt key is unique', new Set(keys).size, keys.length);
  t('data-fmt is therefore a usable selector',
    keys.filter((k) => k === 'toml').length, 1);
}

// ── one order, previews last ───────────────────────────────────────────────
{
  t('implemented formats in canonical order',
    _EXPORT_FORMATS.map((f) => f.fmt).join(','), 'json,html,txt,yaml,toml');
  t('no implemented format is a stub',
    _EXPORT_FORMATS.some((f) => f.stub), false);
  t('every preview is flagged',
    _EXPORT_STUB_FORMATS.every((f) => f.stub === true), true);

  const firstStub = _EXPORT_CARD_FORMATS.findIndex((f) => f.stub);
  const lastLive = _EXPORT_CARD_FORMATS.reduce(
    (acc, f, i) => (f.stub ? acc : i), -1);
  t('previews are empty or come after every implemented format',
    firstStub === -1 || firstStub > lastLive, true);

  // The card order must be the dropdown order with previews appended — not a
  // re-sort, not a reversal, not a bookend arrangement.
  t('card order extends dropdown order',
    _EXPORT_CARD_FORMATS.slice(0, _EXPORT_FORMATS.length).map((f) => f.fmt).join(','),
    _EXPORT_FORMATS.map((f) => f.fmt).join(','));
}

// ── both surfaces read the registry, neither keeps a private copy ──────────
{
  // A second `var formats = [` anywhere would be a re-introduced local array.
  t('no surface-local format array remains',
    (src.match(/var formats = \[/g) || []).length, 0);
  // Both surfaces iterate the SAME array, so they render the same set in the
  // same order by construction rather than by anyone remembering to.
  t('both surfaces iterate the card registry',
    (src.match(/_EXPORT_CARD_FORMATS\.forEach\(/g) || []).length, 2);
  t('neither surface iterates a narrower list',
    (src.match(/_EXPORT_FORMATS\.forEach\(/g) || []).length, 0);
}

// ── preview behaviour is shared, not re-implemented per surface ────────────
{
  // The defect class this whole refactor exists to prevent is "two surfaces,
  // two copies of the same rule". Assert the rule has exactly ONE definition
  // and that both surfaces call it.
  t('preview semantics defined once',
    (src.match(/function _applyExportPreviewSemantics\(/g) || []).length, 1);
  t('both surfaces apply the shared semantics',
    (src.match(/_applyExportPreviewSemantics\(/g) || []).length, 3);  // 1 def + 2 uses

  t('the refusal sentence is defined once',
    (src.match(/function _exportPreviewNotice\(/g) || []).length, 1);
  t('no surface hard-codes a refusal sentence',
    (src.match(/export is not available yet\. /g) || []).length, 1);

  t('the soon badge is built once',
    (src.match(/function _createExportSoonBadge\(/g) || []).length, 1);
  t('both surfaces use the shared badge',
    (src.match(/_createExportSoonBadge\(/g) || []).length, 3);

  t('the live region is built once',
    (src.match(/function _createExportLiveRegion\(/g) || []).length, 1);
  t('both surfaces mount a live region',
    (src.match(/_createExportLiveRegion\(/g) || []).length, 3);

  t('the accessible name is computed once',
    (src.match(/function _exportFormatAccessibleLabel\(/g) || []).length, 1);
  t('both surfaces name formats the same way',
    (src.match(/_exportFormatAccessibleLabel\(/g) || []).length, 3);
}

// ── the refusal sentence names what IS ready ───────────────────────────────
{
  globalThis._EXPORT_FORMATS = _EXPORT_FORMATS;
  const notice = (0, eval)('(' + extract('_exportPreviewNotice') + ')');
  const msg = notice('CSV');
  t('notice names the refused format', msg.startsWith('CSV'), true);
  t('notice says it is unavailable', /not available yet/.test(msg), true);
  for (const f of _EXPORT_FORMATS) {
    t('notice lists ' + f.label, msg.includes(f.label), true);
  }
  t('notice does not list the synthetic preview itself',
    msg.split('not available yet.')[1].includes('CSV'), false);
}

// ── the dropdown menu keeps preview rows reachable too ─────────────────────
{
  const i = src.indexOf('_EXPORT_CARD_FORMATS.forEach(');
  const j = src.indexOf('menu.appendChild(menuLive);', i);
  const body = src.slice(i, j);
  t('menu previews are not `disabled`', /item\.disabled\s*=\s*true/.test(body), false);
  t('menu previews get the stub class',
    /ai-assistant-export-menu-item--stub/.test(body), true);
  t('menu previews get a soon badge',
    /ai-assistant-export-menu-soon/.test(body), true);
  // A refused click must not close the menu — nothing happened, and closing
  // would tear down the live region before it could be read.
  const stubBranch = body.slice(body.indexOf('if (opt.stub) {', body.indexOf('mousedown')));
  const elseAt = stubBranch.indexOf('} else {');
  t('refused click does not close the menu',
    stubBranch.slice(0, elseAt).includes('_closeExportMenu'), false);
}

// ── every entry is complete ────────────────────────────────────────────────
{
  let bad = 0;
  for (const f of _EXPORT_CARD_FORMATS) {
    if (!f.fmt || !f.label || !f.hint || !f.icon) bad++;
    if (!_exportFormatDesc(f)) bad++;
    if (typeof f.icon !== 'string' || !f.icon.startsWith('<svg')) bad++;
  }
  t('no incomplete registry entry', bad, 0);
}

// ── implemented formats own download + share serialization metadata ───────
{
  let bad = 0;
  for (const f of _EXPORT_FORMATS) {
    if (!f.shareDesc || typeof f.shareDesc !== 'string') bad++;
    if (!f.mime || typeof f.mime !== 'string') bad++;
    if (!f.ext || typeof f.ext !== 'string' || !f.ext.startsWith('.')) bad++;
    if (typeof f.buildStr !== 'function') bad++;
  }
  t('every live format owns share/download metadata', bad, 0);
  t('JSON MIME is canonical', _EXPORT_FORMATS[0].mime, 'application/json;charset=utf-8');
  t('HTML MIME is canonical', _EXPORT_FORMATS[1].mime, 'text/html;charset=utf-8');
  t('TXT MIME is canonical', _EXPORT_FORMATS[2].mime, 'text/plain;charset=utf-8');
  t('YAML MIME is canonical', _EXPORT_FORMATS[3].mime, 'application/yaml');
  t('TOML MIME is canonical', _EXPORT_FORMATS[4].mime, 'application/toml');
  t('live extensions stay aligned with format keys',
    _EXPORT_FORMATS.map((f) => f.ext).join(','), '.json,.html,.txt,.yaml,.toml');
}

// ── the stub template: adding a format is one line ─────────────────────────
{
  const yaml = _stubFormat('yaml', 'YAML', 'Human-readable config for CI.');
  t('template sets the key', yaml.fmt, 'yaml');
  t('template sets the label', yaml.label, 'YAML');
  t('template flags it as a preview', yaml.stub, true);
  t('template supplies a generic icon', yaml.icon, globalThis._EXPORT_STUB_ICON);
  t('template needs no icon argument', yaml.icon.startsWith('<svg'), true);
  t('template desc is carried through',
    _exportFormatDesc(yaml), 'Human-readable config for CI.');

  const custom = _stubFormat('csv', 'CSV', 'Tabular.', { icon: '<svg id="x"/>' });
  t('template accepts a custom icon', custom.icon, '<svg id="x"/>');

  // Promotion path: dropping the flag must yield a valid implemented entry.
  delete yaml.stub;
  t('dropping the flag yields a live format', !!yaml.stub, false);
  t('promoted entry is still complete',
    !!(yaml.fmt && yaml.label && yaml.hint && yaml.icon), true);
}

// ── accessible names ───────────────────────────────────────────────────────
{
  const json = _EXPORT_FORMATS[0];
  const stub = _stubFormat('csv', 'CSV', 'Tabular preview.');
  t('live card name carries the hint',
    _exportFormatAccessibleLabel(json).includes(json.hint), true);
  t('preview card name says it is unavailable',
    /not available yet/.test(_exportFormatAccessibleLabel(stub)), true);
  t('preview card name still names the format',
    _exportFormatAccessibleLabel(stub).startsWith(stub.label), true);
  t('names differ between a live and a preview card',
    _exportFormatAccessibleLabel(json) === _exportFormatAccessibleLabel(stub), false);

  // desc falls back to hint so an entry never repeats itself to satisfy layout.
  t('desc falls back to hint',
    _exportFormatDesc({ fmt: 'x', label: 'X', hint: 'short' }), 'short');
}

// ── preview controls stay reachable ────────────────────────────────────────
{
  // The shared helper is where reachability is decided for BOTH surfaces.
  const helper = extract('_applyExportPreviewSemantics');
  t('helper announces unavailability', /aria-disabled/.test(helper), true);
  t('helper does not set `disabled`', /\.disabled\s*=\s*true/.test(helper), false);
  t('helper does not remove from the tab order',
    /tabindex/.test(helper), false);
  t('helper refuses activation', /preventDefault\(\)/.test(helper), true);
  t('helper announces the reason', /_exportPreviewNotice\(/.test(helper), true);

  const region = extract('_createExportLiveRegion');
  t('live region is polite', /aria-live'\s*,\s*'polite/.test(region), true);
  t('live region is visually hidden',
    /ai-assistant-visually-hidden/.test(region), true);

  // Neither surface may re-introduce the old treatment in its own stub branch.
  //
  // Scoped to the branches on purpose: the menu applies tabindex="-1" to EVERY
  // item as its roving-tabindex pattern, which is correct and unrelated. What
  // must never come back is a preview being singled out for removal from the
  // tab order or the accessibility tree.
  const stubBranches = src.split('if (opt.stub) {').slice(1).map(function (chunk) {
    const end = chunk.indexOf('} else {');
    return end < 0 ? chunk.slice(0, 900) : chunk.slice(0, end);
  });
  t('both surfaces have a stub branch', stubBranches.length >= 2, true);
  let singledOut = 0;
  for (const b of stubBranches) {
    if (/\.disabled\s*=\s*true/.test(b)) singledOut++;
    if (/setAttribute\('tabindex'/.test(b)) singledOut++;
  }
  t('no preview is singled out for removal', singledOut, 0);

  // Every stub branch must delegate rather than hand-roll the behaviour.
  let delegating = 0;
  for (const b of stubBranches) {
    if (/_applyExportPreviewSemantics\(|_createExportSoonBadge\(/.test(b)) delegating++;
  }
  t('every stub branch delegates to the shared helpers',
    delegating, stubBranches.length);
}


// ── mode-aware trigger/chooser icon synchronization ─────────────────────────
{
  t('one shared export action icon resolver exists',
    (src.match(/function _exportActionModeIcon\(/g) || []).length, 1);
  t('share-link trigger uses the dedicated local Octicon glyph',
    /return linkMode \? ICONS\.linkMode : ICONS\.exportTxt/.test(src), true);
  t('link-mode uses the supplied Octicon share geometry',
    /class="octicon octicon-share"[\s\S]*M3\.75 6\.5a\.25\.25 0 0 0-\.25\.25/.test(src), true);
  t('link-mode has no external sprite dependency',
    /sprites-core-ceb34a6c/.test(src), false);
  t('both trigger surfaces subscribe their icon to shared export state',
    (src.match(/innerHTML = _exportActionModeIcon\(state\.linkMode\)/g) || []).length, 2);
  t('both trigger surfaces initialize from persisted mode',
    (src.match(/innerHTML = _exportActionModeIcon\(_exportLinkMode\)/g) || []).length, 2);
  t('mode chooser icon is semantic in both modes',
    /icon\.innerHTML = on \? ICONS\.linkChain : ICONS\.exportTxt/.test(src), true);
  t('Contribute registry points at the dedicated dataset icon',
    /icon: 'dataset', label: 'Contribute'/.test(src), true);
  t('dataset icon is defined in the JS icon registry',
    /dataset: '<svg viewBox="0 0 16 16"/.test(src), true);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
