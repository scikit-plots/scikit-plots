// Responsive contract for model-management launchers and row actions.
//
//   node tests/test_model_responsive_actions.mjs _static/ai-assistant.js
//
// The same Edit/Delete/Reset controls are reused at every breakpoint. CSS
// changes presentation only: a state-aware vertical icon rail + floating labels at every
// width >=500px, and a vertical-ellipsis popover <500px. Normal rows expose Edit/Delete;
// edited rows expose Edited/Delete/Reset. The same DOM/handlers power both breakpoints.
import fs from 'node:fs';
import path from 'node:path';

const jsPath = process.argv[2];
const src = fs.readFileSync(jsPath, 'utf8');
const css = fs.readFileSync(path.join(path.dirname(jsPath), 'ai-assistant.css'), 'utf8');

function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { depth++; started = true; }
    else if (src[j] === '}') {
      depth--;
      if (started && depth === 0) return src.slice(i, j + 1);
    }
  }
  throw new Error('unbalanced: ' + name);
}

let pass = 0, fail = 0;
function t(name, got, want) {
  if (got === want) pass++;
  else {
    fail++;
    console.log('  FAIL ' + name + '\n       got  ' + got + '\n       want ' + want);
  }
}

const row = extract('_buildModelRowV2');
const manager = extract('_appendModelCustomSection');

t('launcher grid has three columns for Add/Revert/Customize',
  /grid-template-columns:\s*minmax\(0, 1fr\) auto auto;/.test(css), true);
t('count is moved out of primary launcher into editor title row',
  manager.indexOf('editorTitleRow.appendChild(countBadge)') > -1 &&
  manager.indexOf('header.appendChild(countBadge)') === -1, true);
t('count remains accessible and announces updates politely',
  /aria-label', 'Custom model count/.test(manager) &&
  /aria-live', 'polite/.test(manager) &&
  /aria-atomic', 'true/.test(manager), true);
t('Revert is appended before Customize',
  manager.indexOf('header.appendChild(revertBtn)') < manager.indexOf('header.appendChild(editorToggle)'), true);

t('row builds one shared action wrapper', /ai-assistant-panel-model-actions/.test(row), true);
t('row builds vertical ellipsis menu button',
  /ai-assistant-panel-model-menu-btn/.test(row) && /\\u22ee/.test(row), true);
t('menu button exposes expanded state', /aria-expanded/.test(row), true);
t('menu button is labelled as a popup', /aria-haspopup/.test(row), true);
t('edit action uses shared icon+label renderer',
  /_setActionContent\(editBtn,\s*'\\u270e',\s*'Edit'\)/.test(row), true);
t('delete action uses shared icon+label renderer',
  /_setActionContent\(removeBtn,\s*'\\ud83d\\udd25',\s*'Delete'\)/.test(row), true);
t('reset action uses shared icon+label renderer',
  /_setActionContent\(rowResetBtn,\s*'\\u27f2',\s*'Reset'\)/.test(row), true);
t('edited state is a real clickable DOM action',
  /editedStatus = document\.createElement\('button'\)/.test(row) &&
  /ai-assistant-panel-model-edited-status/.test(row), true);
t('edited row helper exposes reset/status without rebuild',
  /row\._ensureOverrideActions\s*=\s*_ensureOverrideActions/.test(row), true);
t('reset helper clears live override UI',
  /row\._clearOverrideActions\s*=\s*_removeOverrideActions/.test(row), true);
t('manager Save rebuilds the canonical live row',
  /_refreshManagedRow\(editedId\)/.test(manager), true);
t('manager Reset uses row-local clear UI hook',
  /resetRow\._clearOverrideActions/.test(manager), true);

t('normal Edit/Delete icons remain visible',
  /model-edit-btn,[\s\S]*model-remove-btn[\s\S]*opacity:\s*1/.test(css), true);
t('edited state replaces Edit rather than duplicating it',
  /data-overridden="true"[\s\S]*model-edit-btn[\s\S]*display:\s*none/.test(css), true);
t('state-aware action order is Edit-or-Edited, Delete, Reset',
  /model-edit-btn\s*\{\s*order:\s*1;/.test(css) &&
  /model-edited-status\s*\{\s*order:\s*1;/.test(css) &&
  /model-remove-btn\s*\{\s*order:\s*2;/.test(css) &&
  /model-row-reset-btn\s*\{\s*order:\s*3;/.test(css), true);
t('edited action is keyboard-accessible and labelled for continuing edit',
  /editedStatus\.type\s*=\s*'button'/.test(row) &&
  /Continue editing configuration for/.test(row), true);
t('edited action reuses the normal Edit handler',
  /editBtn\.addEventListener\('click', _requestModelEdit\)/.test(row) &&
  /editedStatus\.addEventListener\('click', _requestModelEdit\)/.test(row), true);
t('all widths at or above 500px hide inline action labels',
  /@media \(min-width:\s*500px\)[\s\S]*?\.ai-assistant-panel-model-action-text\s*\{\s*display:\s*none;/s.test(css), true);
t('non-mobile action rail stays vertical',
  /\.ai-assistant-panel-model-actions\s*\{[\s\S]*?flex-direction:\s*column;/s.test(css), true);
t('non-mobile widths use non-reflow floating labels',
  /data-action-label\]::after[\s\S]*content:\s*attr\(data-action-label\)[\s\S]*position:\s*absolute/s.test(css), true);
t('floating labels reveal on hover and keyboard focus',
  /data-action-label\]:hover::after[\s\S]*button\[data-action-label\]:focus-visible::after/s.test(css), true);
t('action renderer supplies short tooltip labels',
  /setAttribute\('data-action-label',\s*labelText\)/.test(row), true);
t('edited action supplies the same short tooltip renderer',
  /_setActionContent\(editedStatus,\s*'\\u270d\\ufe0e',\s*'Edited'\)/.test(row), true);
t('mobile breakpoint is below 500px', /@media \(max-width:\s*499\.98px\)/.test(css), true);
t('mobile breakpoint shows ellipsis trigger',
  /@media \(max-width:\s*499\.98px\)[\s\S]*?\.ai-assistant-panel-model-menu-btn\s*\{[\s\S]*?display:\s*inline-flex;/s.test(css), true);
t('mobile actions are collapsed by default',
  /@media \(max-width:\s*499\.98px\)[\s\S]*?\.ai-assistant-panel-model-actions\s*\{[\s\S]*?display:\s*none;/s.test(css), true);
t('mobile open state reveals same action wrapper',
  /data-actions-open="true"[\s\S]*?\.ai-assistant-panel-model-actions\s*\{\s*display:\s*flex;/s.test(css), true);
t('mobile popover restores text labels',
  /@media \(max-width:\s*499\.98px\)[\s\S]*?model-action-text[\s\S]*?display:\s*inline;/s.test(css), true);

console.log(`\n${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
