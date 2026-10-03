import fs from 'node:fs';

const js = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let pass = 0, fail = 0;
const ok = (cond, name) => {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
};

ok(/var _FEATURE_DEFS = \[[\s\S]*capLabel: 'Learn'[\s\S]*suffix: '\/v1\/learn'/.test(js), 'canonical endpoint registry carries display and route metadata');
ok(/function _makeEndpointTableShell/.test(js), 'shared semantic endpoint table shell exists');
ok(/function _buildEndpointResolutionTable/.test(js), 'shared resolution table renderer exists');
ok((js.match(/_buildEndpointResolutionTable\(/g) || []).length >= 3, 'resolution renderer is reused by active and profile URL views');
ok(/Feature'[\s\S]*Configuration'[\s\S]*Health'[\s\S]*Endpoint'[\s\S]*Actions'/.test(js), 'active resolved endpoint table separates configuration and health before endpoint/actions');
ok(/\\u2713 Configured/.test(js) && !/\\u2713 Ready/.test(js.slice(js.indexOf('function _makeEndpointStatus'), js.indexOf('function _makeEndpointActionGroup'))), 'configured status does not overclaim runtime health');
ok(/Advanced endpoint routing/.test(js) && /Configured value/.test(js) && /Effective endpoint/.test(js), 'Advanced routing uses the table vocabulary');
ok(/_FEATURE_DEFS\.map\(function \(fd\)/.test(js), 'Add Custom Profile Advanced fields derive from canonical endpoint registry');
ok(/for \(var _ci = 0; _ci < _FEATURE_DEFS\.length; _ci\+\+\)/.test(js), 'capability chips derive from canonical endpoint registry');
ok(!/var _capDefs = \[/.test(js), 'legacy partial capability list is retired');
ok(!/ai-assistant-panel-ep-card-detail-row/.test(js), 'legacy profile URL row renderer is retired');
ok(!/ai-assistant-panel-ep-resolved-row/.test(js), 'legacy active URL row renderer is retired');
ok(!/urlTxt\.appendChild\(_makeCopyBtn/.test(js), 'copy control is never nested inside endpoint anchor');
ok(/var card = document\.createElement\('div'\)/.test(js), 'profile card is not a label wrapping interactive descendants');
ok(/target\.closest\('button, a, input, textarea, select'\)/.test(js), 'full-card selection excludes nested interactive controls');
ok(/detailToggle\.setAttribute\('aria-controls', detailWrap\.id\)/.test(js), 'Show URLs disclosure is connected by aria-controls');
ok(/var ftd = document\.createElement\('th'\)/.test(js) && /ftd\.setAttribute\('scope', 'row'\)/.test(js), 'comparison feature cells use row headers');
ok(!/table\.setAttribute\('role', 'grid'\)/.test(js.slice(js.indexOf('function _rebuildCompareGrid'), js.indexOf('/** Update the status bar'))), 'comparison uses native table semantics rather than interactive grid role');

ok(/function _resolveAdvancedBaseDraft/.test(js), 'Advanced Base draft has a use-time validation gate');
ok(/!_advBaseInp\.readOnly[\s\S]*_epSafe\.validateUrl/.test(js), 'editable Base drafts must pass URL validation before actions consume them');
ok(/!input\.readOnly[\s\S]*_epSafe\.validateEndpoint/.test(js), 'editable route drafts must pass endpoint validation before actions consume them');
ok(/aria-invalid/.test(js) && /Invalid draft/.test(js), 'invalid Advanced drafts are exposed accessibly instead of becoming clickable URLs');
ok(/_advBaseActionGroup = _makeEndpointActionGroup\(_resolveAdvancedBaseDraft, 'Base', true\)/.test(js), 'Base copy/open/health actions consume validated draft authority');
ok(/return _resolveAdvancedDraftEndpoint\(fd\.key, inp\.value\)/.test(js), 'feature actions consume validated effective endpoint authority');
ok(/function _setEndpointActionGroupEnabled/.test(js) && /_setEndpointActionGroupEnabled\(actions, false\)/.test(js), 'invalid Advanced drafts disable authority-bearing action buttons');
ok(/_setEndpointActionGroupEnabled\(actions, !!url\)/.test(js), 'Advanced actions re-enable only when an effective URL resolves');
ok(/_advBaseInp\.addEventListener\('input', _refreshAdvancedEffectiveEndpoints\)/.test(js), 'Base draft edits refresh inherited effective endpoints live');
ok(/inp\.addEventListener\('input', function \(\) \{ _renderAdvancedEffectiveCell\(fd\.key\); \}\)/.test(js), 'feature draft edits refresh their effective endpoint live');

ok(/\.ai-assistant-panel-ep-route-scroll \{/.test(css), 'shared endpoint table has a scroll host');
ok(/\.ai-assistant-panel-ep-route-table--advanced \{[\s\S]*min-width: 48rem/.test(css), 'Advanced table has deliberate horizontal overflow geometry');
ok(/\.ai-assistant-panel-ep-route-th--feature,[\s\S]*position: sticky/.test(css), 'shared endpoint tables keep route names visible while scrolling');
ok(/\.ai-assistant-panel-ep-grid-feature \{[\s\S]*position: sticky/.test(css), 'profile comparison keeps feature names visible while scrolling');
ok(/\.ai-assistant-panel-ep-route-actions \.ai-assistant-panel-ep-copy-btn[\s\S]*opacity: 1[\s\S]*pointer-events: auto/.test(css), 'table copy actions are always visible and touch-operable');
ok(/\.ai-assistant-panel-ep-route-actions button:disabled[\s\S]*pointer-events: none/.test(css), 'invalid draft actions have an explicit disabled affordance');
ok(/@media \(pointer: coarse\)[\s\S]*ai-assistant-panel-ep-route-actions[\s\S]*width: 2\.25rem/.test(css), 'coarse-pointer endpoint actions receive larger touch targets');
ok(!/\.ai-assistant-panel-ep-resolved-row \.ai-assistant-panel-ep-copy-btn/.test(css), 'obsolete hover-only resolved-row copy treatment is removed');
ok(/\.ai-assistant-panel-ep-route-invalid \{/.test(css), 'invalid draft state has dedicated visual treatment');
ok(/\.ai-assistant-panel-ep-route-input\[aria-invalid="true"\]/.test(css), 'invalid draft inputs have visible error affordance');
ok(!/\.ai-assistant-panel-ep-card-detail-row \{/.test(css), 'dead legacy card-detail row CSS is removed');
ok(!/\.ai-assistant-panel-ep-indicator \{/.test(css), 'dead legacy resolved indicator CSS is removed');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
