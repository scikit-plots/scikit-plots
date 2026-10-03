import fs from 'node:fs';

const js = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let pass = 0, fail = 0;
const ok = (cond, name) => {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
};

ok(/if \(healthController\) \{ columns\.push\(\{ key: 'health', label: 'Health' \}\); \}/.test(js), 'Health is an optional column of the shared resolution-table renderer');
ok(/key: 'status', label: 'Configuration'/.test(js), 'configuration state is named independently from transport health');
ok(/ai-assistant-panel-ep-route-table--health/.test(js) && /ai-assistant-panel-ep-route-table--health/.test(css), 'active health table has deliberate geometry modifier');
ok(/_activeHealthController/.test(js) && /makeStatus: function \(url, label\)/.test(js), 'active table receives one centralized health controller');
ok(/_connectionHealthCache = Object\.create\(null\)/.test(js), 'health observations are ephemeral in-memory state');
ok(!/localStorage[\s\S]{0,120}_connectionHealthCache/.test(js), 'health observations are not persisted as endpoint authority');
ok(/_connectionHealthBindings = Object\.create\(null\)/.test(js), 'health cells bind to URL-keyed fan-out state');
ok(/_setEndpointHealthState\(url, state\)[\s\S]*bindings\[i\]\.el/.test(js), 'one URL observation fans out to every feature row sharing that URL');
ok(/if \(!seen\[url\]\)[\s\S]*urls\.push\(url\)/.test(js), 'bulk connection check deduplicates network probes by URL');
ok(!/\? 'Service' : _tfd\.label/.test(js), 'duplicate URLs no longer collapse visible feature identity into a generic Service row');
ok(!/testResultsEl/.test(js), 'legacy duplicate connectivity-result list is retired');
ok(!/ai-assistant-panel-ep-test-results/.test(css), 'legacy result-panel CSS is retired');
ok(!/ai-assistant-panel-ep-health-result/.test(css), 'legacy per-result row CSS is retired');
ok(/browser-level reachability only; they do not verify API semantics, credentials, or model capability/.test(js), 'connection UI states the limits of reachability evidence');
ok(/Not tested/.test(js) && /Checking\\u2026|Checking\u2026/.test(js) && /Reachable/.test(js) && /Timeout/.test(js) && /Unreachable/.test(js), 'health column models idle/checking/reachable/timeout/unreachable states');
ok(/_makeEndpointActionGroup\([\s\S]*!!healthController[\s\S]*healthController \? healthController\.makeHooks/.test(js), 'active route actions feed individual probes into the same table health state');
ok(/_makeHealthBtn\(urlGetter, label, healthHooks\)/.test(js), 'shared action factory forwards health hooks rather than creating a second result model');
ok(/btn\.setAttribute\('aria-busy', 'true'\)/.test(js) && /pointer-events: none/.test(css), 'per-row probe busy state blocks duplicate clicks without stealing validation disabled authority');
ok(!/_makeHealthBtn[\s\S]{0,2200}btn\.disabled = false/.test(js), 'health completion cannot re-enable an Advanced action invalidated while the probe is in flight');
ok(/function _cancelConnectionRun\(restorePrevious\)/.test(js), 'bulk connection runs have explicit cancellation lifecycle');
ok(/_connectionRunCancels\[i\]\(\)/.test(js), 'profile/rerun cancellation aborts outstanding transport work');
ok(/return _cancel;/.test(js.slice(js.indexOf('function _pingUrl'), js.indexOf('function _fallbackCopy'))), 'low-level ping exposes cancellation instead of leaking background requests');
ok(/if \(runId !== _connectionRunSeq\) \{ return; \}/.test(js), 'stale bulk callbacks cannot overwrite a newer profile or rerun');
ok(/_connectionRunPrevious\[urls\[u\]\] = _connectionHealthCache\[urls\[u\]\] \|\| null/.test(js), 'bulk run snapshots stable health state before entering checking');
ok(/_cancelConnectionRun\(true\);[\s\S]*_connectionHealthBindings = Object\.create\(null\)/.test(js), 'profile refresh cancels work and rebinds the table atomically');
ok(/_refreshConnectionSummaryForProfile\(activeKey\)/.test(js), 'profile refresh derives Test connection availability/summary from current resolved topology');
ok(/role', 'status'/.test(js) && /aria-live', 'polite'/.test(js) && /aria-atomic', 'true'/.test(js), 'bulk summary is an accessible live status region');
ok(/_testConnectionBtn\.disabled = unique\.length === 0/.test(js), 'Test connection is disabled when the profile has no endpoint addresses');
ok(/checkedAt: Date\.now\(\)/.test(js) && /Checked ' \+ new Date\(state\.checkedAt\)\.toLocaleTimeString/.test(js), 'latest health observations retain a sheet-local check timestamp');
ok(/_isEndpointUrlActive\(url\)/.test(js), 'late individual health callbacks cannot announce status for an unrelated active profile');
ok(/var _resetTimer = null/.test(js) && /clearTimeout\(_resetTimer\)/.test(js) && /_resetTimer = setTimeout/.test(js), 'repeated per-row probes cannot let an older visual reset timer erase a newer result');
ok(/\.ai-assistant-panel-ep-route-th--health,[\s\S]*width: 7\.4rem/.test(css), 'Health has a stable readable table column');
ok(/\.ai-assistant-panel-ep-route-health--checking/.test(css) && /\.ai-assistant-panel-ep-route-health--ok/.test(css) && /\.ai-assistant-panel-ep-route-health--timeout/.test(css), 'table health states have dedicated visual semantics');
ok(/@keyframes ep-pulse/.test(css), 'checking animation keyframes remain defined after retiring the old result list');
ok(/\.ai-assistant-panel-ep-health-row--table-control[\s\S]*grid-template-columns: auto minmax\(0, 1fr\)/.test(css), 'bulk action and summary use compact table-control layout');
ok(/@media \(max-width: 520px\)[\s\S]*ai-assistant-panel-ep-health-row--table-control[\s\S]*grid-template-columns: 1fr/.test(css), 'connection controls collapse cleanly on narrow panels');

ok(/testBtn\.setAttribute\('aria-controls', 'ai-assistant-panel-ep-active-routes'\)/.test(js), 'bulk control is explicitly connected to the route table it updates');
ok(/testBtn\.setAttribute\('aria-describedby', 'ai-assistant-panel-ep-test-hint'\)/.test(js), 'bulk control is described by the reachability limitation text');
ok(/_urlDisplay\.setAttribute\('aria-busy', 'true'\)/.test(js) && /_urlDisplay\.setAttribute\('aria-busy', 'false'\)/.test(js), 'route region exposes bulk update busy state to assistive technology');
ok(/_openStateObserver[\s\S]*sheet\.getAttribute\('data-open'\) !== 'true'[\s\S]*_cancelConnectionRun\(true\)[\s\S]*_cancelIndividualHealthProbes\(\)/.test(js), 'every Endpoint-sheet close path aborts bulk and individual diagnostics');
ok(/function _cancelIndividualHealthProbes/.test(js) && /pending\[i\]\.cancel\(\)/.test(js) && /pending\[i\]\.reset\(\)/.test(js), 'individual probes have explicit abort and UI-state restoration');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
