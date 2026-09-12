import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
const ok = (cond, name) => {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
};

ok(/var _SCHEMA_VER\s*=\s*3;/.test(src), 'runtime profile storage schema v3 strips persisted tokens');
ok(/\['base', 'chat', 'share', 'feedback', 'training'\]/.test(src), 'runtime URL validator accepts base');
ok(/profile\[feature\] \|\| profile\.base/.test(src), 'feature resolution falls back to base');
ok(/resolveBaseFor/.test(src), 'canonical base resolver exists');
ok(/datasetRepo/.test(src), 'profile dataset metadata is retained');
ok(/Base endpoint/.test(src), 'simple UI exposes Base endpoint');
ok(/Configure one service endpoint/.test(src), 'simple UI explains one-service topology');
ok(/Auto-discovered from service/.test(src), 'simple dataset communicates discovery');
ok(/Save simple profile/.test(src), 'custom simple profiles are editable');
ok(/chat: '', share: '', feedback: '', training: ''/.test(src), 'simple save clears route overrides');
ok(/fd\.label \+ ' endpoint override'/.test(src), 'active Advanced UI uses endpoint override terminology');
ok(/Base endpoint \*/.test(src), 'advanced form requires base');
ok(/Absolute URL, relative v1\/chat\/completions, or blank to inherit/.test(src), 'advanced form accepts absolute, relative, or inherited endpoints');
ok(/Dataset override/.test(src), 'advanced form supports dataset override');
ok(/_seenTestUrls/.test(src), 'connectivity test deduplicates inherited service bases');
ok(/testBtn\.textContent = 'Test connection'/.test(src), 'connectivity CTA is singular');
ok(/Runtime & Data/.test(src), 'lower operator area is named Runtime & Data');
ok(/bodyEl\.insertBefore\(extSection, addSection\)/.test(src), 'Runtime & Data mounts before Service diagnostics / Add Custom Profile');
ok(/_buildSheetSection\('Service diagnostics'\)/.test(src), 'service diagnostics is its own sheet section');
ok(/bodyEl\.insertBefore\(diagnosticsSection, addSection\)/.test(src), 'service diagnostics mounts before Add Custom Profile');
ok(/profileRepo \|\| explicitRepo \|\| customRepo/.test(src), 'dataset priority is profile then conf then compatibility fallback');
ok(/prof\.base/.test(src) && /prof\.datasetRepo/.test(src), 'conf.py snippet preserves topology fields');
ok(/conf\.py helper/.test(src), 'conf.py helper uses the compact helper label');
ok(/snippetMode = 'recommended'/.test(src) && /Expanded/.test(src) && /Advanced/.test(src), 'conf.py helper offers recommended, expanded, and advanced modes');
ok(/Endpoint route forms accepted by every feature/.test(src), 'Advanced snippet documents accepted endpoint route forms');
ok(/Base-relative endpoint — leading \/ is optional/.test(src), 'Advanced snippet explains relative routes');
ok(/Inherit .*None \/ \"\" \/ omitted/.test(src), 'Advanced snippet explains inherited routes');
ok(/mode === 'advanced'/.test(src) && /snippetAdvancedBtn/.test(src), 'Advanced snippet mode is interactive');
ok(/ai_assistant_endpoint_default_profile/.test(src), 'generated snippet persists the active default profile');
ok(/_explicit && \(!base \|\| _explicit !== base\)/.test(src), 'recommended snippet emits only true route overrides');
ok(/prof\.ttlDays !== 30/.test(src), 'recommended snippet omits the default TTL');
ok(/_pyDqEscape\(key\)/.test(src), 'generated profile key is Python-string escaped');
ok(/Secrets\/tokens are intentionally excluded/.test(src), 'generated snippet explicitly excludes secrets');
ok(/_epSafe\.resolveFor\(_cd\.key, key\)/.test(src), 'capability pills use resolved inherited routes');
ok(/resolveEndpoint:\s+resolveEndpoint/.test(src), 'active registry exposes complete endpoint resolver');
ok(/resolveEndpointFor/.test(src), 'arbitrary-profile complete endpoint resolver exists');
ok(/Save routing/.test(src), 'runtime profiles can save Advanced routing changes');
ok(/_advBaseInp\.readOnly = !canEditSimple/.test(src), 'runtime Advanced Base endpoint is editable');
ok(/_advInputs\[_ofd\.key\]\.readOnly = !canEditSimple/.test(src), 'runtime Advanced feature endpoints are editable');
ok(/_epSafe\.resolveEndpointFor\(_sf, key\)/.test(src), 'Expanded conf.py snippet emits complete resolved routes');


ok(!/Browser-wide dataset override/.test(src), 'legacy dataset editor is removed from visible UI');
ok(!/Share Configuration/.test(src), 'duplicate Share configuration section is removed');
ok(!/Training Configuration/.test(src), 'duplicate Training configuration section is removed');
ok(!/Rating scale selector|Feedback question text/.test(src), 'endpoint coming-soon placeholders are removed');
console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
