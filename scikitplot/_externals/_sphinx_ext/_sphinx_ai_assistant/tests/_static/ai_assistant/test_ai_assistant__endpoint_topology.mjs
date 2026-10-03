import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
const ok = (cond, name) => {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
};

ok(/var _SCHEMA_VER\s*=\s*9;/.test(src), 'runtime profile storage schema v9 carries current routes and retires Assistant feedback endpoint fields');
ok(/\['base', 'chat', 'share', 'training', 'image', 'video', 'audio', 'document', 'publication'\]/.test(src), 'runtime URL validator accepts only current Assistant routes');
ok(/return resolveFor\(feature, getActive\(\)\)/.test(src) && /profile\.base/.test(src), 'feature resolution falls back to current profile base');

ok(/AI_ASSISTANT_ENDPOINT_API/.test(src), 'read-only endpoint bridge remains available to AI Learn');
ok(/_PUBLIC_ENDPOINT_FEATURES[\s\S]{0,240}'publication'/.test(src), 'public endpoint bridge exposes current publication routing');
ok(!/_PUBLIC_ENDPOINT_FEATURES[\s\S]{0,240}'feedback'/.test(src), 'public endpoint bridge excludes retired feedback feature');
ok(/resolveBaseFor/.test(src), 'canonical base resolver exists');
ok(/datasetRepo/.test(src), 'profile dataset metadata is retained');
ok(/Base endpoint/.test(src), 'simple UI exposes Base endpoint');
ok(/Configure one service endpoint/.test(src), 'simple UI explains one-service topology');
ok(/Auto-discovered from service/.test(src), 'simple dataset communicates discovery');
ok(/Save simple profile/.test(src), 'custom simple profiles are editable');
ok(/chat: '', share: '', training: '', image: '', video: '', audio: '', document: '', publication: ''/.test(src), 'simple save clears current route overrides');
ok(/Advanced endpoint routing/.test(src) && /Configured value/.test(src) && /Effective endpoint/.test(src), 'active Advanced UI uses semantic routing table terminology');
ok(/Base endpoint \*/.test(src), 'advanced form requires base');
ok(/ph: 'Absolute URL, relative ' \+ fd\.suffix\.replace/.test(src) && /or blank to inherit/.test(src), 'advanced form derives absolute, relative, or inherited endpoint guidance from the canonical registry');
ok(/Dataset override/.test(src), 'advanced form supports dataset override');
ok(/var seen = Object\.create\(null\);[\s\S]*if \(!seen\[url\]\)[\s\S]*urls\.push\(url\)/.test(src), 'connectivity test deduplicates transport probes by effective URL');
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
ok(/_resolveFeatureEndpoint\(_cd, key\)/.test(src), 'capability pills use the shared complete-endpoint resolver');
ok(/resolveEndpoint:\s+resolveEndpoint/.test(src), 'active registry exposes complete endpoint resolver');
ok(/image:\s+'\/v1\/image'/.test(src), 'image route inherits the canonical short generation endpoint');
ok(/video:\s+'\/v1\/video'/.test(src), 'video route inherits the canonical short generation endpoint');
ok(/audio:\s+'\/v1\/audio'/.test(src), 'audio route inherits the canonical short generation endpoint');
ok(/document:\s+'\/v1\/document'/.test(src), 'document route inherits the canonical generation endpoint');
ok(/publication:\s+'\/v1\/learn'/.test(src), 'publication route inherits canonical /v1/learn');
ok(!/feedback:\s+'\/v1\/feedback'/.test(src), 'Assistant endpoint registry does not own generic page feedback');
ok(/Test AI Learn publication/.test(src), 'endpoint sheet exposes a non-mutating reviewed-publication policy test');
ok(/AI_ASSISTANT_ENDPOINT_API/.test(src) && /onProfileChange/.test(src), 'read-only sibling endpoint bridge exposes routing and profile changes');
ok(/AI_ASSISTANT_MODEL_API/.test(src), 'sibling model bridge is installed');
ok(/listModels:\s*function/.test(src) && /getState:\s*function/.test(src), 'model bridge exposes read-only model state');
ok(/selectModel:\s*function \(id\) \{ return _selectQuickModel/.test(src), 'model bridge selects through the canonical assistant model path');
ok(/openPicker:\s*function/.test(src) && /ai-assistant-open-model-configuration/.test(src), 'model bridge can open the canonical Assistant model configuration sheet');
ok(/onChange:\s*function/.test(src) && /ai-assistant-effort-change/.test(src), 'model bridge synchronizes model and effort changes');
ok(/function _publicModelSnapshot/.test(src) && !/function _publicModelSnapshot[\s\S]{0,1200}endpoint:/.test(src), 'public model snapshot does not expose endpoint routing');
ok(/resolveEndpointFor/.test(src), 'arbitrary-profile complete endpoint resolver exists');
ok(/Save routing/.test(src), 'runtime profiles can save Advanced routing changes');
ok(/_advBaseInp\.readOnly = !canEditSimple/.test(src), 'runtime Advanced Base endpoint is editable');
ok(/_advInputs\[_ofd\.key\]\.readOnly = !canEditSimple/.test(src), 'runtime Advanced feature endpoints are editable');
ok(/_epSafe\.resolveEndpointFor\(_sf, key\)/.test(src), 'Expanded conf.py snippet emits complete resolved routes');


ok(/function _buildEndpointResolutionTable/.test(src), 'shared endpoint-resolution table renderer exists');
ok((src.match(/_buildEndpointResolutionTable\(/g) || []).length >= 3, 'shared resolution renderer powers active and per-profile URL surfaces');
ok(/_makeEndpointTableShell/.test(src), 'endpoint table shell is centralized');
ok(/_refreshAdvancedEffectiveEndpoints/.test(src), 'Advanced routing previews effective endpoints');
ok(/inp\.addEventListener\('input', function \(\) \{ _renderAdvancedEffectiveCell/.test(src), 'Advanced per-route effective endpoint updates live');
ok(/_advBaseInp\.addEventListener\('input', _refreshAdvancedEffectiveEndpoints\)/.test(src), 'Base edits refresh every inherited effective endpoint');
ok(/document\.createElement\('div'\);\n            card\.className = 'ai-assistant-panel-ep-card'/.test(src), 'profile card no longer wraps nested actions in a label element');
ok(/target\.closest\('button, a, input, textarea, select'\)/.test(src), 'profile card click delegation preserves nested interactive controls');
ok(!/urlTxt\.appendChild\(_makeCopyBtn/.test(src), 'copy button is not nested inside endpoint anchor');
ok(/capLabel: 'Learn'/.test(src) && /for \(var _ci = 0; _ci < _FEATURE_DEFS\.length; _ci\+\+\)/.test(src), 'all capability chips derive from the canonical feature registry');

ok(!/Browser-wide dataset override/.test(src), 'legacy dataset editor is removed from visible UI');
ok(!/Share Configuration/.test(src), 'duplicate Share configuration section is removed');
ok(!/Training Configuration/.test(src), 'duplicate Training configuration section is removed');
ok(!/Rating scale selector|Feedback question text/.test(src), 'endpoint coming-soon placeholders are removed');
console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
