// Run 8 contract harness for the unified Share conversation architecture.
import fs from 'node:fs';
import path from 'node:path';
const target=process.argv[2]; const src=fs.readFileSync(target,'utf8');
const siblingCss=path.join(path.dirname(target),'ai-assistant.css');
const suppliedCss=process.argv[3];
const css=fs.readFileSync(suppliedCss || siblingCss,'utf8');
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,s=false,q=null,line=false,block=false;for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1];if(line){if(c==='\n')line=false;continue}if(block){if(c==='*'&&n==='/'){block=false;j++}continue}if(q){if(c==='\\'){j++;continue}if(c===q)q=null;continue}if(c==='/'&&n==='/'){line=true;j++;continue}if(c==='/'&&n==='*'){block=true;j++;continue}if(c==='"'||c==="'"||c==='`'){q=c;continue}if(c==='{'){d++;s=true}else if(c==='}'){d--;if(s&&d===0)return src.slice(i,j+1)}}throw new Error('unbalanced '+name)}
let pass=0,fail=0;function t(name,got,want=true){if(got===want)pass++;else{fail++;console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`)}}
const outer=extract('_buildConversationShareSheet');
const panel=extract('_buildFmtSharePanel');
const mode=extract('_buildExportModeControl');
const setter=extract('_setExportLinkMode');
const openSheet=extract('_openSheet');
const openShare=extract('_openConversationShare');
const clearConversation=extract('clearConversation');
const downloadFormat=extract('_downloadConversationFormat');
const saveGlobalSS=extract('_buildConversationShareSheet').slice(extract('_buildConversationShareSheet').indexOf('function _saveGlobalSS('), extract('_buildConversationShareSheet').indexOf('_globalShareState = _loadGlobalSS();'));

// One shell; format is representation only, destination is sheet-level state.
t('one unified Share sheet builder',(src.match(/function _buildConversationShareSheet\(/g)||[]).length,1);
t('conversation artifact filename helper exists',src.includes('function _conversationArtifactFilename('));
t('local Save filenames encode lifecycle role and format',src.includes("'ai-conversation-' + normalizedRole + '-' + meta.fmt + stamp + meta.ext"));
t('direct downloads use provenance filename helper',downloadFormat.includes("_conversationArtifactFilename('local-save', meta.fmt)"));
t('legacy ambiguous conversation timestamp filename is absent',src.includes("'ai-conversation-' + _isoFileStamp() + meta.ext"),false);
t('one lightweight format descriptor builder',(src.match(/function _buildFmtSharePanel\(/g)||[]).length,1);
t('legacy format-specific share sheet absent',src.includes('function _buildFmtShareSheet('),false);
t('one unified sheet instance',(src.match(/var convShareSheet = _buildConversationShareSheet\(/g)||[]).length,1);
t('sheet stable id',outer.includes("sheet.id = 'ai-assistant-panel-conv-share-sheet';"));
t('sheet is modal dialog',outer.includes("sheet.setAttribute('role', 'dialog');"));
t('outer defaults to local Save file destination',outer.includes("var selectedDestination = 'download';"));
// The preset now follows the destination's exposure: a local device file
// keeps its provenance, anything published keeps the privacy-lighter default.
// A real local save produced a file whose every timestamp, model field,
// session id and page url was null.
t('outer owns content preset',outer.includes("var contentPreset = _conversationPresetForDestination('download');"));
t('local destinations default to complete provenance',src.includes("return (destination === 'download' || destination === 'local')") && src.includes("? 'complete' : 'standard';"));
t('switching destination re-defaults an unchosen preset',outer.includes("if (contentPreset !== 'custom') {"));
t('a chosen preset survives a destination change',outer.includes("contentPreset !== 'custom'"));
t('outer owns one result state',outer.includes('var resultState = null;'));
t('page-memory managed artifact registry exists',src.includes('var _managedConversationArtifacts = [];'));
t('Share sheet consumes page-memory artifact registry',outer.includes('var managedArtifacts = _managedConversationArtifacts;'));
t('format panel is description-only',panel.includes('meta.shareDesc || meta.desc'));
t('format panel does not own Global save',panel.includes('_postGlobalShare'),false);
t('format panel does not own contribution',panel.includes('_postTrainingContribution'),false);

// Five live formats from one registry.
t('live format order is JSON HTML Text YAML TOML',/fmt:\s*'json'[\s\S]*fmt:\s*'html'[\s\S]*fmt:\s*'txt'[\s\S]*fmt:\s*'yaml'[\s\S]*fmt:\s*'toml'/.test(src));
t('YAML live serializer registered',src.includes('buildStr: function (snapshot) { return _buildConvYamlString(snapshot); }'));
t('TOML live serializer registered',src.includes('buildStr: function (snapshot) { return _buildConvTomlString(snapshot); }'));
t('YAML canonical MIME',src.includes("mime: 'application/yaml'"));
t('TOML canonical MIME',src.includes("mime: 'application/toml'"));
t('no live TOML preview remains',/_stubFormat\(\s*'toml'/.test(src),false);
t('format switcher is tablist',outer.includes("nav.setAttribute('role', 'tablist');"));
t('format buttons are tabs',outer.includes("btn.setAttribute('role', 'tab');"));
t('format tabs support arrows',/ArrowRight[\s\S]*ArrowLeft/.test(outer));
t('format tabs support Home End',/event\.key === 'Home'[\s\S]*event\.key === 'End'/.test(outer));

// Destination mental model.
t('Save file destination exists',outer.includes("_makeDestination('download', 'Save file'"));
t('Local preview destination exists',outer.includes("_makeDestination('local', 'Local preview'"));
t('Self-contained destination exists',outer.includes("_makeDestination('self_contained', 'Self-contained link'"));
t('Self-contained Open preserves canonical artifact URL', /function _openArtifact\(artifact\)[\s\S]*window\.open\(artifact\.url/.test(outer));
t('Self-contained Open does not rebuild data artifact as Blob', !/function _openArtifact\(artifact\)[\s\S]*?URL\.createObjectURL[\s\S]*?function _globalLifecycleText/.test(outer));
t('Global destination exists',outer.includes("_makeDestination('global', 'Global link'"));
t('new destination enum avoids private/public transport labels',/selectedDestination\s*=\s*'private'/.test(outer),false);
t('self-contained copy says not encrypted',/not encrypted/i.test(outer));
t('self-contained copy says copied links cannot be revoked',outer.includes('copied links cannot be revoked'));
t('Global copy says revocable with edit capability',outer.includes('revocable while the private edit capability is available'));
t('primary labels follow destination',/Save [\s\S]*Open preview[\s\S]*Create link[\s\S]*Create global link/.test(outer));

// Content/privacy before serialization.
t('content presets exist',outer.includes("['standard','minimal','complete','custom']"));
t('content option checklist is always visible for preset inspection',/customGrid\.style\.display = ''/.test(outer));
t('content preset hint explains automatic Customize transition',outer.includes('Preset selections are shown below. Change any option to switch to Customize.'));
t('preset selection never hides content option checklist',/customGrid\.style\.display = key === 'custom'/.test(outer),false);
t('editing a checkbox automatically transitions to Customize',/customControls\[key\]\.addEventListener\('change'[\s\S]*contentPreset = 'custom'[\s\S]*_setPreset\('custom'\)/.test(outer));
t('granular timestamp control exists',outer.includes("['includeTimestamps','Timestamps']"));
t('granular model control exists',outer.includes("['includeModel','Model and provider']"));
t('granular ratings control exists',outer.includes("['includeRatings','Ratings and feedback']"));
t('safe source URLs control names URL scope clearly',outer.includes("['includeSafeSourcePage','Safe source URLs']"));
t('Advanced no longer owns a duplicate download path',outer.includes('Download current snapshot'),false);
t('granular session id opt-in exists',outer.includes("['includeSessionId','Session identifier']"));
t('locked URL sanitization row exists',outer.includes('never exports local filesystem/custom-scheme paths'));
t('current snapshot uses content options before serializer',/function _currentSnapshot\(\)[\s\S]*_buildConversationSnapshot\(contentOptions\)/.test(outer));
t('share preflight reviews canonical snapshot',/await _privacyPreflightReview\(snapshot/.test(outer));
t('redacted review value becomes outbound snapshot',outer.includes('return reviewed.value;'));

// Size/preflight helpers.
t('UTF8 byte helper exists',src.includes('function _utf8ByteLength('));
t('self-contained warning budget exists',outer.includes('48 * 1024'));
t('self-contained hard budget exists',outer.includes('256 * 1024'));
t('self-contained encoded URL cap exists',outer.includes('384 * 1024'));
t('current self-contained transport is portable base64 data HTML',outer.includes('_buildPortableSelfContainedDataUrl(snapshot, meta.fmt)'));
t('current self-contained path no longer generates c2',!/selectedDestination === 'self_contained'[\s\S]{0,1400}_buildSelfContainedHashUrl/.test(outer));
t('oversize self-contained blocks',outer.includes('too large for the configured portable data-link budget'));
t('long link warning exists',outer.includes('some messaging apps may truncate it'));

// Result component and truthful lifecycle.
t('one Result component exists',outer.includes("resultWrap.className = 'ai-assistant-conv-share-result';"));
t('self-contained huge URL hidden by default',/resultInput\.style\.display = 'none'/.test(outer));
t('local result exposes Copy link',/resultState\.kind === 'local'[\s\S]*copyResultBtn\.textContent = 'Copy link'[\s\S]*inspectResultBtn\.style\.display = ''/.test(outer));
t('local result labels removal as browser removal',/resultState\.kind === 'local'[\s\S]*removeResultBtn\.textContent = 'Remove from browser'/.test(outer));
t('managed local artifacts expose Copy link',/artifact\.kind === 'global' \|\| artifact\.kind === 'self_contained' \|\| artifact\.kind === 'local'/.test(outer));
t('Inspect reveals local or self-contained URL',/resultState\.kind !== 'self_contained' && resultState\.kind !== 'local'[\s\S]*resultInput\.style\.display/.test(outer));
t('stale result warning exists',outer.includes('Conversation or options changed'));
t('format/content changes mark old result stale',outer.includes('function _markStale()'));

// Artifact lifecycle — user requested every managed creation have removal semantics.
t('managed artifacts section exists',outer.includes("_sectionTitle('Created artifacts')"));
t('local artifact revokes Blob URL',/artifact\.kind === 'local'[\s\S]*URL\.revokeObjectURL\(artifact\.url\)/.test(outer));
t('self-contained removal is local-only truthful',outer.includes('Already copied self-contained links cannot be revoked'));
t('download artifact explicitly leaves browser control',outer.includes('The file has left this page’s control'));
t('download removal says delete file on device',outer.includes('Delete the file itself from your device'));
t('direct toolbar downloads register lifecycle artifacts',downloadFormat.includes('_registerManagedConversationArtifact({'));
t('direct download record says external device file',downloadFormat.includes('external device file · delete with file manager'));
t('artifact registry is not Web Storage persisted',/_ss(Set|Get)\([^\n]*managedConversationArtifacts/.test(src),false);
t('legacy IndexedDB clear helper exists',src.includes('function _idbClearShares('));
t('Advanced exposes legacy local deletion',outer.includes('Delete legacy local Share artifacts'));
t('Global revoke helper exists',src.includes('function _deleteGlobalShare('));
t('Global artifact uses server DELETE capability',/artifact\.kind === 'global'[\s\S]*_deleteGlobalShare\(/.test(outer));
t('Global local-only fallback is truthful when edit token absent',outer.includes('This does not prove or perform remote revocation'));
t('edit token remains memory-only',/Mutation\/revoke authority[\s\S]{0,160}page-memory only/.test(outer));
t('session storage serialization omits editToken',/editToken\s*:/.test(saveGlobalSS),false);
t('new chat preserves artifact registry for old-link revocation',outer.includes('Do not discard managed artifacts or the public Global ledger'));

// Global structured contract remains Run-3 server authority.
t('Global payload is snapshot format ttl only',/var payload = recoveringGlobal \? _pendingGlobalCreate\.payload : \{ snapshot: snapshot, format: meta\.fmt, ttlDays: g\.ttlDays \}/.test(outer));
t('Global PATCH requires live edit capability',/_globalShareState\.uuid && _globalShareState\.editToken/.test(outer));
t('Global DELETE passes artifact edit capability',outer.includes('artifact.editToken'));
t('Global POST does not send MIME authority',/payload\s*=\s*\{[^}]*mime/.test(outer),false);

// Contribution is now a separate first-class control plane. Share must own none of it.
t('Share has no contribution More actions',outer.includes("_collapsible('More actions'"),false);
t('Share has no rated-answer contribution button',outer.includes('Contribute rated answers…'),false);
t('Share has no contribution endpoint resolver',outer.includes('_resolveContributionEndpoint'),false);
t('Share has no contribution receipt capability',outer.includes('X-Contribution-Delete-Token'),false);
t('dedicated contribution sheet exists outside Share',src.includes("id = 'ai-assistant-panel-contribution-sheet'"));

// Conversation identity and async race protection.
t('explicit conversation ID infrastructure remains',src.includes("var _CONVERSATION_ID_KEY = 'ai-assistant-conversation-id';"));
t('clear rotates conversation id',clearConversation.includes('var nextConversationId = _rotateConversationId();'));
t('Share binds to explicit conversation id',outer.includes('var boundConversationId = _getConversationId();'));
t('Share captures operation conversation id',outer.includes('var opConversationId = boundConversationId;'));
t('async review verifies conversation id',outer.includes('opConversationId !== _getConversationId()'));
t('Global success verifies conversation id',outer.includes("function success(res) {\n                if (opConversationId !== boundConversationId || opConversationId !== _getConversationId()) return;"));

// Existing Download/Share top-level mode remains one observable state.
t('shared mode control builder exists once',(src.match(/function _buildExportModeControl\(/g)||[]).length,1);
t('menu mode uses menuitemcheckbox',mode.includes("row.setAttribute('role', 'menuitemcheckbox');"));
t('setter notifies observers',setter.includes('_notifyExportState();'));
t('setter has no singleton DOM authority',setter.includes('getElementById'),false);
t('mode builder has no singleton DOM id',mode.includes("row.id = 'ai-assistant-export-link-toggle'"),false);
t('central Share dispatcher exists once',(src.match(/function _openConversationShare\(/g)||[]).length,1);
t('dispatcher selects requested format',openShare.includes('convShareSheet._selectExportFormat(fmt)'));
t('dispatcher opens unified sheet',openShare.includes('_openSheet(convShareSheet, opener);'));
t('open sweep consumes sheet registry',openSheet.includes('_allPanelSheets().forEach'));
t('sheet registry includes unified Share',src.includes("{ key: 'conversation-share', sheet: convShareSheet,   toolbarId: 'conv-share' }"));

// CSS for new IA + responsive layout.
t('v2 body styled',css.includes('.ai-assistant-conv-share-v2-body'));
t('destination cards styled',css.includes('.ai-assistant-conv-share-destinations'));
t('result component styled',css.includes('.ai-assistant-conv-share-result'));
t('artifact lifecycle list styled',css.includes('.ai-assistant-conv-share-artifacts'));
t('narrow destination layout becomes vertical',/@media \(max-width:720px\)[\s\S]*ai-assistant-conv-share-destinations/.test(css));

// ── Destination layout ────────────────────────────────────────────────────
//
// Four destinations in a three-column track always produced a 3 + 1 row: three
// cards of one width and a fourth, alone, stretched across. That reads as a
// hierarchy the destinations do not have -- Global link is a peer of the other
// three, not a summary of them.
const dest_css = fs.readFileSync(process.argv[3], 'utf8');
const destRule = (dest_css.match(/\.ai-assistant-conv-share-destinations \{[^}]*\}/) || [''])[0];
t('destinations use a two-column track', /grid-template-columns:\s*repeat\(2,\s*minmax\(0,1fr\)\)/.test(destRule), true);
t('no three-column track remains', !/repeat\(3,/.test(destRule), true);
t('rows share a height so a wrapped description does not unbalance them', /grid-auto-rows:\s*1fr/.test(destRule), true);
t('cards fill their track, or equal rows would not show', /\.ai-assistant-conv-share-destination \{[^}]*height:\s*100%/.test(dest_css), true);
t('a narrow panel drops to one column', /\.ai-assistant-conv-share-destinations \{ grid-template-columns:1fr; grid-auto-rows:auto; \}/.test(dest_css), true);
// Two columns divide four evenly and stay even at three or two, so the layout
// does not depend on how many destinations the build happens to offer.
const destCount = (src.match(/destWrap\.appendChild\(/g) || []).length;
t('destination count is not assumed by the track', destCount === 0 || /repeat\(2,/.test(destRule), true);

// ── Collapsible sections ──────────────────────────────────────────────────
//
// Content-and-privacy was already a bordered card while its siblings got a
// single top rule, so one control looked like a panel and the rest looked like
// separators. A reader scanning for "where do I change what gets sent" had to
// learn which rows were interactive by clicking them.
const col_css = fs.readFileSync(process.argv[3], 'utf8');
const collapseRule = (col_css.match(/\.ai-assistant-conv-share-collapse \{[^}]*\}/) || [''])[0];
t('every collapsible section is a bordered card', /border:1px solid/.test(collapseRule) && /border-radius:\.8rem/.test(collapseRule), true);
t('no section is only a top rule', !/^[^}]*border-top:1px solid rgba\(127,127,127,\.18\)/.test(collapseRule), true);
t('adjacent sections are spaced, not fused', /\.ai-assistant-conv-share-collapse \+ \.ai-assistant-conv-share-collapse \{ margin-top/.test(col_css), true);
t('keyboard focus is visible on the card, not only its button', /\.ai-assistant-conv-share-collapse:focus-within/.test(col_css), true);

// The section governing what leaves the device is not a peer of the others.
const privRule = (col_css.match(/\.ai-assistant-panel-review-content-privacy \{[^}]*\}/) || [''])[0];
t('the privacy section keeps the shared card shape', /border-radius: \.8rem/.test(privRule), true);
t('and takes an accent leading edge', /border-inline-start: 3px solid var\(--ai-artifact-accent\)/.test(privRule), true);
t('the distinction survives forced-colours mode', /forced-colors: active[\s\S]{0,220}?review-content-privacy[\s\S]{0,80}?border-inline-start-width/.test(col_css), true);
// The border is a signpost, not a safeguard: what makes the section safe is
// that the snapshot is built from the reviewed options, asserted separately.
t('the reviewed options still drive the snapshot', src.includes('_conversationContentPreset(contentPreset)'), true);

// ── Width containment and section separation ─────────────────────────────
const lay_css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');

// A flex item's default `min-width: auto` refuses to shrink below its content,
// so `overflow-x: auto` on its own did nothing: the switcher grew to fit all
// five tabs and widened the sheet instead of scrolling.
const swRule = (lay_css.match(/\.ai-assistant-conv-share-format-switcher \{[^}]*\}/) || [''])[0];
t('the tab switcher can shrink below its content', /min-width:\s*0/.test(swRule), true);
t('and is capped at its container width', /max-width:\s*100%/.test(swRule), true);
// Superseded: the switcher wraps instead of scrolling. Horizontal scrolling
// hid TOML behind the edge with nothing indicating it existed, and a format the
// reader cannot see is one they will not choose.
t('the switcher wraps into rows rather than scrolling', /display:\s*grid/.test(swRule) && !/overflow-x:\s*auto/.test(swRule), true);
t('every tab is the same width, so rows line up as a grid', /grid-template-columns:\s*repeat\(auto-fit,\s*minmax\(min\(6rem,\s*100%\),\s*1fr\)\)/.test(swRule), true);
// auto-fit collapses empty tracks, so a trailing row of two sits at two
// columns rather than one tab stretching across -- the orphan R173T36 fixed
// for the destination cards, avoided here without fixing a column count.
t('no fixed column count is assumed', !/repeat\(\d+,/.test(swRule), true);
t('a single column never exceeds a narrow container', /minmax\(min\(6rem,\s*100%\)/.test(swRule), true);
t('tabs fill their track and centre their label', /\.ai-assistant-conv-share-format-switcher > \* \{[^}]*text-align:\s*center/.test(lay_css), true);
t('the narrow override no longer forces scrolling back on', !/\.ai-assistant-conv-share-format-switcher \{ overflow-x:auto; flex-wrap:nowrap; \}/.test(lay_css), true);

// Managed artifact metadata and actions are separate layout groups. This avoids
// shrinking the description to zero merely to keep direct-child buttons inline.
const artRule = (lay_css.match(/\.ai-assistant-conv-share-artifact \{[^}]*\}/) || [''])[0];
t('the artifact card can shrink inside its own parent', /min-width:0/.test(artRule) && /max-width:100%/.test(artRule), true);
t('its text keeps one meaningful flex basis', /\.ai-assistant-conv-share-artifact-text \{[^}]*flex:1 1 12rem/.test(lay_css), true);
t('artifact actions are an explicit group', /ai-assistant-conv-share-artifact-actions/.test(outer), true);
t('narrow artifact cards stack from actual container width', /@container conv-share-artifacts \(max-width: 30rem\)[\s\S]*?flex-direction:column/.test(lay_css), true);

// A title after a bordered full-height preview read as the preview's caption.
t('a section title after the format preview is separated from it', /\.ai-assistant-conv-share-format-host \+ \.ai-assistant-conv-share-section-title[\s\S]{0,140}?margin-top:\s*1rem/.test(lay_css), true);
t('the base title spacing is unchanged for ordinary siblings', /\.ai-assistant-conv-share-section-title \{ margin-top:\.2rem/.test(lay_css), true);

console.log(`\n${pass} passed, ${fail} failed`);process.exit(fail?1:0);
