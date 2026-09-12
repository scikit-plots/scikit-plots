// Run 102 regression — slash command discovery/autocomplete palette.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3],'utf8');
let passed=0,failed=0; function ok(c,n){if(c)passed++;else{failed++;console.error('FAIL '+n);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,start=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;start=true;}else if(src[j]==='}'){d--;if(start&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}

ok(src.includes('function _localSlashCommandDefinitions()'),'one canonical local command registry exists');
ok(src.includes("id: 'skill-creator'") && src.includes('command: _SKILL_CREATOR_COMMAND'),'skill creator is registered, not separately copied into palette');
ok(src.includes("id: 'add-current-page-context'") && src.includes('command: _ADD_CURRENT_PAGE_CONTEXT_COMMAND'),'current-page context command is registered canonically');
ok(src.includes("id: 'pin-current-page'") && src.includes('command: _PIN_CURRENT_PAGE_COMMAND'),'pin-current-page command is registered canonically');
ok(src.includes("id: 'add-files'") && src.includes('command: _ADD_FILES_COMMAND'),'file-picker command is registered canonically');
ok(src.includes('function _availableLocalSlashCommands()'),'availability filtering is centralized');
ok(src.includes("skill.enabled = _cfg().panelSkillGenerator !== false"),'registry follows Skill Generator feature flag without making base discovery depend on dynamic config');
ok(src.includes('function _slashCommandDiscoveryAt(value, selectionStart, selectionEnd)'),'slash discovery is range-aware and side-effect free');
ok(src.includes('function _slashCommandFilterQuery(value)'),'compatibility query wrapper remains available');
ok(src.includes('function _slashCommandMatches(item, query)'),'slash filtering is a reusable pure matcher');

const normSrc=extract('_normalizeSlashCommandText');
const discoverySrc=extract('_slashCommandDiscoveryAt');
const filterSrc=extract('_slashCommandFilterQuery');
const filter=new Function(normSrc+'\n'+discoverySrc+'\n'+filterSrc+'\nreturn _slashCommandFilterQuery;')();
ok(filter('/')==='', 'bare slash activates discovery with empty query');
ok(filter('/skill')==='skill','slash token becomes lowercase filter text');
ok(filter('/SKILL')==='skill','filtering is case tolerant');
ok(filter('/ ')===null,'slash followed immediately by space exits discovery and becomes literal text');
ok(filter('/Add current')==='add current','multi-word friendly command phrases remain discoverable');
ok(filter('hello /')==='', 'slash after existing prose opens discovery');
ok(filter(' /skill')==='skill','leading whitespace before slash is accepted');
ok(filter('hello/')===null,'slash inside an ordinary word does not open discovery');
ok(filter('')===null,'empty composer does not open palette');

const matchSrc=extract('_slashCommandMatches');
const match=new Function(normSrc+'\n'+matchSrc+'\nreturn _slashCommandMatches;')();
const def={command:'/skill-creator',title:'Skill Creator',keywords:['agent','eval','benchmark']};
ok(match(def,''),'empty query lists every available command');
ok(match(def,'skill'),'command prefix matches');
ok(match(def,'ski'),'partial command prefix matches');
ok(match(def,'agent'),'keyword prefix matches');
ok(match(def,'eval'),'secondary keyword matches');
ok(!match(def,'memory'),'unrelated query is filtered out');

ok(src.includes("slashInlineHint.textContent = 'Type to filter'"),'bare slash gets inline Type to filter hint');
ok(src.includes("slashPalette.setAttribute('role', 'menu')") && src.includes("slashPalette.setAttribute('aria-label', 'Slash commands')"),'palette exposes menu semantics');
ok(src.includes("input.setAttribute('aria-autocomplete', 'list')") && src.includes("input.setAttribute('aria-controls', 'ai-assistant-panel-slash-command-menu')"),'composer exposes autocomplete ownership');
ok(src.includes("input.setAttribute('aria-activedescendant', selected.id)"),'keyboard highlight is announced without moving textarea focus');
ok(src.includes("slashPaletteStatus.setAttribute('aria-live', 'polite')"),'result count is announced accessibly');
ok(src.includes("slashPaletteList.replaceChildren()"),'filter rendering is idempotent rather than append-only');
ok(src.includes("_availableLocalSlashCommands().filter(function (item)"),'palette lists only currently available commands');
ok(src.includes("empty.textContent = 'No matching local commands'"),'no-match state is explicit');

const keySrc=extract('_handleSlashCommandKeydown');
ok(keySrc.includes("e.key === 'ArrowDown'") && keySrc.includes("e.key === 'ArrowUp'"),'arrow navigation is supported');
ok(keySrc.includes("e.key === 'Home'") && keySrc.includes("e.key === 'End'"),'Home/End navigation is supported');
ok(keySrc.includes("e.key === 'Escape'") && keySrc.includes('_closeSlashCommandPalette(true)'),'Escape dismisses without mutating input');
ok(keySrc.includes("e.key === 'Tab'") && keySrc.includes("e.key === 'Enter'"),'Tab and Enter own completion/run semantics');
ok(keySrc.includes("if (e.key === 'Tab') _replaceActiveSlashCommand(item)") && keySrc.includes('else _executeSlashCommand(item)'),'Tab completes while Enter runs the selected local action');
ok(src.includes('value.slice(0, active.start) + inserted + value.slice(active.end)'),'autocomplete replaces only the active slash range and preserves surrounding prose');
ok(src.includes("slashPalette.addEventListener('pointerdown'") && src.includes('e.preventDefault()'),'pointer selection preserves mobile textarea/caret focus');
ok(src.includes("row.addEventListener('click', function () { _executeSlashCommand(item); })"),'pointer selection executes the same local command authority as Enter');
ok(src.includes("document.addEventListener('pointerdown'") && src.includes("slashPalette.getAttribute('data-open') !== 'true'"),'outside pointer closes open slash palette');
ok(src.includes('_closeSlashCommandPalette(false);\n            _syncCurrentPageAttachMenuItem();'),'opening Attach menu closes slash palette');
const renderPaletteSrc=extract('_renderSlashCommandPalette');
ok(renderPaletteSrc.includes('_closeAttachMenu(false);') && renderPaletteSrc.includes('slashPalette.hidden = false'),'opening slash palette closes Attach menu before commit');
ok(src.includes("window.visualViewport.addEventListener('resize', _positionSlashCommandPalette"),'virtual-keyboard/viewport resize repositions palette');
ok(src.includes('groupRect.top - bodyRect.top - 8'),'palette max height is bounded by available panel-body space');
ok(src.includes('function _composerSlashFilterQuery()'),'composer eligibility is separate from raw slash token parsing');
ok(src.includes('return _slashCommandDiscoveryAt(value, start, end);'),'composer discovery follows the active collapsed caret instead of requiring character zero/end-of-text');
ok(src.includes("input.addEventListener('compositionstart'") && src.includes('slashImeComposing = true'),'IME composition suspends slash discovery');
ok(src.includes("input.addEventListener('compositionend'") && src.includes('slashImeComposing = false'),'IME completion safely re-evaluates discovery');
ok(src.includes("input.addEventListener('click', function () { _renderSlashCommandPalette(false); })"),'pointer caret changes re-evaluate palette ownership');
ok(src.includes("e.key === 'ArrowLeft' || e.key === 'ArrowRight'"),'horizontal caret movement re-evaluates slash ownership');

ok(css.includes('.ai-assistant-panel-slash-command-menu'),'palette has dedicated styling');
ok(css.includes('bottom: calc(100% + 0.45rem)'),'palette opens upward from bottom-owned composer');
ok(css.includes('overscroll-behavior: contain'),'long command lists scroll internally');
ok(css.includes('.ai-assistant-panel-slash-command-item[data-highlighted="true"]'),'keyboard highlight has visible state');
ok(css.includes('.ai-assistant-panel-slash-inline-hint'),'inline filtering hint is overlay-only');
ok(css.includes('@media (max-width: 480px)') && css.includes('.ai-assistant-panel-slash-command-item { min-height: 44px; }'),'mobile rows preserve touch target size');
ok(css.includes('.ai-assistant-panel-slash-command-copy small,\n    .ai-assistant-panel-slash-command-arg,\n    .ai-assistant-panel-slash-command-foot { display: none; }'),'compact mobile palette removes secondary descriptions, badges, and help before commands');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
