// Run 103 regression — slash discovery anywhere + friendly local actions.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3],'utf8');
let passed=0,failed=0; function ok(c,n){if(c)passed++;else{failed++;console.error('FAIL '+n);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,start=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;start=true;}else if(src[j]==='}'){d--;if(start&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}

ok(src.includes("var _ADD_CURRENT_PAGE_CONTEXT_COMMAND = '/Add current page context';"),'friendly current-page command exists');
ok(src.includes("var _PIN_CURRENT_PAGE_COMMAND = '/Pin current page';"),'friendly pin command exists');
ok(src.includes("var _ADD_FILES_COMMAND = '/Add files or photos';"),'friendly file command exists');
ok(src.includes("var _SKILL_CREATOR_COMMAND = '/skill-creator';"),'skill-creator command remains canonical');

const normSrc=extract('_normalizeSlashCommandText');
const discoverySrc=extract('_slashCommandDiscoveryAt');
const discovery=new Function(normSrc+'\n'+discoverySrc+'\nreturn _slashCommandDiscoveryAt;')();
let d=discovery('/',1,1);
ok(d && d.start===0 && d.end===1 && d.query==='', 'bare slash at start opens discovery');
d=discovery('hello /',7,7);
ok(d && d.start===6 && d.query==='', 'slash after prose opens discovery');
d=discovery(' /skill',7,7);
ok(d && d.start===1 && d.query==='skill', 'leading whitespace before slash is accepted');
d=discovery('hello /Pin current',18,18);
ok(d && d.start===6 && d.query==='pin current', 'multi-word command filtering works after prose');
d=discovery('/Add current page context',25,25);
ok(d && d.query==='add current page context', 'complete multi-word command remains one slash phrase');
ok(discovery('/ ',2,2)===null, 'immediate space after slash makes slash literal');
ok(discovery('hello / ',8,8)===null, 'immediate space after mid-sentence slash also makes it literal');
ok(discovery('hello/skill',11,11)===null, 'slash embedded in word does not steal ordinary text');
ok(discovery('https://example.test',20,20)===null, 'URL slashes do not open local command discovery');
d=discovery('one /old two /pin',17,17);
ok(d && d.start===13 && d.query==='pin', 'nearest eligible slash owns discovery');
d=discovery('before /skill after',13,13);
ok(d && d.start===7 && d.query==='skill', 'caret can own slash phrase before later text');
ok(discovery('before /skill after',7,10)===null, 'non-collapsed selection suspends discovery');
d=discovery('first line\n /pin',16,16);
ok(d && d.start===12 && d.query==='pin', 'discovery works independently on later lines');

const matchSrc=extract('_slashCommandMatches');
const match=new Function(normSrc+'\n'+matchSrc+'\nreturn _slashCommandMatches;')();
const add={command:'/Add current page context',title:'Add current page context',keywords:['page','context','current','documentation','add']};
const files={command:'/Add files or photos',title:'Add files or photos',keywords:['file','files','photo','upload','attach']};
ok(match(add,'add cur'),'multi-word prefix filters current-page action');
ok(match(add,'current page'),'phrase contained after leading verb can still filter');
ok(match(files,'files'),'non-leading command word can filter');
ok(match(files,'photo'),'keyword prefix filters file action');
ok(!match(files,'pin'),'unrelated command is excluded');

const defs=extract('_localSlashCommandDefinitions');
ok(src.indexOf("id: 'add-current-page-context'") < src.indexOf("id: 'pin-current-page'"),'current-page command precedes pin in static registry');
ok(src.indexOf("id: 'pin-current-page'") < src.indexOf("id: 'add-files'"),'pin precedes files in static registry');
ok(src.indexOf("id: 'add-files'") < src.indexOf("id: 'skill-creator'"),'files precede skill creator in static registry');
ok(defs.includes("current.badge = currentStaged ? 'Staged' : 'Context'"),'current context exposes one-turn staged state badge');
ok(defs.includes("pin.badge = currentPinned ? (currentConsumed ? 'Saved' : 'Staged') : 'Save'"),'pin command distinguishes saved source from one-turn staged state');
ok(src.includes("badge: 'Local'"),'file picker exposes local-only badge in static command catalog');

const replaceSrc=extract('_replaceActiveSlashCommand');
ok(replaceSrc.includes('value.slice(0, active.start) + inserted + value.slice(active.end)'),'Tab completion replaces only active slash range');
ok(replaceSrc.includes("item.argumentHint ? ' ' : ''"),'argument-taking commands keep an argument caret position');
const removeSrc=extract('_removeActiveSlashPhrase');
ok(removeSrc.includes('value.slice(0, active.start) + value.slice(active.end)'),'executing a utility removes only its slash phrase');

const execSrc=extract('_executeSlashCommand');
ok(execSrc.includes("commandId === 'add-current-page-context'") && execSrc.includes('_addCurrentPageContextFromCommand()'),'current-page command routes to local context authority');
ok(execSrc.includes("commandId === 'pin-current-page'") && execSrc.includes('_pinCurrentPageFromCommand()'),'pin command routes to local pin authority');
ok(execSrc.includes("commandId === 'add-files'") && execSrc.includes('_openAttachmentPicker()'),'file command routes directly to native picker');
ok(execSrc.includes("commandId === 'skill-creator'") && execSrc.includes('_requestSkillGeneratorOpen'),'skill command routes to Skill Studio');
ok(execSrc.includes("var goal = String(remaining.text || '').trim()"),'surrounding composer draft becomes Skill Creator goal');
ok(execSrc.includes('selectComposerFiles: true'),'Skill Creator keeps staged composer files selected');
ok(!execSrc.includes('fetch('),'local slash execution introduces no network authority');

const addCtxSrc=extract('_addCurrentPageContextFromCommand');
ok(addCtxSrc.includes("'Current page is already staged for the next question.'"),'Add current page is idempotent when already staged');
ok(addCtxSrc.includes('_setCurrentPageContextInTab(true)'),'Add current page can enable the existing tab preference when globally off');
ok(addCtxSrc.includes('_setCurrentPageContextExcluded(sourceUrl, false, false)'),'Add current page reverses a per-page exclusion');
const pinSrc=extract('_pinCurrentPageFromCommand');
ok(pinSrc.includes("'Current page is already pinned and staged for the next question.'"),'Pin current page is idempotent when already pinned and staged');
ok(pinSrc.includes('_pinCurrentPageContext()'),'Pin uses canonical async pin lifecycle');
ok(pinSrc.includes("err.message === 'PAGE_CONTEXT_PIN_STALE'"),'Pin respects stale async cancellation authority');

const renderSrc=extract('_renderSlashCommandPalette');
const rowBuilderSrc=extract('_buildSlashCommandPaletteRows');
ok(renderSrc.includes('slashPaletteDiscovery = active'),'palette retains active range for pointer selection');
ok(renderSrc.includes("slashInlineHint.hidden = !(active.start === 0 && active.raw === '')"),'inline hint stays truthful while mid-text discovery uses palette header');
ok(renderSrc.includes('_slashCommandCommittedItem(active.raw, availableCommands)'),'completed command + trailing text stays committed out of discovery');
ok(rowBuilderSrc.includes('(item.argumentHint || item.badge)'),'rows show argument or state/local badge through detached builder');
ok(rowBuilderSrc.includes("row.addEventListener('click', function () { _executeSlashCommand(item); })"),'click runs highlighted local action through canonical detached row builder');

const keySrc=extract('_handleSlashCommandKeydown');
ok(keySrc.includes("if (e.key === 'Tab') _replaceActiveSlashCommand(item)"),'Tab is explicit autocomplete gesture');
ok(keySrc.includes('else _executeSlashCommand(item)'),'Enter executes highlighted command');
ok(keySrc.includes("e.key === 'Escape'") && keySrc.includes('_closeSlashCommandPalette(true)'),'Escape remains literal-text escape hatch');

ok(src.includes("slashPaletteFoot.innerHTML = '<span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>"),'palette retains keyboard help');
ok(src.includes("input.addEventListener('compositionstart'") && src.includes('slashImeComposing = true'),'IME composition still suspends discovery');
ok(src.includes("document.addEventListener('pointerdown'") && src.includes("slashPalette.getAttribute('data-open') !== 'true'"),'outside pointer still dismisses palette');
ok(css.includes('.ai-assistant-panel-slash-command-arg'),'state/argument badge has bounded styling');
ok(css.includes('overscroll-behavior: contain'),'palette remains short-height safe');
ok(css.includes('.ai-assistant-panel-slash-command-item { min-height: 44px; }'),'mobile command rows preserve touch target');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
