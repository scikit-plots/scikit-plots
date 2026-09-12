// Run 101 regression — /skill-creator local command + inline Skills attach submenu.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3],'utf8');
let passed=0,failed=0; function ok(c,n){if(c)passed++;else{failed++;console.error('FAIL '+n);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,start=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;start=true;}else if(src[j]==='}'){d--;if(start&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}

ok(src.includes("var _SKILL_CREATOR_COMMAND = '/skill-creator'"),'slash command has one canonical token');
ok(src.includes('function _parseSkillCreatorCommand(value)'),'slash command has side-effect-free parser');
ok(src.includes("type: 'ai-assistant-open-skill-generator'") && src.includes('function _requestSkillGeneratorOpen(detail)'),'UI and command paths share private open event');
ok(src.includes("source: 'slash-command'") && src.includes("mode: 'guide'") && src.includes('selectComposerFiles: true'),'slash command launches Guided studio and preserves composer files as explicit sources');
ok(src.includes('sheet._launch = function (detail)'),'Skill Studio exposes one launch adapter');
ok(src.includes("sheet._skillPreferredFocus = (detail.source === 'slash-command') ? objectiveF.input : null"),'slash launch focuses intent field');
ok(src.includes("typeof target._takePreferredFocus === 'function'"),'generic sheet opener honors preferred focus without special-case DOM poking');

const submit=extract('handleAIPanelSubmit');
const commandIdx=submit.indexOf('var skillCreatorCommand = _parseSkillCreatorCommand(rawText)');
ok(commandIdx>=0,'submit parses skill command');
ok(commandIdx < submit.indexOf('var attachmentPlan = await _prepareComposerEffectiveAttachmentPlan(attachmentSnapshot)'),'command interception precedes outbound attachment composition');
ok(commandIdx < submit.indexOf('_privacyPrepareDocumentationContext'),'command interception precedes privacy/network preflight');
ok(commandIdx < submit.indexOf('_appendPanelMessage'),'command interception precedes transcript mutation');
ok(commandIdx < submit.indexOf('_fetchAbortController'),'command interception does not cancel an active chat request');
ok(submit.includes("input.value = '';\n            _updateSendBtnState();\n            _requestSkillGeneratorOpen"),'command clears only composer text before local navigation');
ok(submit.includes("Skill Generator is disabled by this documentation site."),'disabled feature is handled locally instead of sending slash text to model');

ok(src.includes("text such as") || src.includes('explain /skill-creator'),'parser documents ordinary-chat non-capture contract');
ok(src.includes("ai-assistant-panel-attach-menu-item--skills-toggle"),'attach menu has Skills disclosure row');
ok(src.includes("skillsItem.setAttribute('aria-haspopup', 'menu')") && src.includes("skillsItem.setAttribute('aria-expanded', 'false')"),'Skills row exposes accessible submenu semantics');
ok(src.includes("skillsSubmenu.id = 'ai-assistant-panel-attach-skills-menu'") && src.includes("skillsSubmenu.setAttribute('role', 'menu')"),'nested Skills menu has stable accessible identity');
ok(src.includes("<strong>Open Skill Generator</strong>") && src.includes("<strong>/skill-creator</strong>"),'Skills submenu provides direct open and slash-command discovery');
const skillPos=src.indexOf("skillsItem.className = 'ai-assistant-panel-attach-menu-item ai-assistant-panel-attach-menu-item--skills-toggle'");
const integrationPos=src.indexOf("integrationLabel.textContent = 'Integration'");
ok(skillPos>=0 && integrationPos>skillPos,'Skills submenu is inserted before Integration');
ok(src.includes("_requestSkillGeneratorOpen({ source: 'attach-menu' })"),'attach submenu opens canonical Skill Studio path');
ok(src.includes("input.value = _SKILL_CREATOR_COMMAND + (existing ? (' ' + existing) : ' ')"),'slash helper converts an existing composer draft into an optional skill goal');
ok(src.includes('function _attachMenuVisibleItems()'),'keyboard traversal filters hidden nested-menu rows');
ok(src.includes("e.key === 'ArrowRight'") && src.includes("e.key === 'ArrowLeft'"),'nested menu supports directional keyboard navigation');
ok(src.includes("_setAttachSkillsOpen(false, false);\n            attachMenu.hidden = true"),'closing parent attach menu atomically collapses Skills submenu');

ok(css.includes('.ai-assistant-panel-attach-skills-menu'),'inline Skills submenu is styled');
ok(css.includes('transform: rotate(-90deg)') && css.includes('[aria-expanded="true"] .ai-assistant-panel-attach-menu-chevron'),'chevron communicates collapsed/expanded state');
ok(css.includes('@media (prefers-reduced-motion: reduce)') && css.includes('.ai-assistant-panel-attach-menu-chevron { transition: none; }'),'submenu affordance respects reduced motion');
ok(css.includes('.ai-assistant-panel-attach-menu-item') && css.includes('min-height: 44px'),'existing coarse-pointer touch target contract covers nested Skills rows');

const parserSrc=extract('_parseSkillCreatorCommand');
const parse=new Function('_SKILL_CREATOR_COMMAND','_SKILL_CREATOR_COMMAND_GOAL_MAX_CHARS',parserSrc+'\nreturn _parseSkillCreatorCommand;')('/skill-creator',4000);
let r=parse('/skill-creator'); ok(r&&r.goal===''&&r.command==='/skill-creator','exact slash command parses');
r=parse('  /skill-creator   build a robust mlops evaluator  '); ok(r&&r.goal==='build a robust mlops evaluator','optional goal parses and trims');
ok(parse('/SKILL-CREATOR api migration')?.goal==='api migration','command is case-tolerant while canonical output remains lowercase');
ok(parse('/skill-creator-help')===null,'lookalike command is not intercepted');
ok(parse('/skill-creator/advanced')===null,'slash path suffix is not intercepted');
ok(parse('please explain /skill-creator')===null,'ordinary message mentioning command is not intercepted');
const longGoal='x'.repeat(4500); ok(parse('/skill-creator '+longGoal).goal.length===4000,'command goal is bounded');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
