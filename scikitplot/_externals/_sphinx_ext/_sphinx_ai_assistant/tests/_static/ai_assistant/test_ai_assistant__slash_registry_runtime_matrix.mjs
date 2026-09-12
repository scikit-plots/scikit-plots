// Run 104 regression — execute the real slash registry + matcher together.
// This covers the live failure where `/` opened the shell but rendered zero rows.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let passed=0,failed=0; function ok(c,n){if(c)passed++;else{failed++;console.error('FAIL '+n);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,start=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;start=true;}else if(src[j]==='}'){d--;if(start&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}
function extractVar(name){const i=src.indexOf('var '+name+' =');if(i<0)throw new Error('missing '+name);const e=src.indexOf('\n    ];',i);if(e<0)throw new Error('unbalanced '+name);return src.slice(i,e+'\n    ];'.length);}

const base=extractVar('_LOCAL_SLASH_COMMAND_BASE');
const baseFn=extract('_localSlashCommandBaseDefinitions');
const defsFn=extract('_localSlashCommandDefinitions');
const availFn=extract('_availableLocalSlashCommands');
const normFn=extract('_normalizeSlashCommandText');
const matchFn=extract('_slashCommandMatches');
const resultsFn=extract('_slashCommandResults');
const constants=`
var _ADD_CURRENT_PAGE_CONTEXT_COMMAND='/Add current page context';
var _PIN_CURRENT_PAGE_COMMAND='/Pin current page';
var _ADD_FILES_COMMAND='/Add files or photos';
var _SKILL_CREATOR_COMMAND='/skill-creator';`;

function makeRuntime(opts={}) {
  const panelSkillGenerator = opts.skill !== false;
  const stateThrows = !!opts.stateThrows;
  const cfgThrows = !!opts.cfgThrows;
  const currentEnabled = opts.currentEnabled !== false;
  const excluded = !!opts.excluded;
  const pinned = !!opts.pinned;
  const consumed = !!opts.consumed;
  const code=`${constants}\n${base}\n${baseFn}\n${defsFn}\n${availFn}\n${normFn}\n${matchFn}\n${resultsFn}
function _currentContextPageUrl(){ if (${stateThrows}) throw new Error('page state unavailable'); return 'https://example.test/docs/a.html'; }
function _currentPageContextEnabled(){ if (${stateThrows}) throw new Error('page state unavailable'); return ${currentEnabled}; }
function _isCurrentPageContextExcluded(){ if (${stateThrows}) throw new Error('page state unavailable'); return ${excluded}; }
function _findPinnedPageContext(){ if (${stateThrows}) throw new Error('page state unavailable'); return ${pinned} ? {sourceUrl:'https://example.test/docs/a.html'} : null; }
function _pageContextConsumed(){ if (${stateThrows}) throw new Error('page state unavailable'); return ${consumed}; }
function _cfg(){ if (${cfgThrows}) throw new Error('cfg unavailable'); return {panelSkillGenerator:${panelSkillGenerator}}; }
return {defs:_localSlashCommandDefinitions, available:_availableLocalSlashCommands, results:_slashCommandResults};`;
  return new Function(code)();
}

const expected=['/Add current page context','/Pin current page','/Add files or photos','/skill-creator'];
let rt=makeRuntime();
ok(JSON.stringify(rt.results('').map(x=>x.command))===JSON.stringify(expected),'bare slash resolver returns all four commands in canonical order');
ok(rt.results('add').length===2,'/add resolves both Add commands');
ok(rt.results('pin').map(x=>x.command).join('')==='/Pin current page','/pin resolves Pin current page');
ok(rt.results('skill').map(x=>x.command).join('')==='/skill-creator','/skill resolves skill creator');
ok(rt.results('files').map(x=>x.command).join('')==='/Add files or photos','/files resolves file picker');
ok(rt.results('definitely-no-match').length===0,'unknown filter returns true empty-result state');

let defs=rt.defs();
ok(defs[0].badge==='Staged','current-page active state decorates command as Staged');
ok(defs[1].badge==='Save','unpinned state decorates command as Save');
rt=makeRuntime({excluded:true,pinned:true}); defs=rt.defs();
ok(defs[0].badge==='Context','excluded current page stays available as Add context');
ok(defs[1].badge==='Staged','pinned-and-staged state decorates command without removing it');
rt=makeRuntime({pinned:true,consumed:true}); defs=rt.defs();
ok(defs[1].badge==='Saved' && defs[1].description.includes('stage it for the next question only'),'consumed pin remains discoverable as a saved source without implying active transport');
rt=makeRuntime({currentEnabled:false}); defs=rt.defs();
ok(defs[0].badge==='Context' && defs[0].description.includes('Enable the current-page default'),'global current-page OFF still lists Add current page context');

rt=makeRuntime({skill:false});
ok(JSON.stringify(rt.results('').map(x=>x.command))===JSON.stringify(expected.slice(0,3)),'Skill Generator OFF removes only /skill-creator, never utility commands');
rt=makeRuntime({stateThrows:true});
ok(JSON.stringify(rt.results('').map(x=>x.command))===JSON.stringify(expected),'page-context state failure cannot empty bare slash palette');
let fallbackDefs=rt.defs();
ok(fallbackDefs[0].badge==='Context' && fallbackDefs[1].badge==='Save','state failure uses truthful static badges');
rt=makeRuntime({cfgThrows:true});
ok(JSON.stringify(rt.results('').map(x=>x.command))===JSON.stringify(expected),'temporary config read failure cannot empty command catalog');

const availableSrc=availFn;
ok(availableSrc.includes('_localSlashCommandBaseDefinitions()'),'availability has static-catalog fallback');
ok(availableSrc.includes('!Array.isArray(definitions) || !definitions.length'),'availability repairs invalid/empty registry results');
ok(src.includes('var commands = _slashCommandResults(active.query);'),'palette renderer consumes the integration-tested resolver directly');
ok(src.includes("if (!commands.length && active.query === '') commands = availableCommands;"),'bare slash has an explicit non-empty catalog fallback');
ok(src.includes('var _LOCAL_SLASH_COMMAND_BASE = ['),'command catalog is static rather than dependent on page state');
ok(src.includes('catch (_stateError)'),'dynamic page-state decoration is failure-contained');
ok(src.includes('catch (_cfgError)'),'feature decoration is independently failure-contained');

// Exhaust the state flags that can decorate/disable entries.  None of these
// combinations may make the three utility commands disappear.
for (const currentEnabled of [false,true]) {
  for (const excluded of [false,true]) {
    for (const pinned of [false,true]) {
      for (const skill of [false,true]) {
        const combo=makeRuntime({currentEnabled,excluded,pinned,skill});
        const rows=combo.results('').map(x=>x.command);
        ok(rows.includes('/Add current page context'),`matrix current=${currentEnabled} excluded=${excluded} pinned=${pinned} skill=${skill}: Add current page remains`);
        ok(rows.includes('/Pin current page'),`matrix current=${currentEnabled} excluded=${excluded} pinned=${pinned} skill=${skill}: Pin remains`);
        ok(rows.includes('/Add files or photos'),`matrix current=${currentEnabled} excluded=${excluded} pinned=${pinned} skill=${skill}: Files remains`);
        ok(rows.includes('/skill-creator')===skill,`matrix current=${currentEnabled} excluded=${excluded} pinned=${pinned} skill=${skill}: skill flag is exact`);
      }
    }
  }
}

// Parser + resolver combinations matching real composer expectations.
const discoveryFn=extract('_slashCommandDiscoveryAt');
const discovery=new Function(normFn+'\n'+discoveryFn+'\nreturn _slashCommandDiscoveryAt;')();
for (const [text, caret, count, label] of [
  ['/',1,4,'bare slash'],
  ['hello /',7,4,'slash after prose'],
  [' /skill',7,1,'leading-space skill filter'],
  ['/Add',4,2,'Add prefix'],
  ['/Pin current',12,1,'multi-word pin prefix']
]) {
  const active=discovery(text,caret,caret);
  ok(!!active,label+' has active discovery');
  if (active) ok(makeRuntime().results(active.query).length===count,label+' resolves expected row count');
}
ok(discovery('/ ',2,2)===null,'literal / + space exits discovery');
ok(discovery('hello / ',8,8)===null,'mid-sentence literal / + space exits discovery');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
