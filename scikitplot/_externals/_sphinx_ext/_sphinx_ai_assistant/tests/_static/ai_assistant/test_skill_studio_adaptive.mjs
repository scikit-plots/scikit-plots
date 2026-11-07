// Run 100 regression — adaptive Skill Studio, advanced bundle anatomy, roles and idempotency.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3],'utf8');
let passed=0,failed=0; function ok(c,n){if(c)passed++;else{failed++;console.error('FAIL '+n);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,start=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;start=true;}else if(src[j]==='}'){d--;if(start&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}

ok(src.includes("advanced.textContent = 'Advanced'") && src.includes("data-skill-mode"), 'studio has progressive Quick/Guide/Advanced depth');
ok(src.includes("'api-library'") && src.includes("'data-science'") && src.includes("'model-evaluation'") && src.includes("'rag-llm'"), 'workflow lenses cover API, data science, evaluation and RAG/LLM work');
ok(src.includes("label: 'MLOps / model operations'") || src.includes("label: 'MLOps / model operations'"), 'MLOps workflow lens exists');
ok(src.includes('Version / change awareness') && src.includes('Migration / change tracking'), 'advanced mode exposes version and migration policy');
ok(src.includes('classes/functions') && src.includes('renamed imports'), 'migration guidance explicitly handles symbols/import changes');
ok(src.includes("'BSD-3-Clause', 'MIT', 'Apache-2.0'") && src.includes("licenseF.input.value = 'BSD-3-Clause'"), 'common SPDX selector defaults to BSD-3-Clause');
ok(src.includes("['script', 'Script']") && src.includes("['asset', 'Asset']") && src.includes("['agent', 'Agent helper']") && src.includes("['eval-viewer', 'Eval viewer resource']"), 'source roles support extended bundle anatomy');
ok(src.includes('_SKILL_GENERATOR_ASSET_MAX_BYTES') && src.includes('item.file.arrayBuffer'), 'explicit binary assets are locally packageable with bounds');
ok(extract('_buildZipBlob').includes('f.bytes instanceof Uint8Array'), 'offline ZIP writer remains backward compatible and supports binary assets');
ok(src.includes('Trigger eval lab') && src.includes('Should not trigger · near misses'), 'description/trigger optimization has positive and near-miss eval inputs');
ok(src.includes('Bundle agents/grader.md') && src.includes('eval-viewer/README.md'), 'optional evaluation scaffold is explicit and isolated');
ok(src.includes('skills-ref validate ./') && src.includes('Bundle VALIDATION.md'), 'validation workflow is visible and exportable');
ok(src.includes('## Contents') && extract('_skillReferenceToc').includes('<= 300'), 'large references gain a bounded contents map');
ok(src.includes("refs.length > 5") && src.includes("references/INDEX.md"), 'many references get an adaptive routing index');
ok(src.includes('Draft trigger-rich description') && src.includes('_skillSuggestedDescription'), 'description helper improves trigger specificity without hidden model calls');
ok(!extract('_buildSkillGeneratorSheet').includes('fetch('), 'advanced studio still creates no parallel fetch authority');
ok(css.includes('.ai-assistant-panel-skill-modes--three') && css.includes('.ai-assistant-panel-skill-source-role'), 'three-depth and source-role UI is styled');
ok(css.includes('@media (pointer:coarse)') && css.includes('min-height:2.5rem'), 'touch targets are hardened for mobile/tablet');
ok(css.includes('@media (max-width: 520px)') && css.includes('grid-column:2 / -1'), 'advanced role selector stacks safely on narrow panels');

const hs=src.indexOf('var _SKILL_GENERATOR_ASSET_MAX_BYTES'); const he=src.indexOf('function _buildSkillGeneratorSheet()',hs);
const rt=new Function(src.slice(hs,he)+`\nreturn {_skillSlugify,_skillSafeResourceName,_skillInferProfile,_skillBuildBundle,_skillValidateBundle,_skillBundleTree,_skillSuggestedDescription};`)();

const big=['# API Reference']; for(let i=0;i<340;i++){ if(i%40===0) big.push('## Section '+i); big.push('line '+i); }
const state={
 name:'api-migration-review', title:'API Migration Review', description:'Use this skill to review library APIs and migration changes. Use when users need imports, classes, functions, signatures, deprecations, or version upgrade guidance.',
 profile:'api-library', license:'BSD-3-Clause', version:'2.0.0', changeMode:'migration', targetVersion:'v2', controlLevel:'strict',
 triggerPositive:'Update these imports for v2\nWhich class replaced LegacyClient?', triggerNegative:'Explain what an API is',
 evalPrompts:'Migrate this v1 example to v2', successCriteria:'Uses verified v2 imports and identifies changed defaults.', includeEvalScaffold:true, includeEvalViewerNotes:true, includeValidationGuide:true
};
const sources=[
 {title:'API Reference',sourceUrl:'https://docs.test/api',text:big.join('\n'),_skillRole:'api'},
 {title:'Migration Guide',sourceUrl:'https://docs.test/migrate',text:'# Migration\nOldClient -> NewClient',_skillRole:'changelog'},
 {name:'validate.py',text:'print("ok")\n',_skillRole:'script'},
 {name:'diagram.png',_skillBytes:new Uint8Array([0,1,2,255]),_skillRole:'asset'}
];
const b1=rt._skillBuildBundle(state,sources); const b2=rt._skillBuildBundle(state,sources);
ok(b1.profile==='api-library','explicit API lens is preserved');
ok(b1.skillMd.includes('## Version and change policy') && b1.skillMd.includes('module and import paths'), 'migration policy appears in SKILL.md');
ok(b1.skillMd.includes('## Script execution rules'), 'script presence adds agentic script execution rules');
ok(b1.files.some(f=>f.name.endsWith('/scripts/validate.py')), 'script resource routes to scripts/');
ok(b1.files.some(f=>f.name.endsWith('/assets/diagram.png') && f.bytes instanceof Uint8Array), 'binary asset routes to assets/ without text coercion');
ok(b1.files.some(f=>f.name.endsWith('/evals/trigger-evals.json')), 'trigger eval JSON is generated');
ok(b1.files.some(f=>f.name.endsWith('/evals/evals.json')), 'output eval JSON is generated');
ok(b1.files.some(f=>f.name.endsWith('/agents/grader.md')), 'grader scaffold is generated only when requested');
ok(b1.files.some(f=>f.name.endsWith('/eval-viewer/README.md')), 'eval viewer extension note is generated only when requested');
ok(b1.files.some(f=>f.name.endsWith('/VALIDATION.md')), 'validation guide is generated only when requested');
const apiRef=b1.files.find(f=>/references\/api-reference\.md$/.test(f.name));
ok(apiRef && apiRef.content.includes('## Contents'), 'large API reference contains generated contents map');
ok(JSON.stringify(b1.files.map(f=>({name:f.name,content:f.content||null,bytes:f.bytes?Array.from(f.bytes):null})))===JSON.stringify(b2.files.map(f=>({name:f.name,content:f.content||null,bytes:f.bytes?Array.from(f.bytes):null}))), 'same state+sources generates byte-for-byte deterministic file entries');
const val=rt._skillValidateBundle(state,b1);
ok(val.errors.length===0,'valid advanced bundle passes local structural validation');
ok(val.metrics.files===b1.files.length && val.metrics.references===2,'validation metrics count files and references');
ok(rt._skillBundleTree(b1).includes('scripts/') && rt._skillBundleTree(b1).includes('assets/') && rt._skillBundleTree(b1).includes('evals/'),'bundle tree reflects multi-directory anatomy');

const single=rt._skillBuildBundle({name:'introduction',title:'Introduction',description:'Use this skill for introduction tasks. Use when users need the project introduction.',license:'BSD-3-Clause'},[{title:'Introduction',text:'# Intro\nHello',_skillRole:'primary'}]);
ok(single.files.length===2,'simple one-reference skill stays minimal');
ok(!single.files.some(f=>/references\/INDEX\.md$/.test(f.name)),'single reference does not create unnecessary index');
ok(single.skillMd.includes('references/introduction.md'),'single reference remains directly discoverable');

const many=[]; for(let i=0;i<7;i++) many.push({title:'Reference '+i,text:'# R'+i,_skillRole:i===0?'primary':'reference'});
const manyBundle=rt._skillBuildBundle({name:'many-refs',title:'Many refs',description:'Use this skill for many-reference tasks. Use when users need these references.'},many);
ok(manyBundle.files.some(f=>/references\/INDEX\.md$/.test(f.name)),'6+ references generate routing index');
ok(manyBundle.skillMd.includes('when you need to choose among the bundled documentation references'),'SKILL.md explains when to load the index');

ok(rt._skillSafeResourceName('../../My Fancy SCRIPT.PY',0)==='my-fancy-script.py','resource names strip traversal and normalize extension');
const autoProfile=rt._skillInferProfile({name:'model-registry-deploy',description:'Deploy model registry pipeline and monitor rollback'},[]);
ok(autoProfile==='mlops','auto lens detects MLOps intent');
const desc=rt._skillSuggestedDescription({name:'rag-review',title:'RAG Review',profile:'rag-llm',triggers:'retrieval quality\nprompt grounding',nonGoals:'generic copywriting'},[]);
ok(desc.startsWith('Use this skill') && desc.includes('retrieval quality') && desc.length<=1024,'description helper is trigger-oriented and bounded');

const _crc32=eval('('+extract('_crc32')+')');
const _buildZipBlob=eval('('+extract('_buildZipBlob')+')');
const zip1=_buildZipBlob([{name:'skill/SKILL.md',content:'hello'},{name:'skill/assets/raw.bin',bytes:new Uint8Array([0,255,1,254])}]);
const zip2=_buildZipBlob([{name:'skill/SKILL.md',content:'hello'},{name:'skill/assets/raw.bin',bytes:new Uint8Array([0,255,1,254])}]);
const z1=new Uint8Array(await zip1.arrayBuffer()), z2=new Uint8Array(await zip2.arrayBuffer());
ok(z1.length===z2.length && z1.every((v,i)=>v===z2[i]),'ZIP output is deterministic for identical text+binary inputs');
let hasBinary=false; for(let i=0;i<z1.length-3;i++){if(z1[i]===0&&z1[i+1]===255&&z1[i+2]===1&&z1[i+3]===254){hasBinary=true;break;}}
ok(hasBinary,'ZIP STORE writer preserves raw binary asset bytes');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
