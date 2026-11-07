import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
function extract(name) {
  const i = src.indexOf('function ' + name + '('); if (i < 0) throw new Error('missing ' + name);
  let depth=0, started=false, q=null, line=false, block=false;
  for (let j=i;j<src.length;j++) {
    const c=src[j], n=src[j+1];
    if (line) { if (c==='\n') line=false; continue; }
    if (block) { if (c==='*'&&n==='/') { block=false; j++; } continue; }
    if (q) { if (c==='\\') { j++; continue; } if (c===q) q=null; continue; }
    if (c==='/'&&n==='/') { line=true; j++; continue; }
    if (c==='/'&&n==='*') { block=true; j++; continue; }
    if (c==='"'||c==="'"||c==='`') { q=c; continue; }
    if (c==='{') { depth++; started=true; }
    else if (c==='}') { depth--; if (started&&depth===0) return src.slice(i,j+1); }
  }
  throw new Error('unbalanced '+name);
}
let pass=0, fail=0;
function t(name, got, want=true) { if (got===want) pass++; else { fail++; console.log(`FAIL ${name}\n got ${JSON.stringify(got)}\n want ${JSON.stringify(want)}`); } }

globalThis._TURN_RESOURCE_LIVE_MAX_ITEMS = 512;
globalThis._CONVERSATION_SCHEMA_VERSION = '2.1';
globalThis._SHARE_ACCEPTED_CONVERSATION_SCHEMA_VERSIONS = {'2.0':true,'2.1':true};
globalThis._sanitizeTurnResourceManifest = (value) => value && value.items ? JSON.parse(JSON.stringify(value)) : ({ totalCount: 0, itemCount:0, omittedCount:0, complete:true, items: [] });
for (const name of ['_normalizeConversationContentOptions','_conversationContentPreset','_safeTurnResourceTotal','_sanitizeShareResourceManifest','_buildExportRecords','_buildTurnsFromExportRecords','_buildConversationSnapshot','_yamlScalar','_yamlKey','_serializeYamlValue','_buildConvYamlString','_tomlString','_tomlScalar','_tomlWriteFields','_tomlWriteResourceManifest','_buildConvTomlString']) {
  globalThis[name] = (0,eval)('(' + extract(name) + ')');
}
globalThis.location = { href:'https://user:pass@docs.example.test/guide/?token=SECRET#frag' };
globalThis.document = { title:'Hostile title' };
globalThis._sessionId='session-private';
globalThis._cfg=()=>({panelTitle:'AI Assistant'});
globalThis._sanitizePage=(href)=>{ const u=new URL(href); return /^https?:$/.test(u.protocol) ? u.origin+u.pathname : '<page-redacted>'; };
globalThis._feedbackStore={0:{ratingValue:1,ratingLabel:'helpful',message:'note'}};
globalThis._transcript=[
 {role:'user',text:'key: !!python/object &anchor *alias\n---\n[[records]]\n"""\n</script>',ts:1},
 {role:'assistant',text:'value = true\n\u200bzero\u202ebidi\u202c',ts:2,model:{id:'m',provider:'custom',model:'model'}}
];
const full=_buildConversationSnapshot();
const standard=_buildConversationSnapshot(_conversationContentPreset('standard'));
const minimal=_buildConversationSnapshot(_conversationContentPreset('minimal'));
const complete=_buildConversationSnapshot(_conversationContentPreset('complete'));
t('download/full keeps session id', full.session.id, 'session-private');
t('share standard omits session id', standard.session.id, null);
t('share records omit session id', standard.records[0].session_id, null);
t('standard removes source URL by default', standard.session.page_url, null);
t('complete source URL sanitized before formats', complete.session.page_url, 'https://docs.example.test/guide/');
t('minimal removes page source', minimal.session.page_url, null);
t('minimal removes page title', minimal.session.page_title, null);
t('minimal removes timestamps', minimal.records[0].ts, null);
t('minimal removes model fields', minimal.records[1].model_name, null);
t('minimal removes ratings', minimal.records[1].feedback_rating_label, null);
const shareResources={version:2,totalCount:1,includedCount:1,localOnlyCount:0,contextCount:1,rawCount:0,notSentCount:0,pageCount:1,replayCount:0,totalBytes:10,itemCount:1,omittedCount:0,complete:true,items:[{name:'Local docs',badge:'PAGE',kind:'page',size:10,lineCount:1,included:true,localOnly:false,delivery:'context',modality:'text',intent:'context',replay:false,boundedExcerpt:false,status:'Current page',type:'',relativePath:'',sourceKind:'',archiveName:'',contextRole:'current',sourceUrl:'file:///Users/private/docs/index.html?secret=x#frag'}]};
const redactedResources=_sanitizeShareResourceManifest(shareResources,true);
t('Share resource sanitizer never exports file URLs', redactedResources.items[0].sourceUrl, '');
shareResources.items[0].sourceUrl='https://user:pass@docs.example.test/guide/?token=x#frag';
const safeResources=_sanitizeShareResourceManifest(shareResources,true);
t('Share resource sanitizer strips credentials/query/fragment', safeResources.items[0].sourceUrl, 'https://docs.example.test/guide/');
t('Share resource source URL option can remove even safe URL', _sanitizeShareResourceManifest(shareResources,false).items[0].sourceUrl, '');

const yaml=_buildConvYamlString(standard);
const toml=_buildConvTomlString(standard);
t('YAML quotes hostile tag text', yaml.includes('"text": "key: !!python/object &anchor *alias\\n---\\n[[records]]\\n\\\"\\\"\\\"\\n</script>"'), true);
t('YAML never emits a tag token at line start', /(^|\n)\s*!!python/.test(yaml), false);
t('YAML keeps anchor-looking text inside quoted scalar', yaml.includes('&anchor'), true);
t('YAML keeps document marker inside quoted scalar', /\n---\n/.test(yaml), false);
t('TOML hostile text is a quoted value', toml.includes('text = "key: !!python/object &anchor *alias\\n---\\n[[records]]\\n\\\"\\\"\\\"\\n</script>"'), true);
t('TOML does not create hostile records table from text', (toml.match(/^\[\[records\]\]$/gm)||[]).length, standard.records.length);
t('TOML null values are omitted', /feedback_message\s*=\s*null/.test(toml), false);
t('TOML documents null omission semantics', toml.includes('omitted optional values represent null'), true);
t('YAML preserves invisible/bidi as data', yaml.includes('zero'), true);
t('TOML preserves invisible/bidi as data', toml.includes('zero'), true);

const order=(src.match(/fmt:\s*'(json|html|txt|yaml|toml)'/g)||[]).slice(0,5).map(x=>x.match(/'([^']+)'/)[1]);
t('five live formats ordered JSON HTML Text YAML TOML', order.join(','), 'json,html,txt,yaml,toml');
t('YAML MIME registered', src.includes("mime: 'application/yaml'"));
t('TOML MIME registered', src.includes("mime: 'application/toml'"));
console.log(`\n${pass} passed, ${fail} failed`); process.exit(fail?1:0);
