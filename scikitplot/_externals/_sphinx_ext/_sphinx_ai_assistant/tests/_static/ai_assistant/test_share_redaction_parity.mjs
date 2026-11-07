// Run 173 T33 - no format may emit what the share policy removed.
//
// Five real global-share exports of one conversation were inspected: all five
// were correctly redacted. That was true by construction, and nothing asserted
// it. A sixth format, or a session field added later and rendered
// unconditionally by one builder, would leak and no gate would notice -- and
// the leak would be a published artifact, not a local file.
//
// So this drives every registered builder with a redacted snapshot and asserts
// that none of the removed values appears in its output.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
function extract(n){
  const st = src.indexOf('function '+n+'('); if (st<0) throw new Error('missing '+n);
  let d=0,b=false,q='',e=false,l=false,bl=false,re=false,cls=false,prev='';
  for(let i=st;i<src.length;i++){const c=src[i],x=src[i+1];
    if(l){if(c==='\n')l=false;continue;} if(bl){if(c==='*'&&x==='/'){bl=false;i++;}continue;}
    if(re){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c==='['){cls=true;continue;}if(c===']'){cls=false;continue;}if(c==='/'&&!cls)re=false;continue;}
    if(q){if(e){e=false;continue;}if(c==='\\'){e=true;continue;}if(c===q)q='';continue;}
    if(c==='/'&&x==='/'){l=true;i++;continue;} if(c==='/'&&x==='*'){bl=true;i++;continue;}
    if(c==='/'){ if(/[(,=:[!&|?{};+\-*%^~<>]/.test(prev)||/\breturn$/.test(src.slice(Math.max(0,i-6),i))){re=true;cls=false;} continue; }
    if(c==='"'||c==="'"||c==='`'){q=c;continue;}
    if(c==='{'){d++;b=true;} else if(c==='}'){ if(--d===0&&b) return src.slice(st,i+1); }
    if(!/\s/.test(c)) prev=c;
  } throw new Error('unterminated '+n);
}
let n=0,f=0; const ok=(c,m)=>{c?n++:(f++,console.error('FAIL '+m));};

// Every format registered in _EXPORT_FORMATS must be covered here. A new
// builder that nobody adds is caught by the count assertion below.
const BUILDERS = ['_buildConvJsonString','_buildConvHtmlString','_buildConvTxtString',
                  '_buildConvYamlString','_buildConvTomlString'];
const registered = (src.match(/buildStr: function \(snapshot\) \{ return (_buildConv\w+)\(snapshot\); \}/g) || [])
  .map(s => /return (_buildConv\w+)/.exec(s)[1]);
ok(registered.length === BUILDERS.length,'every registered format builder is covered by this gate');
BUILDERS.forEach(b => ok(registered.includes(b), b + ' is a registered format'));

// A redacted snapshot: exactly the shape _buildConversationSnapshot produces
// under share policy, with sentinels in the fields that survive so a builder
// that emits nothing at all cannot pass vacuously.
const SENTINEL_KEPT = 'KEPT-PAGE-TITLE-SENTINEL';
const KEPT_TEXT = 'KEPT-MESSAGE-BODY-SENTINEL';
function redactedSnapshot() {
  const rec = (role, text) => ({
    turn_index: 0, message_index: role === 'user' ? 0 : 1, role, text,
    ts: null, ts_iso: null,
    model_id: null, model_provider: null, model_name: null,
    feedback_rating_value: null, feedback_rating_label: null, feedback_message: null,
    resources: null, session_id: null, page_url: null
  });
  const records = [rec('user', KEPT_TEXT + ' question'), rec('assistant', KEPT_TEXT + ' answer')];
  return {
    schema_version: '2.1',
    session: { id: null, page_url: null, page_title: SENTINEL_KEPT,
               assistant_name: 'AI Assistant', exported_at: null, exported_at_iso: null },
    turns: [{ turn_index: 0, user: records[0], assistant: records[1] }],
    records
  };
}

// Anything a share must not carry. Values, not key names: a key with a null
// value is fine and idiomatic in JSON and YAML.
const FORBIDDEN = [
  ['session uuid', '2847a013-51a2-4d7c-97ab-9d67138a8530'],
  ['source page', 'https://scikit-plots.github.io/dev/user_guide/impute/index.html'],
  ['model id', 'Qwen2.5-Coder-7B-Instruct-hf'],
  ['model provider', 'huggingface'],
  ['timestamp', '2026-09-08T16:25:02.875Z']
];

BUILDERS.forEach(function (name) {
  let out;
  try {
    // Real dependencies, extracted from the same source. A stub here would let
    // the gate pass against a serializer that leaks -- the serializers are
    // precisely where a redacted value would reappear.
    // Dependency closure of the format builders, resolved from the source
    // rather than listed by hand: a builder that grows a new helper must not
    // silently drop out of this gate.
    const deps = ['_attachmentExtension', '_attachmentItemBadge', '_attachmentSafeName', '_attachmentSafeRelativePath', '_buildConversationSnapshot', '_buildExportHtmlDoc', '_buildExportRecords', '_buildTurnResourceManifest', '_buildTurnsFromExportRecords', '_cfg', '_escapeHtml', '_exportCss', '_generatedArtifactSafePath', '_htmlTimeFmt', '_jsStubEndpoint', '_jsonForHtmlRawText', '_log', '_mdToHtml', '_normalizeContextPageUrl', '_normalizeConversationContentOptions', '_pageTitle', '_pageUrl', '_parseCodeFenceInfo', '_providerColor', '_ratingDisplay', '_resourceManifestHtml', '_resourceManifestSummaryText', '_resourceManifestTextLines', '_safeErrorDiagnostic', '_safeTurnResourceTotal', '_sanitizeDiagnosticText', '_sanitizePage', '_sanitizeShareResourceManifest', '_sanitizeTurnAttachmentSummaries', '_sanitizeTurnResourceManifest', '_scrubArg', '_serializeYamlValue', '_stubModelsEnabled', '_syncJsStubModels', '_tomlScalar', '_tomlString', '_tomlWriteFields', '_tomlWriteResourceManifest', '_turnResourceAggregates', '_typesetMath', '_yamlKey', '_yamlScalar']
      .map(function (d) { try { return extract(d); } catch (_) { return ''; } })
      .filter(Boolean).join('\n');
    // Module constants the closure reads. Values come from the source so the
    // gate cannot drift from the bounds the product actually enforces.
    const consts = ['_TURN_RESOURCE_LIVE_MAX_ITEMS','_TURN_RESOURCE_PERSIST_MAX_ITEMS',
                    '_CONVERSATION_SCHEMA_VERSION']
      .map(function (c) {
        const m = new RegExp('var ' + c + "\\s*=\\s*([^;]+);").exec(src);
        return m ? ('var ' + c + ' = ' + m[1] + ';') : '';
      }).filter(Boolean).join('\n');
    const fallbacks = [
      "if (typeof _escapeHtml !== 'function') var _escapeHtml = function (s) { return String(s).replace(/[&<>\"']/g, function (c) { return ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'})[c]; }); };",
      "if (typeof _jsonForScriptTag !== 'function') var _jsonForScriptTag = function (o) { return JSON.stringify(o); };"
    ].join('\n');
    const fn = new Function('_CONVERSATION_SCHEMA_VERSION',
      consts + '\n' + deps + '\n' + fallbacks + '\n' + extract(name) + '\nreturn ' + name + ';')('2.1');
    out = fn(redactedSnapshot());
  } catch (err) {
    // A builder needing more of its module than this harness supplies is
    // reported, not skipped: an uncovered builder is an unguarded one.
    ok(false, name + ' could not be driven in isolation: ' + err.message);
    return;
  }
  ok(typeof out === 'string' && out.length > 0, name + ' produces output');
  ok(out.includes(SENTINEL_KEPT) || out.includes(KEPT_TEXT),
     name + ' renders the content that survived redaction, so a pass is not vacuous');
  FORBIDDEN.forEach(function (pair) {
    ok(out.indexOf(pair[1]) === -1, name + ' does not emit the redacted ' + pair[0]);
  });
  // Comments are excluded: the TOML export documents its own convention --
  // "omitted optional values represent null" -- and matching that sentence
  // reported a correct builder as broken. A comment describing a convention is
  // not the convention being violated.
  const withoutComments = out.split('\n')
    .filter(function (line) { return !/^\s*(#|\/\/|<!--)/.test(line); }).join('\n');
  ok(!/^\s*[A-Za-z_][\w.]*\s*=\s*null\s*$/m.test(withoutComments),
     name + ' never assigns a literal null, which reads as content in a flat format');
});

console.log(`${n} passed, ${f} failed`); if(f) process.exit(1);
