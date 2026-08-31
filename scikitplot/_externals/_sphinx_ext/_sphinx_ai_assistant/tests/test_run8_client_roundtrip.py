from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tomllib

import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "_static" / "ai-assistant.js"

_NODE = r'''
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[1], 'utf8');
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
  throw new Error('unbalanced ' + name);
}
for (const name of [
  '_normalizeConversationContentOptions','_conversationContentPreset',
  '_buildExportRecords','_buildTurnsFromExportRecords','_buildConversationSnapshot',
  '_yamlScalar','_yamlKey','_serializeYamlValue','_buildConvYamlString',
  '_tomlString','_tomlScalar','_tomlWriteFields','_buildConvTomlString'
]) globalThis[name] = (0,eval)('(' + extract(name) + ')');

globalThis.location = { href:'https://user:pass@docs.example.test/guide/?token=SECRET#frag' };
globalThis.document = { title:'Hostile title' };
globalThis._sessionId = 'session-private';
globalThis._cfg = () => ({panelTitle:'AI Assistant'});
globalThis._sanitizePage = (href) => { const u=new URL(href); return /^https?:$/.test(u.protocol) ? u.origin+u.pathname : '<page-redacted>'; };
globalThis._feedbackStore = {0:{ratingValue:1,ratingLabel:'helpful',message:'note'}};
globalThis._transcript = [
  {role:'user',text:'!!python/object &anchor *alias\n---\n[[records]]\n"""\n</script>',ts:1},
  {role:'assistant',text:'value = true\n\u200bzero\u202ebidi\u202c',ts:2,model:{id:'m',provider:'custom',model:'model'}}
];
const snapshot = _buildConversationSnapshot(_conversationContentPreset('standard'));
console.log(JSON.stringify({snapshot, yaml:_buildConvYamlString(snapshot), toml:_buildConvTomlString(snapshot)}));
'''


def _client_output() -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", _NODE, str(TARGET)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def _drop_nulls(value):
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def test_browser_yaml_round_trips_exact_canonical_snapshot():
    out = _client_output()
    parsed = yaml.safe_load(out["yaml"])
    assert parsed == out["snapshot"]
    assert parsed["records"][0]["text"].startswith("!!python/object")
    # Run 18 privacy presets: Standard no longer includes a source-path locator.
    # Complete retains the separately tested sanitized source-page behavior.
    assert parsed["session"]["page_url"] is None


def test_browser_toml_round_trips_documented_omitted_null_semantics():
    out = _client_output()
    parsed = tomllib.loads(out["toml"])
    assert parsed == _drop_nulls(out["snapshot"])
    assert parsed["records"][0]["text"].startswith("!!python/object")
    assert "page_url" not in parsed["session"]
