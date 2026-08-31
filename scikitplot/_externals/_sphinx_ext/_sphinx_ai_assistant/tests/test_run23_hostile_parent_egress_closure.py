from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "_static"
INIT = ROOT / "__init__.py"
MAIN = STATIC / "ai-assistant.js"
HOST = STATIC / "ai-assistant-isolation-host.js"
FRAME = STATIC / "ai-assistant-isolated-frame.js"
POLICY = STATIC / "ai-assistant-isolation-policy.json"
GUIDE = ROOT / "ISOLATION_DEPLOYMENT.md"


def _module():
    return importlib.import_module("scikitplot._externals._sphinx_ext._sphinx_ai_assistant")


def test_b42_protocol_nonce_is_frame_generated_and_not_in_iframe_url():
    host = HOST.read_text()
    frame = FRAME.read_text()
    assert "var PROTOCOL = '2.0.0'" in host
    assert "var PROTOCOL = '2.0.0'" in frame
    assert "channel: _channelId" not in host[host.index("var fragment"):host.index("frameUrl.hash")]
    assert "channelId = randomHex(16)" in frame
    assert "Math.random" not in host
    assert "Math.random" not in frame
    assert "CRYPTO_UNAVAILABLE" in frame
    assert "event.stopImmediatePropagation" in host
    assert "_nativeAddMessageListener('message', onMessage, true)" in host
    assert "msg.seq !== _rxSeq + 1" in host
    assert "msg.seq !== rxSeq + 1" in frame


def test_b42_default_source_policy_is_deny_all_and_closed_schema():
    policy = json.loads(POLICY.read_text())
    assert policy == {
        "allowedParentOrigins": [],
        "isolationOrigin": "",
        "protocolVersion": "2.0.0",
        "schemaVersion": 1,
    }
    frame = FRAME.read_text()
    assert "POLICY_PARENT_DENIED" in frame
    assert "keys.length !== expected.length" in frame
    assert "credentials:'omit', redirect:'error', cache:'no-store'" in frame


def test_b42_parent_origin_validator_and_generated_policy(tmp_path: Path):
    m = _module()
    class _Log:
        def warning(self, *args, **kwargs):
            return None
    m._logger = _Log()
    assert m._validate_isolation_parent_origin("https://docs.example.com") == "https://docs.example.com"
    assert m._validate_isolation_parent_origin("http://localhost:8000") == "http://localhost:8000"
    assert m._validate_isolation_parent_origin("http://docs.example.com") == ""
    assert m._validate_isolation_parent_origin("https://docs.example.com/path") == ""
    cfg = SimpleNamespace(
        ai_assistant_isolation_origin="https://assistant.example.com",
        ai_assistant_isolation_parent_origins=["https://docs.example.com", "https://docs.example.com"],
        html_baseurl="",
        ai_assistant_base_url="",
    )
    app = SimpleNamespace(config=cfg, outdir=str(tmp_path))
    m.generate_isolation_policy(app, None)
    policy = json.loads((tmp_path / "_static" / "ai-assistant-isolation-policy.json").read_text())
    assert policy == {
        "schemaVersion": 1,
        "protocolVersion": "2.0.0",
        "isolationOrigin": "https://assistant.example.com",
        "allowedParentOrigins": ["https://docs.example.com"],
    }


def test_b42_parent_policy_derives_safe_html_baseurl_and_denies_missing_origin(tmp_path: Path):
    m = _module()
    cfg = SimpleNamespace(
        ai_assistant_isolation_origin="https://assistant.example.com",
        ai_assistant_isolation_parent_origins=[],
        html_baseurl="https://docs.example.com/project/v1/",
        ai_assistant_base_url="",
    )
    app = SimpleNamespace(config=cfg, outdir=str(tmp_path / "a"))
    m.generate_isolation_policy(app, None)
    policy = json.loads((tmp_path / "a" / "_static" / "ai-assistant-isolation-policy.json").read_text())
    assert policy["allowedParentOrigins"] == ["https://docs.example.com"]

    cfg2 = SimpleNamespace(
        ai_assistant_isolation_origin="https://assistant.example.com",
        ai_assistant_isolation_parent_origins=[],
        html_baseurl="",
        ai_assistant_base_url="",
    )
    app2 = SimpleNamespace(config=cfg2, outdir=str(tmp_path / "b"))
    m.generate_isolation_policy(app2, None)
    policy2 = json.loads((tmp_path / "b" / "_static" / "ai-assistant-isolation-policy.json").read_text())
    assert policy2["allowedParentOrigins"] == []


def test_b42_sandbox_blocks_self_boundary_escape_and_popup_escape():
    host = HOST.read_text()
    frame = FRAME.read_text()
    assert "allow-scripts allow-same-origin allow-downloads allow-popups'" in host
    assert "allow-popups-to-escape-sandbox" not in host
    assert "allow-top-navigation" not in host
    assert "installNavigationGuard(cfg)" in frame
    assert "event.preventDefault();" in frame
    assert "window.open(destination.href, '_blank', 'noopener,noreferrer')" in frame
    assert "destination.protocol !== 'http:' && destination.protocol !== 'https:'" in frame


def test_b42_cross_origin_microphone_is_separate_site_owner_opt_in():
    init = INIT.read_text()
    host = HOST.read_text()
    assert 'app.add_config_value("ai_assistant_isolation_allow_microphone", False, "html")' in init
    assert '"isolationAllowMicrophone": _cfg_bool(' in init
    assert "cfg.isolationAllowMicrophone === true && cfg.panelSpeakBanner !== false" in host
    assert "if (cfg.isolationAllowMicrophone !== true) initConfig.panelSpeakBanner = false;" in host


def test_b42_assistant_service_fetches_omit_ambient_credentials_by_default():
    init = INIT.read_text()
    main = MAIN.read_text()
    assert 'app.add_config_value("ai_assistant_allow_credentialed_fetch", False, "html")' in init
    assert '"allowCredentialedFetch": _cfg_bool(' in init
    assert "safe.credentials = (_cfg().allowCredentialedFetch === true) ? 'same-origin' : 'omit';" in main
    assert "if (input.credentials === 'omit')" in main
    assert "credentials = 'include'" not in main
    assert "safe.credentials = 'include'" not in main


def test_b42_canonical_page_read_is_stream_bounded_and_deliberately_same_origin():
    host = HOST.read_text()
    assert "_readBoundedResponseText" in host
    assert "response.body.getReader" in host
    assert "bytes > maxChars * 4" in host
    assert "out.length > maxChars" in host
    assert "credentials: 'same-origin', redirect: 'error', cache:'no-store'" in host


def test_b42_docs_root_crossing_boundary_is_same_origin_query_fragment_free():
    host = HOST.read_text()
    assert "root.origin === location.origin" in host
    assert "root.search = ''; root.hash = '';" in host
    assert "scriptRoot.origin === location.origin" in host


def test_b42_setup_connects_policy_generator_and_protocol_version_is_serialized():
    init = INIT.read_text()
    assert '"isolationProtocolVersion": "2.0.0"' in init
    assert 'app.add_config_value("ai_assistant_isolation_parent_origins", [], "html")' in init
    assert 'app.connect("build-finished", generate_isolation_policy)' in init


def test_b42_guide_mentions_parent_policy_cookie_and_navigation_boundaries():
    guide = GUIDE.read_text().lower()
    for phrase in (
        "parent-origin policy",
        "ambient cookies",
        "microphone",
        "frame-self navigation",
        "2.0.0",
    ):
        assert phrase in guide
