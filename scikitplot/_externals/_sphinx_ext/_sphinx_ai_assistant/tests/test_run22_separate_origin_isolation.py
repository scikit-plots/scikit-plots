from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "_static"
INIT = ROOT / "__init__.py"
MAIN = STATIC / "ai-assistant.js"
HOST = STATIC / "ai-assistant-isolation-host.js"
FRAME = STATIC / "ai-assistant-isolated-frame.js"
HTML = STATIC / "ai-assistant-isolated.html"
ISO_CSS = STATIC / "ai-assistant-isolated.css"
GUIDE = ROOT / "ISOLATION_DEPLOYMENT.md"


def _load_validator(name: str):
    tree = ast.parse(INIT.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])

    class _Log:
        def warning(self, *args: Any, **kwargs: Any) -> None:
            return None

    ns: dict[str, Any] = {
        "Any": Any,
        "urlsplit": urlsplit,
        "_get_logger": lambda: _Log(),
    }
    exec(compile(module, str(INIT), "exec"), ns)
    return ns[name]


def test_isolation_origin_validator_is_exact_origin_and_https_first():
    validate = _load_validator("_validate_isolation_origin")
    assert validate("https://assistant.example.com") == "https://assistant.example.com"
    assert validate("https://assistant.example.com/") == "https://assistant.example.com"
    assert validate("http://localhost:8000") == "http://localhost:8000"
    assert validate("http://assistant.example.com") == ""
    assert validate("https://user:pw@assistant.example.com") == ""
    assert validate("https://assistant.example.com/path") == ""
    assert validate("https://assistant.example.com/?x=1") == ""
    assert validate("javascript:alert(1)") == ""


def test_frame_path_validator_is_root_relative_and_traversal_safe():
    validate = _load_validator("_validate_isolation_frame_path")
    assert validate("/ai-assistant-isolated.html") == "/ai-assistant-isolated.html"
    assert validate("/frames/assistant.html") == "/frames/assistant.html"
    assert validate("../secret.html") == "/ai-assistant-isolated.html"
    assert validate("/x/../secret") == "/ai-assistant-isolated.html"
    assert validate("https://evil.example/frame") == "/ai-assistant-isolated.html"
    assert validate("/frame.html?x=1") == "/ai-assistant-isolated.html"


def test_setup_loads_fail_closed_host_bridge_before_full_runtime():
    src = INIT.read_text()
    host_pos = src.index('app.add_js_file("ai-assistant-isolation-host.js"')
    main_pos = src.index('"ai-assistant.js", loading_method="defer"')
    assert host_pos < main_pos
    assert 'app.add_config_value("ai_assistant_isolation_origin", "", "html")' in src
    assert 'app.add_config_value("ai_assistant_isolation_context_max_chars", 200000, "html")' in src
    main = MAIN.read_text()
    assert "window.AI_ASSISTANT_CONFIG && window.AI_ASSISTANT_CONFIG.isolationOrigin" in main
    assert main.count("!window.SphinxAIAssistantIsolationFrame) return;") >= 2


def test_message_protocol_uses_port_after_exact_window_handshake():
    host = HOST.read_text()
    frame = FRAME.read_text()
    assert "event.source !== iframe.contentWindow || event.origin !== isolationOrigin" in host
    assert "var channel = new _NativeMessageChannel();" in host
    assert "_postToFrame(iframe.contentWindow, init, isolationOrigin, [channel.port2])" in host
    assert "_nativeRemoveMessageListener('message', onMessage, true);" in host
    assert "event.source !== parent || event.origin !== parentOrigin" in frame
    assert "event.ports.length !== 1" in frame
    assert "port.onmessage = onPortMessage" in frame
    assert "msg.seq !== rxSeq + 1" in frame
    assert "msg.seq !== _rxSeq + 1" in host


def test_bridge_bootstrap_snapshots_globals_and_blocks_prototype_pollution_keys():
    host = HOST.read_text()
    frame = FRAME.read_text()
    assert "var cfg = _snapshotBridgeValue(window.AI_ASSISTANT_CONFIG || {}, 0, [])" in host
    assert "var _endpointSnapshot = _snapshotBridgeValue(window.AI_ASSISTANT_ENDPOINTS || {}, 0, [])" in host
    assert "endpointDefault:_endpointDefaultSnapshot" in host
    assert "_SNAPSHOT_FORBIDDEN_KEY" in host
    assert "__proto__|prototype|constructor" in host
    assert "Object.create(null)" in host
    assert "__proto__|prototype|constructor" in frame
    assert "var out = Object.create(null);" in frame


def test_host_capabilities_are_closed_and_bounded():
    host = HOST.read_text()
    for cap in (
        "page.context.read",
        "page.canonical.read",
        "page.print",
        "ui.resize",
        "page.integration.emit",
    ):
        assert f"cap === '{cap}'" in host
    assert "else throw new Error('CAPABILITY_DENIED')" in host
    assert "MAX_MESSAGE_CHARS = 262144" in host
    assert "MAX_CANONICAL_CHARS = 1048576" in host
    assert "MAX_PUBLIC_EVENT_CHARS = 8192" in host
    assert "u.search = '';" in host and "u.hash = '';" in host
    assert "credentials: 'same-origin', redirect: 'error'" in host


def test_context_adapter_removes_active_forms_hidden_and_dangerous_attributes():
    host = HOST.read_text()
    for token in ("'script'", "'iframe'", "'object'", "'embed'", "'form'", "'input'", "'[hidden]'", "'[aria-hidden=\"true\"]'"):
        assert token in host
    for attr in ("n === 'srcdoc'", "n === 'style'", "n === 'nonce'", "n === 'formaction'"):
        assert attr in host
    assert "Oversize pages" in host
    assert "format: 'text'" in host


def test_isolated_runtime_storage_is_namespaced_by_parent_site_root():
    main = MAIN.read_text()
    frame = FRAME.read_text()
    assert "cfg.isolationStorageScope = storageScope(init.page || {})" in frame
    assert "return 'host-site:' + parentOrigin + '|' + path" in frame
    assert "function _scopedStorage(nativeStore, scope)" in main
    assert "ai-assistant-isolated:" in main
    assert "var localStorage = _scopedStorage" in main
    assert "var sessionStorage = _scopedStorage" in main


def test_isolated_html_has_no_inline_script_or_style_and_restrictive_meta_csp():
    html = HTML.read_text()
    assert "default-src 'none'" in html
    assert "script-src 'self'" in html
    assert "style-src 'self'" in html
    assert "object-src 'none'" in html
    assert "form-action 'none'" in html
    assert "worker-src 'none'" in html
    assert "unsafe-inline" not in html
    assert "<style" not in html.lower()
    assert "<script>" not in html.lower()
    assert ISO_CSS.is_file()


def test_public_event_bridge_revalidates_b40_projection_at_host_boundary():
    main = MAIN.read_text()
    host = HOST.read_text()
    assert "isolationBridge.notify('page.integration.emit'" in main
    assert "if (!_feedbackDomIntegrationEnabled) return internalResult;" in main
    assert "PUBLIC_EVENT_TYPES" in host
    assert "FORBIDDEN_PUBLIC_KEY" in host
    assert "PUBLIC_EVENT_DETAIL_DENIED" in host
    assert "document.dispatchEvent(new CustomEvent(payload.eventType" in host


def test_deployment_guide_keeps_residual_and_header_requirements_explicit():
    guide = GUIDE.read_text()
    assert "frame-ancestors" in guide
    assert "Referrer-Policy: no-referrer" in guide
    assert "X-Content-Type-Options: nosniff" in guide
    assert "fully compromised parent page" in guide
    assert "cannot" in guide
    assert "No host capability returns transcript" in guide
