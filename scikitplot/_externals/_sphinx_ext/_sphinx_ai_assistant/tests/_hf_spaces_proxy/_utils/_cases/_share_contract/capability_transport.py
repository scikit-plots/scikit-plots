from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
pytestmark = pytest.mark.usefixtures('_reset_capability_transport')


HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

share_contract = importlib.import_module("_utils._share_contract")
proxy_app = importlib.import_module("app")


def _snapshot(text: str = "hello") -> dict:
    return {
        "schema_version": "2.1",
        "session": {
            "id": "session-run13",
            "page_url": "https://docs.example.test/page?private=x#frag",
            "page_title": "Run 13",
            "assistant_name": "AI Assistant",
            "exported_at": 1,
            "exported_at_iso": "2026-08-29T00:00:00Z",
        },
        "records": [
            {
                "turn_index": 0,
                "message_index": 0,
                "role": "user",
                "text": "question",
                "ts": 1,
                "resources": {
                    "version": 2,
                    "totalCount": 1,
                    "includedCount": 1,
                    "localOnlyCount": 0,
                    "contextCount": 1,
                    "rawCount": 0,
                    "notSentCount": 0,
                    "pageCount": 1,
                    "replayCount": 0,
                    "totalBytes": 42,
                    "itemCount": 1,
                    "omittedCount": 0,
                    "complete": True,
                    "items": [{
                        "name": "Docs",
                        "badge": "PAGE",
                        "kind": "page",
                        "size": 42,
                        "lineCount": 2,
                        "included": True,
                        "localOnly": False,
                        "delivery": "context",
                        "modality": "text",
                        "intent": "context",
                        "replay": False,
                        "boundedExcerpt": False,
                        "status": "Current page",
                        "type": "",
                        "relativePath": "",
                        "sourceKind": "",
                        "archiveName": "",
                        "contextRole": "current",
                        "sourceUrl": "https://user:pass@docs.example.test/resource/?token=SECRET#frag",
                    }],
                },
            },
            {
                "turn_index": 0,
                "message_index": 1,
                "role": "assistant",
                "text": text,
                "ts": 2,
                "feedback_rating_value": 1,
                "feedback_rating_label": "helpful",
                "feedback_message": "useful answer",
            },
        ],
    }


@pytest.fixture
def _reset_capability_transport(monkeypatch):
    proxy_app._share_store.clear()
    proxy_app._share_rl.clear()
    monkeypatch.setattr(proxy_app, "SHARE_WRITE_TOKEN", "")
    monkeypatch.setattr(proxy_app, "SHARE_PUBLIC_BASE_URL", "https://share.example.test")
    monkeypatch.setattr(proxy_app, "SHARE_MAX_BODY_BYTES", 512_000)
    monkeypatch.setattr(proxy_app, "SHARE_MAX_ENTRIES", 256)
    monkeypatch.setattr(proxy_app, "SHARE_MAX_TOTAL_BYTES", 16 * 1024 * 1024)
    yield
    proxy_app._share_store.clear()
    proxy_app._share_rl.clear()


def test_new_global_link_keeps_capability_in_fragment_only():
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/share", json={"snapshot": _snapshot(), "format": "html", "ttlDays": 7})
    assert created.status_code == 200
    data = created.json()
    share_id = data["uuid"]
    assert data["url"] == f"https://share.example.test/v1/share#share={share_id}"
    assert f"/v1/share/{share_id}" not in data["url"]
    assert data["editToken"] not in data["url"]


def test_fixed_viewer_is_static_no_store_and_dom_safe():
    with TestClient(proxy_app.app) as client:
        viewer = client.get("/v1/share")
    assert viewer.status_code == 200
    assert viewer.headers["cache-control"] == "private, no-store"
    assert viewer.headers["referrer-policy"] == "no-referrer"
    assert "connect-src 'self'" in viewer.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in viewer.headers["content-security-policy"]
    assert "location.hash" in viewer.text
    assert "'/v1/share/read'" in viewer.text or '"/v1/share/read"' in viewer.text
    assert "textContent" in viewer.text
    assert "appendResources" in viewer.text
    assert "feedback_rating_label" in viewer.text
    assert "resource-source" in viewer.text
    assert "innerHTML=data.content" not in viewer.text.replace(" ", "")
    assert ".innerHTML" not in viewer.text


def test_fixed_read_status_update_revoke_never_need_capability_path():
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/share", json={"snapshot": _snapshot("first"), "format": "html"}).json()
        share_id = created["uuid"]
        token = created["editToken"]

        status = client.post("/v1/share/status", json={"shareId": share_id})
        assert status.status_code == 200 and status.content == b""

        read = client.post("/v1/share/read", json={"shareId": share_id})
        assert read.status_code == 200
        body = read.json()
        assert body["format"] == "html"
        assert body["snapshot"]["records"][1]["text"] == "first"
        assert "content" not in body

        denied = client.post(
            "/v1/share/update",
            headers={"X-Share-Edit-Token": "wrong"},
            json={"shareId": share_id, "snapshot": _snapshot("second"), "format": "txt"},
        )
        assert denied.status_code == 403

        updated = client.post(
            "/v1/share/update",
            headers={"X-Share-Edit-Token": token},
            json={"shareId": share_id, "snapshot": _snapshot("second"), "format": "txt"},
        )
        assert updated.status_code == 200
        assert updated.json()["url"].endswith(f"/v1/share#share={share_id}")
        assert "editToken" not in updated.json()

        read2 = client.post("/v1/share/read", json={"shareId": share_id})
        assert read2.status_code == 200
        assert read2.json()["format"] == "txt"
        assert "second" in read2.json()["content"]

        revoked = client.post(
            "/v1/share/revoke",
            headers={"X-Share-Edit-Token": token},
            json={"shareId": share_id},
        )
        assert revoked.status_code == 200
        assert client.post("/v1/share/status", json={"shareId": share_id}).status_code == 404
        assert client.post("/v1/share/read", json={"shareId": share_id}).status_code == 404



@pytest.mark.parametrize("fmt", ["json", "html", "txt", "yaml", "toml"])
def test_fixed_share_round_trip_supports_every_export_format(fmt):
    with TestClient(proxy_app.app) as client:
        created = client.post(
            "/v1/share",
            json={"snapshot": _snapshot(), "format": fmt, "ttlDays": 7},
        )
        assert created.status_code == 200
        share_id = created.json()["uuid"]
        read = client.post("/v1/share/read", json={"shareId": share_id})
        downloaded = client.post("/v1/share/download", json={"shareId": share_id})
    assert read.status_code == 200
    body = read.json()
    assert body["format"] == fmt
    expected_name = f"ai-conversation-global-share-{fmt}." + ("yaml" if fmt == "yaml" else fmt)
    assert body["filename"] == expected_name
    assert downloaded.status_code == 200
    assert downloaded.headers["content-disposition"] == f'attachment; filename="{expected_name}"'
    assert share_id not in downloaded.headers["content-disposition"]
    assert downloaded.headers["cache-control"] == "private, no-store"
    assert downloaded.headers["x-content-type-options"] == "nosniff"
    if fmt == "html":
        snap = body["snapshot"]
        assert snap["schema_version"] == "2.1"
        assert snap["records"][0]["resources"]["items"][0]["sourceUrl"] == "https://docs.example.test/resource/"
        assert snap["records"][1]["feedback_rating_label"] == "helpful"
        assert "<!DOCTYPE html>" in downloaded.text or "<!doctype html>" in downloaded.text
        assert "Content-Security-Policy" in downloaded.text
        assert downloaded.headers["content-security-policy"].startswith("sandbox;")
        return

    content = body["content"]
    assert downloaded.text == content
    assert "user:pass" not in content
    assert "token=SECRET" not in content
    if fmt == "json":
        snap = json.loads(content)
        assert snap["schema_version"] == "2.1"
        assert snap["records"][0]["resources"]["items"][0]["sourceUrl"] == "https://docs.example.test/resource/"
    elif fmt == "txt":
        assert "Schema: 2.1" in content
        assert "Resources used for this question: 1" in content
        assert "Rating: helpful" in content
    elif fmt == "yaml":
        assert '"schema_version": "2.1"' in content
        assert '"sourceUrl": "https://docs.example.test/resource/"' in content
    else:
        assert 'schema_version = "2.1"' in content
        assert '[[records.resources.items]]' in content
        assert 'sourceUrl = "https://docs.example.test/resource/"' in content

def test_fixed_locator_endpoints_are_body_limited():
    huge = {"shareId": "a" * 32, "padding": "x" * 10_000}
    with TestClient(proxy_app.app) as client:
        assert client.post("/v1/share/read", json=huge).status_code == 413
        assert client.post("/v1/share/download", json=huge).status_code == 413
        assert client.post("/v1/share/status", json=huge).status_code == 413
        assert client.post("/v1/share/revoke", json=huge).status_code == 413


def test_viewer_helper_does_not_embed_snapshot_or_capability():
    shell = share_contract.render_share_viewer_shell()
    assert "__READ_PATH__" not in shell
    assert "/v1/share/read" in shell
    assert "/v1/share/download" in shell
    assert "Download "+"" in shell
    assert "ai-conversation-global-share-" in shell
    assert "shareId:raw" in shell
    assert "textContent" in shell
    assert "snapshot.records" in shell
    assert "appendResources" in shell
    assert "feedback_rating_label" in shell
    assert ".innerHTML" not in shell


def test_worker_and_wrangler_have_same_fixed_path_contract():
    worker = (ROOT / "_cf_worker" / "index.js").read_text(encoding="utf-8")
    wrangler = (ROOT / "_cf_worker" / "wrangler.toml").read_text(encoding="utf-8")
    for route in ("/v1/share/read", "/v1/share/status", "/v1/share/update", "/v1/share/revoke"):
        assert route in worker
    assert "/v1/share#share=${shareUuid}" in worker
    assert "location.hash" in worker
    assert "textContent" in worker
    assert "invocation_logs = false" in wrangler


def test_current_client_uses_fixed_paths_not_capability_paths():
    js = (ROOT / "_static" / "ai-assistant.js").read_text(encoding="utf-8")
    patch = js[js.index("function _patchGlobalShare"):js.index("function _utf8ByteLength")]
    assert "+ '/update'" in patch
    assert "+ '/revoke'" in patch
    assert "+ '/status'" in patch
    assert "JSON.stringify({ shareId: loc.id })" in patch
    assert "base + '/' + encodeURIComponent(_globalShareState.uuid)" not in js
    assert "base + '/' + encodeURIComponent(revokeUuid)" not in js
