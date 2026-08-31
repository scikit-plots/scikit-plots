from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

share_contract = importlib.import_module("_utils._share_contract")
proxy_app = importlib.import_module("app")


def _snapshot(text: str = "hello") -> dict:
    return {
        "schema_version": "2.0",
        "session": {
            "id": "session-run13",
            "page_url": "https://docs.example.test/page?private=x#frag",
            "page_title": "Run 13",
            "assistant_name": "AI Assistant",
            "exported_at": 1,
            "exported_at_iso": "2026-08-29T00:00:00Z",
        },
        "records": [
            {"turn_index": 0, "message_index": 0, "role": "user", "text": "question", "ts": 1},
            {"turn_index": 0, "message_index": 1, "role": "assistant", "text": text, "ts": 2},
        ],
    }


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
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
    assert "innerHTML=data.content" not in viewer.text.replace(" ", "")


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


def test_fixed_locator_endpoints_are_body_limited():
    huge = {"shareId": "a" * 32, "padding": "x" * 10_000}
    with TestClient(proxy_app.app) as client:
        assert client.post("/v1/share/read", json=huge).status_code == 413
        assert client.post("/v1/share/status", json=huge).status_code == 413
        assert client.post("/v1/share/revoke", json=huge).status_code == 413


def test_viewer_helper_does_not_embed_snapshot_or_capability():
    shell = share_contract.render_share_viewer_shell()
    assert "__READ_PATH__" not in shell
    assert "/v1/share/read" in shell
    assert "shareId:raw" in shell
    assert "textContent" in shell
    assert "snapshot.records" in shell


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
