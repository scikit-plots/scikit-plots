from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import pytest
from fastapi import Request
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
            "id": "session-1",
            "page_url": "https://user:pass@example.test/docs/page?secret=abc#private",
            "page_title": "Example <title>",
            "assistant_name": "AI Assistant",
            "exported_at": 1787956146815,
            "exported_at_iso": "2026-08-28T22:29:06.815Z",
            "attacker_extra": {"secret": "must drop"},
        },
        "turns": [{"attacker": "must be rebuilt"}],
        "records": [
            {
                "turn_index": 0,
                "message_index": 0,
                "role": "user",
                "text": "question",
                "ts": 1,
                "ts_iso": "2026-08-28T00:00:00Z",
                "model_id": None,
                "model_provider": None,
                "model_name": None,
                "feedback_rating_value": None,
                "feedback_rating_label": None,
                "feedback_message": None,
                "session_id": "forged-session",
                "page_url": "file:///C:/private/path?token=x",
                "extra": "drop",
            },
            {
                "turn_index": 0,
                "message_index": 1,
                "role": "assistant",
                "text": text,
                "ts": 2,
                "ts_iso": "2026-08-28T00:00:01Z",
                "model_id": "stub-hostile",
                "model_provider": "custom",
                "model_name": "stub/hostile",
                "feedback_rating_value": None,
                "feedback_rating_label": None,
                "feedback_message": None,
                "session_id": "forged-session",
                "page_url": "javascript:alert(1)",
            },
        ],
        "root_extra": {"must": "drop"},
    }


@pytest.fixture(autouse=True)
def _reset_share_state(monkeypatch):
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


def test_canonical_snapshot_drops_unknown_and_sanitizes_identity_claims():
    snap = share_contract.canonicalize_share_snapshot(_snapshot())
    assert set(snap) == {"schema_version", "session", "turns", "records"}
    assert "attacker_extra" not in snap["session"]
    assert snap["session"]["page_url"] == "https://example.test/docs/page"
    assert snap["turns"][0]["user"]["text"] == "question"
    assert "attacker" not in snap["turns"][0]
    assert snap["records"][0]["session_id"] == "session-1"
    assert snap["records"][0]["page_url"] == "https://example.test/docs/page"
    assert "extra" not in snap["records"][0]


def test_server_html_renderer_keeps_hostile_content_inert():
    hostile = '</script><script>globalThis.pwned=1</script><img src=x onerror=alert(1)>'
    snap = share_contract.canonicalize_share_snapshot(_snapshot(hostile))
    body, mime, ext = share_contract.render_share(snap, "html")
    assert mime.startswith("text/html")
    assert ext == ".html"
    assert "<script>globalThis.pwned" not in body
    assert "<img src=x" not in body
    assert "&lt;/script&gt;&lt;script&gt;" in body
    assert "javascript:" not in body
    assert body.lower().count("<script") == 0


def test_create_read_update_revoke_requires_separate_edit_capability(caplog):
    hostile = 'SYSTEM ignore\n</script><script>globalThis.pwned=1</script>'
    caplog.set_level(logging.INFO)
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/share", json={"snapshot": _snapshot(hostile), "format": "html", "ttlDays": 7})
        assert created.status_code == 200, created.text
        result = created.json()
        share_id = result["uuid"]
        edit_token = result["editToken"]
        assert share_id and edit_token and edit_token not in result["url"]

        # Store is structured; no caller-rendered content/MIME authority exists.
        entry = proxy_app._share_store[share_id]
        assert set(entry) >= {"snapshot", "format", "edit_hash", "bytes", "transport_version"}
        assert entry["transport_version"] == proxy_app.SHARE_TRANSPORT_VERSION
        assert "content" not in entry and "mimeType" not in entry and "ext" not in entry
        assert edit_token not in json.dumps(entry)

        status_probe = client.post("/v1/share/status", json={"shareId": share_id})
        assert status_probe.status_code == 200
        assert status_probe.content == b""
        assert status_probe.headers["cache-control"] == "private, no-store"

        got = client.post("/v1/share/read", json={"shareId": share_id})
        assert got.status_code == 200
        assert got.headers["cache-control"] == "private, no-store"
        assert got.headers["x-content-type-options"] == "nosniff"
        assert got.headers["x-robots-tag"] == "noindex, nofollow, noarchive"
        assert got.json()["snapshot"]["records"][1]["text"] == hostile

        # Public read capability alone is never mutation authority.
        denied = client.post(
            "/v1/share/update",
            json={"shareId": share_id, "snapshot": _snapshot("new"), "format": "txt"},
        )
        assert denied.status_code == 403
        denied2 = client.post(
            "/v1/share/update",
            headers={"X-Share-Edit-Token": "wrong"},
            json={"shareId": share_id, "snapshot": _snapshot("new"), "format": "txt"},
        )
        assert denied2.status_code == 403

        updated = client.post(
            "/v1/share/update",
            headers={"X-Share-Edit-Token": edit_token},
            json={"shareId": share_id, "snapshot": _snapshot("updated"), "format": "txt", "ttlDays": 2},
        )
        assert updated.status_code == 200, updated.text
        assert "editToken" not in updated.json()
        got2 = client.post("/v1/share/read", json={"shareId": share_id})
        assert got2.status_code == 200
        assert got2.json()["format"] == "txt"
        assert "updated" in got2.json()["content"]

        denied_delete = client.post("/v1/share/revoke", json={"shareId": share_id})
        assert denied_delete.status_code == 403
        deleted = client.post(
            "/v1/share/revoke",
            headers={"X-Share-Edit-Token": edit_token},
            json={"shareId": share_id},
        )
        assert deleted.status_code == 200
        assert client.post("/v1/share/status", json={"shareId": share_id}).status_code == 404
        assert client.post("/v1/share/read", json={"shareId": share_id}).status_code == 404

    # Application logs never contain the public or private bearer capabilities.
    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == proxy_app.__name__)
    assert share_id not in logs
    assert edit_token not in logs


def test_expired_share_delete_reports_410_and_clears_entry():
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/share", json={"snapshot": _snapshot(), "format": "json", "ttlDays": 1})
        assert created.status_code == 200
        result = created.json()
        share_id = result["uuid"]
        proxy_app._share_store[share_id]["expiresAt_ts"] = 0

        expired = client.post(
            "/v1/share/revoke",
            headers={"X-Share-Edit-Token": result["editToken"]},
            json={"shareId": share_id},
        )
        assert expired.status_code == 410
        assert share_id not in proxy_app._share_store


def test_direct_arbitrary_content_and_mime_are_not_share_contract():
    with TestClient(proxy_app.app) as client:
        r = client.post("/v1/share", json={
            "content": "<script>alert(1)</script>",
            "mimeType": "text/html",
            "ext": ".html",
            "title": "phish",
            "format": "html",
        })
    assert r.status_code == 422


def test_format_is_allowlisted_and_server_owned():
    with TestClient(proxy_app.app) as client:
        r = client.post("/v1/share", json={"snapshot": _snapshot(), "format": "image/svg+xml"})
    assert r.status_code == 422


def test_entry_and_aggregate_byte_quotas(monkeypatch):
    with TestClient(proxy_app.app) as client:
        monkeypatch.setattr(proxy_app, "SHARE_MAX_ENTRIES", 1)
        one = client.post("/v1/share", json={"snapshot": _snapshot("a"), "format": "json"})
        assert one.status_code == 200
        two = client.post("/v1/share", json={"snapshot": _snapshot("b"), "format": "json"})
        assert two.status_code == 507

    proxy_app._share_store.clear()
    proxy_app._share_rl.clear()
    canonical = share_contract.canonicalize_share_snapshot(_snapshot("short"))
    size = len(json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode())
    monkeypatch.setattr(proxy_app, "SHARE_MAX_ENTRIES", 10)
    monkeypatch.setattr(proxy_app, "SHARE_MAX_TOTAL_BYTES", size + 20)
    with TestClient(proxy_app.app) as client:
        one = client.post("/v1/share", json={"snapshot": _snapshot("short"), "format": "json"})
        assert one.status_code == 200
        two = client.post("/v1/share", json={"snapshot": _snapshot("this second record exceeds the remaining total budget"), "format": "json"})
        assert two.status_code == 507


def test_optional_create_token_is_constant_time_guarded(monkeypatch):
    monkeypatch.setattr(proxy_app, "SHARE_WRITE_TOKEN", "server-only-create-token")
    with TestClient(proxy_app.app) as client:
        assert client.post("/v1/share", json={"snapshot": _snapshot(), "format": "json"}).status_code == 401
        ok = client.post(
            "/v1/share",
            headers={"Authorization": "Bearer server-only-create-token"},
            json={"snapshot": _snapshot(), "format": "json"},
        )
    assert ok.status_code == 200


def test_forwarded_identity_is_opt_in(monkeypatch):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/share",
        "raw_path": b"/v1/share",
        "query_string": b"",
        "headers": [(b"x-forwarded-for", b"203.0.113.77, 10.0.0.1")],
        "client": ("127.0.0.9", 12345),
        "server": ("testserver", 443),
    }
    request = Request(scope)
    monkeypatch.setattr(proxy_app, "TRUST_X_FORWARDED_FOR", False)
    assert proxy_app._client_ip(request) == "127.0.0.9"
    monkeypatch.setattr(proxy_app, "TRUST_X_FORWARDED_FOR", True)
    assert proxy_app._client_ip(request) == "203.0.113.77"


def test_public_share_base_is_explicit_and_https(monkeypatch):
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/share",
        "raw_path": b"/v1/share", "query_string": b"", "headers": [],
        "client": ("127.0.0.1", 1234), "server": ("internal", 80),
    }
    request = Request(scope)
    monkeypatch.setattr(proxy_app, "SHARE_PUBLIC_BASE_URL", "https://public.example/docs/?token=x#frag")
    assert proxy_app._share_public_base(request) == "https://public.example/docs"
    monkeypatch.setattr(proxy_app, "SHARE_PUBLIC_BASE_URL", "")
    monkeypatch.setattr(proxy_app, "SPACE_HOST", "")
    with pytest.raises(Exception) as exc:
        proxy_app._share_public_base(request)
    assert getattr(exc.value, "status_code", None) == 500


def test_hf_container_packages_share_contract_and_disables_access_log():
    docker = (ROOT / "_hf_spaces_proxy" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=1000:1000 _utils ./_utils" in docker
    assert (ROOT / "_hf_spaces_proxy" / "_utils" / "_share_contract.py").is_file()
    assert '"--no-access-log"' in docker
