from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

proxy_app = importlib.import_module("app")


def _snapshot(text: str = "hello") -> dict:
    return {
        "schema_version": "2.0",
        "session": {
            "id": "session-run14",
            "page_url": "https://docs.example.test/page",
            "page_title": "Run 14",
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


def test_generation2_share_cannot_be_served_or_mutated_through_legacy_path():
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/share", json={"snapshot": _snapshot(), "format": "html"})
        assert created.status_code == 200
        data = created.json()
        share_id = data["uuid"]
        token = data["editToken"]

        assert proxy_app._share_store[share_id]["transport_version"] == 2
        assert client.head(f"/v1/share/{share_id}").status_code == 404
        assert client.get(f"/v1/share/{share_id}").status_code == 404
        assert client.patch(
            f"/v1/share/{share_id}",
            headers={"X-Share-Edit-Token": token},
            json={"snapshot": _snapshot("legacy patch must not run"), "format": "txt"},
        ).status_code == 404
        assert client.delete(
            f"/v1/share/{share_id}", headers={"X-Share-Edit-Token": token}
        ).status_code == 404

        # The supported fixed-path lifecycle remains fully usable.
        assert client.post("/v1/share/status", json={"shareId": share_id}).status_code == 200
        assert client.post("/v1/share/read", json={"shareId": share_id}).status_code == 200


def test_pre_generation_share_get_head_are_deprecated_and_legacy_patch_is_retired():
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/share", json={"snapshot": _snapshot("old"), "format": "html", "ttlDays": 7})
        data = created.json()
        share_id = data["uuid"]
        token = data["editToken"]

        # Model a still-live object created before transport-generation metadata existed.
        proxy_app._share_store[share_id].pop("transport_version", None)

        head = client.head(f"/v1/share/{share_id}")
        assert head.status_code == 200
        assert head.headers["deprecation"] == "@1787961600"
        assert "successor-version" in head.headers["link"]
        assert "sunset" in head.headers

        legacy_get = client.get(f"/v1/share/{share_id}")
        assert legacy_get.status_code == 200
        assert legacy_get.headers["deprecation"] == "@1787961600"
        assert "successor-version" in legacy_get.headers["link"]

        retired = client.patch(
            f"/v1/share/{share_id}",
            headers={"X-Share-Edit-Token": token},
            json={"snapshot": _snapshot("must not update"), "format": "txt", "ttlDays": 365},
        )
        assert retired.status_code == 410
        assert retired.headers["deprecation"] == "@1787961600"
        assert "POST /v1/share/update" in retired.json()["error"]
        assert "transport_version" not in proxy_app._share_store[share_id]
        assert client.get(f"/v1/share/{share_id}").status_code == 200


def test_fixed_update_also_migrates_pre_generation_share():
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/share", json={"snapshot": _snapshot("old"), "format": "json"}).json()
        share_id = created["uuid"]
        proxy_app._share_store[share_id].pop("transport_version", None)

        updated = client.post(
            "/v1/share/update",
            headers={"X-Share-Edit-Token": created["editToken"]},
            json={"shareId": share_id, "snapshot": _snapshot("new"), "format": "json"},
        )
        assert updated.status_code == 200
        assert proxy_app._share_store[share_id]["transport_version"] == 2
        assert client.get(f"/v1/share/{share_id}").status_code == 404
        assert client.post("/v1/share/read", json={"shareId": share_id}).status_code == 200


def test_pre_generation_share_can_still_be_revoked_during_bounded_migration_window():
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/share", json={"snapshot": _snapshot(), "format": "json"}).json()
        share_id = created["uuid"]
        proxy_app._share_store[share_id].pop("transport_version", None)

        denied = client.delete(f"/v1/share/{share_id}")
        assert denied.status_code == 403
        revoked = client.delete(
            f"/v1/share/{share_id}", headers={"X-Share-Edit-Token": created["editToken"]}
        )
        assert revoked.status_code == 200
        assert revoked.headers["deprecation"] == "@1787961600"
        assert share_id not in proxy_app._share_store


def test_worker_source_stamps_generation_and_legacy_routes_are_gated():
    worker = (ROOT / "_cf_worker" / "index.js").read_text(encoding="utf-8")
    assert "const SHARE_TRANSPORT_VERSION = 2;" in worker
    assert "transportVersion: SHARE_TRANSPORT_VERSION" in worker
    assert "function _legacyShareEntryAllowed(entry)" in worker
    assert worker.count("_legacyShareEntryAllowed(") >= 5  # helper + PATCH/DELETE/HEAD/GET
    assert "Deprecation: '@1787961600'" in worker
    assert 'rel="successor-version"' in worker
