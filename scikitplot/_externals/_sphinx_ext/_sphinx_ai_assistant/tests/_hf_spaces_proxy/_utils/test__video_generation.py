# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Video generation contract, test backend, and proxy activation gates."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ...._hf_spaces_proxy import app
from ...._hf_spaces_proxy._utils._video_generation import (
    StubVideoGenerationStore,
    VideoGenerationError,
    normalize_video_job,
    normalize_video_upstream_url,
    parse_video_generation_request,
)


def _request() -> dict:
    return {
        "contract": "learn.video-generation-request.v1",
        "client_request_id": "req-video-1",
        "input": {"mode": "prompt", "prompt": "Explain Lasso regularization."},
        "instructions": "Beginner-friendly.",
        "presentation": {
            "style": "explainer",
            "length": "short",
            "language": "en",
            "voice": "default",
            "aspect_ratio": "16:9",
            "captions": True,
            "branding": True,
        },
        "model_selection": {
            "source": "ai-assistant",
            "id": "stub/mirror",
            "label": "Stub mirror",
            "provider": "custom",
            "model": "stub/mirror",
            "effort": {"id": "high", "label": "High", "supported": True},
        },
        "provenance": {
            "site_id": "learn",
            "catalog_revision": "test",
            "source_page": "https://scikit-plots.github.io/dev/learn-ai/videos/new.html",
        },
    }


def test_request_validation_and_private_url_rejection() -> None:
    parsed = parse_video_generation_request(json.dumps(_request()).encode())
    assert parsed["input"]["prompt"] == "Explain Lasso regularization."
    bad = _request()
    bad["input"] = {"mode": "url", "url": "https://127.0.0.1/private"}
    with pytest.raises(VideoGenerationError, match="Local/private"):
        parse_video_generation_request(json.dumps(bad).encode())


def test_upstream_url_is_exact_https_operator_configuration() -> None:
    assert normalize_video_upstream_url(
        "https://video.example.test/v1/video-generations/"
    ) == "https://video.example.test/v1/video-generations"
    for value in (
        "http://video.example.test/v1/video-generations",
        "https://user:pass@video.example.test/v1/video-generations",
        "https://video.example.test/v1/video-generations?token=nope",
    ):
        with pytest.raises(VideoGenerationError):
            normalize_video_upstream_url(value)


def test_stub_store_is_idempotent_and_never_claims_publication() -> None:
    store = StubVideoGenerationStore(ready_seconds=0)
    parsed = parse_video_generation_request(json.dumps(_request()).encode())
    first = store.create(parsed, "idem-1")
    second = store.create(parsed, "idem-1")
    assert first["generation_id"] == second["generation_id"]
    assert second["status"] == "ready"
    assert second["result"]["test_mode"] is True
    assert second["result"]["url"] == ""
    archived = store.action(first["generation_id"], "archive")
    assert archived["status"] == "archived"
    restored = store.action(first["generation_id"], "restore")
    assert restored["status"] == "ready"


def test_upstream_receipt_is_privacy_minimized() -> None:
    job = normalize_video_job(
        {
            "contract": "learn.video-generation-job.v1",
            "generation_id": "vg-1",
            "status": "ready",
            "stage": "ready",
            "progress": 1,
            "title": "Lasso",
            "result": {
                "provider": "youtube",
                "provider_id": "abcdefghijk",
                "url": "https://www.youtube.com/watch?v=abcdefghijk",
                "token": "must-not-pass",
            },
            "execution": {
                "pipeline_version": "video-v1",
                "planner_model": "model/a",
                "renderer": "slides",
                "publish_provider": "youtube",
                "fallback_used": False,
                "secret_trace": "must-not-pass",
            },
            "internal_prompt": "must-not-pass",
        }
    )
    assert "internal_prompt" not in job
    assert "token" not in job["result"]
    assert "secret_trace" not in job["execution"]


def test_proxy_stub_mode_activates_generation_without_publication(monkeypatch) -> None:
    monkeypatch.setattr(app, "VIDEO_GENERATION_MODE", "stub")
    monkeypatch.setattr(app, "_VIDEO_GENERATION_CONFIG_ERROR", "")
    monkeypatch.setattr(app, "_VIDEO_GENERATION_STUB", StubVideoGenerationStore(ready_seconds=0))
    app._video_generation_rl.clear()
    with TestClient(app.app) as client:
        cap = client.get("/health").json()["capabilities"]["video_generation"]
        assert cap["enabled"] is True
        assert cap["test_mode"] is True
        assert cap["publishes_media"] is False
        assert cap["publish_provider"] == "none"
        assert cap["library_scope"] == "browser-local-receipts"
        assert cap["enumeration"] == "test-only"
        assert cap["list_endpoint"] == "/v1/video"
        created = client.post(
            "/v1/video-generations",
            json=_request(),
            headers={"Idempotency-Key": "idem-browser-1"},
        )
        assert created.status_code == 202
        body = created.json()
        assert body["status"] == "ready"
        assert body["result"]["test_mode"] is True
        again = client.post(
            "/v1/video-generations",
            json=_request(),
            headers={"Idempotency-Key": "idem-browser-1"},
        )
        assert again.json()["generation_id"] == body["generation_id"]
        archived = client.post(
            f"/v1/video-generations/{body['generation_id']}/archive"
        )
        assert archived.json()["status"] == "archived"



def test_proxy_upstream_mode_does_not_expose_global_job_enumeration(monkeypatch) -> None:
    monkeypatch.setattr(app, "VIDEO_GENERATION_MODE", "upstream")
    monkeypatch.setattr(app, "_VIDEO_GENERATION_CONFIG_ERROR", "")
    monkeypatch.setattr(app, "_VIDEO_GENERATION_UPSTREAM_URL", "https://video.example.test/v1/video-generations")
    with TestClient(app.app) as client:
        cap = client.get("/health").json()["capabilities"]["video_generation"]
        assert cap["enabled"] is True
        assert cap["enumeration"] == "disabled"
        assert cap["list_endpoint"] is None
        response = client.get("/v1/video-generations")
        assert response.status_code == 405
        assert response.json()["code"] == "VIDEO_ENUMERATION_DISABLED"

def test_proxy_disabled_mode_keeps_execution_closed(monkeypatch) -> None:
    monkeypatch.setattr(app, "VIDEO_GENERATION_MODE", "disabled")
    monkeypatch.setattr(app, "_VIDEO_GENERATION_CONFIG_ERROR", "")
    with TestClient(app.app) as client:
        cap = client.get("/health").json()["capabilities"]["video_generation"]
        assert cap["enabled"] is False
        response = client.post(
            "/v1/video-generations",
            json=_request(),
            headers={"Idempotency-Key": "idem-disabled"},
        )
        assert response.status_code == 503

class _Leaky(VideoGenerationError):
    """An instance whose own rendering is not what its author approved."""

    def __str__(self) -> str:
        return "Traceback (most recent call last): /srv/app/secret.py line 1"


def test_error_response_exposes_the_authored_message_never_the_rendering():
    """
    The response carries ``code`` and the authored ``message``.

    ``str(exc)`` is the exception machinery's rendering of the instance; a
    subclass, a note or a chained cause can put a path or a trace there. The
    handler must not forward it.
    """
    response = app._video_generation_error_response(_Leaky("VIDEO_JOB_NOT_FOUND", "The video job was not found."))
    body = json.loads(response.body)
    assert "Traceback" not in response.body.decode("utf-8")
    assert "secret.py" not in response.body.decode("utf-8")
    assert body["error"]["code"] == 'VIDEO_JOB_NOT_FOUND'
    assert body["error"]["message"] == 'The video job was not found.'


def test_error_message_attribute_is_the_authored_sentence():
    assert VideoGenerationError("VIDEO_JOB_NOT_FOUND", "The video job was not found.").message == 'The video job was not found.'
