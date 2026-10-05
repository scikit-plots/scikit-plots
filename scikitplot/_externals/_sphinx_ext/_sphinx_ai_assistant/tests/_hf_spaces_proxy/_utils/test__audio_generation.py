# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Audio generation workflow authority and proxy integration."""

from __future__ import annotations

import asyncio
import json
import time

from fastapi.testclient import TestClient

from ...._hf_spaces_proxy import app
from ...._hf_spaces_proxy._providers.artifact_output import (
    ProviderArtifactOutputRegistry,
    StubProviderArtifactOutputExecutor,
)
from ...._hf_spaces_proxy._utils._audio_generation import (
    AUDIO_GENERATION_REQUEST_CONTRACT,
    AudioGenerationError,
    AudioGenerationService,
    parse_audio_generation_request,
)


def _request(*, model: str = "stub/mirror", **extra) -> dict:
    body = {
        "contract": AUDIO_GENERATION_REQUEST_CONTRACT,
        "text": "Read this short explanation aloud.",
        "selected_model": model,
        "mode": "narration",
    }
    body.update(extra)
    return body


def test_audio_request_contract_rejects_renderer_authority() -> None:
    parsed = parse_audio_generation_request(json.dumps(_request()).encode())
    assert parsed.mode == "narration"
    assert parsed.selected_model == "stub/mirror"

    for key, value in (
        ("provider", "openai"),
        ("generator_id", "stub/generated-wav"),
        ("deployment_id", "prod"),
        ("upstream_url", "https://example.test"),
        ("voice", "anything"),
    ):
        try:
            parse_audio_generation_request(
                json.dumps(_request(**{key: value})).encode()
            )
        except AudioGenerationError as exc:
            assert exc.code == "REQUEST_INVALID"
        else:  # pragma: no cover
            raise AssertionError(f"audio request unexpectedly accepted {key}")


def test_service_is_idempotent_and_capability_scoped() -> None:
    registry = ProviderArtifactOutputRegistry()
    registry.register(StubProviderArtifactOutputExecutor())
    service = AudioGenerationService(
        registry=registry,
        model_allowed=lambda model: model == "stub/mirror",
    )
    request = parse_audio_generation_request(json.dumps(_request()).encode())

    async def _run() -> None:
        first = await service.submit(request, "audio-idem-1")
        replay = await service.submit(request, "audio-idem-1")
        assert replay.generation_id == first.generation_id
        assert replay.capability == first.capability

        try:
            await service.get(first.generation_id, "wrong")
        except AudioGenerationError as exc:
            assert exc.code == "JOB_NOT_ACCESSIBLE"
        else:  # pragma: no cover
            raise AssertionError("wrong generation capability was accepted")

        deadline = time.monotonic() + 2.0
        artifact = None
        while time.monotonic() < deadline:
            job, artifact = await service.get(first.generation_id, first.capability)
            if job.state == "ready":
                break
            await asyncio.sleep(0.01)
        assert job.state == "ready"
        assert artifact is not None
        assert artifact.data[:4] == b"RIFF"
        assert artifact.data[8:12] == b"WAVE"

        try:
            await service.artifact(artifact.artifact_id, "wrong")
        except AudioGenerationError as exc:
            assert exc.code == "ARTIFACT_NOT_ACCESSIBLE"
        else:  # pragma: no cover
            raise AssertionError("wrong artifact capability was accepted")

        got = await service.artifact(artifact.artifact_id, artifact.capability)
        assert got.sha256 == artifact.sha256

        changed = parse_audio_generation_request(
            json.dumps(_request(text="Different narration text.")).encode()
        )
        try:
            await service.submit(changed, "audio-idem-1")
        except AudioGenerationError as exc:
            assert exc.code == "IDEMPOTENCY_CONFLICT"
        else:  # pragma: no cover
            raise AssertionError("idempotency key was rebound to another request")

    asyncio.run(_run())


def test_proxy_audio_stub_lifecycle_and_artifact_access(monkeypatch) -> None:
    app._audio_generation_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        cap = health.json()["capabilities"]["audio_generation"]
        assert cap["enabled"] is True
        assert cap["endpoint"] == "/v1/audio"
        assert cap["legacy_endpoint"] == "/v1/audio-generations"
        assert cap["operation"] == "speech.synthesize"
        assert cap["job_authority"]["shared"] is False
        assert cap["job_authority"]["durable"] is False

        response = client.post(
            "/v1/audio-generations",
            json=_request(model=app.ALLOWED_MODELS[0]),
            headers={"Idempotency-Key": "audio-route-1"},
        )
        assert response.status_code in {200, 202}, response.text
        job = response.json()
        generation_id = job["generation_id"]
        capability = job["generation_capability"]

        denied = client.get(f"/v1/audio-generations/{generation_id}")
        assert denied.status_code == 404

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status = client.get(
                f"/v1/audio-generations/{generation_id}",
                headers={"X-Generation-Capability": capability},
            )
            assert status.status_code == 200, status.text
            job = status.json()
            if job["execution"]["state"] == "ready":
                break
            time.sleep(0.02)
        assert job["execution"]["state"] == "ready", job
        artifact = job["artifacts"][0]

        denied_artifact = client.get(
            f"/v1/generated-artifacts/{artifact['artifact_id']}"
        )
        assert denied_artifact.status_code == 404

        media = client.get(
            f"/v1/generated-artifacts/{artifact['artifact_id']}",
            headers={"X-Artifact-Capability": artifact["artifact_capability"]},
        )
        assert media.status_code == 200
        assert media.headers["content-type"].startswith("audio/wav")
        assert media.content[:4] == b"RIFF"
        assert media.content[8:12] == b"WAVE"
        assert media.headers["x-content-type-options"] == "nosniff"

        replay = client.post(
            "/v1/audio-generations",
            json=_request(model=app.ALLOWED_MODELS[0]),
            headers={"Idempotency-Key": "audio-route-1"},
        )
        assert replay.status_code == 200
        assert replay.json()["generation_id"] == generation_id


def test_proxy_audio_canonical_create_route(monkeypatch) -> None:
    app._audio_generation_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        response = client.post(
            "/v1/audio",
            json=_request(model=app.ALLOWED_MODELS[0]),
            headers={"Idempotency-Key": "audio-canonical-route-1"},
        )
    assert response.status_code in {200, 202}, response.text
    assert response.json()["contract"]

class _Leaky(AudioGenerationError):
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
    response = app._audio_generation_error_response(_Leaky("QUEUE_FULL"))
    body = json.loads(response.body)
    assert "Traceback" not in response.body.decode("utf-8")
    assert "secret.py" not in response.body.decode("utf-8")
    assert body["error"]["code"] == 'QUEUE_FULL'
    assert body["error"]["message"] == 'Audio generation request could not be completed.'


def test_error_message_attribute_is_the_authored_sentence():
    assert AudioGenerationError("QUEUE_FULL").message == ''
