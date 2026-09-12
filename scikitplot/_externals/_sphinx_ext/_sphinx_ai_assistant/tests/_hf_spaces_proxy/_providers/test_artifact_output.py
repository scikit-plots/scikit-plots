# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Run 143 — first-class provider-generated binary artifact output."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json

import httpx
from fastapi.testclient import TestClient

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.artifact_output import (
    OpenAIProviderArtifactOutputExecutor,
    ProviderArtifactOutputRegistry,
    StubProviderArtifactOutputExecutor,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._provider_artifact import (
    PROVIDER_ARTIFACT_CONTRACT,
    PROVIDER_ARTIFACT_RECEIPT_CONTRACT,
    ProviderArtifactError,
    parse_provider_artifact_request,
    validate_provider_artifact_signature,
)


def _body(**overrides) -> bytes:
    raw = {
        "contract": PROVIDER_ARTIFACT_CONTRACT,
        "generator_id": "stub/generated-png",
        "kind": "image",
        "prompt": "A compact blue chart icon",
        "mime_type": "image/png",
        "options": {},
    }
    raw.update(overrides)
    return json.dumps(raw).encode()


def test_request_contract_is_strict_and_does_not_accept_archive_authority() -> None:
    parsed = parse_provider_artifact_request(_body())
    assert parsed.generator_id == "stub/generated-png"
    assert parsed.kind == "image"
    assert parsed.prompt_sha256 == hashlib.sha256(parsed.prompt.encode()).hexdigest()

    raw = json.loads(_body())
    raw["path"] = "pkg/logo.png"
    try:
        parse_provider_artifact_request(json.dumps(raw).encode())
    except ProviderArtifactError as exc:
        assert exc.code == "PROVIDER_ARTIFACT_REQUEST_INVALID"
    else:  # pragma: no cover
        raise AssertionError("provider output unexpectedly accepted ZIP path authority")


def test_input_capability_does_not_create_output_generator() -> None:
    registry = ProviderArtifactOutputRegistry()
    assert registry.public_specs() == ()
    assert registry.spec("openai/gpt-image-2") is None


def test_stub_generators_are_real_verified_binary_artifacts() -> None:
    registry = ProviderArtifactOutputRegistry()
    registry.register(StubProviderArtifactOutputExecutor())
    image_request = parse_provider_artifact_request(_body())
    image = asyncio.run(registry.generate(image_request))
    try:
        image.file.seek(0)
        payload = image.file.read()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert image.receipt.kind == "image"
        assert image.receipt.output_sha256 == hashlib.sha256(payload).hexdigest()
        assert image.receipt.prompt_sha256 == image_request.prompt_sha256
    finally:
        image.close()

    audio_request = parse_provider_artifact_request(
        _body(
            generator_id="stub/generated-wav",
            kind="audio",
            mime_type="audio/wav",
            prompt="A short diagnostic tone",
        )
    )
    audio = asyncio.run(registry.generate(audio_request))
    try:
        audio.file.seek(0)
        payload = audio.file.read(12)
        assert payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"
    finally:
        audio.close()


def test_signature_verification_rejects_mime_confusion() -> None:
    import io

    fake = io.BytesIO(b"not a png")
    try:
        validate_provider_artifact_signature(fake, "image/png", len(fake.getvalue()))
    except ProviderArtifactError as exc:
        assert exc.code == "PROVIDER_ARTIFACT_OUTPUT_TYPE_MISMATCH"
    else:  # pragma: no cover
        raise AssertionError("mime-confused generated output unexpectedly accepted")


def test_provider_output_endpoint_returns_binary_with_bounded_provenance(monkeypatch) -> None:
    app._provider_artifact_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        response = client.post(
            "/v1/artifacts/provider-output",
            content=_body(),
            headers={"content-type": "application/json"},
        )
        caps = client.get("/health").json()["capabilities"]["provider_artifact_output"]
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["x-ai-artifact-contract"] == PROVIDER_ARTIFACT_RECEIPT_CONTRACT
    assert response.headers["x-ai-artifact-sha256"] == hashlib.sha256(response.content).hexdigest()
    receipt = json.loads(response.headers["x-ai-artifact-receipt"])
    assert receipt["contract"] == PROVIDER_ARTIFACT_RECEIPT_CONTRACT
    assert receipt["provider"] == "stub"
    assert receipt["generator_id"] == "stub/generated-png"
    assert receipt["output_size"] == len(response.content)
    assert "A compact blue chart icon" not in response.headers["x-ai-artifact-receipt"]
    assert "request_id" not in response.headers["x-ai-artifact-receipt"]
    ids = {row["id"] for row in caps["generators"]}
    assert "stub/generated-png" in ids and "stub/generated-wav" in ids
    assert caps["chat_text_is_output_authority"] is False
    assert caps["resource_input_is_output_authority"] is False


def test_provider_output_endpoint_rejects_unknown_generator(monkeypatch) -> None:
    app._provider_artifact_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        response = client.post(
            "/v1/artifacts/provider-output",
            content=_body(generator_id="unknown/generator"),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 404
    assert response.json()["code"] == "PROVIDER_ARTIFACT_GENERATOR_UNAVAILABLE"


def test_openai_image_executor_uses_dedicated_image_api_and_binds_output() -> None:
    png_registry = ProviderArtifactOutputRegistry()
    stub = ProviderArtifactOutputRegistry()
    stub.register(StubProviderArtifactOutputExecutor())
    req = parse_provider_artifact_request(_body())
    fixture = asyncio.run(stub.generate(req))
    try:
        fixture.file.seek(0)
        png = fixture.file.read()
    finally:
        fixture.close()

    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert str(request.url) == "https://api.openai.com/v1/images/generations"
        doc = json.loads(request.content)
        assert doc["model"] == "gpt-image-2"
        assert doc["output_format"] == "png"
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = OpenAIProviderArtifactOutputExecutor(api_key="secret", client=client, timeout_seconds=30)
            png_registry.register(executor)
            request = parse_provider_artifact_request(
                _body(generator_id="openai/gpt-image-2", options={"quality": "low", "size": "1024x1024"})
            )
            return await png_registry.generate(request)

    artifact = asyncio.run(run())
    try:
        artifact.file.seek(0)
        assert artifact.file.read() == png
        assert artifact.receipt.provider == "openai"
        assert artifact.receipt.model == "gpt-image-2"
    finally:
        artifact.close()
    assert len(seen) == 1


def test_openai_tts_executor_streams_dedicated_speech_api() -> None:
    # Minimal valid WAV header/body sufficient for the server signature boundary.
    import io
    import wave

    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000); wav.writeframes(b"\x00\x00" * 80)
    wav_bytes = out.getvalue()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/audio/speech"
        doc = json.loads(request.content)
        assert doc["model"] == "gpt-4o-mini-tts"
        assert doc["response_format"] == "wav"
        assert doc["voice"] == "coral"
        return httpx.Response(200, content=wav_bytes, headers={"content-type": "audio/wav"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            registry = ProviderArtifactOutputRegistry()
            registry.register(OpenAIProviderArtifactOutputExecutor(api_key="secret", client=client, timeout_seconds=30))
            request = parse_provider_artifact_request(
                _body(
                    generator_id="openai/gpt-4o-mini-tts",
                    kind="audio",
                    mime_type="audio/wav",
                    prompt="Say hello",
                    options={"voice": "coral", "instructions": "Speak clearly."},
                )
            )
            return await registry.generate(request)

    artifact = asyncio.run(run())
    try:
        artifact.file.seek(0)
        assert artifact.file.read() == wav_bytes
        assert artifact.receipt.model == "gpt-4o-mini-tts"
    finally:
        artifact.close()


def test_public_capability_does_not_expose_credentials(monkeypatch) -> None:
    monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "super-secret-token")
    monkeypatch.setattr(app, "PROVIDER_ARTIFACT_OPENAI_TOKEN", "artifact-secret-token")
    registry = ProviderArtifactOutputRegistry()
    registry.register(StubProviderArtifactOutputExecutor())
    monkeypatch.setattr(app, "_PROVIDER_ARTIFACT_OUTPUT_REGISTRY", registry)
    rendered = json.dumps(app._public_capabilities()["provider_artifact_output"], sort_keys=True)
    assert "super-secret-token" not in rendered
    assert "artifact-secret-token" not in rendered
    assert "Authorization" not in rendered
    assert "api_key" not in rendered


def test_stub_generators_are_marked_diagnostic() -> None:
    rows = StubProviderArtifactOutputExecutor().specs()
    assert rows
    assert all(row.diagnostic is True for row in rows)
    assert all(row.as_public_dict()["diagnostic"] is True for row in rows)


def test_openai_output_requires_separate_explicit_deployment_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "openai")
    monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "chat-token")
    monkeypatch.setattr(app, "BACKEND_URL", "https://api.openai.com/v1/chat/completions")
    monkeypatch.setattr(app, "PROVIDER_ARTIFACT_OUTPUT_ADAPTERS", ())
    monkeypatch.setattr(app, "PROVIDER_ARTIFACT_OPENAI_TOKEN", "")
    assert app._openai_resource_executor_configured() is True
    assert app._openai_provider_artifact_output_configured() is False

    # Enabling the adapter alone is still insufficient: output uses a separate
    # server credential and never silently reuses the chat/resource token.
    monkeypatch.setattr(app, "PROVIDER_ARTIFACT_OUTPUT_ADAPTERS", ("openai",))
    assert app._openai_provider_artifact_output_configured() is False
    monkeypatch.setattr(app, "PROVIDER_ARTIFACT_OPENAI_TOKEN", "artifact-token")
    assert app._openai_provider_artifact_output_configured() is True

    monkeypatch.setattr(app, "BACKEND_URL", "https://example.com/v1/chat/completions")
    assert app._openai_provider_artifact_output_configured() is False
