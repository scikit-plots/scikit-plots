from __future__ import annotations

import hashlib
import io
import json

import httpx
import pytest
from starlette.datastructures import UploadFile

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.base import ResourceRoute
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.huggingface import HuggingFaceResourceExecutor
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.policy import capabilities_for, huggingface_asr_supported_model
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._chat_contract import ChatRequest
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import ResourceDescriptor, VerifiedResource
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_transport import ResourceUpload


def _upload(data: bytes, *, mime="audio/wav", rid="aud", name="a.wav") -> ResourceUpload:
    desc = ResourceDescriptor(rid, name, mime, len(data), "audio", "raw")
    verified = VerifiedResource(
        desc, len(data), hashlib.sha256(data).hexdigest(), mime, "audio", "test"
    )
    return ResourceUpload(
        verified,
        UploadFile(
            io.BytesIO(data), filename=name, headers={"content-type": mime}
        ),
    )


def _chat(model="openai/whisper-large-v3", *, stream=False):
    return ChatRequest(model=model, user_message="transcribe", page_text="", page_descriptor="", max_tokens=256, stream=stream, effort=None, thinking=False, budget_tokens=None, resources=())


def test_asr_model_policy_is_task_specific_and_fail_closed() -> None:
    assert huggingface_asr_supported_model("openai/whisper-large-v3")
    assert not huggingface_asr_supported_model("openai/whisper-large-v3:fastest")
    caps = capabilities_for(adapter="huggingface", model="openai/whisper-large-v3")
    assert caps.routes_for("audio") == ("native",)
    assert caps.routes_for("text") == ("unsupported",)
    assert caps.routes_for("image") == ("unsupported",)
    assert capabilities_for(
        adapter="huggingface", model="zai-org/GLM-4.5V"
    ).routes_for("image") == ("native",)


@pytest.mark.asyncio
async def test_asr_prepare_preserves_raw_audio_and_cleanup_revokes_it() -> None:
    raw = b"RIFF\x10\x00\x00\x00WAVEraw-audio"
    upload = _upload(raw)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200))
    )
    ex = HuggingFaceResourceExecutor(api_key="hf_private", client=client)
    try:
        handle = await ex.prepare(
            model="openai/whisper-large-v3",
            route=ResourceRoute("aud", "native"),
            upload=upload,
        )
        assert handle.opaque["task"] == "asr" and handle.opaque["audio"] == raw
        assert "hf_private" not in repr(handle.opaque)
        await ex.release(handle)
        assert handle.released and handle.opaque == {}
    finally:
        await upload.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_asr_request_uses_exact_hf_inference_model_url_and_raw_body() -> None:
    raw = b"ID3raw-mp3"
    upload = _upload(raw, mime="audio/mpeg", name="a.mp3")
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["ct"] = request.headers.get("content-type")
        seen["body"] = await request.aread()
        return httpx.Response(200, json={"text": "hello world"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = HuggingFaceResourceExecutor(api_key="hf_private", client=client)
    try:
        handle = await ex.prepare(
            model="openai/whisper-large-v3",
            route=ResourceRoute("aud", "native"),
            upload=upload,
        )
        upstream = await ex.open_chat(chat=_chat(), handles=[handle])
        out = await ex.buffered_chat_response(upstream, maximum=4096)
        assert out["choices"][0]["message"]["content"] == "hello world"
        assert (
            seen["url"]
            == "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
        )
        assert (
            seen["auth"] == "Bearer hf_private"
            and seen["ct"] == "audio/mpeg"
            and seen["body"] == raw
        )
    finally:
        await upload.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_asr_model_without_audio_does_not_fall_through_to_chat_completion() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = HuggingFaceResourceExecutor(api_key="k", client=client)
    try:
        with pytest.raises(Exception, match="requires exactly one audio"):
            await ex.open_chat(chat=_chat(), handles=[])
        assert calls == 0
    finally:
        await client.aclose()


def test_asr_route_constraints_include_single_file_size_and_mime_authority() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(200))
    )
    ex = HuggingFaceResourceExecutor(api_key="k", client=client)
    try:
        assert ex.executable_routes("openai/whisper-large-v3") == {
            "audio": ("native",)
        }
        spec = ex.route_constraints("openai/whisper-large-v3")["audio"]["native"]
        assert spec["max_files"] == 1 and spec["max_file_bytes"] == 25 * 1024 * 1024
        assert {"audio/wav", "audio/mpeg", "audio/flac", "audio/ogg"} <= set(
            spec["mime_types"]
        )
        assert "video" not in ex.executable_routes("openai/whisper-large-v3")
    finally:
        import asyncio

        asyncio.run(client.aclose())


def test_server_route_constraint_gate_rejects_second_audio_before_prepare() -> None:
    class Ex:
        def route_constraints(self, model):
            del model
            return {
                "audio": {
                    "native": {
                        "max_files": 1,
                        "max_file_bytes": 25 * 1024 * 1024,
                        "mime_types": ["audio/wav"],
                    }
                }
            }

    u1 = _upload(b"1", rid="a1")
    u2 = _upload(b"2", rid="a2")
    routes = (
        ResourceRoute("a1", "native", metadata={"modality": "audio"}),
        ResourceRoute("a2", "native", metadata={"modality": "audio"}),
    )
    try:
        with pytest.raises(Exception, match="file-count limit"):
            app._enforce_executor_route_constraints(
                executor=Ex(),
                model="openai/whisper-large-v3",
                routes=routes,
                uploads=(u1, u2),
            )
    finally:
        import asyncio

        asyncio.run(u1.close())
        asyncio.run(u2.close())


def test_public_health_sanitizes_and_exposes_asr_max_files(monkeypatch) -> None:
    monkeypatch.setattr(app, "BACKEND_URL", "")
    monkeypatch.setattr(app, "HF_SPACES_MODEL_URL", "")
    monkeypatch.setattr(app, "HF_TOKEN", "hf_private")
    monkeypatch.setattr(app, "HF_BASE", "https://router.huggingface.co")
    app._configure_resource_executors(object())
    try:
        doc = app._resource_capability_doc("openai/whisper-large-v3")
        assert doc["execution"] == "enabled" and doc["routes"]["audio"] == ["native"]
        spec = doc["route_constraints"]["audio"]["native"]
        assert spec["max_files"] == 1 and spec["max_file_bytes"] == 25 * 1024 * 1024
        assert "audio/wav" in spec["mime_types"]
        assert doc["routes"]["text"] == ["unsupported"]
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("huggingface")


def test_app_passes_exact_asr_deployment_allowlist_to_executor(monkeypatch) -> None:
    class Fake:
        name = "huggingface"
        enabled = True

        def __init__(self, **kw):
            self.kw = kw

        def executable_routes(self, model):
            del model
            return {"audio": ("native",)}

    monkeypatch.setattr(app, "HuggingFaceResourceExecutor", Fake)
    monkeypatch.setattr(app, "BACKEND_URL", "")
    monkeypatch.setattr(app, "HF_TOKEN", "k")
    monkeypatch.setattr(app, "HF_BASE", "https://router.huggingface.co")
    monkeypatch.setattr(app, "HF_RESOURCE_VLM_MODELS", ())
    monkeypatch.setattr(app, "HF_RESOURCE_ASR_MODELS", ("org/custom-asr",))
    try:
        app._configure_resource_executors(object())
        ex = app._RESOURCE_EXECUTOR_REGISTRY.get("huggingface")
        assert ex.kw["asr_models"] == ("org/custom-asr",)
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("huggingface")
