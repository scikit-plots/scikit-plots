# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Hugging Face task-aware Chat Completion resource executor.

Inference Providers are heterogeneous: the OpenAI-compatible Chat Completion
surface covers conversational LLMs and VLMs, while audio/video/other media
families use separate task APIs.  This executor therefore owns ordinary Path-3
chat, but enables raw resources only for exact reviewed/opted-in VLMs and
raster-image inputs.  Browser-to-proxy transport remains raw multipart; the
provider-private ``data:`` URL exists only inside this adapter lifecycle.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Sequence
from urllib.parse import quote, urlparse

import httpx

from .executor import ProviderPrivateResource, ResourceExecutionError
from .policy import huggingface_asr_supported_model, huggingface_vlm_supported_model
from .registry import StaticProviderAdapter

try:
    from .._utils._chat_contract import ChatRequest, build_upstream_payload
    from .._utils._resource_transport import ResourceUpload
except ImportError:  # standalone HF Space: _providers is top-level
    from _utils._chat_contract import ChatRequest, build_upstream_payload
    from _utils._resource_transport import ResourceUpload

_HF_ROUTER_BASE = "https://router.huggingface.co"
_HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_IMAGE_BYTES = 24 * 1024 * 1024
_MAX_SSE_EVENT_BYTES = 1024 * 1024
_MAX_ASR_AUDIO_BYTES = 25 * 1024 * 1024
_AUDIO_MIMES = frozenset(
    {
        "audio/wav",
        "audio/mpeg",
        "audio/flac",
        "audio/ogg",
        "audio/mp4",
        "audio/aac",
        "audio/webm",
    }
)
_MODEL_ID_RE = __import__("re").compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,159}$"
)
_IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/webp"})


class HuggingFaceAdapter(StaticProviderAdapter):
    def __init__(self, **kwargs):
        super().__init__("huggingface", **kwargs)


def official_huggingface_router_base(value: str) -> bool:
    """Return whether *value* is exactly the official credential destination."""
    raw = str(value or "").strip().rstrip("/")
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False
    return bool(
        raw == _HF_ROUTER_BASE
        and parsed.scheme == "https"
        and parsed.hostname == "router.huggingface.co"
        and parsed.username is None
        and parsed.password is None
        and parsed.port in {None, 443}
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def _bounded_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _chat_text(payload: object) -> str:
    if not isinstance(payload, dict) or payload.get("error"):
        raise ResourceExecutionError(
            "Hugging Face Chat Completion reported a provider failure"
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ResourceExecutionError(
            "Hugging Face Chat Completion returned invalid metadata"
        )
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ResourceExecutionError(
            "Hugging Face Chat Completion returned no assistant text"
        )
    text = message["content"]
    if not text:
        raise ResourceExecutionError(
            "Hugging Face Chat Completion returned no assistant text"
        )
    return text


def _private_image(handle: ProviderPrivateResource) -> tuple[str, str]:
    opaque = handle.opaque if isinstance(handle.opaque, dict) else {}
    data_url = opaque.get("data_url")
    mime = opaque.get("mime")
    if (
        handle.route != "native"
        or not isinstance(data_url, str)
        or not data_url.startswith("data:image/")
        or not isinstance(mime, str)
        or mime not in _IMAGE_MIMES
    ):
        raise ResourceExecutionError("Hugging Face private VLM image handle is invalid")
    return data_url, mime


def _private_audio(handle: ProviderPrivateResource) -> tuple[bytes, str]:
    opaque = handle.opaque if isinstance(handle.opaque, dict) else {}
    raw = opaque.get("audio")
    mime = opaque.get("mime")
    if (
        handle.route != "native"
        or opaque.get("task") != "asr"
        or not isinstance(raw, (bytes, bytearray))
        or not isinstance(mime, str)
        or mime not in _AUDIO_MIMES
    ):
        raise ResourceExecutionError("Hugging Face private ASR audio handle is invalid")
    return bytes(raw), mime


def _asr_url(model: str) -> str:
    raw = str(model or "").strip()
    if not _MODEL_ID_RE.fullmatch(raw):
        raise ResourceExecutionError("Hugging Face ASR model id is invalid")
    encoded = "/".join(quote(part, safe="-._~") for part in raw.split("/", 1))
    return f"{_HF_ROUTER_BASE}/hf-inference/models/{encoded}"


class HuggingFaceResourceExecutor:
    """Execute Path-3 HF chat and exact VLM raster-image resources."""

    name = "huggingface"
    enabled = True

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient,
        base_url: str = _HF_ROUTER_BASE,
        response_timeout_seconds: float = 120.0,
        vlm_models: Sequence[str] = (),
        asr_models: Sequence[str] = (),
    ) -> None:
        if not api_key:
            raise ValueError(
                "Hugging Face resource executor requires an inference token"
            )
        if not official_huggingface_router_base(base_url):
            raise ValueError(
                "Hugging Face resource executor requires the official router origin"
            )
        self._api_key = str(api_key)
        self._client = client
        self._response_timeout = max(1.0, float(response_timeout_seconds))
        self._vlm_models = frozenset(
            str(row).strip() for row in vlm_models if str(row).strip()
        )
        self._asr_models = frozenset(
            str(row).strip() for row in asr_models if str(row).strip()
        )

    def _vlm(self, model: str) -> bool:
        return huggingface_vlm_supported_model(model, extra_models=self._vlm_models)

    def _asr(self, model: str) -> bool:
        return huggingface_asr_supported_model(model, extra_models=self._asr_models)

    def executable_routes(self, model: str) -> dict[str, tuple[str, ...]]:
        out: dict[str, tuple[str, ...]] = {}
        if self._vlm(model):
            out["image"] = ("native",)
        if self._asr(model):
            out["audio"] = ("native",)
        return out

    def route_constraints(self, model: str) -> dict[str, dict[str, dict]]:
        out: dict[str, dict[str, dict]] = {}
        if self._vlm(model):
            out["image"] = {
                "native": {
                    "max_file_bytes": _MAX_IMAGE_BYTES,
                    "mime_types": sorted(_IMAGE_MIMES),
                }
            }
        if self._asr(model):
            out["audio"] = {
                "native": {
                    "max_files": 1,
                    "max_file_bytes": _MAX_ASR_AUDIO_BYTES,
                    "mime_types": sorted(_AUDIO_MIMES),
                }
            }
        return out

    async def prepare(
        self, *, model: str, route, upload: ResourceUpload
    ) -> ProviderPrivateResource:
        verified = upload.verified
        mime = str(verified.detected_mime or "").lower()
        size = int(verified.actual_size)
        if route.route != "native":
            raise ResourceExecutionError("Hugging Face resource route is unsupported")
        if verified.detected_modality == "audio":
            if not self._asr(model) or mime not in _AUDIO_MIMES:
                raise ResourceExecutionError(
                    "Hugging Face ASR audio route is unsupported"
                )
            if size <= 0 or size > _MAX_ASR_AUDIO_BYTES:
                raise ResourceExecutionError(
                    "Hugging Face ASR audio exceeds the bounded 25 MiB limit"
                )
            await upload.upload.seek(0)
            raw = await upload.upload.read(_MAX_ASR_AUDIO_BYTES + 1)
            await upload.upload.seek(0)
            if len(raw) != size or len(raw) > _MAX_ASR_AUDIO_BYTES:
                raise ResourceExecutionError(
                    "Hugging Face ASR audio bytes failed bounded verification"
                )
            return ProviderPrivateResource(
                resource_id=verified.id,
                route="native",
                provider=self.name,
                source_sha256=verified.sha256,
                source_size=size,
                opaque={
                    "task": "asr",
                    "audio": bytes(raw),
                    "mime": mime,
                    "modality": "audio",
                },
                cleanup_required=True,
            )
        if not self._vlm(model) or verified.detected_modality != "image":
            raise ResourceExecutionError("Hugging Face VLM image route is unsupported")
        if mime not in _IMAGE_MIMES:
            raise ResourceExecutionError("Hugging Face VLM image MIME is unsupported")
        if size <= 0 or size > _MAX_IMAGE_BYTES:
            raise ResourceExecutionError(
                "Hugging Face VLM image exceeds the bounded 8 MiB limit"
            )
        await upload.upload.seek(0)
        raw = await upload.upload.read(_MAX_IMAGE_BYTES + 1)
        await upload.upload.seek(0)
        if len(raw) != size or len(raw) > _MAX_IMAGE_BYTES:
            raise ResourceExecutionError(
                "Hugging Face VLM image bytes failed bounded verification"
            )
        data_url = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
        return ProviderPrivateResource(
            resource_id=verified.id,
            route="native",
            provider=self.name,
            source_sha256=verified.sha256,
            source_size=size,
            opaque={
                "task": "vlm",
                "data_url": data_url,
                "mime": mime,
                "modality": "image",
            },
            cleanup_required=True,
        )

    async def release(self, handle: ProviderPrivateResource) -> None:
        if handle.released:
            return
        if isinstance(handle.opaque, dict):
            handle.opaque.clear()
        handle.released = True

    def chat_payload(
        self, chat: ChatRequest, handles: Sequence[ProviderPrivateResource]
    ) -> dict:
        payload = build_upstream_payload(chat, reasoning_enabled=False)
        if handles:
            total = sum(int(row.source_size) for row in handles)
            if total > _MAX_TOTAL_IMAGE_BYTES:
                raise ResourceExecutionError(
                    "Hugging Face VLM images exceed the bounded 24 MiB request limit"
                )
            messages = payload.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) < 2  # ruff: ignore[magic-value-comparison]
                or not isinstance(messages[1], dict)
            ):
                raise ResourceExecutionError(
                    "Hugging Face Chat payload authority is invalid"
                )
            user_text = str(messages[1].get("content", ""))
            content: list[dict] = [{"type": "text", "text": user_text}]
            for handle in handles:
                data_url, _mime = _private_image(handle)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            messages[1]["content"] = content
        return payload

    async def open_chat(
        self, *, chat: ChatRequest, handles: Sequence[ProviderPrivateResource]
    ) -> httpx.Response:
        audio_handles = [
            row
            for row in handles
            if isinstance(row.opaque, dict) and row.opaque.get("task") == "asr"
        ]
        if audio_handles:
            if (
                len(handles) != 1
                or len(audio_handles) != 1
                or not self._asr(chat.model)
            ):
                raise ResourceExecutionError(
                    "Hugging Face ASR accepts exactly one audio resource"
                )
            raw, mime = _private_audio(audio_handles[0])
            request = self._client.build_request(
                "POST",
                _asr_url(chat.model),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": mime,
                    "Accept": "application/json",
                },
                content=raw,
                timeout=self._response_timeout,
            )
            try:
                return await self._client.send(request, stream=True)
            except (httpx.HTTPError, OSError) as exc:
                raise ResourceExecutionError("Hugging Face ASR request failed") from exc
        if self._asr(chat.model):
            raise ResourceExecutionError(
                "Hugging Face ASR requires exactly one audio resource"
            )
        payload = self.chat_payload(chat, handles)
        request = self._client.build_request(
            "POST",
            _HF_CHAT_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if chat.stream else "application/json",
            },
            content=_bounded_json_bytes(payload),
            timeout=self._response_timeout,
        )
        try:
            return await self._client.send(request, stream=True)
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "Hugging Face Chat Completion request failed"
            ) from exc

    async def buffered_chat_response(
        self, upstream: httpx.Response, *, maximum: int
    ) -> dict:
        total = 0
        chunks: list[bytes] = []
        async for chunk in upstream.aiter_bytes():
            total += len(chunk)
            if total > maximum:
                raise ResourceExecutionError(
                    "Hugging Face Chat Completion body exceeded the response limit"
                )
            chunks.append(chunk)
        try:
            payload = json.loads(b"".join(chunks))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "Hugging Face Chat Completion returned invalid JSON"
            ) from exc
        if isinstance(payload, dict) and isinstance(payload.get("text"), str):
            text = payload["text"]
            if not text:
                raise ResourceExecutionError(
                    "Hugging Face ASR returned no transcript text"
                )
        else:
            text = _chat_text(payload)
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "model": (
                str(payload.get("model", "") or "") if isinstance(payload, dict) else ""
            ),
        }

    async def chat_sse(  # ruff: ignore[too-many-branches]
        self, upstream: httpx.Response, *, maximum: int
    ) -> AsyncIterator[bytes]:
        total = 0
        buffer = bytearray()
        done = False
        async for chunk in upstream.aiter_bytes():
            total += len(chunk)
            if total > maximum:
                yield b'event: error\ndata: {"error":{"code":"UPSTREAM_RESPONSE_TOO_LARGE","message":"The upstream response exceeded the configured limit."}}\n\n'
                return
            buffer.extend(chunk)
            if (
                len(buffer) > _MAX_SSE_EVENT_BYTES
                and b"\n\n" not in buffer
                and b"\r\n\r\n" not in buffer
            ):
                yield b'event: error\ndata: {"error":{"code":"UPSTREAM_PROTOCOL_ERROR","message":"The upstream stream event exceeded the configured limit."}}\n\n'
                return
            while True:
                marker = buffer.find(b"\n\n")
                marker_len = 2
                if marker < 0:
                    marker = buffer.find(b"\r\n\r\n")
                    marker_len = 4
                if marker < 0:
                    break
                raw = bytes(buffer[:marker])
                del buffer[: marker + marker_len]
                if len(raw) > _MAX_SSE_EVENT_BYTES:
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_PROTOCOL_ERROR","message":"The upstream stream event exceeded the configured limit."}}\n\n'
                    return
                data_lines = [
                    line[5:].lstrip()
                    for line in raw.replace(b"\r\n", b"\n").split(b"\n")
                    if line.startswith(b"data:")
                ]
                if not data_lines:
                    continue
                data = b"\n".join(data_lines)
                if data == b"[DONE]":
                    done = True
                    break
                try:
                    doc = json.loads(data)
                except (json.JSONDecodeError, ValueError):
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_PROTOCOL_ERROR","message":"The upstream stream was malformed."}}\n\n'
                    return
                if not isinstance(doc, dict) or doc.get("error"):
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_SERVICE_ERROR","message":"The upstream AI provider did not complete the streamed response."}}\n\n'
                    return
                choices = doc.get("choices")
                if (
                    isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                ):
                    delta = choices[0].get("delta")
                    if (
                        isinstance(delta, dict)
                        and isinstance(delta.get("content"), str)
                        and delta["content"]
                    ):
                        frame = {"choices": [{"delta": {"content": delta["content"]}}]}
                        yield b"data: " + _bounded_json_bytes(frame) + b"\n\n"
            if done:
                break
        if not done:
            yield b'event: error\ndata: {"error":{"code":"UPSTREAM_PROTOCOL_ERROR","message":"The upstream stream ended before completion."}}\n\n'
            return
        yield b"data: [DONE]\n\n"
