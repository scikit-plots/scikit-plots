# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Anthropic Files + Messages resource executor.

Uploaded file identifiers are workspace-level provider capabilities and remain
strictly request-private.  They never enter browser responses, health metadata,
logs, receipts, persistence, or exports.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator, Sequence
from urllib.parse import quote, urlparse

import httpx

from .executor import ProviderPrivateResource, ResourceExecutionError
from .policy import anthropic_code_execution_supported_model
from .registry import StaticProviderAdapter

try:
    from .._utils._chat_contract import ChatRequest, build_upstream_payload
    from .._utils._resource_transport import ResourceUpload
except ImportError:  # standalone HF Space: _providers is top-level
    from _utils._chat_contract import ChatRequest, build_upstream_payload
    from _utils._resource_transport import ResourceUpload

_ANTHROPIC_BASE = "https://api.anthropic.com"
_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MIN_EXPIRES_SECONDS = 3600
_MAX_EXPIRES_SECONDS = 90 * 24 * 60 * 60
_MAX_FILE_BYTES = 500_000_000
_MAX_SSE_EVENT_BYTES = 1024 * 1024
_CODE_EXECUTION_TOOL = "code_execution_20260521"
_IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


class AnthropicAdapter(StaticProviderAdapter):
    def __init__(self, **kwargs):
        super().__init__("anthropic", **kwargs)


def _validated_anthropic_base(value: str) -> str:
    raw = str(value or _ANTHROPIC_BASE).rstrip("/")
    parsed = urlparse(raw)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.anthropic.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "Anthropic resource executor requires the official HTTPS API origin"
        )
    return _ANTHROPIC_BASE


def official_anthropic_messages_backend(value: str) -> bool:
    """Return whether Path-1 points exactly at the official Messages URL."""
    raw = str(value or "").strip().rstrip("/")
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "api.anthropic.com"
        and parsed.username is None
        and parsed.password is None
        and parsed.port in {None, 443}
        and not parsed.query
        and not parsed.fragment
        and parsed.path.rstrip("/") == "/v1/messages"
        and raw == _ANTHROPIC_MESSAGES_URL
    )


def _safe_provider_file_id(value: object) -> str:
    """Validate one opaque provider file ID without assuming a prefix/format."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512  # ruff: ignore[magic-value-comparison]
    ):
        raise ResourceExecutionError("Anthropic private file handle is invalid")
    if any(
        ord(ch) < 0x20 or ord(ch) == 0x7F  # ruff: ignore[magic-value-comparison]
        for ch in value
    ):
        raise ResourceExecutionError("Anthropic private file handle is invalid")
    if any(ch in value for ch in "/\\?#&"):
        raise ResourceExecutionError("Anthropic private file handle is invalid")
    return value


def _opaque_file_id(handle: ProviderPrivateResource) -> str:
    opaque = handle.opaque if isinstance(handle.opaque, dict) else {}
    return _safe_provider_file_id(opaque.get("file_id"))


def _upload_mime(upload: ResourceUpload) -> str:
    verified = upload.verified
    if verified.detected_modality == "text":
        return "text/plain"
    return verified.detected_mime or "application/octet-stream"


def _multipart_header(
    boundary: str, *, seconds: int, upload: ResourceUpload
) -> tuple[bytes, bytes]:
    verified = upload.verified
    safe_name = (
        verified.name.replace('"', "_").replace("\r", " ").replace("\n", " ")[:500]
        or "resource"
    )
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="expires_in_seconds"\r\n\r\n'
        f"{seconds}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
        f"Content-Type: {_upload_mime(upload)}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    return prefix, suffix


def _bounded_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _message_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ResourceExecutionError("Anthropic Messages returned invalid metadata")
    if payload.get("type") == "error" or payload.get("error"):
        raise ResourceExecutionError("Anthropic Messages reported a provider failure")
    pieces: list[str] = []
    content = payload.get("content")
    if isinstance(content, list):
        pieces.extend(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    text = "".join(pieces)
    if not text:
        raise ResourceExecutionError("Anthropic Messages returned no assistant text")
    return text


class AnthropicResourceExecutor:
    """Upload verified resources to Files and execute Claude Messages."""

    name = "anthropic"
    enabled = True

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient,
        base_url: str = _ANTHROPIC_BASE,
        expires_seconds: int = _MIN_EXPIRES_SECONDS,
        response_timeout_seconds: float = 600.0,
    ) -> None:
        key = str(api_key or "").strip()
        if not key:
            raise ValueError("Anthropic API key is required")
        self._api_key = key
        self._client = client
        self._base = _validated_anthropic_base(base_url)
        self._expires = max(
            _MIN_EXPIRES_SECONDS, min(int(expires_seconds), _MAX_EXPIRES_SECONDS)
        )
        self._response_timeout = httpx.Timeout(
            connect=20.0,
            read=max(30.0, min(float(response_timeout_seconds), 1800.0)),
            write=120.0,
            pool=20.0,
        )

    def executable_routes(self, model: str) -> dict[str, tuple[str, ...]]:
        routes: dict[str, tuple[str, ...]] = {
            "text": ("native",),
            "image": ("native",),
            "animated_image": ("native",),
            "document": ("native",),
        }
        if anthropic_code_execution_supported_model(model):
            for modality in ("vector_image", "data", "archive", "binary"):
                routes[modality] = ("tool",)
        return routes

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Accept": "application/json",
        }

    async def _body(self, prefix: bytes, suffix: bytes, upload: ResourceUpload):
        await upload.upload.seek(0)
        try:
            yield prefix
            while True:
                chunk = await upload.upload.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk
            yield suffix
        finally:
            await upload.upload.seek(0)

    async def prepare(
        self, *, model: str, route, upload: ResourceUpload
    ) -> ProviderPrivateResource:
        del model
        verified = upload.verified
        if route.resource_id != verified.id or route.route not in {"native", "tool"}:
            raise ResourceExecutionError(
                "Anthropic executor received an invalid resource route"
            )
        if int(verified.actual_size) > _MAX_FILE_BYTES:
            raise ResourceExecutionError(
                "Anthropic file exceeds the bounded 500 MB provider limit"
            )
        boundary = "scikitplots-" + secrets.token_hex(16)
        prefix, suffix = _multipart_header(
            boundary, seconds=self._expires, upload=upload
        )
        headers = self._headers()
        headers.update(
            {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(prefix) + verified.actual_size + len(suffix)),
            }
        )
        try:
            response = await self._client.post(
                f"{self._base}/v1/files",
                headers=headers,
                content=self._body(prefix, suffix, upload),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "Anthropic temporary file upload failed"
            ) from exc
        if (
            response.status_code < 200  # ruff: ignore[magic-value-comparison]
            or response.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError("Anthropic temporary file upload was rejected")
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "Anthropic temporary file upload returned invalid metadata"
            ) from exc
        if not isinstance(payload, dict):
            raise ResourceExecutionError(
                "Anthropic temporary file upload returned invalid metadata"
            )
        file_id = _safe_provider_file_id(payload.get("id"))
        returned_bytes = payload.get("size_bytes")
        if returned_bytes is not None:
            try:
                if int(returned_bytes) != int(verified.actual_size):
                    raise ResourceExecutionError(
                        "Anthropic temporary file size did not match source bytes"
                    )
            except (TypeError, ValueError) as exc:
                raise ResourceExecutionError(
                    "Anthropic temporary file size metadata was invalid"
                ) from exc
        return ProviderPrivateResource(
            resource_id=verified.id,
            route=route.route,
            provider=self.name,
            source_sha256=verified.sha256,
            source_size=verified.actual_size,
            opaque={
                "file_id": file_id,
                "modality": verified.detected_modality,
                "mime": _upload_mime(upload),
                "name": verified.name,
            },
            cleanup_required=True,
        )

    async def release(self, handle: ProviderPrivateResource) -> None:
        file_id = _opaque_file_id(handle)
        try:
            response = await self._client.delete(
                f"{self._base}/v1/files/{quote(file_id, safe='')}",
                headers=self._headers(),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "Anthropic temporary file cleanup failed"
            ) from exc
        if response.status_code == 404:  # ruff: ignore[magic-value-comparison]
            handle.released = True
            return
        if (
            response.status_code < 200  # ruff: ignore[magic-value-comparison]
            or response.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError(
                "Anthropic temporary file cleanup was rejected"
            )
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "Anthropic temporary file cleanup returned invalid metadata"
            ) from exc
        if not isinstance(payload, dict) or payload.get("type") != "file_deleted":
            raise ResourceExecutionError(
                "Anthropic temporary file cleanup was not confirmed"
            )
        handle.released = True

    def messages_payload(
        self, chat: ChatRequest, handles: Sequence[ProviderPrivateResource]
    ) -> dict:
        provider_body = build_upstream_payload(chat, reasoning_enabled=False)
        messages = provider_body.get("messages", [])
        system = str(messages[0].get("content", "")) if len(messages) > 0 else ""
        user_text = (
            str(messages[1].get("content", ""))
            if len(messages) > 1
            else chat.user_message
        )
        content: list[dict] = [{"type": "text", "text": user_text}]
        tool_needed = False

        for handle in handles:
            file_id = _opaque_file_id(handle)
            opaque = handle.opaque if isinstance(handle.opaque, dict) else {}
            modality = str(opaque.get("modality", ""))
            mime = str(opaque.get("mime", ""))
            if handle.route == "tool":
                if not anthropic_code_execution_supported_model(chat.model):
                    raise ResourceExecutionError(
                        "Anthropic Code Execution is not enabled for this model"
                    )
                content.append({"type": "container_upload", "file_id": file_id})
                tool_needed = True
                continue
            if handle.route != "native":
                raise ResourceExecutionError(
                    "Anthropic Messages received a non-executable route"
                )
            if modality in {"text", "document"}:
                if modality == "document" and mime != "application/pdf":
                    raise ResourceExecutionError(
                        "Anthropic native document route supports PDF or plain text only"
                    )
                content.append(
                    {"type": "document", "source": {"type": "file", "file_id": file_id}}
                )
                continue
            if modality in {"image", "animated_image"} and mime in _IMAGE_MIMES:
                content.append(
                    {"type": "image", "source": {"type": "file", "file_id": file_id}}
                )
                continue
            raise ResourceExecutionError(
                "Anthropic native route does not support this modality"
            )

        payload: dict = {
            "model": chat.model,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": chat.max_tokens,
            "stream": bool(chat.stream),
        }
        if tool_needed:
            payload["tools"] = [
                {"type": _CODE_EXECUTION_TOOL, "name": "code_execution"}
            ]
        return payload

    async def open_chat(
        self, *, chat: ChatRequest, handles: Sequence[ProviderPrivateResource]
    ) -> httpx.Response:
        payload = self.messages_payload(chat, handles)
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        request = self._client.build_request(
            "POST",
            f"{self._base}/v1/messages",
            headers=headers,
            content=_bounded_json_bytes(payload),
            timeout=self._response_timeout,
        )
        try:
            return await self._client.send(request, stream=True)
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError("Anthropic Messages request failed") from exc

    async def buffered_chat_response(
        self, upstream: httpx.Response, *, maximum: int
    ) -> dict:
        total = 0
        chunks: list[bytes] = []
        async for chunk in upstream.aiter_bytes():
            total += len(chunk)
            if total > maximum:
                raise ResourceExecutionError(
                    "Anthropic Messages body exceeded the configured response limit"
                )
            chunks.append(chunk)
        try:
            payload = json.loads(b"".join(chunks))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "Anthropic Messages returned invalid JSON"
            ) from exc
        text = _message_text(payload)
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
        completed = False
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
                yield b'event: error\ndata: {"error":{"code":"UPSTREAM_EVENT_TOO_LARGE","message":"The upstream stream event exceeded the configured limit."}}\n\n'
                return
            while True:
                pos_lf = buffer.find(b"\n\n")
                pos_crlf = buffer.find(b"\r\n\r\n")
                positions = [(pos_lf, 2), (pos_crlf, 4)]
                positions = [(pos, width) for pos, width in positions if pos >= 0]
                if not positions:
                    break
                pos, width = min(positions, key=lambda row: row[0])
                if pos > _MAX_SSE_EVENT_BYTES:
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_EVENT_TOO_LARGE","message":"The upstream stream event exceeded the configured limit."}}\n\n'
                    return
                raw = bytes(buffer[:pos])
                del buffer[: pos + width]
                event = ""
                data_lines: list[bytes] = []
                for line in raw.replace(b"\r\n", b"\n").split(b"\n"):
                    if line.startswith(b"event:"):
                        event = line[6:].strip().decode("utf-8", "replace")
                    elif line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip())
                if event == "message_stop":
                    completed = True
                    yield b"data: [DONE]\n\n"
                    return
                if event == "error":
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_PROVIDER_ERROR","message":"The upstream provider stream failed."}}\n\n'
                    return
                if not data_lines:
                    continue
                try:
                    payload = json.loads(b"\n".join(data_lines))
                except (json.JSONDecodeError, ValueError):
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_STREAM_INVALID","message":"The upstream provider stream was malformed."}}\n\n'
                    return
                if event == "content_block_delta" and isinstance(payload, dict):
                    delta = payload.get("delta")
                    if (
                        isinstance(delta, dict)
                        and delta.get("type") == "text_delta"
                        and isinstance(delta.get("text"), str)
                    ):
                        frame = json.dumps(
                            {"choices": [{"delta": {"content": delta["text"]}}]},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        yield b"data: " + frame + b"\n\n"
        if not completed:
            yield b'event: error\ndata: {"error":{"code":"UPSTREAM_STREAM_INCOMPLETE","message":"The upstream provider stream ended before completion."}}\n\n'
