# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
OpenAI resource planning, temporary files, and Responses execution.

The provider file identifier is request-private capability state.  Browser
metadata, health documents, logs, receipts, and persisted transcripts never
contain it.  Run 130 adds the full execution boundary: verified resources are
uploaded first, a server-owned Responses request consumes those private handles,
and temporary provider files are deleted after buffered or streaming completion.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator, Sequence
from urllib.parse import quote, urlparse

import httpx

from .executor import ProviderPrivateResource, ResourceExecutionError
from .registry import StaticProviderAdapter

try:
    from .._utils._chat_contract import ChatRequest, build_upstream_payload
    from .._utils._resource_transport import ResourceUpload
except ImportError:  # standalone HF Space: _providers is top-level
    from _utils._chat_contract import ChatRequest, build_upstream_payload
    from _utils._resource_transport import ResourceUpload

_OPENAI_BASE = "https://api.openai.com"
_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MIN_EXPIRES_SECONDS = 3600
_MAX_EXPIRES_SECONDS = 30 * 24 * 60 * 60
_DIRECT_FILE_LIMIT_BYTES = 50 * 1024 * 1024
_MAX_SSE_EVENT_BYTES = 1024 * 1024


class OpenAIAdapter(StaticProviderAdapter):
    def __init__(self, **kwargs):
        super().__init__("openai", **kwargs)


def _validated_openai_base(value: str) -> str:
    raw = str(value or _OPENAI_BASE).rstrip("/")
    parsed = urlparse(raw)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.openai.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "OpenAI resource executor requires the official HTTPS API origin"
        )
    return _OPENAI_BASE


def official_openai_chat_backend(value: str) -> bool:
    """Return whether Path-1 points exactly at the official Chat Completions URL."""
    raw = str(value or "").strip().rstrip("/")
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "api.openai.com"
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and parsed.port in {None, 443}
        and parsed.path.rstrip("/") == "/v1/chat/completions"
        and raw == _OPENAI_CHAT_URL
    )


def _multipart_header(
    boundary: str, *, purpose: str, seconds: int, upload: ResourceUpload
) -> tuple[bytes, bytes]:
    verified = upload.verified
    safe_name = verified.name.replace('"', "_").replace("\r", " ").replace("\n", " ")
    mime = verified.detected_mime or "application/octet-stream"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        f"{purpose}\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="expires_after[anchor]"\r\n\r\n'
        "created_at\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="expires_after[seconds]"\r\n\r\n'
        f"{seconds}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    return prefix, suffix


def _opaque_file_id(handle: ProviderPrivateResource) -> str:
    opaque = handle.opaque if isinstance(handle.opaque, dict) else {}
    file_id = str(opaque.get("file_id", "")).strip()
    if (
        not file_id
        or len(file_id) > 256  # ruff: ignore[magic-value-comparison]
        or not file_id.startswith("file-")
    ):
        raise ResourceExecutionError("OpenAI private file handle is invalid")
    return file_id


def _bounded_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _response_text(payload: object) -> str:
    """Extract only assistant-visible text from one bounded Responses JSON body."""
    if not isinstance(payload, dict):
        raise ResourceExecutionError("OpenAI Responses returned invalid metadata")
    status = str(payload.get("status", "") or "")
    if status in {"failed", "incomplete", "cancelled"} or payload.get("error"):
        raise ResourceExecutionError("OpenAI Responses reported a provider failure")
    top = payload.get("output_text")
    if isinstance(top, str) and top:
        return top
    pieces: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for row in content:
                if not isinstance(row, dict):
                    continue
                if row.get("type") == "output_text" and isinstance(
                    row.get("text"), str
                ):
                    pieces.append(row["text"])
                elif row.get("type") == "refusal" and isinstance(
                    row.get("refusal"), str
                ):
                    pieces.append(row["refusal"])
    text = "".join(pieces)
    if not text:
        raise ResourceExecutionError("OpenAI Responses returned no assistant text")
    return text


class OpenAIResourceExecutor:
    """Stream verified resources to OpenAI Files and consume them with Responses."""

    name = "openai"
    enabled = True

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient,
        base_url: str = _OPENAI_BASE,
        expires_seconds: int = _MIN_EXPIRES_SECONDS,
        response_timeout_seconds: float = 600.0,
    ) -> None:
        key = str(api_key or "").strip()
        if not key:
            raise ValueError("OpenAI API key is required")
        self._api_key = key
        self._client = client
        self._base = _validated_openai_base(base_url)
        self._expires = max(
            _MIN_EXPIRES_SECONDS,
            min(int(expires_seconds), _MAX_EXPIRES_SECONDS),
        )
        self._response_timeout = httpx.Timeout(
            connect=20.0,
            read=max(30.0, min(float(response_timeout_seconds), 1800.0)),
            write=120.0,
            pool=20.0,
        )

    def executable_routes(self, model: str) -> dict[str, tuple[str, ...]]:
        del model
        return {
            "text": ("native",),
            "image": ("native",),
            "document": ("native",),
            "vector_image": ("tool",),
            "data": ("tool",),
            "archive": ("tool",),
            "binary": ("tool",),
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
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
                "OpenAI executor received an invalid resource route"
            )
        purpose = "vision" if verified.detected_modality == "image" else "user_data"
        boundary = "scikitplots-" + secrets.token_hex(16)
        prefix, suffix = _multipart_header(
            boundary,
            purpose=purpose,
            seconds=self._expires,
            upload=upload,
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
            raise ResourceExecutionError("OpenAI temporary file upload failed") from exc
        if (
            response.status_code < 200  # ruff: ignore[magic-value-comparison]
            or response.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError("OpenAI temporary file upload was rejected")
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "OpenAI temporary file upload returned invalid metadata"
            ) from exc
        file_id = (
            str(payload.get("id", "")).strip() if isinstance(payload, dict) else ""
        )
        returned_bytes = payload.get("bytes") if isinstance(payload, dict) else None
        if (
            not file_id
            or len(file_id) > 256  # ruff: ignore[magic-value-comparison]
            or not file_id.startswith("file-")
        ):
            raise ResourceExecutionError(
                "OpenAI temporary file upload returned an invalid file identifier"
            )
        if returned_bytes is not None:
            try:
                if int(returned_bytes) != int(verified.actual_size):
                    raise ResourceExecutionError(
                        "OpenAI temporary file size did not match source bytes"
                    )
            except (TypeError, ValueError) as exc:
                raise ResourceExecutionError(
                    "OpenAI temporary file size metadata was invalid"
                ) from exc
        return ProviderPrivateResource(
            resource_id=verified.id,
            route=route.route,
            provider=self.name,
            source_sha256=verified.sha256,
            source_size=verified.actual_size,
            opaque={
                "file_id": file_id,
                "purpose": purpose,
                "modality": verified.detected_modality,
                "mime": verified.detected_mime,
                "name": verified.name,
            },
            cleanup_required=True,
        )

    async def release(self, handle: ProviderPrivateResource) -> None:
        file_id = _opaque_file_id(handle)
        try:
            response = await self._client.delete(
                f"{self._base}/v1/files/{quote(file_id, safe='-_.~')}",
                headers=self._headers(),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "OpenAI temporary file cleanup failed"
            ) from exc
        if response.status_code == 404:  # ruff: ignore[magic-value-comparison]
            handle.released = True
            return
        if (
            response.status_code < 200  # ruff: ignore[magic-value-comparison]
            or response.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError("OpenAI temporary file cleanup was rejected")
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "OpenAI temporary file cleanup returned invalid metadata"
            ) from exc
        if not isinstance(payload, dict) or payload.get("deleted") is not True:
            raise ResourceExecutionError(
                "OpenAI temporary file cleanup was not confirmed"
            )
        handle.released = True

    def responses_payload(
        self,
        chat: ChatRequest,
        handles: Sequence[ProviderPrivateResource],
    ) -> dict:
        """Build one server-authoritative Responses request from private file handles."""
        provider_body = build_upstream_payload(chat, reasoning_enabled=False)
        messages = provider_body.get("messages", [])
        instructions = str(messages[0].get("content", "")) if len(messages) > 0 else ""
        user_text = (
            str(messages[1].get("content", ""))
            if len(messages) > 1
            else chat.user_message
        )
        content: list[dict] = [{"type": "input_text", "text": user_text}]
        tool_file_ids: list[str] = []
        direct_file_bytes = 0

        for handle in handles:
            file_id = _opaque_file_id(handle)
            opaque = handle.opaque if isinstance(handle.opaque, dict) else {}
            modality = str(opaque.get("modality", ""))
            if handle.route == "tool":
                tool_file_ids.append(file_id)
                continue
            if handle.route != "native":
                raise ResourceExecutionError(
                    "OpenAI Responses received a non-executable route"
                )
            if modality == "image":
                content.append(
                    {"type": "input_image", "file_id": file_id, "detail": "auto"}
                )
                continue
            if modality not in {"text", "document"}:
                raise ResourceExecutionError(
                    "OpenAI direct file route does not support this modality"
                )
            direct_file_bytes += int(handle.source_size)
            if (
                int(handle.source_size) > _DIRECT_FILE_LIMIT_BYTES
                or direct_file_bytes > _DIRECT_FILE_LIMIT_BYTES
            ):
                raise ResourceExecutionError(
                    "OpenAI direct file input exceeds the bounded 50 MiB request limit"
                )
            content.append({"type": "input_file", "file_id": file_id})

        payload: dict = {
            "model": chat.model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": chat.max_tokens,
            "stream": bool(chat.stream),
            "store": False,
        }
        if chat.effort:
            payload["reasoning"] = {
                "effort": {
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "extra": "high",
                    "max": "high",
                }[chat.effort]
            }
        if tool_file_ids:
            payload["tools"] = [
                {
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "memory_limit": "1g",
                        "file_ids": tool_file_ids,
                    },
                }
            ]
            payload["tool_choice"] = "required"
            content[0]["text"] += (
                "\n\nProvider tool resources are attached to the code interpreter container. "
                "Inspect them as needed to answer the user."
            )
        return payload

    async def open_chat(
        self,
        *,
        chat: ChatRequest,
        handles: Sequence[ProviderPrivateResource],
    ) -> httpx.Response:
        payload = self.responses_payload(chat, handles)
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        request = self._client.build_request(
            "POST",
            f"{self._base}/v1/responses",
            headers=headers,
            content=_bounded_json_bytes(payload),
            timeout=self._response_timeout,
        )
        try:
            return await self._client.send(request, stream=True)
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError("OpenAI Responses request failed") from exc

    async def open_responses(
        self,
        chat: ChatRequest,
        handles: Sequence[ProviderPrivateResource],
    ) -> httpx.Response:
        """Internally compatibility alias for focused provider tests/tools."""
        return await self.open_chat(chat=chat, handles=handles)

    async def buffered_chat_response(
        self, upstream: httpx.Response, *, maximum: int
    ) -> dict:
        total = 0
        chunks: list[bytes] = []
        async for chunk in upstream.aiter_bytes():
            total += len(chunk)
            if total > maximum:
                raise ResourceExecutionError(
                    "OpenAI Responses body exceeded the configured response limit"
                )
            chunks.append(chunk)
        try:
            payload = json.loads(b"".join(chunks))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "OpenAI Responses returned invalid JSON"
            ) from exc
        text = _response_text(payload)
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "model": str(payload.get("model", "") or ""),
        }

    async def chat_sse(  # ruff: ignore[too-many-branches]
        self,
        upstream: httpx.Response,
        *,
        maximum: int,
    ) -> AsyncIterator[bytes]:
        """Bridge Responses SSE into the panel's Chat-Completions delta vocabulary."""
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
                event_name = ""
                data_lines: list[bytes] = []
                for line in raw.replace(b"\r\n", b"\n").split(b"\n"):
                    if line.startswith(b"event:"):
                        event_name = line[6:].strip().decode("utf-8", "replace")[:128]
                    elif line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip())
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
                kind = (
                    str(doc.get("type", "") or event_name)
                    if isinstance(doc, dict)
                    else event_name
                )
                if kind in {"response.output_text.delta", "response.refusal.delta"}:
                    delta = doc.get("delta", "") if isinstance(doc, dict) else ""
                    if isinstance(delta, str) and delta:
                        frame = {"choices": [{"delta": {"content": delta}}]}
                        yield b"data: " + _bounded_json_bytes(frame) + b"\n\n"
                elif kind in {"response.failed", "response.incomplete", "error"}:
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_SERVICE_ERROR","message":"The upstream AI provider did not complete the streamed response."}}\n\n'
                    return
                elif kind == "response.completed":
                    done = True
                    break
            if done:
                break
        if not done:
            # A Responses stream is successful only after an explicit completed
            # terminal event.  Partial output followed by EOF must not be promoted
            # into a Chat-Completions-style success marker.
            yield b'event: error\ndata: {"error":{"code":"UPSTREAM_PROTOCOL_ERROR","message":"The upstream stream ended before completion."}}\n\n'
            return
        yield b"data: [DONE]\n\n"
