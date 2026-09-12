# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded multipart transport for first-class chat resources."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import AsyncIterator

from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request

from ._chat_contract import ChatRequest, parse_chat_request
from ._resource_contract import (
    MAX_RESOURCE_COUNT,
    RESOURCE_SIGNATURE_BYTES,
    VerifiedResource,
    verified_descriptor,
)

REQUEST_FIELD = "request"
RESOURCE_FIELD_PREFIX = "resource:"
DEFAULT_MAX_RESOURCE_FILE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_RESOURCE_TOTAL_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_MULTIPART_BYTES = DEFAULT_MAX_RESOURCE_TOTAL_BYTES + 2 * 1024 * 1024
MAX_REQUEST_FIELD_BYTES = 512 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024


class ResourceTransportError(ValueError):
    """Bounded multipart/resource verification failed."""


@dataclass
class ResourceUpload:
    """A verified spooled file whose bytes remain server-owned."""

    verified: VerifiedResource
    upload: UploadFile

    async def close(self) -> None:
        await self.upload.close()


class _BoundedMultiPartParser(MultiPartParser):
    """Starlette multipart parser with active per-file + aggregate file limits."""

    def __init__(self, *args, max_file_bytes: int, max_resource_bytes: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_file_bytes = max(1, int(max_file_bytes))
        self._max_resource_bytes = max(1, int(max_resource_bytes))
        self._current_file_bytes = 0
        self._resource_bytes = 0

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._current_file_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        nbytes = max(0, end - start)
        if self._current_part.file is not None:
            self._current_file_bytes += nbytes
            self._resource_bytes += nbytes
            if self._current_file_bytes > self._max_file_bytes:
                raise MultiPartException(
                    "Resource file exceeded configured byte limit."
                )
            if self._resource_bytes > self._max_resource_bytes:
                raise MultiPartException(
                    "Aggregate resource bytes exceeded configured limit."
                )
        super().on_part_data(data, start, end)


async def _bounded_stream(
    request: Request, maximum: int, stats: dict[str, object]
) -> AsyncIterator[bytes]:
    total = 0
    digest = hashlib.sha256()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > maximum:
                raise ResourceTransportError(
                    "multipart request exceeds configured byte limit"
                )
        except ValueError as exc:
            raise ResourceTransportError("invalid Content-Length") from exc
    async for chunk in request.stream():
        total += len(chunk)
        digest.update(chunk)
        if total > maximum:
            raise ResourceTransportError(
                "multipart request exceeds configured byte limit"
            )
        yield chunk
    stats["wire_body_bytes"] = total
    stats["wire_body_sha256"] = digest.hexdigest()


async def parse_multipart_form(
    request: Request,
    *,
    max_request_bytes: int = DEFAULT_MAX_MULTIPART_BYTES,
    max_file_bytes: int = DEFAULT_MAX_RESOURCE_FILE_BYTES,
    max_resource_bytes: int = DEFAULT_MAX_RESOURCE_TOTAL_BYTES,
    max_files: int = MAX_RESOURCE_COUNT,
) -> tuple[FormData, dict[str, object]]:
    content_type = request.headers.get("content-type", "").lower()
    if not content_type.startswith("multipart/form-data"):
        raise ResourceTransportError("resource request must use multipart/form-data")
    stats: dict[str, object] = {"multipart": True}
    parser = _BoundedMultiPartParser(
        headers=request.headers,
        stream=_bounded_stream(request, max_request_bytes, stats),
        max_files=max_files,
        max_fields=2,
        max_part_size=MAX_REQUEST_FIELD_BYTES,
        max_file_bytes=max_file_bytes,
        max_resource_bytes=max_resource_bytes,
    )
    try:
        return await parser.parse(), stats
    except MultiPartException as exc:
        raise ResourceTransportError(str(exc)) from exc


def _request_bytes_from_form(form: FormData) -> bytes:
    values = form.getlist(REQUEST_FIELD)
    if len(values) != 1 or not isinstance(values[0], str):
        raise ResourceTransportError(
            "multipart request must contain exactly one JSON request field"
        )
    try:
        encoded = values[0].encode("utf-8")
    except UnicodeError as exc:
        raise ResourceTransportError("request field must be UTF-8") from exc
    if len(encoded) > MAX_REQUEST_FIELD_BYTES:
        raise ResourceTransportError("request metadata exceeds configured byte limit")
    # Parse now to reject surprising non-object JSON before contract validation.
    try:
        if not isinstance(json.loads(encoded), dict):
            raise ResourceTransportError("request metadata must be a JSON object")
    except json.JSONDecodeError as exc:
        raise ResourceTransportError("request metadata must be valid JSON") from exc
    return encoded


async def _measure_upload(upload: UploadFile) -> tuple[int, str, bytes]:
    digest = hashlib.sha256()
    prefix = bytearray()
    total = 0
    await upload.seek(0)
    while True:
        chunk = await upload.read(_HASH_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        digest.update(chunk)
        if len(prefix) < RESOURCE_SIGNATURE_BYTES:
            need = RESOURCE_SIGNATURE_BYTES - len(prefix)
            prefix.extend(chunk[:need])
    await upload.seek(0)
    return total, digest.hexdigest(), bytes(prefix)


async def parse_resource_chat_request(
    request: Request,
    *,
    allowed_models,
    allowed_namespaces=(),
    max_request_bytes: int = DEFAULT_MAX_MULTIPART_BYTES,
    max_file_bytes: int = DEFAULT_MAX_RESOURCE_FILE_BYTES,
    max_resource_bytes: int = DEFAULT_MAX_RESOURCE_TOTAL_BYTES,
    max_files: int = MAX_RESOURCE_COUNT,
) -> tuple[bytes, ChatRequest, tuple[ResourceUpload, ...], dict[str, object]]:
    """Parse, cross-check and hash a multipart ``scikitplot-chat-v1`` request."""
    form, transport_stats = await parse_multipart_form(
        request,
        max_request_bytes=max_request_bytes,
        max_file_bytes=max_file_bytes,
        max_resource_bytes=max_resource_bytes,
        max_files=max_files,
    )
    uploads_seen: list[UploadFile] = [
        value for _key, value in form.multi_items() if isinstance(value, UploadFile)
    ]
    try:
        request_bytes = _request_bytes_from_form(form)
        chat = parse_chat_request(
            request_bytes,
            allowed_models=allowed_models,
            allowed_namespaces=allowed_namespaces,
        )
        descriptors = {row.id: row for row in chat.resources}

        uploads_by_id: dict[str, UploadFile] = {}
        for key, value in form.multi_items():
            if key == REQUEST_FIELD:
                continue
            if not key.startswith(RESOURCE_FIELD_PREFIX) or not isinstance(
                value, UploadFile
            ):
                raise ResourceTransportError("unsupported multipart field")
            rid = key[len(RESOURCE_FIELD_PREFIX) :]
            if rid not in descriptors or rid in uploads_by_id:
                raise ResourceTransportError(
                    "multipart resource field does not match request metadata"
                )
            uploads_by_id[rid] = value

        if set(uploads_by_id) != set(descriptors):
            raise ResourceTransportError(
                "every declared resource must have exactly one file part"
            )

        out: list[ResourceUpload] = []
        for descriptor in chat.resources:
            upload = uploads_by_id[descriptor.id]
            actual_size, sha256, prefix = await _measure_upload(upload)
            if actual_size != descriptor.size:
                raise ResourceTransportError(
                    "resource byte length does not match declared size"
                )
            actual_mime = str(upload.content_type or "").strip().lower()
            verified = verified_descriptor(
                descriptor,
                actual_size=actual_size,
                sha256=sha256,
                prefix=prefix,
                actual_mime=actual_mime,
            )
            out.append(ResourceUpload(verified=verified, upload=upload))
        return request_bytes, chat, tuple(out), transport_stats
    except Exception:
        # Starlette may already have spooled one or more file parts before a
        # later metadata/field cross-check fails. Close every UploadFile found
        # in the parsed form, not only the subset that passed ID matching.
        for upload in uploads_seen:
            try:  # ruff: ignore[suppressible-exception]
                await upload.close()
            except Exception:  # ruff: ignore[blind-except, try-except-in-loop]
                pass
        raise
