# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Gemini Files + Interactions first-class resource executor.

Verified browser resources are uploaded byte-for-byte through Gemini's provider
facilities.  Provider names/URIs/store names are request-private capabilities,
never public receipts.  Image/GIF/audio/video/PDF use native Files + Interactions
content blocks; reviewed text/ZIP/XLS/XLSX resources may use one request-scoped
File Search store with mandatory cleanup and route-specific MIME/size limits.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
from collections.abc import AsyncIterator, Sequence
from urllib.parse import quote, urlparse

import httpx

from .executor import ProviderPrivateResource, ResourceExecutionError
from .policy import (
    gemini_file_search_supported_model,
    gemini_native_media_supported_model,
)
from .registry import StaticProviderAdapter

try:
    from .._utils._chat_contract import ChatRequest, build_upstream_payload
    from .._utils._resource_transport import ResourceUpload
except ImportError:  # standalone HF Space: _providers is top-level
    from _utils._chat_contract import ChatRequest, build_upstream_payload
    from _utils._resource_transport import ResourceUpload

_GEMINI_BASE = "https://generativelanguage.googleapis.com"
_GEMINI_INTERACTIONS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_PDF_BYTES = 50 * 1024 * 1024
_FILE_SEARCH_MAX_BYTES = 100 * 1024 * 1024
_MAX_SSE_EVENT_BYTES = 1024 * 1024
_FILE_NAME_RE = re.compile(r"^files/[A-Za-z0-9._~-]{1,240}$")
_FILE_SEARCH_STORE_RE = re.compile(
    r"^fileSearchStores/[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?$"
)
_FILE_SEARCH_OPERATION_RE = re.compile(
    r"^fileSearchStores/[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?/upload/operations/[A-Za-z0-9._~-]{1,240}$"
)
_FILE_SEARCH_MIMES = {
    "archive": {"application/zip"},
    "data": {
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
}
_NATIVE_MIMES = {
    "image": {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
        "image/bmp",
        "image/tiff",
    },
    "animated_image": {"image/gif"},
    "audio": {
        "audio/wav",
        "audio/mp3",
        "audio/aiff",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/mpeg",
        "audio/m4a",
        "audio/l16",
        "audio/opus",
        "audio/alaw",
        "audio/mulaw",
        "audio/webm",
    },
    "video": {
        "video/mp4",
        "video/mpeg",
        "video/mpg",
        "video/mov",
        "video/avi",
        "video/x-flv",
        "video/webm",
        "video/wmv",
        "video/3gpp",
    },
    "document": {"application/pdf"},
}

_MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "video/quicktime": "video/mov",
    "video/x-msvideo": "video/avi",
    "video/x-ms-wmv": "video/wmv",
    "audio/x-wav": "audio/wav",
    "audio/x-aiff": "audio/aiff",
    "audio/x-flac": "audio/flac",
    "audio/x-m4a": "audio/m4a",
    "audio/mp4": "audio/m4a",
}


def _gemini_mime(upload: ResourceUpload) -> str:
    raw = str(upload.verified.detected_mime or "application/octet-stream").lower()
    return _MIME_ALIASES.get(raw, raw)


class GeminiAdapter(StaticProviderAdapter):
    def __init__(self, **kwargs):
        super().__init__("gemini", **kwargs)


def _validated_gemini_base(value: str) -> str:
    raw = str(value or _GEMINI_BASE).rstrip("/")
    parsed = urlparse(raw)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "generativelanguage.googleapis.com"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Gemini executor requires the official HTTPS API origin")
    return _GEMINI_BASE


def official_gemini_interactions_backend(value: str) -> bool:
    raw = str(value or "").strip().rstrip("/")
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "generativelanguage.googleapis.com"
        and parsed.port in {None, 443}
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and parsed.path.rstrip("/") == "/v1beta/interactions"
        and raw == _GEMINI_INTERACTIONS_URL
    )


def _validated_upload_url(value: str) -> str:
    raw = str(value or "").strip()
    if len(raw) > 4096 or any(  # ruff: ignore[magic-value-comparison]
        ord(ch) < 32  # ruff: ignore[magic-value-comparison]
        or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in raw
    ):
        raise ResourceExecutionError("Gemini resumable upload URL was invalid")
    try:
        parsed = urlparse(raw)
    except ValueError as exc:
        raise ResourceExecutionError("Gemini resumable upload URL was invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "generativelanguage.googleapis.com"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not parsed.path.startswith("/upload/v1beta/files")
    ):
        raise ResourceExecutionError("Gemini resumable upload URL was invalid")
    return raw


def _safe_file_name(value: object) -> str:
    name = str(value or "").strip()
    if not _FILE_NAME_RE.fullmatch(name):
        raise ResourceExecutionError("Gemini Files returned an invalid file name")
    return name


def _safe_file_uri(value: object) -> str:
    raw = str(value or "").strip()
    if len(raw) > 4096 or any(  # ruff: ignore[magic-value-comparison]
        ord(ch) < 32  # ruff: ignore[magic-value-comparison]
        or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in raw
    ):
        raise ResourceExecutionError("Gemini Files returned an invalid file URI")
    try:
        parsed = urlparse(raw)
    except ValueError as exc:
        raise ResourceExecutionError(
            "Gemini Files returned an invalid file URI"
        ) from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "generativelanguage.googleapis.com"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/v1beta/files/")
    ):
        raise ResourceExecutionError("Gemini Files returned an invalid file URI")
    return raw


def _native_opaque(handle: ProviderPrivateResource) -> dict:
    data = handle.opaque if isinstance(handle.opaque, dict) else {}
    name = _safe_file_name(data.get("file_name"))
    uri = _safe_file_uri(data.get("uri"))
    return {**data, "file_name": name, "uri": uri}


def _safe_store_name(value: object) -> str:
    name = str(value or "").strip()
    if not _FILE_SEARCH_STORE_RE.fullmatch(name):
        raise ResourceExecutionError(
            "Gemini File Search returned an invalid store name"
        )
    return name


def _safe_operation_name(value: object, store_name: str) -> str:
    name = str(value or "").strip()
    if not _FILE_SEARCH_OPERATION_RE.fullmatch(name) or not name.startswith(
        store_name + "/upload/operations/"
    ):
        raise ResourceExecutionError(
            "Gemini File Search returned an invalid operation name"
        )
    return name


def _validated_file_search_upload_url(value: str, store_name: str) -> str:
    raw = str(value or "").strip()
    if len(raw) > 4096 or any(  # ruff: ignore[magic-value-comparison]
        ord(ch) < 32  # ruff: ignore[magic-value-comparison]
        or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in raw
    ):
        raise ResourceExecutionError("Gemini File Search upload URL was invalid")
    try:
        parsed = urlparse(raw)
    except ValueError as exc:
        raise ResourceExecutionError(
            "Gemini File Search upload URL was invalid"
        ) from exc
    expected = "/upload/v1beta/" + store_name + ":uploadToFileSearchStore"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "generativelanguage.googleapis.com"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.path != expected
    ):
        raise ResourceExecutionError("Gemini File Search upload URL was invalid")
    return raw


def _tool_opaque(handle: ProviderPrivateResource) -> dict:
    data = handle.opaque if isinstance(handle.opaque, dict) else {}
    return {**data, "store_name": _safe_store_name(data.get("store_name"))}


def _file_search_mime(upload: ResourceUpload) -> str:
    modality = str(upload.verified.detected_modality or "")
    raw = str(upload.verified.detected_mime or "application/octet-stream").lower()
    if modality == "text":
        if raw.startswith("text/") or raw in {"application/json", "application/xml"}:
            return raw
        # The verifier already classified the resource as text. Normalizing
        # uncommon textual application/* labels to text/plain changes only
        # provider metadata, never the source bytes.
        return "text/plain"
    return raw


def _provider_hash_hex(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ResourceExecutionError(
            "Gemini Files returned invalid SHA-256 metadata"
        ) from exc
    if len(decoded) != 32:  # ruff: ignore[magic-value-comparison]
        raise ResourceExecutionError("Gemini Files returned invalid SHA-256 metadata")
    return decoded.hex()


def _file_metadata(payload: object) -> dict:
    if isinstance(payload, dict) and isinstance(payload.get("file"), dict):
        return payload["file"]
    if isinstance(payload, dict):
        return payload
    raise ResourceExecutionError("Gemini Files returned invalid metadata")


def _response_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ResourceExecutionError("Gemini Interactions returned invalid metadata")
    status = str(payload.get("status", "") or "")
    if status != "completed" or payload.get("error"):
        raise ResourceExecutionError(
            "Gemini Interactions did not complete successfully"
        )
    pieces: list[str] = []
    steps = payload.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            content = step.get("content")
            if not isinstance(content, list):
                continue
            pieces.extend(
                row["text"]
                for row in content
                if isinstance(row, dict)
                and row.get("type") == "text"
                and isinstance(row.get("text"), str)
            )
    text = "".join(pieces)
    if not text:
        raise ResourceExecutionError("Gemini Interactions returned no assistant text")
    return text


class GeminiResourceExecutor:
    """Upload verified native media to Gemini Files and execute Interactions."""

    name = "gemini"
    enabled = True

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient,
        base_url: str = _GEMINI_BASE,
        response_timeout_seconds: float = 600.0,
        processing_timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        key = str(api_key or "").strip()
        if not key:
            raise ValueError("Gemini API key is required")
        self._api_key = key
        self._client = client
        self._base = _validated_gemini_base(base_url)
        self._response_timeout = httpx.Timeout(
            connect=20.0,
            read=max(30.0, min(float(response_timeout_seconds), 1800.0)),
            write=180.0,
            pool=20.0,
        )
        self._processing_timeout = max(
            1.0, min(float(processing_timeout_seconds), 900.0)
        )
        self._poll_interval = max(0.05, min(float(poll_interval_seconds), 5.0))

    def executable_routes(self, model: str) -> dict[str, tuple[str, ...]]:
        routes: dict[str, tuple[str, ...]] = {}
        if gemini_native_media_supported_model(model):
            routes.update(
                {
                    "image": ("native",),
                    "animated_image": ("native",),
                    "audio": ("native",),
                    "video": ("native",),
                    "document": ("native",),
                }
            )
        if gemini_file_search_supported_model(model):
            routes.update(
                {
                    "text": ("tool",),
                    "archive": ("tool",),
                    "data": ("tool",),
                }
            )
        return routes

    def route_constraints(self, model: str) -> dict[str, dict[str, dict]]:
        """
        Return browser-visible limits for exact executable routes.

        Limits are route-specific because Gemini native media and File Search
        have different provider ceilings and MIME vocabularies.  This data is
        non-secret and is re-checked server-side before provider I/O.
        """
        constraints: dict[str, dict[str, dict]] = {}
        if gemini_native_media_supported_model(model):
            constraints["document"] = {
                "native": {
                    "max_file_bytes": _MAX_PDF_BYTES,
                    "mime_types": ["application/pdf"],
                }
            }
        if gemini_file_search_supported_model(model):
            constraints["text"] = {"tool": {"max_file_bytes": _FILE_SEARCH_MAX_BYTES}}
            constraints["archive"] = {
                "tool": {
                    "max_file_bytes": _FILE_SEARCH_MAX_BYTES,
                    "mime_types": sorted(_FILE_SEARCH_MIMES["archive"]),
                }
            }
            constraints["data"] = {
                "tool": {
                    "max_file_bytes": _FILE_SEARCH_MAX_BYTES,
                    "mime_types": sorted(_FILE_SEARCH_MIMES["data"]),
                }
            }
        return constraints

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key, "Accept": "application/json"}

    async def _raw_body(self, upload: ResourceUpload):
        await upload.upload.seek(0)
        try:
            while True:
                chunk = await upload.upload.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk
        finally:
            await upload.upload.seek(0)

    def _validate_metadata(
        self, meta: dict, upload: ResourceUpload
    ) -> tuple[str, str, str]:
        verified = upload.verified
        name = _safe_file_name(meta.get("name"))
        uri = _safe_file_uri(meta.get("uri"))
        state = str(meta.get("state", "") or "").upper()
        size = meta.get("sizeBytes", meta.get("size_bytes"))
        if size is not None:
            try:
                if int(size) != int(verified.actual_size):
                    raise ResourceExecutionError(
                        "Gemini Files size did not match source bytes"
                    )
            except (TypeError, ValueError) as exc:
                raise ResourceExecutionError(
                    "Gemini Files returned invalid size metadata"
                ) from exc
        provider_hash = _provider_hash_hex(
            meta.get("sha256Hash", meta.get("sha256_hash"))
        )
        if provider_hash and provider_hash != verified.sha256:
            raise ResourceExecutionError(
                "Gemini Files SHA-256 did not match source bytes"
            )
        returned_mime = str(
            meta.get("mimeType", meta.get("mime_type", "")) or ""
        ).lower()
        expected_mime = _gemini_mime(upload)
        if returned_mime and returned_mime != expected_mime:
            raise ResourceExecutionError(
                "Gemini Files MIME metadata did not match verified bytes"
            )
        return name, uri, state

    async def _delete_name(self, name: str) -> None:
        safe = _safe_file_name(name)
        try:
            response = await self._client.delete(
                f"{self._base}/v1beta/{quote(safe, safe='/')}", headers=self._headers()
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "Gemini temporary file cleanup failed"
            ) from exc
        if response.status_code == 404:  # ruff: ignore[magic-value-comparison]
            return
        if (
            response.status_code < 200  # ruff: ignore[magic-value-comparison]
            or response.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError("Gemini temporary file cleanup was rejected")

    async def _wait_active(
        self, *, name: str, upload: ResourceUpload, initial: dict
    ) -> dict:
        meta = initial
        deadline = asyncio.get_running_loop().time() + self._processing_timeout
        while True:
            got_name, _uri, state = self._validate_metadata(meta, upload)
            if got_name != name:
                raise ResourceExecutionError(
                    "Gemini Files identity changed while processing"
                )
            if state == "ACTIVE":
                return meta
            if state == "FAILED":
                raise ResourceExecutionError("Gemini Files processing failed")
            if state not in {"PROCESSING", "STATE_UNSPECIFIED", ""}:
                raise ResourceExecutionError(
                    "Gemini Files returned an unexpected processing state"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise ResourceExecutionError("Gemini Files processing timed out")
            await asyncio.sleep(self._poll_interval)
            try:
                response = await self._client.get(
                    f"{self._base}/v1beta/{quote(name, safe='/')}",
                    headers=self._headers(),
                )
            except (httpx.HTTPError, OSError) as exc:
                raise ResourceExecutionError(
                    "Gemini Files processing check failed"
                ) from exc
            if (
                response.status_code < 200  # ruff: ignore[magic-value-comparison]
                or response.status_code >= 300  # ruff: ignore[magic-value-comparison]
            ):
                raise ResourceExecutionError(
                    "Gemini Files processing check was rejected"
                )
            try:
                meta = _file_metadata(response.json())
            except (json.JSONDecodeError, ValueError) as exc:
                raise ResourceExecutionError(
                    "Gemini Files processing metadata was invalid"
                ) from exc

    async def _create_file_search_store(self) -> str:
        payload = {
            "displayName": "scikit-plots-ephemeral",
            "embeddingModel": "models/gemini-embedding-2",
        }
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        try:
            response = await self._client.post(
                f"{self._base}/v1beta/fileSearchStores",
                headers=headers,
                content=json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8"),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "Gemini File Search store creation failed"
            ) from exc
        if (
            response.status_code < 200  # ruff: ignore[magic-value-comparison]
            or response.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError(
                "Gemini File Search store creation was rejected"
            )
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "Gemini File Search store metadata was invalid"
            ) from exc
        if not isinstance(data, dict):
            raise ResourceExecutionError(
                "Gemini File Search store metadata was invalid"
            )
        return _safe_store_name(data.get("name"))

    async def _delete_store(self, store_name: str) -> None:
        safe = _safe_store_name(store_name)
        try:
            response = await self._client.delete(
                f"{self._base}/v1beta/{quote(safe, safe='/')}?force=true",
                headers=self._headers(),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "Gemini File Search store cleanup failed"
            ) from exc
        if response.status_code == 404:  # ruff: ignore[magic-value-comparison]
            return
        if (
            response.status_code < 200  # ruff: ignore[magic-value-comparison]
            or response.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError(
                "Gemini File Search store cleanup was rejected"
            )

    async def _wait_file_search_operation(
        self, *, store_name: str, payload: object
    ) -> None:
        if not isinstance(payload, dict):
            raise ResourceExecutionError(
                "Gemini File Search upload returned invalid operation metadata"
            )
        operation_name = _safe_operation_name(payload.get("name"), store_name)
        operation = payload
        deadline = asyncio.get_running_loop().time() + self._processing_timeout
        while True:
            if operation.get("error"):
                raise ResourceExecutionError("Gemini File Search indexing failed")
            if operation.get("done") is True:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise ResourceExecutionError("Gemini File Search indexing timed out")
            await asyncio.sleep(self._poll_interval)
            try:
                response = await self._client.get(
                    f"{self._base}/v1beta/{quote(operation_name, safe='/')}",
                    headers=self._headers(),
                )
            except (httpx.HTTPError, OSError) as exc:
                raise ResourceExecutionError(
                    "Gemini File Search indexing check failed"
                ) from exc
            if (
                response.status_code < 200  # ruff: ignore[magic-value-comparison]
                or response.status_code >= 300  # ruff: ignore[magic-value-comparison]
            ):
                raise ResourceExecutionError(
                    "Gemini File Search indexing check was rejected"
                )
            try:
                operation = response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise ResourceExecutionError(
                    "Gemini File Search operation metadata was invalid"
                ) from exc
            if (
                not isinstance(operation, dict)
                or _safe_operation_name(operation.get("name"), store_name)
                != operation_name
            ):
                raise ResourceExecutionError(
                    "Gemini File Search operation identity changed"
                )

    async def _upload_to_file_search_store(
        self, *, store_name: str, upload: ResourceUpload
    ) -> None:
        verified = upload.verified
        modality = str(verified.detected_modality or "")
        mime = _file_search_mime(upload)
        if int(verified.actual_size) > _FILE_SEARCH_MAX_BYTES:
            raise ResourceExecutionError(
                "Gemini File Search resource exceeds the 100 MB document limit"
            )
        if modality == "archive":
            if mime not in _FILE_SEARCH_MIMES["archive"]:
                raise ResourceExecutionError(
                    "Gemini File Search does not support this archive MIME type"
                )
        elif modality == "data":
            if mime not in _FILE_SEARCH_MIMES["data"]:
                raise ResourceExecutionError(
                    "Gemini File Search does not support this data MIME type"
                )
        elif modality != "text":
            raise ResourceExecutionError(
                "Gemini File Search does not support this resource modality"
            )

        safe_store = _safe_store_name(store_name)
        start_headers = self._headers()
        start_headers.update(
            {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(verified.actual_size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            }
        )
        display_name = str(verified.name or "resource")[:512]
        try:
            start = await self._client.post(
                f"{self._base}/upload/v1beta/{quote(safe_store, safe='/')}:uploadToFileSearchStore",
                headers=start_headers,
                content=json.dumps(
                    {"displayName": display_name},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "Gemini File Search resumable upload could not start"
            ) from exc
        if (
            start.status_code < 200  # ruff: ignore[magic-value-comparison]
            or start.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError(
                "Gemini File Search resumable upload start was rejected"
            )
        upload_url = _validated_file_search_upload_url(
            start.headers.get("x-goog-upload-url", ""), safe_store
        )
        try:
            finish = await self._client.post(
                upload_url,
                headers={
                    "Content-Length": str(verified.actual_size),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                    "Content-Type": mime,
                },
                content=self._raw_body(upload),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "Gemini File Search resource upload failed"
            ) from exc
        if (
            finish.status_code < 200  # ruff: ignore[magic-value-comparison]
            or finish.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError(
                "Gemini File Search resource upload was rejected"
            )
        try:
            operation = finish.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "Gemini File Search upload returned invalid operation metadata"
            ) from exc
        await self._wait_file_search_operation(store_name=safe_store, payload=operation)

    async def prepare_many(  # ruff: ignore[too-many-branches]
        self, *, model: str, routes: Sequence, uploads: Sequence[ResourceUpload]
    ):
        if len(routes) != len(uploads):
            raise ResourceExecutionError(
                "Gemini batch prepare route/upload count mismatch"
            )
        pairs = tuple(zip(routes, uploads))
        tool_pairs = []
        for route, upload in pairs:
            if route.resource_id != upload.verified.id:
                raise ResourceExecutionError(
                    "Gemini batch prepare resource identity mismatch"
                )
            modality = str(upload.verified.detected_modality or "")
            allowed = self.executable_routes(model).get(modality, ())
            if route.route not in allowed:
                raise ResourceExecutionError(
                    "Gemini executor does not implement this modality/route"
                )
            if route.route == "tool":
                # Validate provider limits/MIME before any provider object is
                # created. The upload helper repeats these checks defensively.
                mime = _file_search_mime(upload)
                if int(upload.verified.actual_size) > _FILE_SEARCH_MAX_BYTES:
                    raise ResourceExecutionError(
                        "Gemini File Search resource exceeds the 100 MB document limit"
                    )
                if modality == "archive" and mime not in _FILE_SEARCH_MIMES["archive"]:
                    raise ResourceExecutionError(
                        "Gemini File Search does not support this archive MIME type"
                    )
                if modality == "data" and mime not in _FILE_SEARCH_MIMES["data"]:
                    raise ResourceExecutionError(
                        "Gemini File Search does not support this data MIME type"
                    )
                tool_pairs.append((route, upload))

        store_name = ""
        native_handles: list[ProviderPrivateResource] = []
        output: dict[str, ProviderPrivateResource] = {}
        try:
            if tool_pairs:
                if not gemini_file_search_supported_model(model):
                    raise ResourceExecutionError(
                        "Gemini model does not support File Search"
                    )
                store_name = await self._create_file_search_store()
                for _route, upload in tool_pairs:
                    await self._upload_to_file_search_store(
                        store_name=store_name, upload=upload
                    )

            owner_id = tool_pairs[0][0].resource_id if tool_pairs else ""
            for route, upload in pairs:
                if route.route == "native":
                    handle = await self.prepare(model=model, route=route, upload=upload)
                    native_handles.append(handle)
                    output[route.resource_id] = handle
                else:
                    output[route.resource_id] = ProviderPrivateResource(
                        resource_id=upload.verified.id,
                        route="tool",
                        provider=self.name,
                        source_sha256=upload.verified.sha256,
                        source_size=upload.verified.actual_size,
                        opaque={
                            "store_name": store_name,
                            "modality": upload.verified.detected_modality,
                            "mime": _file_search_mime(upload),
                            "name": upload.verified.name,
                        },
                        cleanup_required=(route.resource_id == owner_id),
                    )
            return tuple(output[route.resource_id] for route, _upload in pairs)
        except BaseException:
            for handle in reversed(native_handles):
                try:  # ruff: ignore[suppressible-exception]
                    await asyncio.shield(self.release(handle))
                except Exception:  # ruff: ignore[blind-except, try-except-in-loop]
                    pass
            if store_name:
                try:  # ruff: ignore[suppressible-exception]
                    await asyncio.shield(self._delete_store(store_name))
                except Exception:  # ruff: ignore[blind-except]
                    pass
            raise

    async def prepare(  # ruff: ignore[too-many-branches]
        self, *, model: str, route, upload: ResourceUpload
    ) -> ProviderPrivateResource:
        verified = upload.verified
        modality = verified.detected_modality
        mime = _gemini_mime(upload)
        if route.resource_id != verified.id or route.route != "native":
            raise ResourceExecutionError(
                "Gemini executor received an invalid resource route"
            )
        if modality not in self.executable_routes(model):
            raise ResourceExecutionError(
                "Gemini native route does not support this modality"
            )
        if mime not in _NATIVE_MIMES.get(modality, set()):
            raise ResourceExecutionError(
                "Gemini native route does not support this MIME type"
            )
        if (
            modality == "document"
            and mime == "application/pdf"
            and int(verified.actual_size) > _MAX_PDF_BYTES
        ):
            raise ResourceExecutionError(
                "Gemini PDF exceeds the bounded 50 MB provider limit"
            )

        start_headers = self._headers()
        start_headers.update(
            {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(verified.actual_size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            }
        )
        display_name = str(verified.name or "resource")[:512]
        try:
            start = await self._client.post(
                f"{self._base}/upload/v1beta/files",
                headers=start_headers,
                content=json.dumps(
                    {"file": {"display_name": display_name}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError(
                "Gemini resumable upload could not start"
            ) from exc
        if (
            start.status_code < 200  # ruff: ignore[magic-value-comparison]
            or start.status_code >= 300  # ruff: ignore[magic-value-comparison]
        ):
            raise ResourceExecutionError("Gemini resumable upload start was rejected")
        upload_url = _validated_upload_url(start.headers.get("x-goog-upload-url", ""))

        provider_name = ""
        try:
            try:
                finish = await self._client.post(
                    upload_url,
                    headers={
                        "Content-Length": str(verified.actual_size),
                        "X-Goog-Upload-Offset": "0",
                        "X-Goog-Upload-Command": "upload, finalize",
                        "Content-Type": mime,
                    },
                    content=self._raw_body(upload),
                )
            except (httpx.HTTPError, OSError) as exc:
                raise ResourceExecutionError("Gemini resource upload failed") from exc
            if (
                finish.status_code < 200  # ruff: ignore[magic-value-comparison]
                or finish.status_code >= 300  # ruff: ignore[magic-value-comparison]
            ):
                raise ResourceExecutionError("Gemini resource upload was rejected")
            try:
                meta = _file_metadata(finish.json())
            except (json.JSONDecodeError, ValueError) as exc:
                raise ResourceExecutionError(
                    "Gemini resource upload returned invalid metadata"
                ) from exc
            # Capture the provider cleanup authority before validating optional
            # metadata such as URI/hash/state. Any later rejection can now roll
            # back the already-created provider file.
            provider_name = _safe_file_name(meta.get("name"))
            checked_name, _uri, _state = self._validate_metadata(meta, upload)
            if checked_name != provider_name:
                raise ResourceExecutionError(
                    "Gemini Files identity changed after upload"
                )
            meta = await self._wait_active(
                name=provider_name, upload=upload, initial=meta
            )
            provider_name, provider_uri, state = self._validate_metadata(meta, upload)
            if state != "ACTIVE":
                raise ResourceExecutionError("Gemini resource did not become active")
        except BaseException:
            if provider_name:
                try:  # ruff: ignore[suppressible-exception]
                    await asyncio.shield(self._delete_name(provider_name))
                except Exception:  # ruff: ignore[blind-except]
                    pass
            raise

        return ProviderPrivateResource(
            resource_id=verified.id,
            route=route.route,
            provider=self.name,
            source_sha256=verified.sha256,
            source_size=verified.actual_size,
            opaque={
                "file_name": provider_name,
                "uri": provider_uri,
                "modality": modality,
                "mime": mime,
                "name": verified.name,
            },
            cleanup_required=True,
        )

    async def release(self, handle: ProviderPrivateResource) -> None:
        if handle.route == "tool":
            data = _tool_opaque(handle)
            if handle.cleanup_required:
                await self._delete_store(data["store_name"])
            handle.released = True
            return
        data = _native_opaque(handle)
        await self._delete_name(data["file_name"])
        handle.released = True

    def interactions_payload(
        self, chat: ChatRequest, handles: Sequence[ProviderPrivateResource]
    ) -> dict:
        upstream = build_upstream_payload(chat, reasoning_enabled=False)
        messages = upstream.get("messages", [])
        system = str(messages[0].get("content", "")) if len(messages) > 0 else ""
        user_text = (
            str(messages[1].get("content", ""))
            if len(messages) > 1
            else chat.user_message
        )
        content: list[dict] = [{"type": "text", "text": user_text}]
        file_search_stores: list[str] = []
        for handle in handles:
            if handle.route == "native":
                data = _native_opaque(handle)
                modality = str(data.get("modality", ""))
                mime = str(data.get("mime", ""))
                block_type = (
                    "image" if modality in {"image", "animated_image"} else modality
                )
                if block_type not in {"image", "audio", "video", "document"}:
                    raise ResourceExecutionError(
                        "Gemini Interactions cannot map this resource modality"
                    )
                block = {"type": block_type, "uri": data["uri"], "mime_type": mime}
                content.append(block)
            elif handle.route == "tool":
                store = _tool_opaque(handle)["store_name"]
                if store not in file_search_stores:
                    file_search_stores.append(store)
            else:
                raise ResourceExecutionError(
                    "Gemini Interactions received an unsupported resource route"
                )
        generation: dict = {
            "max_output_tokens": chat.max_tokens,
            "thinking_summaries": "none",
        }
        if chat.effort:
            effort_map = {
                "low": "low",
                "medium": "medium",
                "high": "high",
                "extra": "high",
                "max": "high",
            }
            generation["thinking_level"] = effort_map.get(chat.effort, "medium")
        payload: dict = {
            "model": chat.model,
            "input": content,
            "system_instruction": system,
            "stream": bool(chat.stream),
            "store": False,
            "generation_config": generation,
        }
        if file_search_stores:
            payload["tools"] = [
                {
                    "type": "file_search",
                    "file_search_store_names": file_search_stores,
                }
            ]
        return payload

    async def open_chat(
        self, *, chat: ChatRequest, handles: Sequence[ProviderPrivateResource]
    ) -> httpx.Response:
        payload = self.interactions_payload(chat, handles)
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        if chat.stream:
            headers["Accept"] = "text/event-stream"
        request = self._client.build_request(
            "POST",
            f"{self._base}/v1beta/interactions",
            headers=headers,
            content=json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
            timeout=self._response_timeout,
        )
        try:
            return await self._client.send(request, stream=True)
        except (httpx.HTTPError, OSError) as exc:
            raise ResourceExecutionError("Gemini Interactions request failed") from exc

    async def buffered_chat_response(
        self, upstream: httpx.Response, *, maximum: int
    ) -> dict:
        total = 0
        chunks: list[bytes] = []
        async for chunk in upstream.aiter_bytes():
            total += len(chunk)
            if total > maximum:
                raise ResourceExecutionError(
                    "Gemini Interactions body exceeded the configured response limit"
                )
            chunks.append(chunk)
        try:
            payload = json.loads(b"".join(chunks))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResourceExecutionError(
                "Gemini Interactions returned invalid JSON"
            ) from exc
        text = _response_text(payload)
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "model": (
                str(payload.get("model", "") or "") if isinstance(payload, dict) else ""
            ),
        }

    async def chat_sse(  # ruff: ignore[too-many-branches, too-many-return-statements]
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
                candidates = [(pos_lf, 2), (pos_crlf, 4)]
                candidates = [(pos, width) for pos, width in candidates if pos >= 0]
                if not candidates:
                    break
                pos, width = min(candidates, key=lambda row: row[0])
                if pos > _MAX_SSE_EVENT_BYTES:
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_EVENT_TOO_LARGE","message":"The upstream stream event exceeded the configured limit."}}\n\n'
                    return
                raw = bytes(buffer[:pos])
                del buffer[: pos + width]
                data_lines: list[bytes] = []
                event_name = ""
                for line in raw.replace(b"\r\n", b"\n").split(b"\n"):
                    if line.startswith(b"event:"):
                        event_name = line[6:].strip().decode("utf-8", "replace")
                    elif line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip())
                if not data_lines:
                    continue
                joined = b"\n".join(data_lines)
                if joined == b"[DONE]":
                    continue
                try:
                    payload = json.loads(joined)
                except (json.JSONDecodeError, ValueError):
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_STREAM_INVALID","message":"The upstream provider stream was malformed."}}\n\n'
                    return
                if not isinstance(payload, dict):
                    continue
                event_type = str(payload.get("event_type", "") or event_name)
                if event_type == "interaction.completed":
                    interaction = payload.get("interaction")
                    if (
                        isinstance(interaction, dict)
                        and interaction.get("status") == "completed"
                    ):
                        completed = True
                        yield b"data: [DONE]\n\n"
                        return
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_STREAM_INCOMPLETE","message":"The upstream interaction did not complete successfully."}}\n\n'
                    return
                if event_type in {
                    "interaction.failed",
                    "interaction.cancelled",
                    "interaction.incomplete",
                    "interaction.budget_exceeded",
                    "error",
                }:
                    yield b'event: error\ndata: {"error":{"code":"UPSTREAM_PROVIDER_ERROR","message":"The upstream provider stream failed."}}\n\n'
                    return
                if event_type == "step.delta":
                    delta = payload.get("delta")
                    if (
                        isinstance(delta, dict)
                        and delta.get("type") == "text"
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
