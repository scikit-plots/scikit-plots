# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Server-authoritative ZIP edit artifact orchestration.

Run 139 keeps model/provider output outside archive authority.  A caller must
supply three independently validated things:

* the exact source ZIP generation (size + SHA-256),
* an explicit set of existing paths authorized for modification, and
* a bounded proposal whose replacement parts are independently size/hash bound.

The proposal can name only authorized paths.  The source archive remains the
final tree authority because all mutation is delegated to ``_zip_workspace``.
No provider identifier, model metadata, credential, URL, or model-selected
archive structure participates in the rewrite contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, BinaryIO, Mapping

from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request

from ._zip_workspace import (
    DEFAULT_ZIP_WORKSPACE_LIMITS,
    ZipWorkspaceArtifact,
    ZipWorkspaceError,
    ZipWorkspaceLimits,
    rewrite_zip_workspace,
    validate_zip_workspace_file_path,
)

ZIP_EDIT_CONTRACT = "scikitplot-zip-edit-v1"
ZIP_EDIT_RECEIPT_CONTRACT = "scikitplot-zip-edit-receipt-v1"
ZIP_EDIT_MANIFEST_FIELD = "manifest"
ZIP_EDIT_SOURCE_FIELD = "archive"
ZIP_EDIT_REPLACEMENT_PREFIX = "replacement:"
ZIP_EDIT_MAX_MANIFEST_BYTES = 512 * 1024
ZIP_EDIT_MAX_AUTHORIZED_PATHS = 4096
ZIP_EDIT_MAX_REPLACEMENTS = 256
# Browser capability discovery must learn the same byte ceilings enforced by
# the workspace rather than duplicating them in UI code.  These aliases are
# intentionally derived from the immutable default limit object.
ZIP_EDIT_MAX_SOURCE_BYTES = DEFAULT_ZIP_WORKSPACE_LIMITS.max_source_bytes
ZIP_EDIT_MAX_ENTRY_BYTES = DEFAULT_ZIP_WORKSPACE_LIMITS.max_entry_uncompressed_bytes
ZIP_EDIT_MAX_REPLACEMENT_TOTAL_BYTES = (
    DEFAULT_ZIP_WORKSPACE_LIMITS.max_replacement_total_bytes
)
ZIP_EDIT_REQUEST_OVERHEAD_BYTES = 2 * 1024 * 1024
ZIP_EDIT_MAX_REQUEST_BYTES = (
    DEFAULT_ZIP_WORKSPACE_LIMITS.max_source_bytes
    + DEFAULT_ZIP_WORKSPACE_LIMITS.max_replacement_total_bytes
    + ZIP_EDIT_REQUEST_OVERHEAD_BYTES
)
_HASH_CHUNK_BYTES = 1024 * 1024
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,47}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LIFECYCLE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class ZipArtifactError(ValueError):
    """ZIP edit artifact request failed before an artifact could be returned."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "ZIP_EDIT_INVALID")


@dataclass(frozen=True)
class ZipEditSource:
    size: int
    sha256: str


@dataclass(frozen=True)
class ZipEditReplacement:
    id: str
    path: str
    size: int
    sha256: str
    provider_artifact_id: str = ""


@dataclass(frozen=True)
class ZipEditManifest:
    source: ZipEditSource
    authorized_paths: tuple[str, ...]
    replacements: tuple[ZipEditReplacement, ...]


@dataclass(frozen=True)
class ZipEditReceipt:
    """Bounded public receipt; authorized/proposed file bodies are never copied."""

    source_sha256: str
    output_sha256: str
    entry_count: int
    authorized_count: int
    applied_count: int
    unchanged_count: int
    tree_preserved: bool
    unchanged_content_preserved: bool
    metadata_preserved: bool
    provider_artifact_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "contract": ZIP_EDIT_RECEIPT_CONTRACT,
            "source_sha256": self.source_sha256,
            "output_sha256": self.output_sha256,
            "entry_count": self.entry_count,
            "authorized_count": self.authorized_count,
            "applied_count": self.applied_count,
            "unchanged_count": self.unchanged_count,
            "tree_preserved": self.tree_preserved,
            "unchanged_content_preserved": self.unchanged_content_preserved,
            "metadata_preserved": self.metadata_preserved,
        }
        if self.provider_artifact_ids:
            out["provider_artifact_ids"] = list(self.provider_artifact_ids)
        return out


@dataclass
class ZipEditArtifact:
    file: BinaryIO
    filename: str
    receipt: ZipEditReceipt

    def close(self) -> None:
        self.file.close()


@dataclass
class ParsedZipEditRequest:
    manifest: ZipEditManifest
    source: UploadFile
    replacements: dict[str, UploadFile]
    wire_body_bytes: int
    wire_body_sha256: str

    async def close(self) -> None:
        rows = [self.source, *self.replacements.values()]
        for upload in rows:
            try:  # ruff: ignore[suppressible-exception]
                await upload.close()
            except Exception:  # ruff: ignore[blind-except, try-except-in-loop]
                pass


class _ZipEditMultiPartParser(MultiPartParser):
    """Multipart parser with role-aware active byte ceilings."""

    def __init__(
        self,
        *args,
        limits: ZipWorkspaceLimits,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._limits = limits
        self._current_file_bytes = 0
        self._replacement_bytes = 0
        self._source_bytes = 0
        self._seen_manifest = False
        self._seen_source = False
        self._seen_replacement_ids: set[str] = set()

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._current_file_bytes = 0

    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        field = self._current_part.field_name
        if self._current_part.file is None:
            if field != ZIP_EDIT_MANIFEST_FIELD:
                raise MultiPartException(
                    "Unsupported ZIP edit multipart metadata field."
                )
            if self._seen_manifest:
                raise MultiPartException("Duplicate ZIP edit manifest field.")
            self._seen_manifest = True
            return
        if field == ZIP_EDIT_SOURCE_FIELD:
            if self._seen_source:
                raise MultiPartException("Duplicate ZIP edit archive field.")
            self._seen_source = True
            return
        if field.startswith(ZIP_EDIT_REPLACEMENT_PREFIX):
            rid = field[len(ZIP_EDIT_REPLACEMENT_PREFIX) :]
            if _ID_RE.fullmatch(rid):
                if rid in self._seen_replacement_ids:
                    raise MultiPartException("Duplicate ZIP edit replacement field.")
                self._seen_replacement_ids.add(rid)
                return
        raise MultiPartException("Unsupported ZIP edit multipart file field.")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        nbytes = max(0, end - start)
        if self._current_part.file is not None:
            self._current_file_bytes += nbytes
            field = self._current_part.field_name
            if field == ZIP_EDIT_SOURCE_FIELD:
                self._source_bytes += nbytes
                if self._current_file_bytes > self._limits.max_source_bytes:
                    raise MultiPartException(
                        "ZIP source exceeded configured byte limit."
                    )
            elif field.startswith(ZIP_EDIT_REPLACEMENT_PREFIX):
                self._replacement_bytes += nbytes
                if self._current_file_bytes > self._limits.max_entry_uncompressed_bytes:
                    raise MultiPartException(
                        "ZIP replacement exceeded configured per-entry byte limit."
                    )
                if self._replacement_bytes > self._limits.max_replacement_total_bytes:
                    raise MultiPartException(
                        "ZIP replacements exceeded configured aggregate byte limit."
                    )
            else:
                raise MultiPartException("Unsupported ZIP edit multipart file field.")
        super().on_part_data(data, start, end)


async def _bounded_stream(request: Request, maximum: int, stats: dict[str, object]):
    total = 0
    digest = hashlib.sha256()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > maximum:
                raise MultiPartException(
                    "ZIP edit request exceeded configured byte limit."
                )
        except ValueError as exc:
            raise MultiPartException("Invalid ZIP edit Content-Length.") from exc
    async for chunk in request.stream():
        total += len(chunk)
        if total > maximum:
            raise MultiPartException("ZIP edit request exceeded configured byte limit.")
        digest.update(chunk)
        yield chunk
    stats["wire_body_bytes"] = total
    stats["wire_body_sha256"] = digest.hexdigest()


def _object(value: Any, *, field: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID", f"{field} must be an object"
        )
    unknown = set(value) - allowed
    if unknown:
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID",
            f"unsupported {field} field(s): " + ", ".join(sorted(unknown)),
        )
    return value


def _positive_size(value: Any, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise ZipArtifactError("ZIP_EDIT_MANIFEST_INVALID", f"{field} is invalid")
    return value


def _sha256(value: Any, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID", f"{field} must be a SHA-256 hex digest"
        )
    return text


def _provider_artifact_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text and not _LIFECYCLE_ID_RE.fullmatch(text):
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID", "provider artifact lifecycle id is invalid"
        )
    return text


def _safe_edit_path(value: Any, *, limits: ZipWorkspaceLimits) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID", "ZIP edit paths must be strings"
        )
    try:
        path = validate_zip_workspace_file_path(value, limits=limits)
    except ZipWorkspaceError as exc:
        raise ZipArtifactError("ZIP_EDIT_MANIFEST_INVALID", str(exc)) from exc
    # Canonical key is request-only duplicate protection. The workspace still
    # requires exact source-entry spelling when the replacement is applied.
    import unicodedata  # ruff: ignore[import-outside-top-level]

    canonical = unicodedata.normalize("NFC", path).casefold()
    return path, canonical


def parse_zip_edit_manifest(  # ruff: ignore[too-many-branches]
    value: bytes | str | Mapping[str, Any],
    *,
    limits: ZipWorkspaceLimits = DEFAULT_ZIP_WORKSPACE_LIMITS,
) -> ZipEditManifest:
    """Parse a strict model-neutral authorization/proposal manifest."""
    if isinstance(value, Mapping):
        raw = dict(value)
        try:
            encoded = json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ZipArtifactError(
                "ZIP_EDIT_MANIFEST_INVALID", "ZIP edit manifest must be JSON-compatible"
            ) from exc
        if len(encoded) > ZIP_EDIT_MAX_MANIFEST_BYTES:
            raise ZipArtifactError(
                "ZIP_EDIT_MANIFEST_TOO_LARGE",
                "ZIP edit manifest exceeds configured limit",
            )
    else:
        if isinstance(value, bytes):
            if len(value) > ZIP_EDIT_MAX_MANIFEST_BYTES:
                raise ZipArtifactError(
                    "ZIP_EDIT_MANIFEST_TOO_LARGE",
                    "ZIP edit manifest exceeds configured limit",
                )
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ZipArtifactError(
                    "ZIP_EDIT_MANIFEST_INVALID", "ZIP edit manifest must be UTF-8"
                ) from exc
        elif not isinstance(value, str):
            raise ZipArtifactError(
                "ZIP_EDIT_MANIFEST_INVALID", "ZIP edit manifest must be JSON"
            )
        if len(value.encode("utf-8")) > ZIP_EDIT_MAX_MANIFEST_BYTES:
            raise ZipArtifactError(
                "ZIP_EDIT_MANIFEST_TOO_LARGE",
                "ZIP edit manifest exceeds configured limit",
            )
        try:
            raw = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ZipArtifactError(
                "ZIP_EDIT_MANIFEST_INVALID", "ZIP edit manifest must be valid JSON"
            ) from exc

    root = _object(
        raw,
        field="manifest",
        allowed={"contract", "source", "authorization", "proposal"},
    )
    if root.get("contract") != ZIP_EDIT_CONTRACT:
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID", f"contract must be {ZIP_EDIT_CONTRACT!r}"
        )

    source_raw = _object(root.get("source"), field="source", allowed={"size", "sha256"})
    source = ZipEditSource(
        size=_positive_size(
            source_raw.get("size"), field="source.size", maximum=limits.max_source_bytes
        ),
        sha256=_sha256(source_raw.get("sha256"), field="source.sha256"),
    )

    auth_raw = _object(
        root.get("authorization"), field="authorization", allowed={"paths"}
    )
    auth_value = auth_raw.get("paths")
    if not isinstance(auth_value, list) or not auth_value:
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID", "authorization.paths must be a non-empty array"
        )
    if len(auth_value) > min(ZIP_EDIT_MAX_AUTHORIZED_PATHS, limits.max_entries):
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID", "authorization.paths exceeds configured count"
        )
    authorized: list[str] = []
    auth_canonical: set[str] = set()
    for value in auth_value:
        path, canonical = _safe_edit_path(value, limits=limits)
        if canonical in auth_canonical:
            raise ZipArtifactError(
                "ZIP_EDIT_MANIFEST_INVALID",
                "authorization.paths contains duplicate or aliased paths",
            )
        auth_canonical.add(canonical)
        authorized.append(path)

    proposal_raw = _object(
        root.get("proposal"), field="proposal", allowed={"replacements"}
    )
    proposal_value = proposal_raw.get("replacements")
    if not isinstance(proposal_value, list) or not proposal_value:
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID",
            "proposal.replacements must be a non-empty array",
        )
    if len(proposal_value) > ZIP_EDIT_MAX_REPLACEMENTS:
        raise ZipArtifactError(
            "ZIP_EDIT_MANIFEST_INVALID",
            "proposal.replacements exceeds configured count",
        )

    replacements: list[ZipEditReplacement] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    authorized_exact = set(authorized)
    for row in proposal_value:
        item = _object(
            row,
            field="proposal replacement",
            allowed={"id", "path", "size", "sha256", "provider_artifact_id"},
        )
        rid = str(item.get("id") or "")
        if not _ID_RE.fullmatch(rid) or rid in seen_ids:
            raise ZipArtifactError(
                "ZIP_EDIT_MANIFEST_INVALID",
                "proposal replacement id is invalid or duplicate",
            )
        seen_ids.add(rid)
        path, canonical = _safe_edit_path(item.get("path"), limits=limits)
        if canonical in seen_paths:
            raise ZipArtifactError(
                "ZIP_EDIT_MANIFEST_INVALID",
                "proposal contains duplicate or aliased paths",
            )
        seen_paths.add(canonical)
        if path not in authorized_exact:
            raise ZipArtifactError(
                "ZIP_EDIT_NOT_AUTHORIZED",
                "proposal path is not present in authorization.paths",
            )
        replacements.append(
            ZipEditReplacement(
                id=rid,
                path=path,
                size=_positive_size(
                    item.get("size"),
                    field="proposal replacement size",
                    maximum=limits.max_entry_uncompressed_bytes,
                ),
                sha256=_sha256(item.get("sha256"), field="proposal replacement sha256"),
                provider_artifact_id=_provider_artifact_id(
                    item.get("provider_artifact_id")
                ),
            )
        )

    if sum(row.size for row in replacements) > limits.max_replacement_total_bytes:
        raise ZipArtifactError(
            "ZIP_EDIT_TOO_LARGE",
            "proposal replacement bytes exceed configured aggregate limit",
        )
    return ZipEditManifest(
        source=source,
        authorized_paths=tuple(authorized),
        replacements=tuple(replacements),
    )


async def _measure_upload(upload: UploadFile, *, maximum: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    await upload.seek(0)
    while True:
        chunk = await upload.read(_HASH_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > maximum:
            raise ZipArtifactError(
                "ZIP_EDIT_TOO_LARGE", "ZIP edit upload exceeds configured byte limit"
            )
        digest.update(chunk)
    await upload.seek(0)
    return total, digest.hexdigest()


async def parse_zip_edit_request(  # ruff: ignore[too-many-branches]
    request: Request,
    *,
    limits: ZipWorkspaceLimits = DEFAULT_ZIP_WORKSPACE_LIMITS,
    max_request_bytes: int = ZIP_EDIT_MAX_REQUEST_BYTES,
) -> ParsedZipEditRequest:
    """Parse and independently bind every multipart byte generation."""
    content_type = request.headers.get("content-type", "").lower()
    if not content_type.startswith("multipart/form-data"):
        raise ZipArtifactError(
            "ZIP_EDIT_INVALID", "ZIP edit request must use multipart/form-data"
        )
    stats: dict[str, object] = {}
    parser = _ZipEditMultiPartParser(
        headers=request.headers,
        stream=_bounded_stream(request, max_request_bytes, stats),
        max_files=ZIP_EDIT_MAX_REPLACEMENTS + 1,
        max_fields=1,
        max_part_size=ZIP_EDIT_MAX_MANIFEST_BYTES,
        limits=limits,
    )
    try:
        form: FormData = await parser.parse()
    except ZipArtifactError:
        raise
    except MultiPartException as exc:
        text = str(exc)
        code = (
            "ZIP_EDIT_TOO_LARGE"
            if "limit" in text.lower() or "exceeded" in text.lower()
            else "ZIP_EDIT_INVALID"
        )
        raise ZipArtifactError(code, text) from exc

    uploads_seen = [
        value for _key, value in form.multi_items() if isinstance(value, UploadFile)
    ]
    try:
        manifest_values = form.getlist(ZIP_EDIT_MANIFEST_FIELD)
        if len(manifest_values) != 1 or not isinstance(manifest_values[0], str):
            raise ZipArtifactError(
                "ZIP_EDIT_INVALID",
                "multipart request must contain exactly one manifest field",
            )
        manifest = parse_zip_edit_manifest(manifest_values[0], limits=limits)

        source: UploadFile | None = None
        replacements: dict[str, UploadFile] = {}
        for key, value in form.multi_items():
            if key == ZIP_EDIT_MANIFEST_FIELD:
                continue
            if not isinstance(value, UploadFile):
                raise ZipArtifactError(
                    "ZIP_EDIT_INVALID", "unsupported ZIP edit multipart field"
                )
            if key == ZIP_EDIT_SOURCE_FIELD:
                if source is not None:
                    raise ZipArtifactError(
                        "ZIP_EDIT_INVALID",
                        "ZIP edit request contains duplicate archive fields",
                    )
                source = value
                continue
            if not key.startswith(ZIP_EDIT_REPLACEMENT_PREFIX):
                raise ZipArtifactError(
                    "ZIP_EDIT_INVALID", "unsupported ZIP edit multipart file field"
                )
            rid = key[len(ZIP_EDIT_REPLACEMENT_PREFIX) :]
            if not _ID_RE.fullmatch(rid) or rid in replacements:
                raise ZipArtifactError(
                    "ZIP_EDIT_INVALID",
                    "ZIP edit replacement multipart id is invalid or duplicate",
                )
            replacements[rid] = value
        if source is None:
            raise ZipArtifactError(
                "ZIP_EDIT_INVALID", "ZIP edit request is missing the archive file"
            )

        expected_ids = {row.id for row in manifest.replacements}
        if set(replacements) != expected_ids:
            raise ZipArtifactError(
                "ZIP_EDIT_INVALID",
                "replacement multipart parts do not match proposal ids",
            )

        source_size, source_sha = await _measure_upload(
            source, maximum=limits.max_source_bytes
        )
        if source_size != manifest.source.size or source_sha != manifest.source.sha256:
            raise ZipArtifactError(
                "ZIP_EDIT_SOURCE_MISMATCH",
                "source ZIP does not match manifest generation",
            )

        by_id = {row.id: row for row in manifest.replacements}
        for rid, upload in replacements.items():
            descriptor = by_id[rid]
            actual_size, actual_sha = await _measure_upload(
                upload, maximum=limits.max_entry_uncompressed_bytes
            )
            if actual_size != descriptor.size or actual_sha != descriptor.sha256:
                raise ZipArtifactError(
                    "ZIP_EDIT_REPLACEMENT_MISMATCH",
                    "replacement bytes do not match proposal descriptor",
                )

        return ParsedZipEditRequest(
            manifest=manifest,
            source=source,
            replacements=replacements,
            wire_body_bytes=int(stats.get("wire_body_bytes", 0)),
            wire_body_sha256=str(stats.get("wire_body_sha256", "")),
        )
    except Exception:
        for upload in uploads_seen:
            try:  # ruff: ignore[suppressible-exception]
                await upload.close()
            except Exception:  # ruff: ignore[blind-except, try-except-in-loop]
                pass
        raise


def _safe_output_filename(filename: str | None) -> str:
    raw = str(filename or "project.zip").replace("\\", "/").split("/")[-1]
    raw = _CONTROL_RE.sub("", raw).strip()
    stem = raw[:-4] if raw.lower().endswith(".zip") else raw
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:120]
    if not stem:
        stem = "project"
    return f"{stem}.modified.zip"


def build_zip_edit_artifact(
    parsed: ParsedZipEditRequest,
    *,
    limits: ZipWorkspaceLimits = DEFAULT_ZIP_WORKSPACE_LIMITS,
) -> ZipEditArtifact:
    """Apply one verified authorization/proposal set to the complete source ZIP."""
    by_id = {row.id: row for row in parsed.manifest.replacements}
    replacements: dict[str, BinaryIO] = {
        by_id[rid].path: upload.file for rid, upload in parsed.replacements.items()
    }
    try:
        workspace: ZipWorkspaceArtifact = rewrite_zip_workspace(
            parsed.source.file,
            replacements,
            authorized_paths=parsed.manifest.authorized_paths,
            expected_source_sha256=parsed.manifest.source.sha256,
            expected_source_size=parsed.manifest.source.size,
            replacement_expectations={
                row.path: (row.size, row.sha256) for row in parsed.manifest.replacements
            },
            limits=limits,
        )
    except ZipWorkspaceError as exc:
        message = str(exc)
        if message.startswith("ZIP source generation does not match expected"):
            raise ZipArtifactError("ZIP_EDIT_SOURCE_MISMATCH", message) from exc
        if message == "ZIP replacement generation does not match expected identity":
            raise ZipArtifactError("ZIP_EDIT_REPLACEMENT_MISMATCH", message) from exc
        raise ZipArtifactError("ZIP_EDIT_WORKSPACE_REJECTED", message) from exc

    if workspace.receipt.source_sha256 != parsed.manifest.source.sha256:
        workspace.close()
        raise ZipArtifactError(
            "ZIP_EDIT_SOURCE_MISMATCH", "workspace source generation changed"
        )
    if set(workspace.receipt.changed_paths) != {
        row.path for row in parsed.manifest.replacements
    }:
        workspace.close()
        raise ZipArtifactError(
            "ZIP_EDIT_VERIFICATION_FAILED", "workspace changed-path receipt mismatch"
        )

    receipt = ZipEditReceipt(
        source_sha256=workspace.receipt.source_sha256,
        output_sha256=workspace.receipt.output_sha256,
        entry_count=workspace.receipt.entry_count,
        authorized_count=len(parsed.manifest.authorized_paths),
        applied_count=len(parsed.manifest.replacements),
        unchanged_count=workspace.receipt.unchanged_count,
        tree_preserved=workspace.receipt.tree_preserved,
        unchanged_content_preserved=workspace.receipt.unchanged_content_preserved,
        metadata_preserved=workspace.receipt.metadata_preserved,
    )
    return ZipEditArtifact(
        file=workspace.file,
        filename=_safe_output_filename(parsed.source.filename),
        receipt=receipt,
    )


def encode_zip_edit_receipt_header(receipt: ZipEditReceipt) -> str:
    """Encode a path-free receipt small enough for a browser-visible response header."""
    return json.dumps(
        receipt.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


__all__ = (
    "ZIP_EDIT_CONTRACT",
    "ZIP_EDIT_MAX_ENTRY_BYTES",
    "ZIP_EDIT_MAX_REPLACEMENT_TOTAL_BYTES",
    "ZIP_EDIT_MAX_REQUEST_BYTES",
    "ZIP_EDIT_MAX_SOURCE_BYTES",
    "ZIP_EDIT_RECEIPT_CONTRACT",
    "ParsedZipEditRequest",
    "ZipArtifactError",
    "ZipEditArtifact",
    "ZipEditManifest",
    "ZipEditReceipt",
    "build_zip_edit_artifact",
    "encode_zip_edit_receipt_header",
    "parse_zip_edit_manifest",
    "parse_zip_edit_request",
)
