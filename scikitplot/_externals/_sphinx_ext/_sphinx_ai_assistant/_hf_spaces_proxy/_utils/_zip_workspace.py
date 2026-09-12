# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tree-preserving, bounded ZIP rewrite workspace.

This module is deliberately provider-neutral.  It accepts an original ZIP as
server-owned bytes, authorizes replacement of *existing regular files only*,
and returns a complete rewritten ZIP after independently verifying its tree and
content.

Security invariants
-------------------
* Never extract an archive to the filesystem and never call ``extractall()``.
* The original archive is the tree authority; replacements cannot add, delete,
  rename, or replace directories.
* ZIP paths are canonicalized for Unicode/case aliases and rejected when they
  are unsafe on common filesystems.
* Symlinks and special Unix filesystem entries are rejected.
* Entry count, entry size, aggregate size, replacement size, compression ratio,
  and output growth are bounded before the rewrite is committed.
* Unchanged local records (local header + compressed payload + optional data
  descriptor) are copied byte-for-byte; only changed records are regenerated.
* The central directory is rebuilt with fresh local-header offsets and bounded
  non-ZIP64 transport fields while preserving portable metadata.
* Unchanged entries still receive an independent decompression/CRC verification
  pass before the artifact is returned, so raw-copy fidelity never weakens the
  Run 137 corrupt-payload rejection contract.
* Source SHA-256 is rechecked before and after the rewrite to detect mutation of
  a seekable source between inspection and artifact creation.
* ZIP64 transport-derived extra records are not copied into the rewritten
  bounded archive; other well-formed extra records are preserved.

Run 138 is the surgical successor to the Run 137 reference rewrite.  It removes
recompression of untouched entries while retaining the same tree authority,
resource ceilings, corruption detection, and output verification boundary.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import struct
import tempfile
import unicodedata
import zipfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Mapping

from typing_extensions import Self

ZIP_CHUNK_BYTES = 1024 * 1024
ZIP_OUTPUT_SPOOL_BYTES = 8 * 1024 * 1024
ZIP_MAX_ENTRIES = 4096
ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
ZIP_MAX_REPLACEMENT_TOTAL_BYTES = 256 * 1024 * 1024
ZIP_MAX_COMPRESSION_RATIO = 500.0
ZIP_MAX_SOURCE_BYTES = 512 * 1024 * 1024
ZIP_MAX_PATH_CHARS = 4096

_LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50
_CENTRAL_DIRECTORY_SIGNATURE = 0x02014B50
_END_CENTRAL_DIRECTORY_SIGNATURE = 0x06054B50
_DATA_DESCRIPTOR_SIGNATURE = 0x08074B50
_LOCAL_FILE_HEADER_SIZE = 30
_CENTRAL_DIRECTORY_HEADER_SIZE = 46
_END_CENTRAL_DIRECTORY_SIZE = 22
_ALLOWED_FLAG_MASK = 0x080E  # deflate hints, data descriptor, UTF-8

_ZIP64_EXTRA_ID = 0x0001
_ALLOWED_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_BIDI_CONTROLS = frozenset(
    {
        "\u061c",  # Arabic letter mark
        "\u200e",
        "\u200f",  # LRM/RLM
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",  # embeddings/override
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",  # isolates
    }
)


class ZipWorkspaceError(ValueError):
    """The ZIP cannot safely participate in the edit-and-return workspace."""


@dataclass(frozen=True)
class ZipWorkspaceLimits:
    """Resource ceilings for a ZIP rewrite workspace."""

    max_entries: int = ZIP_MAX_ENTRIES
    max_entry_uncompressed_bytes: int = ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES
    max_total_uncompressed_bytes: int = ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES
    max_replacement_total_bytes: int = ZIP_MAX_REPLACEMENT_TOTAL_BYTES
    max_compression_ratio: float = ZIP_MAX_COMPRESSION_RATIO
    max_source_bytes: int = ZIP_MAX_SOURCE_BYTES
    chunk_bytes: int = ZIP_CHUNK_BYTES
    output_spool_bytes: int = ZIP_OUTPUT_SPOOL_BYTES
    max_path_chars: int = ZIP_MAX_PATH_CHARS

    def __post_init__(self) -> None:
        ints = (
            self.max_entries,
            self.max_entry_uncompressed_bytes,
            self.max_total_uncompressed_bytes,
            self.max_replacement_total_bytes,
            self.max_source_bytes,
            self.chunk_bytes,
            self.output_spool_bytes,
            self.max_path_chars,
        )
        if (
            any(int(value) <= 0 for value in ints)
            or float(self.max_compression_ratio) <= 0
        ):
            raise ValueError("ZIP workspace limits must be positive")


DEFAULT_ZIP_WORKSPACE_LIMITS = ZipWorkspaceLimits()


@dataclass(frozen=True)
class ZipWorkspaceReceipt:
    """Bounded provenance for a successful complete-archive rewrite."""

    source_sha256: str
    output_sha256: str
    entry_count: int
    changed_paths: tuple[str, ...]
    unchanged_count: int
    tree_preserved: bool = True
    unchanged_content_preserved: bool = True
    metadata_preserved: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "output_sha256": self.output_sha256,
            "entry_count": self.entry_count,
            "changed_paths": list(self.changed_paths),
            "unchanged_count": self.unchanged_count,
            "tree_preserved": self.tree_preserved,
            "unchanged_content_preserved": self.unchanged_content_preserved,
            "metadata_preserved": self.metadata_preserved,
        }


@dataclass
class ZipWorkspaceArtifact:
    """Verified output archive and its bounded provenance receipt."""

    file: BinaryIO
    receipt: ZipWorkspaceReceipt

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


@dataclass(frozen=True)
class _EntryPlan:
    info: zipfile.ZipInfo
    safe_extra: bytes
    is_dir: bool
    canonical_key: str


@dataclass
class _Replacement:
    file: BinaryIO
    size: int
    sha256: str

    def close(self) -> None:
        self.file.close()


@dataclass(frozen=True)
class _ArchivePlan:
    comment: bytes
    entries: tuple[_EntryPlan, ...]
    source_total_uncompressed: int
    start_dir: int


@contextmanager
def _source_handle(source: str | os.PathLike[str] | BinaryIO) -> Iterator[BinaryIO]:
    if isinstance(source, (str, os.PathLike)):
        with open(Path(source), "rb") as fh:
            yield fh
        return
    if not hasattr(source, "read") or not hasattr(source, "seek"):
        raise ZipWorkspaceError("ZIP source must be a path or seekable binary file")
    try:
        source.seek(0)
    except Exception as exc:
        raise ZipWorkspaceError("ZIP source must be seekable") from exc
    yield source


def _cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise ZipWorkspaceError("ZIP workspace operation cancelled")


def _hash_source(
    source: str | os.PathLike[str] | BinaryIO,
    *,
    limits: ZipWorkspaceLimits,
    should_cancel: Callable[[], bool] | None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with _source_handle(source) as fh:
        try:
            fh.seek(0)
            while True:
                _cancelled(should_cancel)
                chunk = fh.read(limits.chunk_bytes)
                if not chunk:
                    break
                total += len(chunk)
                if total > limits.max_source_bytes:
                    raise ZipWorkspaceError(
                        "ZIP archive exceeds configured source byte limit"
                    )
                digest.update(chunk)
        finally:
            try:  # ruff: ignore[suppressible-exception]
                fh.seek(0)
            except Exception:  # ruff: ignore[blind-except]
                pass
    return digest.hexdigest(), total


def _snapshot_source(
    source: str | os.PathLike[str] | BinaryIO,
    *,
    limits: ZipWorkspaceLimits,
    should_cancel: Callable[[], bool] | None,
) -> tuple[BinaryIO, str]:
    """Copy the authoritative archive generation into a bounded server spool."""
    snapshot = (
        # lint
        tempfile.SpooledTemporaryFile(  # ruff: ignore[open-file-with-context-handler]
            max_size=limits.output_spool_bytes,
            mode="w+b",
        )
    )
    digest = hashlib.sha256()
    total = 0
    try:
        with _source_handle(source) as fh:
            while True:
                _cancelled(should_cancel)
                chunk = fh.read(limits.chunk_bytes)
                if not chunk:
                    break
                total += len(chunk)
                if total > limits.max_source_bytes:
                    raise ZipWorkspaceError(
                        "ZIP archive exceeds configured source byte limit"
                    )
                snapshot.write(chunk)
                digest.update(chunk)
        snapshot.seek(0)
        return snapshot, digest.hexdigest()
    except Exception:
        snapshot.close()
        raise


def _contains_control(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (
            cp == 0
            or cp < 0x20  # ruff: ignore[magic-value-comparison]
            or 0x7F <= cp <= 0x9F  # ruff: ignore[magic-value-comparison]
            or ch in _BIDI_CONTROLS
        ):
            return True
    return False


def _validate_path(name: str, *, max_chars: int) -> tuple[str, bool]:
    if not isinstance(name, str) or not name:
        raise ZipWorkspaceError("ZIP archive contains an empty entry path")
    if len(name) > max_chars:
        raise ZipWorkspaceError("ZIP archive entry path exceeds configured limit")
    if _contains_control(name):
        raise ZipWorkspaceError("ZIP archive contains a control or bidi-format path")
    if "\\" in name:
        raise ZipWorkspaceError("ZIP archive path must use forward slashes only")
    if name.startswith("/") or _DRIVE_RE.match(name):
        raise ZipWorkspaceError(
            "ZIP archive contains an absolute or drive-qualified path"
        )

    is_dir = name.endswith("/")
    body = name[:-1] if is_dir else name
    if not body:
        raise ZipWorkspaceError("ZIP archive root directory entry is not allowed")
    parts = body.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ZipWorkspaceError(
            "ZIP archive contains traversal or ambiguous path segments"
        )
    if any(part.rstrip(" .") != part for part in parts):
        raise ZipWorkspaceError("ZIP archive contains trailing-dot/space path aliases")

    normalized = unicodedata.normalize("NFC", body)
    canonical = normalized.casefold()
    return canonical, is_dir


def _sanitize_extra(extra: bytes) -> bytes:
    """Preserve well-formed extra TLVs except derived ZIP64 transport data."""
    if not extra:
        return b""
    out = bytearray()
    offset = 0
    size = len(extra)
    while offset < size:
        if size - offset < 4:  # ruff: ignore[magic-value-comparison]
            raise ZipWorkspaceError(
                "ZIP archive contains malformed extra-field metadata"
            )
        field_id, field_size = struct.unpack_from("<HH", extra, offset)
        end = offset + 4 + field_size
        if end > size:
            raise ZipWorkspaceError(
                "ZIP archive contains malformed extra-field metadata"
            )
        if field_id != _ZIP64_EXTRA_ID:
            out.extend(extra[offset:end])
        offset = end
    return bytes(out)


def validate_zip_workspace_file_path(
    name: str,
    *,
    limits: ZipWorkspaceLimits = DEFAULT_ZIP_WORKSPACE_LIMITS,
) -> str:
    """
    Validate one caller-supplied existing-file path for ZIP edit manifests.

    This is intentionally a syntax/canonicalization gate only. Existence and
    regular-file authority remain owned by :func:`rewrite_zip_workspace` and
    the source archive generation.
    """
    _canonical, is_dir = _validate_path(name, max_chars=limits.max_path_chars)
    if is_dir:
        raise ZipWorkspaceError("ZIP edit path must name a regular file")
    return name


def _validate_unix_type(info: zipfile.ZipInfo, *, path_is_dir: bool) -> None:
    if int(info.create_system) != 3:  # ruff: ignore[magic-value-comparison]
        return
    mode = (int(info.external_attr) >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type == 0:
        return
    if file_type == stat.S_IFDIR:
        if not path_is_dir:
            raise ZipWorkspaceError(
                "ZIP Unix type metadata disagrees with entry path shape"
            )
        return
    if file_type == stat.S_IFREG:
        if path_is_dir:
            raise ZipWorkspaceError(
                "ZIP Unix type metadata disagrees with entry path shape"
            )
        return
    if file_type == stat.S_IFLNK:
        raise ZipWorkspaceError("ZIP archive symlinks are not allowed")
    raise ZipWorkspaceError("ZIP archive special filesystem entries are not allowed")


def _validate_structural_collisions(entries: tuple[_EntryPlan, ...]) -> None:
    exact: dict[str, bool] = {}
    files: set[str] = set()
    all_nodes: set[str] = set()
    for row in entries:
        key = row.canonical_key
        if key in exact:
            raise ZipWorkspaceError("ZIP archive contains duplicate or aliased paths")
        exact[key] = row.is_dir
        all_nodes.add(key)
        if not row.is_dir:
            files.add(key)

    for key in all_nodes:
        parts = key.split("/")
        for index in range(1, len(parts)):
            ancestor = "/".join(parts[:index])
            if ancestor in files:
                raise ZipWorkspaceError(
                    "ZIP archive contains a file/descendant path collision"
                )


def _validate_eocd(
    fh: BinaryIO,
    *,
    expected_entries: int,
    expected_comment: bytes,
) -> tuple[int, int, int]:
    """Reject multi-disk/trailing junk and return EOCD/CD geometry."""
    fh.seek(0, os.SEEK_END)
    size = fh.tell()
    tail_size = min(size, _END_CENTRAL_DIRECTORY_SIZE + 0xFFFF)
    fh.seek(size - tail_size)
    tail = _read_exact(fh, tail_size)
    signature = struct.pack("<I", _END_CENTRAL_DIRECTORY_SIGNATURE)
    cursor = len(tail)
    found = None
    while True:
        index = tail.rfind(signature, 0, cursor)
        if index < 0:
            break
        if len(tail) - index >= _END_CENTRAL_DIRECTORY_SIZE:
            fields = struct.unpack_from("<I4H2IH", tail, index)
            comment_len = fields[-1]
            if index + _END_CENTRAL_DIRECTORY_SIZE + comment_len == len(tail):
                found = (index, fields)
                break
        cursor = index
    if found is None:
        raise ZipWorkspaceError(
            "ZIP archive end-of-central-directory record is malformed"
        )
    index, fields = found
    _, disk_no, disk_cd, count_disk, count_total, cd_size, cd_offset, comment_len = (
        fields
    )
    if disk_no != 0 or disk_cd != 0 or count_disk != count_total:
        raise ZipWorkspaceError("Multi-disk ZIP archives are not supported")
    if (
        count_total in {0xFFFF}  # ruff: ignore[single-item-membership-test]
        or cd_size == 0xFFFFFFFF  # ruff: ignore[magic-value-comparison]
        or cd_offset == 0xFFFFFFFF  # ruff: ignore[magic-value-comparison]
    ):
        raise ZipWorkspaceError(
            "Archive-level ZIP64 is outside the bounded workspace contract"
        )
    if count_total != expected_entries:
        raise ZipWorkspaceError(
            "ZIP archive central-directory entry count is inconsistent"
        )
    _x = index + _END_CENTRAL_DIRECTORY_SIZE
    _y = index + _END_CENTRAL_DIRECTORY_SIZE + comment_len
    comment = tail[_x:_y]
    if comment != expected_comment:
        raise ZipWorkspaceError("ZIP archive comment metadata is inconsistent")
    eocd_offset = size - tail_size + index
    return eocd_offset, int(cd_size), int(cd_offset)


def _validate_central_layout(  # ruff: ignore[too-many-branches]
    fh: BinaryIO,
    *,
    rows: tuple[_EntryPlan, ...],
    start_dir: int,
    eocd_offset: int,
    cd_size: int,
    cd_offset: int,
) -> None:
    """Require the central directory to contain only the declared file rows."""
    if start_dir < 0 or eocd_offset < start_dir:
        raise ZipWorkspaceError("ZIP archive central-directory geometry is invalid")
    if eocd_offset - start_dir != cd_size:
        raise ZipWorkspaceError("ZIP archive central-directory size is inconsistent")
    archive_base = start_dir - cd_offset
    if archive_base < 0:
        raise ZipWorkspaceError("ZIP archive central-directory offset is inconsistent")

    fh.seek(start_dir)
    for row in rows:
        fixed = _read_exact(fh, _CENTRAL_DIRECTORY_HEADER_SIZE)
        (
            signature,
            _version_made,
            _version_needed,
            flag_bits,
            compression,
            _mtime,
            _mdate,
            crc,
            csize,
            usize,
            name_len,
            extra_len,
            comment_len,
            disk_start,
            internal_attr,
            external_attr,
            local_offset,
        ) = struct.unpack("<I6H3I5H2I", fixed)
        info = row.info
        if signature != _CENTRAL_DIRECTORY_SIGNATURE:
            raise ZipWorkspaceError(
                "ZIP archive contains unsupported central-directory records"
            )
        if flag_bits != int(info.flag_bits) or compression != int(info.compress_type):
            raise ZipWorkspaceError(
                "ZIP archive central-directory metadata is inconsistent"
            )
        if crc != int(info.CRC):
            raise ZipWorkspaceError(
                "ZIP archive central-directory CRC metadata is inconsistent"
            )
        if csize != 0xFFFFFFFF and csize != int(  # ruff: ignore[magic-value-comparison]
            info.compress_size
        ):
            raise ZipWorkspaceError(
                "ZIP archive central-directory size metadata is inconsistent"
            )
        if usize != 0xFFFFFFFF and usize != int(  # ruff: ignore[magic-value-comparison]
            info.file_size
        ):
            raise ZipWorkspaceError(
                "ZIP archive central-directory size metadata is inconsistent"
            )
        if disk_start not in {0, 0xFFFF}:
            raise ZipWorkspaceError("Multi-disk ZIP archives are not supported")
        if internal_attr != int(info.internal_attr) or external_attr != int(
            info.external_attr
        ):
            raise ZipWorkspaceError(
                "ZIP archive central-directory attributes are inconsistent"
            )
        if (
            local_offset != 0xFFFFFFFF  # ruff: ignore[magic-value-comparison]
            and local_offset + archive_base != int(info.header_offset)
        ):
            raise ZipWorkspaceError(
                "ZIP archive local-header offset metadata is inconsistent"
            )

        raw_name = _read_exact(fh, name_len)
        if _decode_filename(raw_name, flag_bits) != info.filename:
            raise ZipWorkspaceError(
                "ZIP archive central-directory filename is inconsistent"
            )
        raw_extra = _read_exact(fh, extra_len)
        if _sanitize_extra(raw_extra) != row.safe_extra:
            raise ZipWorkspaceError(
                "ZIP archive central-directory extra metadata is inconsistent"
            )
        raw_comment = _read_exact(fh, comment_len)
        if raw_comment != bytes(info.comment):
            raise ZipWorkspaceError(
                "ZIP archive central-directory comment is inconsistent"
            )

    if fh.tell() != eocd_offset:
        raise ZipWorkspaceError(
            "ZIP archive contains unsupported central-directory records"
        )


def _inspect_archive(  # ruff: ignore[too-many-branches]
    source: str | os.PathLike[str] | BinaryIO,
    *,
    limits: ZipWorkspaceLimits,
) -> _ArchivePlan:
    try:
        with _source_handle(source) as fh, zipfile.ZipFile(
            fh, mode="r", allowZip64=True
        ) as zin:
            infos = zin.infolist()
            if len(infos) > limits.max_entries:
                raise ZipWorkspaceError(
                    "ZIP archive exceeds configured entry-count limit"
                )
            rows: list[_EntryPlan] = []
            total = 0
            for info in infos:
                canonical, path_is_dir = _validate_path(
                    info.filename, max_chars=limits.max_path_chars
                )
                _validate_unix_type(info, path_is_dir=path_is_dir)
                if info.flag_bits & ~_ALLOWED_FLAG_MASK:
                    raise ZipWorkspaceError(
                        "ZIP archive uses unsupported general-purpose flags"
                    )
                if info.volume != 0:
                    raise ZipWorkspaceError("Multi-disk ZIP archives are not supported")
                if info.compress_type not in _ALLOWED_COMPRESSION:
                    raise ZipWorkspaceError(
                        "ZIP archive uses an unsupported compression method"
                    )
                if info.file_size < 0 or info.compress_size < 0:
                    raise ZipWorkspaceError("ZIP archive contains invalid entry sizes")
                if path_is_dir and (info.file_size != 0 or info.CRC != 0):
                    raise ZipWorkspaceError(
                        "ZIP directory entries must not carry file payloads"
                    )
                if info.file_size > limits.max_entry_uncompressed_bytes:
                    raise ZipWorkspaceError(
                        "ZIP archive entry exceeds configured uncompressed limit"
                    )
                total += info.file_size
                if total > limits.max_total_uncompressed_bytes:
                    raise ZipWorkspaceError(
                        "ZIP archive exceeds configured total uncompressed limit"
                    )
                if info.file_size > 0:
                    if info.compress_size <= 0:
                        raise ZipWorkspaceError(
                            "ZIP archive exceeds configured compression-ratio limit"
                        )
                    if (
                        info.file_size / info.compress_size
                        > limits.max_compression_ratio
                    ):
                        raise ZipWorkspaceError(
                            "ZIP archive exceeds configured compression-ratio limit"
                        )
                safe_extra = _sanitize_extra(info.extra)
                rows.append(
                    _EntryPlan(
                        info=info,
                        safe_extra=safe_extra,
                        is_dir=path_is_dir,
                        canonical_key=canonical,
                    )
                )
            rows_tuple = tuple(rows)
            eocd_offset, cd_size, cd_offset = _validate_eocd(
                fh, expected_entries=len(infos), expected_comment=bytes(zin.comment)
            )
            _validate_central_layout(
                fh,
                rows=rows_tuple,
                start_dir=int(zin.start_dir),
                eocd_offset=eocd_offset,
                cd_size=cd_size,
                cd_offset=cd_offset,
            )
            plan = _ArchivePlan(
                comment=bytes(zin.comment),
                entries=rows_tuple,
                source_total_uncompressed=total,
                start_dir=int(zin.start_dir),
            )
    except ZipWorkspaceError:
        raise
    except zipfile.BadZipFile as exc:
        if "Corrupt extra field" in str(exc):
            raise ZipWorkspaceError(
                "ZIP archive contains malformed extra-field metadata"
            ) from exc
        raise ZipWorkspaceError("ZIP archive is malformed or corrupt") from exc
    except (zipfile.LargeZipFile, EOFError, OSError, ValueError, struct.error) as exc:
        raise ZipWorkspaceError("ZIP archive is malformed or corrupt") from exc

    _validate_structural_collisions(plan.entries)
    return plan


def _snapshot_replacement_value(  # ruff: ignore[too-many-branches]
    value: bytes | bytearray | memoryview | BinaryIO,
    *,
    limits: ZipWorkspaceLimits,
    aggregate_so_far: int,
    should_cancel: Callable[[], bool] | None,
) -> tuple[_Replacement, int]:
    """Copy one replacement generation into a bounded server-owned spool."""
    spool = (
        # lint
        tempfile.SpooledTemporaryFile(  # ruff: ignore[open-file-with-context-handler]
            max_size=limits.output_spool_bytes,
            mode="w+b",
        )
    )
    digest = hashlib.sha256()
    total = 0
    try:
        if isinstance(value, (bytes, bytearray, memoryview)):
            view = memoryview(value)
            for offset in range(0, len(view), limits.chunk_bytes):
                _cancelled(should_cancel)
                chunk = bytes(view[offset : offset + limits.chunk_bytes])
                total += len(chunk)
                if total > limits.max_entry_uncompressed_bytes:
                    raise ZipWorkspaceError(
                        "ZIP replacement exceeds configured per-entry limit"
                    )
                if aggregate_so_far + total > limits.max_replacement_total_bytes:
                    raise ZipWorkspaceError(
                        "ZIP replacements exceed configured aggregate byte limit"
                    )
                spool.write(chunk)
                digest.update(chunk)
        elif hasattr(value, "read") and hasattr(value, "seek"):
            try:
                value.seek(0)
            except Exception as exc:
                raise ZipWorkspaceError(
                    "ZIP replacement source must be seekable"
                ) from exc
            try:
                while True:
                    _cancelled(should_cancel)
                    chunk = value.read(limits.chunk_bytes)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise ZipWorkspaceError(
                            "ZIP replacement source must yield bytes"
                        )
                    chunk = bytes(chunk)
                    total += len(chunk)
                    if total > limits.max_entry_uncompressed_bytes:
                        raise ZipWorkspaceError(
                            "ZIP replacement exceeds configured per-entry limit"
                        )
                    if aggregate_so_far + total > limits.max_replacement_total_bytes:
                        raise ZipWorkspaceError(
                            "ZIP replacements exceed configured aggregate byte limit"
                        )
                    spool.write(chunk)
                    digest.update(chunk)
            finally:
                try:  # ruff: ignore[suppressible-exception]
                    value.seek(0)
                except Exception:  # ruff: ignore[blind-except]
                    pass
        else:
            raise ZipWorkspaceError(
                "ZIP replacement content must be bytes-like or a seekable binary file"
            )
        spool.seek(0)
        return (
            _Replacement(
                file=spool,
                size=total,
                sha256=digest.hexdigest(),
            ),
            aggregate_so_far + total,
        )
    except Exception:
        spool.close()
        raise


def _close_replacements(rows: Mapping[str, _Replacement]) -> None:
    for replacement in rows.values():
        try:  # ruff: ignore[suppressible-exception]
            replacement.close()
        except Exception:  # ruff: ignore[blind-except, try-except-in-loop]
            pass


def _coerce_replacements(  # ruff: ignore[too-many-branches]
    replacements: Mapping[str, bytes | bytearray | memoryview | BinaryIO],
    *,
    plan: _ArchivePlan,
    limits: ZipWorkspaceLimits,
    authorized_paths: tuple[str, ...] | None = None,
    replacement_expectations: Mapping[str, tuple[int, str]] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, _Replacement]:
    if not isinstance(replacements, Mapping):
        raise ZipWorkspaceError("ZIP replacements must be a path-to-content mapping")

    by_name = {row.info.filename: row for row in plan.entries}
    authorized: set[str] | None = None
    if authorized_paths is not None:
        authorized = set()
        for name in authorized_paths:
            if not isinstance(name, str):
                raise ZipWorkspaceError("ZIP authorized paths must be strings")
            if name in authorized:
                raise ZipWorkspaceError("ZIP authorized paths must be unique")
            target = by_name.get(name)
            if target is None:
                raise ZipWorkspaceError(
                    "ZIP authorized path does not name an existing entry"
                )
            if target.is_dir:
                raise ZipWorkspaceError("ZIP authorized paths must name regular files")
            authorized.add(name)

    expectations: Mapping[str, tuple[int, str]] | None = replacement_expectations
    if expectations is not None and set(expectations) != set(replacements):
        raise ZipWorkspaceError(
            "ZIP replacement expectations must match replacement paths exactly"
        )

    out: dict[str, _Replacement] = {}
    aggregate = 0
    try:
        for name, value in replacements.items():
            if not isinstance(name, str):
                raise ZipWorkspaceError("ZIP replacement paths must be strings")
            if name not in by_name:
                raise ZipWorkspaceError(
                    "ZIP replacement path does not name an existing entry"
                )
            if authorized is not None and name not in authorized:
                raise ZipWorkspaceError(
                    "ZIP replacement path is outside the authorized path set"
                )
            target = by_name[name]
            if target.is_dir:
                raise ZipWorkspaceError("ZIP directories cannot be replaced")
            replacement, aggregate = _snapshot_replacement_value(
                value,
                limits=limits,
                aggregate_so_far=aggregate,
                should_cancel=should_cancel,
            )
            if expectations is not None:
                expected = expectations.get(name)
                if (
                    not isinstance(expected, tuple)
                    or len(expected) != 2  # ruff: ignore[magic-value-comparison]
                    or isinstance(expected[0], bool)
                    or not isinstance(expected[0], int)
                    or int(expected[0]) != replacement.size
                    or str(expected[1]).lower() != replacement.sha256
                ):
                    replacement.close()
                    raise ZipWorkspaceError(
                        "ZIP replacement generation does not match expected identity"
                    )
            out[name] = replacement

        expected_total = plan.source_total_uncompressed
        for name, replacement in out.items():
            expected_total -= by_name[name].info.file_size
            expected_total += replacement.size
        if expected_total > limits.max_total_uncompressed_bytes:
            raise ZipWorkspaceError(
                "Rewritten ZIP would exceed configured total uncompressed limit"
            )
        return out
    except Exception:
        _close_replacements(out)
        raise


def _encode_filename(info: zipfile.ZipInfo) -> bytes:
    try:
        if info.flag_bits & 0x800:
            return info.filename.encode("utf-8")
        return info.filename.encode("cp437")
    except UnicodeEncodeError as exc:
        raise ZipWorkspaceError(
            "ZIP filename encoding is inconsistent with general-purpose flags"
        ) from exc


def _dos_datetime(info: zipfile.ZipInfo) -> tuple[int, int]:
    year, month, day, hour, minute, second = info.date_time
    if not (1980 <= year <= 2107):  # ruff: ignore[magic-value-comparison]
        raise ZipWorkspaceError(
            "ZIP archive timestamp is outside the supported DOS range"
        )
    dos_time = (hour << 11) | (minute << 5) | (second // 2)
    dos_date = ((year - 1980) << 9) | (month << 5) | day
    return dos_time, dos_date


def _read_exact(fh: BinaryIO, size: int) -> bytes:
    data = fh.read(size)
    if len(data) != size:
        raise ZipWorkspaceError("ZIP archive payload is malformed or corrupt")
    return data


def _decode_filename(raw: bytes, flag_bits: int) -> str:
    try:
        return raw.decode("utf-8" if flag_bits & 0x800 else "cp437")
    except UnicodeDecodeError as exc:
        raise ZipWorkspaceError("ZIP archive filename encoding is malformed") from exc


def _descriptor_length(
    fh: BinaryIO,
    *,
    data_end: int,
    boundary: int,
    info: zipfile.ZipInfo,
) -> int:
    available = boundary - data_end
    if available < 12:  # ruff: ignore[magic-value-comparison]
        raise ZipWorkspaceError("ZIP archive data descriptor is malformed or truncated")
    fh.seek(data_end)
    probe = fh.read(min(24, available))

    candidates: list[int] = []
    # Prefer wider descriptors first.  This disambiguates force-ZIP64 records
    # whose bounded sizes still fit in 32 bits.
    if (
        len(probe) >= 24  # ruff: ignore[magic-value-comparison]
        and struct.unpack_from("<I", probe, 0)[0] == _DATA_DESCRIPTOR_SIGNATURE
    ):
        crc, csize, usize = struct.unpack_from("<IQQ", probe, 4)
        if (crc, csize, usize) == (info.CRC, info.compress_size, info.file_size):
            candidates.append(24)
    if len(probe) >= 20:  # ruff: ignore[magic-value-comparison]
        crc, csize, usize = struct.unpack_from("<IQQ", probe, 0)
        if (crc, csize, usize) == (info.CRC, info.compress_size, info.file_size):
            candidates.append(20)
    if (
        len(probe) >= 16  # ruff: ignore[magic-value-comparison]
        and struct.unpack_from("<I", probe, 0)[0] == _DATA_DESCRIPTOR_SIGNATURE
    ):
        crc, csize, usize = struct.unpack_from("<III", probe, 4)
        if (crc, csize, usize) == (info.CRC, info.compress_size, info.file_size):
            candidates.append(16)
    if len(probe) >= 12:  # ruff: ignore[magic-value-comparison]
        crc, csize, usize = struct.unpack_from("<III", probe, 0)
        if (crc, csize, usize) == (info.CRC, info.compress_size, info.file_size):
            candidates.append(12)
    if not candidates:
        raise ZipWorkspaceError(
            "ZIP archive data descriptor is malformed or inconsistent"
        )
    return max(candidates)


def _local_record_span(
    fh: BinaryIO,
    *,
    row: _EntryPlan,
    boundary: int,
) -> tuple[int, int]:
    info = row.info
    start = int(info.header_offset)
    if start < 0 or start >= boundary or boundary > (2**32):
        raise ZipWorkspaceError("ZIP archive contains invalid local-record offsets")
    fh.seek(start)
    header = _read_exact(fh, _LOCAL_FILE_HEADER_SIZE)
    (
        signature,
        _version_needed,
        flag_bits,
        compression,
        _mtime,
        _mdate,
        _crc,
        _csize,
        _usize,
        name_len,
        extra_len,
    ) = struct.unpack("<IHHHHHIIIHH", header)
    if signature != _LOCAL_FILE_HEADER_SIGNATURE:
        raise ZipWorkspaceError("ZIP archive local-file header is malformed")
    if flag_bits & ~_ALLOWED_FLAG_MASK:
        raise ZipWorkspaceError("ZIP archive uses unsupported local-file flags")
    if compression != info.compress_type:
        raise ZipWorkspaceError(
            "ZIP archive local and central compression metadata disagree"
        )
    if (flag_bits & _ALLOWED_FLAG_MASK) != (int(info.flag_bits) & _ALLOWED_FLAG_MASK):
        raise ZipWorkspaceError("ZIP archive local and central flag metadata disagree")
    raw_name = _read_exact(fh, name_len)
    if _decode_filename(raw_name, flag_bits) != info.filename:
        raise ZipWorkspaceError("ZIP archive local and central filenames disagree")
    local_extra = _read_exact(fh, extra_len)
    _sanitize_extra(
        local_extra
    )  # validate TLV structure without changing raw copied bytes

    data_start = start + _LOCAL_FILE_HEADER_SIZE + name_len + extra_len
    data_end = data_start + int(info.compress_size)
    if data_end > boundary:
        raise ZipWorkspaceError("ZIP archive local record overlaps another record")
    end = data_end
    if flag_bits & 0x08:
        end += _descriptor_length(fh, data_end=data_end, boundary=boundary, info=info)
    if end > boundary:
        raise ZipWorkspaceError(
            "ZIP archive local record exceeds its physical boundary"
        )
    return start, end


def _physical_rows(plan: _ArchivePlan) -> tuple[tuple[_EntryPlan, int], ...]:
    ordered = sorted(plan.entries, key=lambda row: int(row.info.header_offset))
    offsets = [int(row.info.header_offset) for row in ordered]
    if len(offsets) != len(set(offsets)):
        raise ZipWorkspaceError("ZIP archive contains duplicate local-header offsets")
    out: list[tuple[_EntryPlan, int]] = []
    for index, row in enumerate(ordered):
        boundary = offsets[index + 1] if index + 1 < len(offsets) else plan.start_dir
        if boundary <= int(row.info.header_offset):
            raise ZipWorkspaceError(
                "ZIP archive contains invalid local-record ordering"
            )
        out.append((row, boundary))
    return tuple(out)


def _copy_raw_record(
    src: BinaryIO,
    dst: BinaryIO,
    *,
    start: int,
    end: int,
    limits: ZipWorkspaceLimits,
    should_cancel: Callable[[], bool] | None,
) -> tuple[int, str]:
    remaining = end - start
    if remaining < 0:
        raise ZipWorkspaceError("ZIP archive contains invalid local-record length")
    digest = hashlib.sha256()
    src.seek(start)
    copied = 0
    while remaining:
        _cancelled(should_cancel)
        chunk = src.read(min(limits.chunk_bytes, remaining))
        if not chunk:
            raise ZipWorkspaceError("ZIP archive local record is truncated")
        dst.write(chunk)
        digest.update(chunk)
        copied += len(chunk)
        remaining -= len(chunk)
    return copied, digest.hexdigest()


def _compress_replacement(  # ruff: ignore[too-many-branches]
    replacement: _Replacement,
    *,
    compression: int,
    limits: ZipWorkspaceLimits,
    should_cancel: Callable[[], bool] | None,
) -> tuple[BinaryIO, int, int, int, str]:
    spool = (
        # lint
        tempfile.SpooledTemporaryFile(  # ruff: ignore[open-file-with-context-handler]
            max_size=limits.output_spool_bytes,
            mode="w+b",
        )
    )
    crc = 0
    digest = hashlib.sha256()
    total = 0
    try:
        compressor = None
        if compression == zipfile.ZIP_DEFLATED:
            compressor = zlib.compressobj(
                zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15
            )
        replacement.file.seek(0)
        while True:
            _cancelled(should_cancel)
            chunk = replacement.file.read(limits.chunk_bytes)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise ZipWorkspaceError("ZIP replacement source must yield bytes")
            chunk = bytes(chunk)
            total += len(chunk)
            if total > replacement.size or total > limits.max_entry_uncompressed_bytes:
                raise ZipWorkspaceError(
                    "ZIP replacement changed while preparing output"
                )
            crc = zlib.crc32(chunk, crc)
            digest.update(chunk)
            if compression == zipfile.ZIP_STORED:
                spool.write(chunk)
            else:
                assert compressor is not None  # ruff: ignore[assert]
                compressed = compressor.compress(chunk)
                if compressed:
                    spool.write(compressed)
        replacement.file.seek(0)
        if compressor is not None:
            tail = compressor.flush()
            if tail:
                spool.write(tail)
        if total != replacement.size or digest.hexdigest() != replacement.sha256:
            raise ZipWorkspaceError("ZIP replacement changed while preparing output")
        compressed_size = spool.tell()
        if (
            compressed_size > 0xFFFFFFFF  # ruff: ignore[magic-value-comparison]
            or total > 0xFFFFFFFF  # ruff: ignore[magic-value-comparison]
        ):
            raise ZipWorkspaceError("ZIP replacement exceeds non-ZIP64 output boundary")
        spool.seek(0)
        return spool, crc & 0xFFFFFFFF, compressed_size, total, digest.hexdigest()
    except Exception:
        spool.close()
        try:  # ruff: ignore[suppressible-exception]
            replacement.file.seek(0)
        except Exception:  # ruff: ignore[blind-except]
            pass
        raise


@dataclass(frozen=True)
class _OutputEntry:
    row: _EntryPlan
    local_offset: int
    crc: int
    compress_size: int
    file_size: int
    flag_bits: int


def _write_changed_record(
    output: BinaryIO,
    *,
    row: _EntryPlan,
    replacement: _Replacement,
    limits: ZipWorkspaceLimits,
    should_cancel: Callable[[], bool] | None,
) -> _OutputEntry:
    info = row.info
    payload, crc, compressed_size, file_size, digest = _compress_replacement(
        replacement,
        compression=info.compress_type,
        limits=limits,
        should_cancel=should_cancel,
    )
    try:
        filename = _encode_filename(info)
        extra = row.safe_extra
        if (
            len(filename) > 0xFFFF  # ruff: ignore[magic-value-comparison]
            or len(extra) > 0xFFFF  # ruff: ignore[magic-value-comparison]
        ):
            raise ZipWorkspaceError("ZIP output local metadata exceeds format limits")
        local_offset = output.tell()
        if local_offset > 0xFFFFFFFF:  # ruff: ignore[magic-value-comparison]
            raise ZipWorkspaceError("ZIP output exceeds non-ZIP64 offset boundary")
        dos_time, dos_date = _dos_datetime(info)
        flag_bits = (
            info.flag_bits & ~0x08
        )  # no descriptor: sizes/CRC are known before write
        version_needed = int(info.extract_version)
        header = struct.pack(
            "<IHHHHHIIIHH",
            _LOCAL_FILE_HEADER_SIGNATURE,
            version_needed,
            flag_bits,
            info.compress_type,
            dos_time,
            dos_date,
            crc,
            compressed_size,
            file_size,
            len(filename),
            len(extra),
        )
        output.write(header)
        output.write(filename)
        output.write(extra)
        while True:
            _cancelled(should_cancel)
            chunk = payload.read(limits.chunk_bytes)
            if not chunk:
                break
            output.write(chunk)
        if digest != replacement.sha256:
            raise ZipWorkspaceError("ZIP replacement changed while writing")
        return _OutputEntry(
            row=row,
            local_offset=local_offset,
            crc=crc,
            compress_size=compressed_size,
            file_size=file_size,
            flag_bits=flag_bits,
        )
    finally:
        payload.close()


def _central_record(entry: _OutputEntry) -> bytes:
    info = entry.row.info
    filename = _encode_filename(info)
    extra = entry.row.safe_extra
    comment = bytes(info.comment)
    if any(
        len(value) > 0xFFFF  # ruff: ignore[magic-value-comparison]
        for value in (filename, extra, comment)
    ):
        raise ZipWorkspaceError(
            "ZIP output central-directory metadata exceeds format limits"
        )
    if (
        entry.local_offset > 0xFFFFFFFF  # ruff: ignore[magic-value-comparison]
        or entry.compress_size > 0xFFFFFFFF  # ruff: ignore[magic-value-comparison]
        or entry.file_size > 0xFFFFFFFF  # ruff: ignore[magic-value-comparison]
    ):
        raise ZipWorkspaceError(
            "ZIP output requires ZIP64 and exceeds workspace contract"
        )
    dos_time, dos_date = _dos_datetime(info)
    version_made = ((int(info.create_system) & 0xFF) << 8) | (
        int(info.create_version) & 0xFF
    )
    fixed = struct.pack(
        "<I6H3I5H2I",
        _CENTRAL_DIRECTORY_SIGNATURE,
        version_made,
        int(info.extract_version),
        entry.flag_bits,
        info.compress_type,
        dos_time,
        dos_date,
        entry.crc,
        entry.compress_size,
        entry.file_size,
        len(filename),
        len(extra),
        len(comment),
        0,  # single-disk output
        int(info.internal_attr),
        int(info.external_attr),
        entry.local_offset,
    )
    return fixed + filename + extra + comment


def _write_central_directory(
    output: BinaryIO,
    *,
    plan: _ArchivePlan,
    entries_by_name: Mapping[str, _OutputEntry],
) -> None:
    central_offset = output.tell()
    if central_offset > 0xFFFFFFFF:  # ruff: ignore[magic-value-comparison]
        raise ZipWorkspaceError(
            "ZIP output central-directory offset exceeds workspace contract"
        )
    output.writelines(
        _central_record(entries_by_name[row.info.filename]) for row in plan.entries
    )
    central_size = output.tell() - central_offset
    if (
        central_size > 0xFFFFFFFF  # ruff: ignore[magic-value-comparison]
        or len(plan.entries) > 0xFFFF  # ruff: ignore[magic-value-comparison]
        or len(plan.comment) > 0xFFFF  # ruff: ignore[magic-value-comparison]
    ):
        raise ZipWorkspaceError(
            "ZIP output central directory exceeds non-ZIP64 format limits"
        )
    output.write(
        struct.pack(
            "<I4H2IH",
            _END_CENTRAL_DIRECTORY_SIGNATURE,
            0,
            0,
            len(plan.entries),
            len(plan.entries),
            central_size,
            central_offset,
            len(plan.comment),
        )
    )
    output.write(plan.comment)


def _verify_raw_records(
    output: BinaryIO,
    *,
    raw_expectations: Mapping[str, tuple[int, int, str]],
    limits: ZipWorkspaceLimits,
    should_cancel: Callable[[], bool] | None,
) -> None:
    for _name, (
        offset,
        length,
        expected_hash,
    ) in raw_expectations.items():  # ruff: ignore[incorrect-dict-iterator]
        output.seek(offset)
        digest = hashlib.sha256()
        remaining = length
        while remaining:
            _cancelled(should_cancel)
            chunk = output.read(min(limits.chunk_bytes, remaining))
            if not chunk:
                raise ZipWorkspaceError("Rewritten ZIP raw-record verification failed")
            digest.update(chunk)
            remaining -= len(chunk)
        if digest.hexdigest() != expected_hash:
            raise ZipWorkspaceError("Rewritten ZIP raw-record verification failed")
    output.seek(0)


def _metadata_tuple(row: _EntryPlan) -> tuple[object, ...]:
    info = row.info
    return (
        info.filename,
        info.compress_type,
        info.date_time,
        bytes(info.comment),
        row.safe_extra,
        info.create_system,
        info.internal_attr,
        info.external_attr,
    )


def _verify_output(  # ruff: ignore[too-many-branches]
    output: BinaryIO,
    *,
    plan: _ArchivePlan,
    replacement_hashes: Mapping[str, str],
    limits: ZipWorkspaceLimits,
    should_cancel: Callable[[], bool] | None,
) -> None:
    try:
        output.seek(0)
        with zipfile.ZipFile(output, mode="r", allowZip64=False) as zout:
            infos = zout.infolist()
            if len(infos) != len(plan.entries):
                raise ZipWorkspaceError("Rewritten ZIP tree verification failed")
            if bytes(zout.comment) != plan.comment:
                raise ZipWorkspaceError("Rewritten ZIP metadata verification failed")

            for expected, actual in zip(plan.entries, infos):
                actual_row = _EntryPlan(
                    info=actual,
                    safe_extra=_sanitize_extra(actual.extra),
                    is_dir=actual.filename.endswith("/"),
                    canonical_key=_validate_path(
                        actual.filename, max_chars=limits.max_path_chars
                    )[0],
                )
                if _metadata_tuple(actual_row) != _metadata_tuple(expected):
                    raise ZipWorkspaceError(
                        "Rewritten ZIP metadata verification failed"
                    )
                if expected.is_dir:
                    continue

                digest = hashlib.sha256()
                total = 0
                try:
                    with zout.open(actual, mode="r") as payload:
                        while True:
                            _cancelled(should_cancel)
                            chunk = payload.read(limits.chunk_bytes)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > limits.max_entry_uncompressed_bytes:
                                raise ZipWorkspaceError(
                                    "Rewritten ZIP verification exceeded entry limit"
                                )
                            digest.update(chunk)
                except ZipWorkspaceError:
                    raise
                except (
                    zipfile.BadZipFile,
                    EOFError,
                    OSError,
                    RuntimeError,
                    zlib.error,
                ) as exc:
                    raise ZipWorkspaceError(
                        "ZIP archive payload is malformed or corrupt"
                    ) from exc
                if total != actual.file_size:
                    raise ZipWorkspaceError("Rewritten ZIP payload verification failed")
                replacement_hash = replacement_hashes.get(actual.filename)
                if (
                    replacement_hash is not None
                    and digest.hexdigest() != replacement_hash
                ):
                    raise ZipWorkspaceError(
                        "Rewritten ZIP replacement verification failed"
                    )
    except ZipWorkspaceError:
        raise
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        EOFError,
        OSError,
        ValueError,
    ) as exc:
        raise ZipWorkspaceError("Rewritten ZIP verification failed") from exc
    finally:
        output.seek(0)


def _hash_output(output: BinaryIO, *, limits: ZipWorkspaceLimits) -> str:
    digest = hashlib.sha256()
    output.seek(0)
    while True:
        chunk = output.read(limits.chunk_bytes)
        if not chunk:
            break
        digest.update(chunk)
    output.seek(0)
    return digest.hexdigest()


def rewrite_zip_workspace(
    source: str | os.PathLike[str] | BinaryIO,
    replacements: Mapping[str, bytes | bytearray | memoryview | BinaryIO],
    *,
    authorized_paths: tuple[str, ...] | None = None,
    expected_source_sha256: str | None = None,
    expected_source_size: int | None = None,
    replacement_expectations: Mapping[str, tuple[int, str]] | None = None,
    limits: ZipWorkspaceLimits = DEFAULT_ZIP_WORKSPACE_LIMITS,
    should_cancel: Callable[[], bool] | None = None,
) -> ZipWorkspaceArtifact:
    """
    Surgically replace selected existing regular files in a complete ZIP.

    When ``authorized_paths`` is supplied, every authorized path must name an
    exact existing regular file and every replacement must be a member of that
    set. Optional source/replacement expectations bind caller-verified byte
    generations again inside the workspace before mutation. Unchanged local
    records are copied byte-for-byte. Changed entries are
    regenerated using their original compression method and portable metadata,
    then the central directory is rebuilt with fresh offsets.  The complete
    output is independently reopened and every unchanged payload is still
    decompressed/CRC-checked before the artifact is accepted.
    """

    _cancelled(should_cancel)
    source_sha256, source_size = _hash_source(
        source, limits=limits, should_cancel=should_cancel
    )
    if (
        expected_source_sha256 is not None
        and source_sha256 != str(expected_source_sha256).lower()
    ):
        raise ZipWorkspaceError("ZIP source generation does not match expected SHA-256")
    if expected_source_size is not None and (
        isinstance(expected_source_size, bool)
        or int(expected_source_size) != source_size
    ):
        raise ZipWorkspaceError("ZIP source generation does not match expected size")
    snapshot, snapshot_sha256 = _snapshot_source(
        source, limits=limits, should_cancel=should_cancel
    )
    if snapshot_sha256 != source_sha256:
        snapshot.close()
        raise ZipWorkspaceError("ZIP source changed while creating server snapshot")
    try:
        plan = _inspect_archive(snapshot, limits=limits)
        replacement_rows = _coerce_replacements(
            replacements,
            plan=plan,
            limits=limits,
            authorized_paths=authorized_paths,
            replacement_expectations=replacement_expectations,
            should_cancel=should_cancel,
        )

        prewrite_sha256, _ = _hash_source(
            source, limits=limits, should_cancel=should_cancel
        )
        if prewrite_sha256 != source_sha256:
            raise ZipWorkspaceError("ZIP source changed after inspection")
    except Exception:
        snapshot.close()
        raise

    output = (
        # lint
        tempfile.SpooledTemporaryFile(  # ruff: ignore[open-file-with-context-handler]
            max_size=limits.output_spool_bytes,
            mode="w+b",
        )
    )
    raw_expectations: dict[str, tuple[int, int, str]] = {}
    entries_by_name: dict[str, _OutputEntry] = {}
    replacement_hashes = {
        name: replacement.sha256 for name, replacement in replacement_rows.items()
    }
    try:
        try:
            with _source_handle(snapshot) as fh:
                for row, boundary in _physical_rows(plan):
                    _cancelled(should_cancel)
                    name = row.info.filename
                    # Validate every original local record, including entries
                    # that will be replaced. Replacement authority must not be
                    # a bypass around malformed local metadata or descriptors.
                    source_start, source_end = _local_record_span(
                        fh, row=row, boundary=boundary
                    )
                    replacement = replacement_rows.get(name)
                    if replacement is not None:
                        entries_by_name[name] = _write_changed_record(
                            output,
                            row=row,
                            replacement=replacement,
                            limits=limits,
                            should_cancel=should_cancel,
                        )
                        continue

                    output_start = output.tell()
                    length, digest = _copy_raw_record(
                        fh,
                        output,
                        start=source_start,
                        end=source_end,
                        limits=limits,
                        should_cancel=should_cancel,
                    )
                    raw_expectations[name] = (output_start, length, digest)
                    entries_by_name[name] = _OutputEntry(
                        row=row,
                        local_offset=output_start,
                        crc=int(row.info.CRC),
                        compress_size=int(row.info.compress_size),
                        file_size=int(row.info.file_size),
                        flag_bits=int(row.info.flag_bits),
                    )

            if len(entries_by_name) != len(plan.entries):
                raise ZipWorkspaceError(
                    "ZIP surgical rewrite did not account for every entry"
                )
            _write_central_directory(output, plan=plan, entries_by_name=entries_by_name)
        except ZipWorkspaceError:
            raise
        except (
            EOFError,
            OSError,
            RuntimeError,
            zlib.error,
            ValueError,
            struct.error,
        ) as exc:
            raise ZipWorkspaceError(
                "ZIP archive payload is malformed or corrupt"
            ) from exc

        postwrite_sha256, _ = _hash_source(
            source, limits=limits, should_cancel=should_cancel
        )
        if postwrite_sha256 != source_sha256:
            raise ZipWorkspaceError("ZIP source changed during rewrite")

        _verify_raw_records(
            output,
            raw_expectations=raw_expectations,
            limits=limits,
            should_cancel=should_cancel,
        )
        _verify_output(
            output,
            plan=plan,
            replacement_hashes=replacement_hashes,
            limits=limits,
            should_cancel=should_cancel,
        )
        output_sha256 = _hash_output(output, limits=limits)
        changed_paths = tuple(
            row.info.filename
            for row in plan.entries
            if row.info.filename in replacement_rows
        )
        receipt = ZipWorkspaceReceipt(
            source_sha256=source_sha256,
            output_sha256=output_sha256,
            entry_count=len(plan.entries),
            changed_paths=changed_paths,
            unchanged_count=len(plan.entries) - len(changed_paths),
        )
        output.seek(0)
        snapshot.close()
        _close_replacements(replacement_rows)
        return ZipWorkspaceArtifact(file=output, receipt=receipt)
    except Exception:
        output.close()
        snapshot.close()
        _close_replacements(locals().get("replacement_rows", {}))
        raise


__all__ = (
    "DEFAULT_ZIP_WORKSPACE_LIMITS",
    "ZIP_CHUNK_BYTES",
    "ZIP_MAX_COMPRESSION_RATIO",
    "ZIP_MAX_ENTRIES",
    "ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES",
    "ZIP_MAX_REPLACEMENT_TOTAL_BYTES",
    "ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES",
    "ZIP_OUTPUT_SPOOL_BYTES",
    "ZipWorkspaceArtifact",
    "ZipWorkspaceError",
    "ZipWorkspaceLimits",
    "ZipWorkspaceReceipt",
    "rewrite_zip_workspace",
    "validate_zip_workspace_file_path",
)
