# scikitplot/corpus/_archive_handler.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._archive_handler
====================================
Safe extraction of archive files (ZIP, TAR, TAR.GZ, TAR.BZ2, TAR.XZ)
for the corpus pipeline.

This module provides a single public function :func:`extract_archive` that:

1. **Detects** whether a file is a supported archive format.
2. **Extracts** all files to a destination directory with security guards.
3. **Returns** a sorted list of extracted file paths that have extensions
   registered in the :class:`~scikitplot.corpus._base.DocumentReader` registry.

Security invariants
-------------------
* **ZipSlip prevention:** Every extracted member's resolved path is checked
  to ensure it falls within the destination directory. Path-traversal
  entries (e.g. ``../../etc/passwd``) are skipped with a warning.
* **Symlink rejection:** Symlinks inside archives are skipped entirely.
* **File count limit:** Archives with more than ``max_files`` members are
  rejected before extraction begins.
* **Total size limit:** Cumulative extracted bytes are tracked; extraction
  halts if ``max_total_bytes`` is exceeded (zip-bomb prevention).
* **Hidden file / ``__pycache__`` exclusion:** Same rules as
  ``CorpusBuilder._expand_sources``.

Design invariants
-----------------
* Zero optional dependencies — ``zipfile``, ``tarfile`` are stdlib.
* Caller owns cleanup of the destination directory.
* Archive-within-archive (nested) is **not** recursed. If a user ships a
  ``.zip`` containing another ``.zip``, only the outer archive is extracted;
  the inner ``.zip`` appears as a regular file and the DocumentReader
  registry handles it via ``ALTOReader`` (which already handles ``zip``
  extension) or is skipped if no reader is registered.

Python compatibility
--------------------
Python 3.8 through 3.15.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import os  # noqa: F401
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Sequence  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "extract_archive",
    "is_archive",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported archive extensions (lowercase, including leading dot).
_ARCHIVE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".zip",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tbz2",
        ".tar.xz",
        ".txz",
    }
)

#: Default maximum number of files inside an archive.
DEFAULT_MAX_FILES: int = 10_000

#: Default maximum total extracted size in bytes (2 GB).
DEFAULT_MAX_TOTAL_BYTES: int = 2 * 1024 * 1024 * 1024

# Hidden-file / __pycache__ predicate — shared logic with _expand_sources.
_SKIP_PARTS: frozenset[str] = frozenset({"__pycache__", "__MACOSX"})


def _should_skip_path(member_path: str) -> bool:
    """
    Return ``True`` if the archive member should be skipped.

    Parameters
    ----------
    member_path : str
        Relative path of the archive member (forward-slash separated).

    Returns
    -------
    bool
        ``True`` if any path component starts with ``"."`` or is in
        ``_SKIP_PARTS``.
    """
    parts = Path(member_path).parts
    return any(part.startswith(".") or part in _SKIP_PARTS for part in parts)


# ===========================================================================
# is_archive
# ===========================================================================


def is_archive(path: str | Path) -> bool:
    """
    Check if a file path has a supported archive extension.

    Parameters
    ----------
    path : str or Path
        File path to check.

    Returns
    -------
    bool
        ``True`` if the file extension (or compound extension) matches
        a supported archive format.

    Examples
    --------
    >>> is_archive("data.zip")
    True
    >>> is_archive("data.tar.gz")
    True
    >>> is_archive("data.pdf")
    False
    """
    p = Path(path)
    name_lower = p.name.lower()

    # Check compound extensions first
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if name_lower.endswith(ext):
            return True

    # Single extension
    return p.suffix.lower() in _ARCHIVE_EXTENSIONS


# ===========================================================================
# extract_archive
# ===========================================================================


def extract_archive(
    archive_path: str | Path,
    dest_dir: str | Path,
    *,
    supported_extensions: frozenset[str] | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> list[Path]:
    """
    Extract an archive to a destination directory.

    Parameters
    ----------
    archive_path : str or Path
        Path to the archive file.
    dest_dir : str or Path
        Directory to extract files into. Created if it does not exist.
    supported_extensions : frozenset[str] or None, optional
        Whitelist of file extensions to include from the archive.
        If ``None``, all files are included (subject to hidden-file
        and ``__pycache__`` filtering). Default: ``None``.
    max_files : int, optional
        Maximum number of files allowed in the archive. Archives
        exceeding this limit are rejected before extraction begins.
        Default: 10,000.
    max_total_bytes : int, optional
        Maximum cumulative extracted size in bytes. Extraction halts
        if this limit is exceeded (zip-bomb prevention).
        Default: 2 GB.

    Returns
    -------
    list[Path]
        Sorted list of extracted file paths (absolute).

    Raises
    ------
    ValueError
        If the file is not a recognised archive format.
    ValueError
        If the archive contains more than *max_files* members.
    ValueError
        If cumulative extracted bytes exceed *max_total_bytes*.
    OSError
        If the archive cannot be opened.

    Notes
    -----
    **ZipSlip prevention:** Every extracted member's resolved path is
    verified to fall within *dest_dir*. Members with path-traversal
    components (``../``) are logged as warnings and skipped.

    **Symlinks:** Symbolic links inside archives are always skipped.

    Examples
    --------
    >>> from pathlib import Path
    >>> files = extract_archive("corpus.zip", "/tmp/corpus_extract")
    >>> [f.suffix for f in files]
    ['.pdf', '.txt', '.txt']
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    name_lower = archive_path.name.lower()

    if zipfile.is_zipfile(archive_path):
        return _extract_zip(
            archive_path,
            dest_dir,
            supported_extensions=supported_extensions,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
        )

    if tarfile.is_tarfile(archive_path) or name_lower.endswith(
        (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
    ):
        return _extract_tar(
            archive_path,
            dest_dir,
            supported_extensions=supported_extensions,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
        )

    raise ValueError(
        f"extract_archive: {archive_path!r} is not a recognised archive "
        f"format. Supported: {sorted(_ARCHIVE_EXTENSIONS)}."
    )


# ===========================================================================
# Internal: ZIP extraction
# ===========================================================================


def _is_within(child: Path, parent: Path) -> bool:
    """
    Check that ``child`` is inside ``parent`` (ZipSlip guard).

    Parameters
    ----------
    child : Path
        Resolved path of the archive member.
    parent : Path
        Resolved path of the destination directory.

    Returns
    -------
    bool
        ``True`` if *child* is a descendant of *parent*.
    """
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _extract_zip(
    archive_path: Path,
    dest_dir: Path,
    *,
    supported_extensions: frozenset[str] | None,
    max_files: int,
    max_total_bytes: int,
) -> list[Path]:
    """
    Extract a ZIP archive.

    Parameters
    ----------
    archive_path : Path
        ZIP file path.
    dest_dir : Path
        Extraction destination.
    supported_extensions : frozenset[str] or None
        Extension whitelist (``None`` = accept all).
    max_files : int
        Maximum member count.
    max_total_bytes : int
        Maximum cumulative extracted size.

    Returns
    -------
    list[Path]
        Sorted list of extracted file paths.
    """
    extracted: list[Path] = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()

        # File count guard
        file_members = [m for m in members if not m.is_dir()]
        if len(file_members) > max_files:
            raise ValueError(
                f"_extract_zip: archive {archive_path!r} contains "
                f"{len(file_members)} files (max_files={max_files})."
            )

        total_bytes = 0

        for info in file_members:
            # Skip directories
            if info.is_dir():
                continue

            # Skip hidden files and __pycache__
            if _should_skip_path(info.filename):
                logger.debug("Skipping hidden/excluded member: %s", info.filename)
                continue

            # ZipSlip guard
            target = (dest_dir / info.filename).resolve()
            if not _is_within(target, dest_dir):
                logger.warning(
                    "ZipSlip detected: %s escapes %s. Skipping.",
                    info.filename,
                    dest_dir,
                )
                continue

            # Extension filter
            if supported_extensions is not None:
                ext = Path(info.filename).suffix.lower()
                if ext not in supported_extensions:
                    logger.debug(
                        "Skipping unsupported extension in archive: %s",
                        info.filename,
                    )
                    continue

            # Size guard (check uncompressed size before extracting)
            total_bytes += info.file_size
            if total_bytes > max_total_bytes:
                raise ValueError(
                    f"_extract_zip: cumulative extracted size "
                    f"({total_bytes} bytes) exceeds "
                    f"max_total_bytes={max_total_bytes}. "
                    f"Possible zip-bomb in {archive_path!r}."
                )

            # Extract
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())

            extracted.append(target)

    extracted.sort()
    logger.info(
        "extract_archive: extracted %d files from %s → %s",
        len(extracted),
        archive_path,
        dest_dir,
    )
    return extracted


# ===========================================================================
# Internal: TAR extraction
# ===========================================================================


def _extract_tar(  # noqa: PLR0912
    archive_path: Path,
    dest_dir: Path,
    *,
    supported_extensions: frozenset[str] | None,
    max_files: int,
    max_total_bytes: int,
) -> list[Path]:
    """
    Extract a TAR archive (optionally compressed).

    Parameters
    ----------
    archive_path : Path
        TAR file path.
    dest_dir : Path
        Extraction destination.
    supported_extensions : frozenset[str] or None
        Extension whitelist (``None`` = accept all).
    max_files : int
        Maximum member count.
    max_total_bytes : int
        Maximum cumulative extracted size.

    Returns
    -------
    list[Path]
        Sorted list of extracted file paths.
    """
    extracted: list[Path] = []

    # Determine open mode
    name_lower = archive_path.name.lower()
    if name_lower.endswith((".tar.gz", ".tgz")):
        mode = "r:gz"
    elif name_lower.endswith((".tar.bz2", ".tbz2")):
        mode = "r:bz2"
    elif name_lower.endswith((".tar.xz", ".txz")):
        mode = "r:xz"
    else:
        mode = "r"

    with tarfile.open(archive_path, mode) as tf:
        file_members = [m for m in tf.getmembers() if m.isfile()]

        # File count guard
        if len(file_members) > max_files:
            raise ValueError(
                f"_extract_tar: archive {archive_path!r} contains "
                f"{len(file_members)} files (max_files={max_files})."
            )

        total_bytes = 0

        for member in file_members:
            # Skip non-files (dirs, symlinks, devices, etc.)
            if not member.isfile():
                continue

            # Skip symlinks explicitly
            if member.issym() or member.islnk():
                logger.debug("Skipping symlink in tar: %s", member.name)
                continue

            # Skip hidden files and __pycache__
            if _should_skip_path(member.name):
                logger.debug("Skipping hidden/excluded member: %s", member.name)
                continue

            # ZipSlip guard (tarfiles can also contain path traversal)
            target = (dest_dir / member.name).resolve()
            if not _is_within(target, dest_dir):
                logger.warning(
                    "Path traversal detected: %s escapes %s. Skipping.",
                    member.name,
                    dest_dir,
                )
                continue

            # Extension filter
            if supported_extensions is not None:
                ext = Path(member.name).suffix.lower()
                if ext not in supported_extensions:
                    logger.debug(
                        "Skipping unsupported extension in archive: %s",
                        member.name,
                    )
                    continue

            # Size guard
            total_bytes += member.size
            if total_bytes > max_total_bytes:
                raise ValueError(
                    f"_extract_tar: cumulative extracted size "
                    f"({total_bytes} bytes) exceeds "
                    f"max_total_bytes={max_total_bytes}. "
                    f"Possible tar-bomb in {archive_path!r}."
                )

            # Extract safely (no extractall to avoid symlink attacks)
            target.parent.mkdir(parents=True, exist_ok=True)
            fileobj = tf.extractfile(member)
            if fileobj is None:
                logger.debug(
                    "Cannot extract %s (extractfile returned None).",
                    member.name,
                )
                continue
            with open(target, "wb") as dst:
                dst.write(fileobj.read())

            extracted.append(target)

    extracted.sort()
    logger.info(
        "extract_archive: extracted %d files from %s → %s",
        len(extracted),
        archive_path,
        dest_dir,
    )
    return extracted
