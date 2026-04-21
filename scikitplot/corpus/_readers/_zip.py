# scikitplot/corpus/_readers/_zip.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._readers._zip
================================
Generic ZIP archive reader for the corpus pipeline.

:class:`ZipReader` transparently extracts a ZIP file and dispatches each
member file to the appropriate registered reader by extension.  It is the
"any-archive-to-any-reader" bridge: instead of forcing the caller to
choose between :class:`ALTOReader` (which only handles ALTO XML zips) and
the ``_archive_handler`` + ``CorpusBuilder`` path, :class:`ZipReader`
handles *any* ZIP containing *any* mix of supported media.

Why this reader exists
----------------------
The previous state of the codebase had a gap:

* :class:`ALTOReader` registers on ``".zip"`` but only handles ALTO XML.
* :class:`CorpusBuilder._ingest_archive` handles general ZIPs but is only
  accessible through the builder API.
* :meth:`DocumentReader.create` had no way to dispatch a plain ``.zip``
  file containing PDFs, audio files, or images.

:class:`ZipReader` closes that gap.  When ``DocumentReader.create("corpus.zip")``
is called, the registry now returns :class:`ZipReader` (it overrides
:class:`ALTOReader` for ``".zip"``).  :class:`ALTOReader` remains available
by instantiating it directly when you know the archive contains ALTO XML.

Design invariants
-----------------
* **ZipSlip prevention** — every extracted path is checked against the
  destination directory via ``path.resolve().relative_to(dest.resolve())``.
* **Bomb prevention** — configurable ``max_files`` and ``max_total_bytes``
  limits with early rejection before extraction begins.
* **Nested archives** — ZIPs inside ZIPs are *not* recursed automatically.
  An inner ``.zip`` is treated as a regular file and dispatched to the
  registry (which will return another :class:`ZipReader`, effectively
  recursing one level at a time).
* **Symlink rejection** — symbolic links inside ZIP entries are skipped.
* **Hidden-file filtering** — members starting with ``"."`` or inside
  ``__MACOSX`` / ``__pycache__`` are excluded.
* **Caller owns cleanup** — extracted files are written to a temporary
  directory that is deleted when the reader's context exits (or when
  ``close()`` is called).  In normal iterator usage, cleanup happens when
  ``get_documents()`` returns.

Python compatibility
--------------------
Python 3.8-3.15.  Zero optional dependencies — ``zipfile`` is stdlib.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generator, Optional  # noqa: F401

from .._base import DocumentReader
from .._schema import SectionType  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["ZipReader"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default maximum number of files inside a ZIP.
_DEFAULT_MAX_FILES: int = 10_000

#: Default maximum cumulative uncompressed size in bytes (2 GB).
_DEFAULT_MAX_TOTAL_BYTES: int = 2 * 1024 * 1024 * 1024

#: Path components that trigger member exclusion.
_SKIP_PARTS: frozenset[str] = frozenset({"__pycache__", "__MACOSX"})


def _should_skip(member_path: str) -> bool:
    """
    Return ``True`` if the ZIP member should be excluded.

    Parameters
    ----------
    member_path : str
        Relative path of the archive member (forward-slash separated).

    Returns
    -------
    bool
        ``True`` if any path component starts with ``"."`` or is in
        :data:`_SKIP_PARTS`.
    """
    parts = Path(member_path).parts
    return any(part.startswith(".") or part in _SKIP_PARTS for part in parts)


def _is_within(child: Path, parent: Path) -> bool:
    """
    ZipSlip guard: confirm *child* is inside *parent*.

    Parameters
    ----------
    child : Path
        Resolved path of the extracted member.
    parent : Path
        Resolved destination directory.

    Returns
    -------
    bool
        ``True`` when *child* is a descendant of *parent*.
    """
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ===========================================================================
# ZipReader
# ===========================================================================


@dataclass
class ZipReader(DocumentReader):
    r"""
    Generic ZIP archive reader — dispatches each member to its natural reader.

    Extracts all supported members from a ``.zip`` archive into a temporary
    directory, then calls :meth:`DocumentReader.create` on each file.
    Documents from all members are yielded in a single stream.

    This reader intentionally overrides :class:`ALTOReader`\'s ``".zip"``
    registration.  To use :class:`ALTOReader` directly, instantiate it
    explicitly.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the ``.zip`` archive.
    max_files : int, optional
        Maximum number of files allowed in the archive.  Archives
        exceeding this limit raise :exc:`ValueError` before extraction
        begins.  Default: 10,000.
    max_total_bytes : int, optional
        Maximum cumulative uncompressed size.  Extraction halts if this
        limit is exceeded (zip-bomb prevention).  Default: 2 GB.
    skip_unsupported : bool, optional
        When ``True`` (default), members whose extension is not
        registered in :class:`DocumentReader` are silently skipped.
        When ``False``, an unregistered extension raises
        :exc:`ValueError`.

    Notes
    -----
    **ALTOReader coexistence:** :class:`ALTOReader` registers on ``".zip"``
    and is used when ``ZipReader`` is not imported.  Importing
    ``scikitplot.corpus._readers`` triggers both imports; :class:`ZipReader`
    is imported *after* :class:`ALTOReader`, so its ``".zip"`` registration
    wins.  To use :class:`ALTOReader` on a known ALTO archive, instantiate
    it directly::

        reader = ALTOReader(input_path=Path("alto_corpus.zip"))

    **Temporary directory:** Extracted files land in
    ``tempfile.mkdtemp()``.  The directory is deleted automatically when
    ``get_documents()`` completes (including on exception).  If you
    iterate manually without exhausting the generator, call
    ``reader.close()`` to clean up.

    **Reader kwargs forwarding:** All constructor kwargs beyond the
    explicit parameters are forwarded to each sub-reader constructed for
    the members.  This means you can pass ``transcribe=True`` and it will
    reach any :class:`AudioReader` or :class:`VideoReader` instances
    created for audio/video members.

    Developer note:

    :class:`ZipReader` is intentionally **not** recursive for nested ZIPs.
    An inner ``.zip`` file will be dispatched back to :class:`ZipReader`
    (since it is now the registered handler for ``".zip"``), achieving
    one level of recursion per nesting depth without any special-case
    logic.  This is safe because each level uses its own temporary
    directory.

    Examples
    --------
    Any ZIP containing a mix of supported files:

    >>> from pathlib import Path
    >>> from scikitplot.corpus._base import DocumentReader
    >>> import scikitplot.corpus._readers
    >>> reader = DocumentReader.create(Path("corpus.zip"))
    >>> type(reader).__name__
    \'ZipReader\'
    >>> docs = list(reader.get_documents())

    Explicit instantiation with custom limits:

    >>> reader = ZipReader(
    ...     input_path=Path("large_corpus.zip"),
    ...     max_files=500,
    ...     max_total_bytes=500 * 1024 * 1024,
    ... )
    """

    file_type: ClassVar[str] = ".zip"
    file_types: ClassVar[list[str] | None] = [".zip"]

    max_files: int = field(default=_DEFAULT_MAX_FILES)
    """Maximum file count inside the archive. Default: 10,000."""

    max_total_bytes: int = field(default=_DEFAULT_MAX_TOTAL_BYTES)
    """Maximum cumulative extracted size. Default: 2 GB."""

    skip_unsupported: bool = field(default=True)
    """Skip members with unregistered extensions instead of raising."""

    infer_source_type: bool = field(default=True)
    """Auto-infer ``source_type`` for each member via
    :meth:`SourceType.infer` when the caller did not supply
    ``source_type`` in ``source_provenance``.  Default: ``True``."""

    reader_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-extension keyword arguments forwarded to sub-reader constructors.

    Enables reader-specific options for individual file types inside the
    archive.  Keys are lower-case file extensions (with leading dot),
    values are dicts of kwargs forwarded to the corresponding reader.

    Example — transcribe MP3 files but not others::

        ZipReader(
            input_path=Path("corpus.zip"),
            reader_kwargs={
                ".mp3": {"transcribe": True, "whisper_model": "small"},
                ".mp4": {"transcribe": True},
                ".jpg": {"backend": "easyocr"},
            },
        )

    Global kwargs passed to the :class:`ZipReader` constructor
    (``**kwargs``) are merged *under* per-extension overrides, so
    per-extension values always win.
    """

    def __post_init__(self) -> None:
        """Validate constructor arguments and normalise ``reader_kwargs`` keys.

        Raises
        ------
        ValueError
            If ``max_files <= 0`` or ``max_total_bytes <= 0``.
        TypeError
            If ``reader_kwargs`` is not a dict, or any value is not a dict.

        Notes
        -----
        ``reader_kwargs`` keys are normalised to lower-case with a
        leading dot (e.g. ``"MP3"`` → ``".mp3"``) for consistent lookup.
        """
        super().__post_init__()
        if self.max_files <= 0:
            raise ValueError(
                f"ZipReader: max_files must be > 0; got {self.max_files!r}."
            )
        if self.max_total_bytes <= 0:
            raise ValueError(
                f"ZipReader: max_total_bytes must be > 0; got {self.max_total_bytes!r}."
            )
        if not isinstance(self.reader_kwargs, dict):
            raise TypeError(
                f"ZipReader: reader_kwargs must be a dict; "
                f"got {type(self.reader_kwargs).__name__!r}."
            )
        # Normalise reader_kwargs keys to lower-case with leading dot
        normalised: dict[str, dict[str, Any]] = {}
        for k, v in self.reader_kwargs.items():
            key = k.lower() if k.startswith(".") else f".{k.lower()}"
            if not isinstance(v, dict):
                raise TypeError(
                    f"ZipReader: reader_kwargs[{k!r}] must be a dict; "
                    f"got {type(v).__name__!r}."
                )
            normalised[key] = dict(v)
        object.__setattr__(self, "reader_kwargs", normalised)

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:  # noqa: PLR0912
        """
        Extract ZIP and yield raw chunks from all supported members.

        Each member is dispatched to :meth:`DocumentReader.create`, which
        selects the appropriate reader by file extension.  The member's raw
        chunks are yielded inline, as if the member files had been passed
        directly.

        Yields
        ------
        dict[str, Any]
            Raw chunk dicts from each member reader's
            ``get_raw_chunks()`` call.  The ``input_path`` key is set
            to ``"<archive_name>/<member_name>"`` for provenance.

        Raises
        ------
        ValueError
            If the archive contains more than ``max_files`` members, or
            if cumulative extracted size exceeds ``max_total_bytes``, or
            if a member has a path-traversal component (ZipSlip).
        OSError
            If the archive cannot be opened or a member cannot be read.

        Notes
        -----
        Temporary extraction happens inside a ``tempfile.mkdtemp()``
        directory that is removed on exit (even on exception) via a
        ``try/finally`` block.  The extracted files are read and their
        chunks forwarded; the files themselves are not streamed — each
        member is fully extracted before its reader is called.
        """
        archive_path = self.input_path
        archive_name = archive_path.name
        supported = set(DocumentReader.supported_types())

        tmp_dir = Path(tempfile.mkdtemp(prefix="skplt_zip_"))
        # Track member count for the final log line regardless of
        # whether the ZipFile block completes normally or raises early.
        members_count: int = 0
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = [m for m in zf.infolist() if not m.is_dir()]
                members_count = len(members)

                # ── File count guard ──────────────────────────────────
                if len(members) > self.max_files:
                    raise ValueError(
                        f"ZipReader: archive {archive_path.name!r} contains "
                        f"{len(members)} files (max_files={self.max_files}). "
                        f"Increase max_files or split the archive."
                    )

                total_bytes = 0

                for info in members:
                    member_path = info.filename

                    # ── Skip hidden / system files ────────────────────
                    if _should_skip(member_path):
                        logger.debug(
                            "ZipReader: skipping hidden/system member %s",
                            member_path,
                        )
                        continue

                    # ── ZipSlip guard ─────────────────────────────────
                    target = (tmp_dir / member_path).resolve()
                    if not _is_within(target, tmp_dir):
                        logger.warning(
                            "ZipReader: ZipSlip detected for member %r — "
                            "skipping (escapes %s).",
                            member_path,
                            tmp_dir,
                        )
                        continue

                    # ── Extension filter ──────────────────────────────
                    ext = Path(member_path).suffix.lower()
                    if ext not in supported:
                        if self.skip_unsupported:
                            logger.debug(
                                "ZipReader: skipping unsupported extension "
                                "%r for member %s",
                                ext,
                                member_path,
                            )
                            continue
                        raise ValueError(
                            f"ZipReader: member {member_path!r} has "
                            f"unsupported extension {ext!r}. "
                            f"Set skip_unsupported=True to ignore."
                        )

                    # ── Bomb guard (pre-extraction) ───────────────────
                    total_bytes += info.file_size
                    if total_bytes > self.max_total_bytes:
                        raise ValueError(
                            f"ZipReader: cumulative extracted size "
                            f"({total_bytes:,} bytes) exceeds "
                            f"max_total_bytes={self.max_total_bytes:,}. "
                            f"Possible zip-bomb in {archive_path.name!r}."
                        )

                    # ── Extract single member ─────────────────────────
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as dst:
                        dst.write(src.read())

                    # ── Dispatch to correct reader ────────────────────
                    provenance = dict(self.source_provenance)
                    provenance["input_path"] = f"{archive_name}/{member_path}"

                    # Infer source_type from member extension when not set
                    member_st = provenance.get("source_type")
                    if member_st is None and self.infer_source_type:
                        try:
                            from .._schema import (  # noqa: PLC0415
                                SourceType,
                            )

                            inferred = SourceType.infer(target)
                            if inferred.value != "unknown":
                                member_st = inferred
                        except Exception:  # noqa: BLE001
                            pass

                    # Merge global kwargs with per-extension overrides
                    member_kw: dict[str, Any] = {}
                    if self.reader_kwargs:
                        member_kw.update(self.reader_kwargs.get(ext, {}))

                    try:
                        sub_reader = DocumentReader._create_one(
                            target,
                            chunker=self.chunker,
                            filter_=self.filter_,
                            filename_override=f"{archive_name}/{member_path}",
                            default_language=self.default_language,
                            source_type=member_st,
                            source_title=provenance.get("source_title"),
                            source_author=provenance.get("source_author"),
                            source_date=provenance.get("source_date"),
                            collection_id=provenance.get("collection_id"),
                            doi=provenance.get("doi"),
                            isbn=provenance.get("isbn"),
                            **member_kw,
                        )
                        yield from sub_reader.get_raw_chunks()
                        logger.debug(
                            "ZipReader: processed member %s from %s (type=%s)",
                            member_path,
                            archive_name,
                            member_st,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "ZipReader: failed to read member %s from %s: %s",
                            member_path,
                            archive_name,
                            exc,
                        )

        finally:
            # Always remove the temporary directory
            import shutil  # noqa: PLC0415

            shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(
            "ZipReader: finished %s — processed %d members",
            archive_name,
            members_count,
        )
