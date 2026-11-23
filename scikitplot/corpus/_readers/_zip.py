# scikitplot/corpus/_readers/_zip.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._readers._zip
===============================

Generic ZIP archive reader for the corpus pipeline.

:class:`ZipReader` transparently extracts a ZIP file and dispatches each
member file to the appropriate registered reader by extension.  It is the
"any-archive-to-any-reader" bridge: instead of forcing the caller to
choose between :class:`ALTOReader` (which only handles ALTO XML zips) and
the ``_archive_handler`` + ``CorpusBuilder`` path, :class:`ZipReader`
handles *any* ZIP containing *any* mix of supported media.

Why this reader exists:

The previous state of the codebase had a gap:

* :class:`ALTOReader` registers on ``".zip"`` but only handles ALTO XML.
* :class:`CorpusBuilder._ingest_archive` handles general ZIPs but is only
  accessible through the builder API.
* :meth:`DocumentReader.create` had no way to dispatch a plain ``.zip``
  file containing PDFs, audio files, or images.

:class:`ZipReader` closes that gap.  When ``DocumentReader.create("corpus.zip")``
is called, the registry now returns :class:`ZipReader` (it overrides
:class:`ALTOReader` for ``".zip"``).  An ALTO page among the members is read
with ALTO's layout by :class:`XMLReader` (CX-43), and members are read in
natural order of their paths (CX-45), so an archive of ALTO pages gives its
text in page order. :class:`ALTOReader` remains available for its extra
options (``granularity``; a page number from the archive position).

Design invariants:

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

Python compatibility:

Python 3.8-3.15.  Zero optional dependencies — ``zipfile`` is stdlib.
"""  # noqa: D205, D400

from __future__ import annotations

import contextvars
import logging
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generator, Optional  # noqa: F401

from .._base import _TEXT_RATIO_FLOOR, DocumentReader
from .._schema import SectionType  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "ArchiveBudgetError",
    "ArchiveNestingError",
    "ZipReader",
]

#: Maximum archive nesting depth (zip-in-zip-in-…) processed before refusing,
#: to bound recursion from a nested-archive bomb / zip quine (CORPUS-ARC-002).
DEFAULT_MAX_ARCHIVE_DEPTH = 8

#: Shared archive-nesting context ``(depth, max_depth)`` propagated through the
#: synchronous re-dispatch of nested archives: ``(depth, max_depth, budget)``.
#: ``None`` marks the top level, where the cap is taken from the top reader's
#: ``max_depth`` and the extraction budget from its size, and both are then
#: shared with every nested level (*global* bounds, not per-reader). Deliberately not
#: annotated with a subscripted generic — that is Python 3.9+ only.
_archive_ctx = contextvars.ContextVar("_scikitplot_corpus_archive_ctx", default=None)


class ArchiveNestingError(ValueError):
    """Raised when archive nesting exceeds ``max_depth`` (nested-archive bomb).

    A subclass of :class:`ValueError` (so existing handlers still catch it), but
    distinct so per-member error handling re-raises it and aborts the whole
    archive rather than skipping a single member (CORPUS-ARC-002).
    """


class ArchiveBudgetError(ValueError):
    """Raised when the members of an archive tree inflate past its extraction budget.

    The budget belongs to the top-level archive and is shared by every
    archive nested in it, so a small file cannot extract more through
    nesting than it could directly. Like :class:`ArchiveNestingError`, it
    aborts the whole tree instead of skipping one member (CX-41). With
    ``on_text_limit="truncate"`` the top-level reader ends the archive with
    a warning instead, keeping what it has read.
    """


@dataclass
class _ExtractBudget:
    """Bytes the whole archive tree may still extract, and how messages name the limit."""

    remaining: int
    described: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default maximum number of files inside a ZIP.
_DEFAULT_MAX_FILES: int = 10_000
#: Most bytes one character of text takes in any Unicode encoding: members
#: are extracted up to this many bytes for each character the ratio limit
#: allows, so plain text refused at extraction would be refused as text too.
_MAX_BYTES_PER_CHAR: int = 4

#: Default maximum cumulative uncompressed size in bytes (2 GB).
_DEFAULT_MAX_TOTAL_BYTES: int = 2 * 1024 * 1024 * 1024

#: Path components that trigger member exclusion.
_SKIP_PARTS: frozenset[str] = frozenset({"__pycache__", "__MACOSX"})


_DIGITS = re.compile(r"(\d+)")


def _natural_key(name: str) -> tuple[tuple[tuple[int, int, str], ...], str]:
    """
    Sort key that orders member paths as a person reads them.

    Parameters
    ----------
    name : str
        Member path inside the archive.

    Returns
    -------
    tuple
        Per path component, digit runs compared as numbers and other text
        case-insensitively; the path itself last, so the order is total and
        the same on every system.

    Examples
    --------
    >>> sorted(["p10.xml", "p2.xml", "P1.xml", "2.xml"], key=_natural_key)
    ['2.xml', 'P1.xml', 'p2.xml', 'p10.xml']
    """
    parts = tuple(
        (0, int(piece), "") if piece.isdigit() else (1, 0, piece.casefold())
        for component in name.split("/")
        for piece in _DIGITS.split(component)
        if piece
    )
    return parts, name


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


def _archive_relative(text: str, extract_dir: Path, archive_name: str) -> str:
    """
    Replace the temporary extraction directory in *text* with the archive.

    Parameters
    ----------
    text : str
        A message that may name an extracted member by its temporary path,
        such as a member reader's exception text.
    extract_dir : Path
        The temporary directory the members were extracted into.
    archive_name : str
        The archive's display name.

    Returns
    -------
    str
        *text* with every path under *extract_dir* replaced by
        ``<archive_name>/<member>``, so ``/tmp/skplt_zip_x1y2/a/b.png`` reads
        ``docs.zip/a/b.png``. On Windows the member's backslashes, single or
        doubled by ``repr()``, become ``/``. The same archive gives the same
        message on every run.

    Notes
    -----
    Both the directory as created and its resolved form are replaced: on
    some systems the temporary directory is reached through a symbolic
    link (macOS ``/var`` -> ``/private/var``), and members are extracted to
    resolved paths. The longer form is replaced first so the shorter one
    cannot match inside it.
    """
    import os  # noqa: PLC0415
    import re  # noqa: PLC0415

    windows = os.sep == "\\"
    forms = set()
    for root in {str(extract_dir), str(extract_dir.resolve())}:
        forms.add((root, os.sep))
        forms.add((root, "/"))
        if windows:
            # repr() of a Windows path doubles each backslash.
            forms.add((root.replace("\\", "\\\\"), "\\\\"))

    def member(match: re.Match) -> str:
        tail = match.group(1)
        if windows:
            tail = tail.replace("\\\\", "/").replace("\\", "/")
        return f"{archive_name}/{tail}"

    for root, sep in sorted(forms, key=lambda f: len(f[0]), reverse=True):
        text = re.sub(re.escape(root + sep) + r"([^'\"\s]*)", member, text)
    return text


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
    max_text_chars : int or None, optional
        Most characters of text the archive may yield, over all members.
        Default: 256 Mi; ``None``: no limit.
    max_text_chars_per_byte : float or None, optional
        Most characters of text per byte of the archive on disk: the limit
        is the smaller of ``max_text_chars`` and
        ``max(max_text_chars_per_byte x archive size, 1 Mi)``. A member is
        read after extraction, so its own reader measures it against its
        inflated size; only the archive's size on disk shows what deflate
        (about 1000 times at most) made of a small file. Default: 50
        (text compresses 3 to 10 times); ``None``: no ratio limit.
        The ratio also bounds extraction: the archive is refused once its
        members inflate past 4 bytes (the most one character takes) for
        each character this ratio allows, by default
        ``max(200 x archive size, 4 Mi)`` bytes and never more than
        ``max_total_bytes``. The budget belongs to the top-level archive:
        archives nested in it draw from the same budget. Plain text refused
        there would be refused by the text limit anyway, without first
        being extracted and read. Markup holds more bytes than text and
        needs no more than this when it compresses less than 200 times
        (honest XML: about 15); a member that compresses more (a blank
        bitmap, a highly repetitive export) needs a larger ratio or
        ``None``.
    on_text_limit : {"refuse", "truncate"}, optional
        What happens past the limit; see
        :class:`~scikitplot.corpus._base.DocumentReader`.
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
    wins. ALTO pages inside the archive are still read as ALTO (by
    :class:`XMLReader`, see its ``alto`` option). For ALTOReader's own
    options, instantiate it directly::

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

    # BUG-03 fix: file_type removed — file_types used for registration.
    file_types: ClassVar[list[str] | None] = [".zip"]

    max_files: int = field(default=_DEFAULT_MAX_FILES)
    """Maximum file count inside the archive. Default: 10,000."""

    max_total_bytes: int = field(default=_DEFAULT_MAX_TOTAL_BYTES)
    """
    Maximum **cumulative uncompressed** bytes across all extracted ZIP members.

    **BUG-10 clarification:** This is a different guard than
    :attr:`~scikitplot.corpus._readers.ALTOReader.max_file_bytes` in
    :class:`ALTOReader`.

    * ``ZipReader.max_total_bytes`` — sum of all member sizes **after
      extraction**.  Guards against ZIP bomb attacks where many small
      compressed members expand to a huge total.

    * ``ALTOReader.max_file_bytes`` — size of the ZIP archive file
      **on disk before opening**.  A coarser, pre-flight guard.

    Both limits exist because different threat models apply:
    pre-download (file size) vs post-extraction (total expanded size).
    With ``max_text_chars_per_byte`` the bound in force is smaller (see
    :meth:`_extract_limit`). Default: 2 GB.
    """

    max_depth: int = field(default=DEFAULT_MAX_ARCHIVE_DEPTH)
    """Most levels of archives nested in one another, the top one included. Default: 8."""

    max_text_chars: int | None = field(default=256 * 1024 * 1024)
    """Most characters of text the archive yields; ``None``: no limit. Default: 256 Mi.

    Enforced by :class:`~scikitplot.corpus._base.DocumentReader` with
    ``on_text_limit``; this reader only sets the default.
    """

    max_text_chars_per_byte: float | None = field(default=50)
    """Most characters of text per byte of the archive (floor: 1 Mi characters).

    Default: 50; ``None``: no ratio limit. See
    :attr:`~scikitplot.corpus._base.DocumentReader.max_text_chars_per_byte`.
    """

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

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield raw chunks of every supported member; see :meth:`_iter_raw_chunks`.

        Yields
        ------
        dict[str, Any]

        Raises
        ------
        ArchiveNestingError
            If the archive nesting depth would exceed ``max_depth``.
        ArchiveBudgetError
            If the archive tree inflates past its extraction budget.
        """
        yield from self._iter_raw_chunks()

    def _iter_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield every member's chunks within the archive's depth, bytes and text bounds.

        This is the one path both :meth:`get_raw_chunks` and
        :meth:`~scikitplot.corpus._base.DocumentReader.get_documents` take,
        so the bounds hold whichever is called:

        - **Depth.** Nested archives are re-dispatched synchronously; a
          shared :data:`_archive_ctx` counter bounds recursion and refuses
          a zip quine or a deeply nested archive (CORPUS-ARC-002).
        - **Extraction bytes.** The top-level archive sets one budget
          (:meth:`_extract_limit` of its size on disk) that every nested
          archive draws from (CX-41).
        - **Text.** This reader's ``max_text_chars`` (and ratio) bound the
          total; each member is also bounded by its own reader.

        With ``on_text_limit="truncate"``, the top-level reader ends the
        archive with a warning when the extraction budget runs out,
        keeping the chunks already yielded.
        """
        ctx = _archive_ctx.get()
        if ctx is None:
            # Top of the chain: this reader sets the shared cap and budget.
            depth, max_depth, budget = 0, self.max_depth, None
        else:
            depth, max_depth, budget = ctx
        if depth >= max_depth:
            raise ArchiveNestingError(
                f"ZipReader: archive nesting depth would exceed "
                f"max_depth={max_depth} (current depth {depth}) while "
                f"processing {str(self.input_path)!r}. Refusing to recurse "
                f"further — possible nested-archive bomb (zip quine)."
            )
        top = budget is None
        if top:
            budget = _ExtractBudget(
                *self._extract_limit(self.input_path.stat().st_size)
            )
        token = _archive_ctx.set((depth + 1, max_depth, budget))
        try:
            yield from self._within_text_limit(self._iter_members(budget))
        except ArchiveBudgetError as exc:
            if not top or self.on_text_limit != "truncate":
                raise
            logger.warning(
                "%s The rest of the archive is not read (on_text_limit='truncate').",
                exc,
            )
        finally:
            _archive_ctx.reset(token)

    def _extract_limit(self, archive_size: int) -> tuple[int, str]:
        """
        Return the extraction limit in force for an archive of *archive_size* bytes.

        Returns
        -------
        tuple of (int, str)
            The most bytes the members may inflate to in total, and how a
            message names it (the option that decides, with its value).

        Notes
        -----
        The smaller of ``max_total_bytes`` and, with a ratio,
        ``4 x max(max_text_chars_per_byte x archive size, 1 Mi)``. Derived
        from the ratio only: an absolute ``max_text_chars`` (as set with
        ``on_text_limit="truncate"`` to sample large documents) says
        nothing about how large the members may be.
        """
        limit = self.max_total_bytes
        described = f"max_total_bytes={limit:,}"
        ratio = self.max_text_chars_per_byte
        if ratio is not None:
            chars = max(math.ceil(ratio * archive_size), _TEXT_RATIO_FLOOR)
            by_ratio = _MAX_BYTES_PER_CHAR * chars
            if by_ratio < limit:
                limit = by_ratio
                described = (
                    f"{by_ratio:,} bytes for an archive of {archive_size:,} bytes"
                    f" ({_MAX_BYTES_PER_CHAR} bytes for each character that"
                    f" max_text_chars_per_byte={ratio} allows); pass a larger"
                    " max_text_chars_per_byte (or None) for an archive of"
                    " highly compressible members"
                )
        return limit, described

    def _iter_members(  # noqa: PLR0912
        self,
        budget: _ExtractBudget,
    ) -> Generator[dict[str, Any], None, None]:
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
        # ``file_name`` preserves a parent ZipReader's filename_override for
        # nested archives, giving compositional provenance such as
        # ``outer.zip/inner.zip/document.txt``.
        archive_name = self.file_name
        supported = set(DocumentReader.supported_types())

        tmp_dir = Path(tempfile.mkdtemp(prefix="skplt_zip_"))
        # Track member count for the final log line regardless of
        # whether the ZipFile block completes normally or raises early.
        members_count: int = 0
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                # Natural order of the member paths (page_2 before page_10),
                # not the order the archiving tool happened to store them in:
                # reading order and outputs are then the same for any copy.
                members = sorted(
                    (m for m in zf.infolist() if not m.is_dir()),
                    key=lambda m: _natural_key(m.filename),
                )
                members_count = len(members)

                # ── File count guard ──────────────────────────────────
                if len(members) > self.max_files:
                    raise ValueError(
                        f"ZipReader: archive {archive_path.name!r} contains "
                        f"{len(members)} files (max_files={self.max_files}). "
                        f"Increase max_files or split the archive."
                    )

                total_bytes = 0
                # This archive's own bound, and the tree's shared budget.
                own_limit, own_described = self._extract_limit(
                    archive_path.stat().st_size
                )

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
                            archive_name,
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

                    # ── Bomb guard: declared pre-check + actual-byte budget ──
                    if own_limit - total_bytes <= budget.remaining:
                        allowed, described = own_limit - total_bytes, own_described
                    else:
                        allowed, described = budget.remaining, budget.described
                    if info.file_size > allowed:
                        raise ArchiveBudgetError(
                            f"ZipReader: member {member_path!r} of"
                            f" {archive_path.name!r} declares {info.file_size:,}"
                            f" bytes, {allowed:,} left of {described}."
                            " Possible zip bomb."
                        )

                    # ── Extract single member (streamed, memory-bounded) ──
                    from .._archive_handler import (  # noqa: PLC0415
                        stream_copy_bounded,
                    )

                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as dst:
                        try:
                            copied = stream_copy_bounded(
                                src,
                                dst,
                                max_bytes=allowed,
                                member_name=member_path,
                                archive_name=archive_path.name,
                            )
                        except ValueError as exc:  # the content outgrew its header
                            raise ArchiveBudgetError(
                                f"ZipReader: {exc} ({described})"
                            ) from exc
                    total_bytes += copied
                    budget.remaining -= copied

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
                        # Preserve child-reader provenance when yielding
                        # through the outer ZipReader.  Without this bridge,
                        # DocumentReader.get_documents() only sees the outer
                        # archive reader and collapses every document to the
                        # archive filename / UNKNOWN source type.
                        member_label = f"{archive_name}/{member_path}"
                        # The member's own text limit (an OfficeReader or
                        # PDFReader default, or reader_kwargs): a refused
                        # member is logged and skipped like any member error.
                        for raw_chunk in sub_reader._within_text_limit(
                            sub_reader.get_raw_chunks()
                        ):
                            bridged = dict(raw_chunk)
                            bridged.setdefault("input_path", member_label)
                            for key, value in sub_reader.source_provenance.items():
                                if value is not None:
                                    bridged.setdefault(key, value)
                            yield bridged
                        logger.debug(
                            "ZipReader: processed member %s from %s (type=%s)",
                            member_path,
                            archive_name,
                            member_st,
                        )
                    except (ArchiveNestingError, ArchiveBudgetError):
                        # Security limits: abort the whole tree, never swallow.
                        raise
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "ZipReader: failed to read member %s from %s: %s",
                            member_path,
                            archive_name,
                            _archive_relative(str(exc), tmp_dir, archive_name),
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
