# scikitplot/corpus/_readers/_tar.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._readers._tar
===============================

Tar archives and single compressed files for the corpus pipeline.

:class:`TarReader` reads a ``.tar`` archive, plain or compressed with gzip,
bzip2 or xz (``.tar.gz``, ``.tgz``, ``.tar.bz2``, ``.tbz2``, ``.tar.xz``,
``.txz``), and a single compressed file (``notes.txt.gz``, ``data.csv.xz``).
Each member is read by the reader registered for its extension, exactly as
:class:`~scikitplot.corpus._readers.ZipReader` does: it is a ``ZipReader``,
so the nesting depth, the shared extraction budget, the path guard, the
natural reading order and the coverage record are the same.

What decides the format:

The file's first bytes, not its name. A ``.gz``, ``.bz2`` or ``.xz`` file
holding a tar archive is read as an archive, whatever its name; any other
holds one compressed file, read under its own name without the compression
suffix (``notes.txt.gz`` is read as ``notes.txt``). A ``.tar``, ``.tgz``,
``.tbz``, ``.tbz2`` or ``.txz`` file must hold a tar archive.

Design invariants (beyond :class:`ZipReader`'s):

* **Regular files only.** Directories are structure; symbolic and hard
  links, devices and FIFOs are skipped (``"not-a-file"``), so a link cannot
  point a member outside the archive or repeat another member's bytes.
* **One member per name.** A tar may store a path more than once (an
  appended update); the last copy wins, as when the archive is unpacked.
  Names that differ only in case are one name, so the result is the same on
  case-insensitive file systems. Earlier copies are skipped (``"duplicate"``).
* **Bounded scan.** A compressed tar has no index: reaching a member means
  decompressing everything stored before it, read or not. The sizes of all
  members, skipped ones included, count against the archive's extraction
  limit, checked at each header before its data is decompressed.
* **One pass.** Members are extracted in the order they are stored, in one
  forward pass (no re-decompression), and read in natural order.

Python compatibility:

Python 3.8-3.15. Standard library only (``tarfile``, ``gzip``, ``bz2``,
``lzma``); a Python built without ``bz2`` or ``lzma`` reads the other
formats and names the missing module for these.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import tarfile
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generator, Optional  # noqa: F401

from .._base import DocumentReader, ReadCoverage
from ._zip import (
    ArchiveBudgetError,
    ZipReader,
    _archive_ctx,
    _ExtractBudget,
    _natural_key,
)

logger = logging.getLogger(__name__)

__all__ = ["TarReader"]

#: Leading bytes of each compression, and the module that reads it.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gz"),
    (b"BZh", "bz2"),
    (b"\xfd7zXZ\x00", "xz"),
)

#: Extensions that name a tar archive; these never hold a single file.
_TAR_ONLY: frozenset[str] = frozenset({".tar", ".tgz", ".tbz", ".tbz2", ".txz"})

#: Most bytes ``tarfile`` may ask for in one read; member data is read in
#: 1 MiB blocks, far below it.
_MAX_READ_BYTES: int = 8 * 1024 * 1024

#: Most header bytes one member may have. ``tarfile`` keeps a pax or GNU
#: long-name header in memory whole (in one read, or in pieces it joins),
#: so this bounds the memory a header takes. Real headers take a few KiB.
_MAX_HEADER_BYTES: int = 1024 * 1024

#: Header bytes allowed for each file a compressed scan may pass, on top of
#: the members' data: a header block plus a pax or long-name record.
_HEADER_BYTES_PER_FILE: int = 4096


class _BoundedStream:
    """
    A read-only stream for ``tarfile`` that refuses what a hostile archive needs.

    Refuses a single read of more than :data:`_MAX_READ_BYTES`; while
    members are listed, more than :data:`_MAX_HEADER_BYTES` of headers for
    one member; and, when *limit* is given, any read or seek past that
    position in the decompressed stream, before decompressing it.

    Parameters
    ----------
    raw : file-like
        The binary stream: the file itself, or its decompressed view.
    name : str
        The archive's name, for messages.
    limit : int or None
        Furthest position the stream may reach; ``None`` for no bound.
    """

    def __init__(self, raw: Any, name: str, limit: int | None) -> None:
        self._raw = raw
        self._name = name
        self._limit = limit
        #: Header bytes read for the member being listed; ``None`` once
        #: listing ends. While listing, ``tarfile`` passes over member data
        #: with ``seek``, so every byte it reads is a header byte.
        self.header_bytes: int | None = 0

    def _refuse_past(self, end: int) -> None:
        if self._limit is not None and end > self._limit:
            raise ArchiveBudgetError(
                f"TarReader: reading {self._name!r} would decompress past"
                f" {self._limit:,} bytes (its members' extraction limit plus"
                f" {_HEADER_BYTES_PER_FILE:,} bytes of headers per file)."
                " Possible tar bomb."
            )

    def read(self, size: int = -1) -> bytes:
        # The header budget, the smaller bound, is checked first: an
        # oversized header is refused as one whether ``tarfile`` asks for it
        # in one read or in bounded chunks (newer CPython releases), so the
        # refusal does not depend on the Python version.
        if self.header_bytes is not None and size is not None and size >= 0:
            self.header_bytes += size
            if self.header_bytes > _MAX_HEADER_BYTES:
                raise ArchiveBudgetError(
                    f"TarReader: a header in {self._name!r} takes more than"
                    f" {_MAX_HEADER_BYTES:,} bytes. Possible tar bomb."
                )
        if size is None or size < 0 or size > _MAX_READ_BYTES:
            raise ArchiveBudgetError(
                f"TarReader: reading {self._name!r} asks for {size!r} bytes at"
                f" once, more than {_MAX_READ_BYTES:,}. Possible tar bomb."
            )
        self._refuse_past(self._raw.tell() + size)
        return self._raw.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._refuse_past(offset)
        elif whence == 1:
            self._refuse_past(self._raw.tell() + offset)
        else:  # tarfile reads never seek from the end
            raise OSError(f"TarReader: unsupported seek whence={whence!r}")
        return self._raw.seek(offset, whence)

    def tell(self) -> int:
        return self._raw.tell()

    def close(self) -> None:
        self._raw.close()

    def __enter__(self) -> _BoundedStream:  # noqa: PYI034  (Self needs 3.11)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _compression(path: Path) -> str | None:
    """
    Return the compression a file starts with: ``"gz"``, ``"bz2"``, ``"xz"`` or ``None``.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str or None
        Decided by the file's first bytes only.
    """
    with open(path, "rb") as fh:
        head = fh.read(6)
    for magic, name in _MAGIC:
        if head.startswith(magic):
            return name
    return None


def _codec_errors() -> tuple[type[BaseException], ...]:
    """Return the exceptions a damaged tar or compressed stream raises."""
    errors: list[type[BaseException]] = [
        tarfile.TarError,
        EOFError,
        OSError,
        zlib.error,
    ]
    try:
        import lzma  # noqa: PLC0415

        errors.append(lzma.LZMAError)
    except ImportError:  # a Python built without lzma
        pass
    return tuple(errors)


def _open_stream(path: Path, compression: str) -> Any:
    """
    Open a single compressed file for reading its decompressed bytes.

    Raises
    ------
    ValueError
        If this Python was built without the module the compression needs.
    """
    try:
        if compression == "gz":
            import gzip  # noqa: PLC0415

            return gzip.open(path, "rb")
        if compression == "bz2":
            import bz2  # noqa: PLC0415

            return bz2.open(path, "rb")
        import lzma  # noqa: PLC0415

        return lzma.open(path, "rb")
    except ImportError as exc:
        raise ValueError(
            f"TarReader: {path.name!r} is {compression}-compressed, and this"
            f" Python was built without the {exc.name!r} module."
        ) from exc


def _inner_name(name: str) -> str:
    """
    Return the name of the file a single compressed file holds.

    Examples
    --------
    >>> _inner_name("notes.txt.gz")
    'notes.txt'
    >>> _inner_name("NOTES.TXT.XZ")
    'NOTES.TXT'
    """
    return Path(name).stem


@dataclass
class TarReader(ZipReader):
    r"""
    Read a tar archive (plain, gzip, bzip2 or xz) or a single compressed file.

    Every member is read by the reader registered for its extension, and
    documents from all members come in one stream labelled
    ``<archive>/<member>``. All options of
    :class:`~scikitplot.corpus._readers.ZipReader` apply with the same
    meaning (``max_files``, ``max_total_bytes``, ``max_depth``,
    ``max_text_chars``, ``max_text_chars_per_byte``, ``on_text_limit``,
    ``skip_unsupported``, ``coverage_warn_share``, ``reader_kwargs``,
    ``infer_source_type``).

    Parameters
    ----------
    input_path : pathlib.Path
        A ``.tar``, ``.tgz``, ``.tbz``, ``.tbz2``, ``.txz``, ``.gz``,
        ``.bz2`` or ``.xz`` file.

    Raises
    ------
    ValueError
        If a ``.tar``, ``.tgz``, ``.tbz``, ``.tbz2`` or ``.txz`` file holds
        no tar archive; if a ``.gz``, ``.bz2`` or ``.xz`` file is neither
        compressed nor a tar archive; if the archive or stream is damaged;
        or on any :class:`ZipReader` limit.
    ArchiveBudgetError
        If the members, or the data decompressed to reach them, exceed the
        extraction limit.

    See Also
    --------
    scikitplot.corpus._readers.ZipReader : The same for ``.zip`` archives.

    Notes
    -----
    **User note.** Read one member's text with its own options through
    ``reader_kwargs``, for example ``{".csv": {"delimiter": ";"}}``. A
    single compressed file whose inner extension has no reader is skipped
    (``coverage`` counts it) unless ``skip_unsupported=False``.

    **Extraction limit in a compressed tar.** Reaching a member decompresses
    every member stored before it, so all members count against
    ``max_total_bytes`` (or the ratio limit), not only those read. An
    archive of large members of other types needs a larger
    ``max_text_chars_per_byte`` (or ``None``). In a plain ``.tar`` other
    members are passed over without being read and do not count.

    **max_files** counts every entry of a tar, directories included
    (``ZipReader`` counts files only): ``tarfile`` keeps a record of each
    entry it lists, so the count bounds that memory.

    **Hostile headers.** One member's headers (a pax or GNU long-name
    record, which ``tarfile`` holds in memory whole) may take at most
    1 MiB, and a compressed stream is never decompressed past the
    extraction limit plus 4 KiB of headers for each entry ``max_files``
    allows. Both refuse with :exc:`ArchiveBudgetError` before the memory
    or the decompression is spent.

    **With** ``on_text_limit="truncate"`` the top-level archive keeps the
    members extracted, in stored order, before the limit was reached, reads
    them, and then logs a warning.

    **Developer note.** Only ``tarfile.TarFile.extractfile`` on regular
    members is used, never ``extract``/``extractall``: no file is created
    from a link, a device or an absolute name, on any Python version.

    Examples
    --------
    >>> from pathlib import Path
    >>> import scikitplot.corpus._readers
    >>> from scikitplot.corpus._base import DocumentReader
    >>> reader = DocumentReader.create(Path("papers.tar.gz"))  # doctest: +SKIP
    >>> type(reader).__name__  # doctest: +SKIP
    'TarReader'
    """

    file_types: ClassVar[list[str] | None] = [
        ".tar",
        ".tgz",
        ".tbz",
        ".tbz2",
        ".txz",
        ".gz",
        ".bz2",
        ".xz",
    ]

    def _iter_members(  # noqa: PLR0912
        self,
        budget: _ExtractBudget,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Extract the archive's members and yield their raw chunks.

        Raises
        ------
        ValueError
            If the file is not what its extension requires, is damaged, or
            holds more than ``max_files`` members.
        ArchiveBudgetError
            If the members exceed the extraction limit.
        """
        path = self.input_path
        compression = _compression(path)
        own_limit, own_described = self._extract_limit(path.stat().st_size)
        # Resolved once: each member path is checked against it.
        tmp_dir = Path(tempfile.mkdtemp(prefix="skplt_tar_")).resolve()
        coverage = ReadCoverage(unit="member")
        self.coverage = coverage
        members_count = 0
        try:
            if path.stat().st_size > 0:  # an empty file has no members
                opened = self._open_tar(path, compression, own_limit)
                if opened is None:
                    members_count = 1
                    yield from self._single(
                        compression=compression,
                        tmp_dir=tmp_dir,
                        coverage=coverage,
                        own_limit=own_limit,
                        own_described=own_described,
                        budget=budget,
                    )
                else:
                    tar, stream = opened
                    with stream, tar:
                        members_count, chunks = self._from_tar(
                            tar=tar,
                            stream=stream,
                            compression=compression,
                            tmp_dir=tmp_dir,
                            coverage=coverage,
                            own_limit=own_limit,
                            own_described=own_described,
                            budget=budget,
                        )
                    yield from chunks
        finally:
            import shutil  # noqa: PLC0415

            shutil.rmtree(tmp_dir, ignore_errors=True)
        self._finish_read(members_count, coverage)

    def _open_tar(self, path: Path, compression: str | None, own_limit: int) -> Any:
        """
        Open *path* as a tar archive, or return ``None`` for a single compressed file.

        Returns ``(archive, stream)``: ``tarfile`` leaves a caller's stream
        open, so the caller closes both.

        ``tarfile`` reads through a :class:`_BoundedStream`: one read is at
        most :data:`_MAX_READ_BYTES`, and a compressed stream is never
        decompressed past *own_limit* plus :data:`_HEADER_BYTES_PER_FILE`
        for each file ``max_files`` allows.

        Raises
        ------
        ValueError
            If the file must be a tar archive (by its extension, or because
            it is not compressed) and is not one, is damaged, or if this
            Python lacks the module its compression needs.
        ArchiveBudgetError
            If a header or the scan exceeds the bounds above.
        """
        if compression is None:
            # Returned with the archive; the caller closes it.
            raw = open(path, "rb")  # noqa: SIM115
            limit = None
        else:
            raw = _open_stream(path, compression)
            limit = own_limit + _HEADER_BYTES_PER_FILE * (self.max_files + 1)
        stream = _BoundedStream(raw, path.name, limit)
        try:
            tar = tarfile.open(fileobj=stream, mode="r:")  # noqa: SIM115  (returned)
        except tarfile.ReadError:
            stream.close()
            if compression is None or path.suffix.lower() in _TAR_ONLY:
                what = (
                    f"a {compression}-compressed file"
                    if compression
                    else "not a tar archive or a gzip, bzip2 or xz file"
                )
                raise ValueError(
                    f"TarReader: {path.name!r} holds no tar archive ({what})."
                ) from None
            return None
        except ArchiveBudgetError:
            stream.close()
            raise
        except _codec_errors() as exc:
            stream.close()
            raise ValueError(f"TarReader: {path.name!r} is damaged: {exc}") from exc
        except BaseException:
            stream.close()
            raise
        return tar, stream

    def _single(
        self,
        *,
        compression: str,
        tmp_dir: Path,
        coverage: ReadCoverage,
        own_limit: int,
        own_described: str,
        budget: _ExtractBudget,
    ) -> Generator[dict[str, Any], None, None]:
        """Decompress a single compressed file and yield its chunks."""
        inner = _inner_name(self.input_path.name)
        placed = self._member_target(
            inner,
            tmp_dir,
            set(DocumentReader.supported_types()),
            coverage,
            self.file_name,
        )
        if placed is None:
            return
        target, ext = placed
        allowed, described = self._allowance(own_limit, own_described, 0, budget)
        dst = self._open_target(target, inner, coverage)
        if dst is None:  # a name this file system cannot hold
            return
        try:
            with dst, _open_stream(self.input_path, compression) as src:
                copied = self._copy_into(src, dst, allowed, described, inner)
        except _codec_errors() as exc:
            raise ValueError(
                f"TarReader: {self.input_path.name!r} is damaged: {exc}"
            ) from exc
        budget.remaining -= copied
        yield from self._read_member(target, inner, ext, tmp_dir, coverage)

    def _from_tar(  # noqa: PLR0912
        self,
        *,
        tar: tarfile.TarFile,
        stream: _BoundedStream,
        compression: str | None,
        tmp_dir: Path,
        coverage: ReadCoverage,
        own_limit: int,
        own_described: str,
        budget: _ExtractBudget,
    ) -> tuple[int, Generator[dict[str, Any], None, None]]:
        """
        List, select and extract the members; return their count and chunks.

        Extraction happens here, before the first chunk: one forward pass in
        stored order. The chunks are then read in natural order.
        """
        name = self.input_path.name
        damaged = _codec_errors()
        # 1. List: count files, bound the data a compressed scan decompresses.
        entries: list[tarfile.TarInfo] = []
        scanned = 0
        try:
            listed = 0
            for info in tar:
                stream.header_bytes = 0  # the next member's headers start here
                # Every entry counts, directories too: tarfile keeps each one.
                listed += 1
                if listed > self.max_files:
                    raise ValueError(
                        f"TarReader: archive {name!r} contains more than"
                        f" {self.max_files} entries (max_files={self.max_files};"
                        " a tar counts directories too). Increase max_files or"
                        " split the archive."
                    )
                if info.isdir():
                    continue
                entries.append(info)
                scanned += info.size
                if compression is not None and scanned > own_limit:
                    raise ArchiveBudgetError(
                        f"TarReader: reaching member {info.name!r} of {name!r}"
                        f" decompresses {scanned:,} bytes, past {own_described}."
                        " Possible tar bomb."
                    )
        except damaged as exc:
            raise ValueError(f"TarReader: {name!r} is damaged: {exc}") from exc

        stream.header_bytes = None  # listing is over: reads are member data now

        # 2. Select: regular files, last copy of each name, readable type.
        last: dict[str, tarfile.TarInfo] = {}
        for info in entries:
            if not info.isreg():
                coverage.skip("not-a-file")
                continue
            key = info.name.casefold()
            if key in last:
                coverage.skip("duplicate")
            last[key] = info
        supported = set(DocumentReader.supported_types())
        chosen: list[tuple[tarfile.TarInfo, Path, str]] = []
        for info in sorted(last.values(), key=lambda m: _natural_key(m.name)):
            placed = self._member_target(
                info.name, tmp_dir, supported, coverage, self.file_name
            )
            if placed is not None:
                chosen.append((info, *placed))

        # 3. Extract in stored order, in one pass. Creating a file is kept
        # out of the damage handler: a full disk is not a damaged archive.
        ctx = _archive_ctx.get()
        keep_partial = (
            ctx is not None and ctx[0] == 1 and self.on_text_limit == "truncate"
        )
        extracted: set[str] = set()
        stopped: ArchiveBudgetError | None = None
        used = 0
        unwritable = 0
        for info, target, _ext in sorted(chosen, key=lambda c: c[0].offset_data):
            allowed, described = self._allowance(own_limit, own_described, used, budget)
            try:
                if info.size > allowed:
                    raise ArchiveBudgetError(
                        f"TarReader: member {info.name!r} of {name!r} declares"
                        f" {info.size:,} bytes, {allowed:,} left of {described}."
                        " Possible tar bomb."
                    )
                dst = self._open_target(target, info.name, coverage)
                if dst is None:  # a name this file system cannot hold
                    unwritable += 1
                    continue
                try:
                    with dst, tar.extractfile(info) as src:
                        copied = self._copy_into(
                            src, dst, allowed, described, info.name
                        )
                except damaged as exc:
                    raise ValueError(f"TarReader: {name!r} is damaged: {exc}") from exc
            except ArchiveBudgetError as exc:
                if not keep_partial:
                    raise
                stopped = exc
                break
            used += copied
            budget.remaining -= copied
            extracted.add(info.name)
        if stopped is not None:
            coverage.skip("over-budget", len(chosen) - len(extracted) - unwritable)

        def chunks() -> Generator[dict[str, Any], None, None]:
            for info, target, ext in chosen:
                if info.name in extracted:
                    yield from self._read_member(
                        target, info.name, ext, tmp_dir, coverage
                    )
            if stopped is not None:
                raise stopped

        return len(entries), chunks()
