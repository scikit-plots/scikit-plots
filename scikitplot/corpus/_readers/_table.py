# scikitplot/corpus/_readers/_table.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Reader for delimited text tables (``.csv``, ``.tsv``).

:class:`CSVReader` decodes the file as
:class:`~scikitplot.corpus._readers.TextReader` does, parses it with the
standard library :mod:`csv` module and yields rows the way
:class:`~scikitplot.corpus._readers.OfficeReader` yields spreadsheet rows:
tab-separated lines, ``rows_per_chunk`` rows a chunk, each chunk opening
with the header row so it reads on its own.
"""

from __future__ import annotations

import csv
import io
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator, Iterator, Optional  # noqa: F401

from .._schema import SectionType, SourceType
from ._text import TextReader

logger = logging.getLogger(__name__)

__all__ = ["CSVReader"]

#: Delimiters detection chooses between, in tie-break order.
DEFAULT_CANDIDATE_DELIMITERS = ",;\t|"
#: Characters of the text detection looks at, cut back to a whole line.
_SAMPLE_CHARS = 64 * 1024
#: Records of the sample detection looks at.
_SAMPLE_RECORDS = 200
#: Separator that never occurs in text: parses a one-column table, whose
#: quoted fields still hold line breaks, without splitting it.
_NO_SPLIT = "\x1f"
#: Largest table read in one file.
_DEFAULT_MAX_FILE_BYTES = 200 * 1024 * 1024


def _detect_delimiter(sample: str, candidates: str) -> str | None:
    r"""
    Return the delimiter that splits *sample* most consistently, or ``None``.

    The rule, deterministic for the same text: parse the sample's records
    with each candidate (quotes honoured); a candidate's *field count* is
    the count most of its records have. A candidate qualifies when that
    count is at least 2 and more than half of the records have it. The
    winner has the largest share of records at its count, then the larger
    count, then comes first in *candidates*. ``None`` means one column.

    Parameters
    ----------
    sample : str
        Start of the table (blank lines are ignored).
    candidates : str
        Candidate delimiters, in tie-break order.

    Returns
    -------
    str or None
        The delimiter, or ``None`` for a one-column table.

    Examples
    --------
    >>> _detect_delimiter("a;b\n1;2,5\n3;4,0\n", ",;")
    ';'
    >>> _detect_delimiter("Hello, world.\nOne line, then another, and more.\n", ",;")
    """
    best: tuple[float, int, int] | None = None
    chosen: str | None = None
    for rank, candidate in enumerate(candidates):
        counts: Counter[int] = Counter()
        reader = csv.reader(io.StringIO(sample, newline=""), delimiter=candidate)
        try:
            for record in reader:
                if any(f.strip() for f in record):
                    counts[len(record)] += 1
                    if sum(counts.values()) >= _SAMPLE_RECORDS:
                        break
        except csv.Error:
            continue
        total = sum(counts.values())
        if not total:
            continue
        width, hits = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        if width < 2 or hits * 2 <= total:  # noqa: PLR2004
            continue
        score = (hits / total, width, -rank)
        if best is None or score > best:
            best, chosen = score, candidate
    return chosen


@dataclass
class CSVReader(TextReader):
    r"""
    Read a delimited text table into chunks of rows.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the ``.csv`` or ``.tsv`` file.
    delimiter : str or None, optional
        One-character field delimiter. ``None`` (default): tab for
        ``.tsv``, otherwise detected among ``candidate_delimiters`` (see
        Notes).
    candidate_delimiters : str, optional
        Delimiters detection chooses between, in tie-break order. Default:
        ``",;\t|"``.
    header : bool, optional
        The first row names the columns: it opens every chunk and is in
        each chunk's ``header``. Default: ``True``.
    rows_per_chunk : int, optional
        Most data rows in one chunk. Default: ``500``.
    encoding : str, sequence of str, callable or None, optional
        As for :class:`~scikitplot.corpus._readers.TextReader`; a
        byte-order mark (common from spreadsheet exports) is dropped.
    max_file_bytes : int, optional
        Larger files are refused. Default: 200 MB.

    Attributes
    ----------
    file_types : list of str
        Class variable. ``[".csv", ".tsv"]``.

    Raises
    ------
    ValueError
        If the file is too large or binary, an option is invalid, or a
        field exceeds the :mod:`csv` field size limit.

    See Also
    --------
    scikitplot.corpus._readers.OfficeReader : ``.xlsx`` and ``.ods`` sheets.

    Notes
    -----
    **User note:** Fields are joined with tabs, as spreadsheet rows are;
    trailing empty fields and blank rows are dropped, and a line break
    inside a quoted field becomes a space, so each record is one line.
    Each chunk carries ``row_start`` and ``row_end`` (1-based record
    numbers, the header is record 1), ``delimiter`` (``None`` for one
    column), ``encoding`` and, with a header, ``header``.
    ``source_type`` is :attr:`SourceType.SPREADSHEET`. Parsing is the
    :mod:`csv` module's lenient default, as spreadsheet applications read
    such files (an unterminated quote runs to the end of the file).

    **Delimiter detection** parses the first 200 records (at most 64 Ki
    characters) with each candidate and keeps the one whose most common
    field count (at least 2) holds for more than half of them, preferring
    the largest such share, then more fields, then the earlier candidate.
    No candidate qualifying means one column: a line of prose with commas
    is not split at them. Pass ``delimiter=`` to skip detection.

    **Developer note:** The bytes are read once to choose the encoding
    (and released), then the file is parsed as a stream, so memory is one
    chunk of rows, not the whole table. The :mod:`csv` module's field size
    limit (128 Ki characters) is kept: it is process-wide state, so the
    reader does not change it. The text limit in force ends a chunk early,
    as for spreadsheets.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = CSVReader(input_path=Path("data.csv"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".csv"
    file_types: ClassVar[list[str] | None] = [".csv", ".tsv"]

    delimiter: str | None = field(default=None)
    """Field delimiter; ``None`` uses tab for ``.tsv`` and detection otherwise."""

    candidate_delimiters: str = field(default=DEFAULT_CANDIDATE_DELIMITERS)
    """Delimiters detection chooses between, in tie-break order."""

    header: bool = field(default=True)
    """Whether the first row names the columns."""

    rows_per_chunk: int = field(default=500)
    """Most data rows in one chunk."""

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """Larger files are refused. Default: 200 MB."""

    def __post_init__(self) -> None:
        """Validate the table options."""
        super().__post_init__()
        if (
            not isinstance(self.candidate_delimiters, str)
            or not self.candidate_delimiters
        ):
            raise ValueError(
                "CSVReader: candidate_delimiters must be a non-empty string;"
                f" got {self.candidate_delimiters!r}."
            )
        checks = [("delimiter", self.delimiter)]
        checks += [("candidate_delimiters", c) for c in self.candidate_delimiters]
        for name, value in checks:
            if value is not None and (
                not isinstance(value, str) or len(value) != 1 or value in "\r\n\"'"
            ):
                raise ValueError(
                    f"CSVReader: {name} must be one character other than a quote"
                    f" or a line break; got {value!r}."
                )
        if not isinstance(self.header, bool):
            raise TypeError(f"CSVReader: header must be a bool; got {self.header!r}.")
        rows = self.rows_per_chunk
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError(
                f"CSVReader: rows_per_chunk must be an int > 0; got {rows!r}."
            )

    def _rows(self, stream: Any, delimiter: str) -> Iterator[tuple[int, str]]:
        """Yield ``(record number, tab-joined line)`` for each non-blank row."""
        reader = csv.reader(stream, delimiter=delimiter)
        number = 0
        try:
            for record in reader:
                number += 1
                if number == 1 and record and record[0].startswith("\ufeff"):
                    record[0] = record[0][1:]
                while record and not record[-1].strip():
                    record.pop()
                if record:
                    # One line per record: a line break in a field is a space.
                    yield (
                        number,
                        "\t".join(
                            " ".join(f.splitlines()) if "\n" in f or "\r" in f else f
                            for f in record
                        ),
                    )
        except csv.Error as exc:
            raise ValueError(
                f"CSVReader: {self.file_name}: malformed table at line"
                f" {reader.line_num}: {exc}."
            ) from exc

    def _delimiter(self, encoding: str) -> str | None:
        """Return the delimiter in force: given, tab for ``.tsv``, or detected."""
        if self.delimiter is not None:
            return self.delimiter
        if self.input_path.suffix.lower() == ".tsv":
            return "\t"
        with open(self.input_path, encoding=encoding, newline="") as fh:
            sample = fh.read(_SAMPLE_CHARS + 1)
        if len(sample) > _SAMPLE_CHARS and "\n" in sample:
            sample = sample[: sample.rindex("\n")]
        return _detect_delimiter(sample, self.candidate_delimiters)

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Parse the table as a stream and yield chunks of ``rows_per_chunk`` rows.

        Yields
        ------
        dict
            ``text``, ``section_type`` (:attr:`SectionType.TABLE`),
            ``source_type``, ``row_start``, ``row_end``, ``delimiter``,
            ``encoding`` and, with a header, ``header``.

        Raises
        ------
        ValueError
            If the file is too large, binary or has an oversized field.
        """
        encoding = self._codec_of_file()
        if encoding is None:
            logger.warning("CSVReader: %s is empty.", self.file_name)
            return
        delimiter = self._delimiter(encoding)
        meta = {"delimiter": delimiter, "encoding": encoding}
        cap = self._text_limit()[0]
        with open(self.input_path, encoding=encoding, newline="") as stream:
            rows = self._rows(stream, delimiter or _NO_SPLIT)
            head: str | None = None
            first: tuple[int, str] | None = None
            if self.header:
                first = next(rows, None)
                if first is None:
                    return
                head = first[1]
            base = len(head) if head is not None else -1
            block: list[tuple[int, str]] = []
            size = base
            data_rows = 0
            for row in rows:
                data_rows += 1
                block.append(row)
                size += len(row[1]) + 1
                if len(block) == self.rows_per_chunk or (
                    cap is not None and size > cap
                ):
                    yield _chunk(block, head, meta)
                    block = []
                    size = base
            if block:
                yield _chunk(block, head, meta)
            elif first is not None and data_rows == 0:
                # A header with no data rows is still the table's content.
                yield _chunk([first], None, meta)


def _chunk(
    block: list[tuple[int, str]], head: str | None, meta: dict[str, Any]
) -> dict[str, Any]:
    """Return one table chunk: *head* (if any) over the lines of *block*."""
    lines = [line for _, line in block]
    made = {
        "text": "\n".join([head, *lines] if head is not None else lines),
        "section_type": SectionType.TABLE.value,
        "source_type": SourceType.SPREADSHEET.value,
        "row_start": block[0][0],
        "row_end": block[-1][0],
        **meta,
    }
    if head is not None:
        made["header"] = head
    return made
