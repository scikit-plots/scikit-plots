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
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator, Iterator, Optional  # noqa: F401

from .._schema import SectionType, SourceType
from ._text import TextReader

logger = logging.getLogger(__name__)

__all__ = ["CSVReader"]

#: Delimiters a sniff chooses between.
_SNIFF_DELIMITERS = ",;\t|"
#: Characters of the text the sniff looks at.
_SNIFF_CHARS = 64 * 1024
#: Largest table text read in one file (as for spreadsheets' part limit).
_DEFAULT_MAX_FILE_BYTES = 200 * 1024 * 1024


def _sniff_delimiter(text: str) -> str:
    r"""
    Return the delimiter of *text*: the sniffed one among ``, ; \\t |``.

    The sniff reads the first 64 Ki characters, cut back to the last whole
    line, and is deterministic for the same text. ``","`` when it cannot
    tell (a one-column table).

    Examples
    --------
    >>> _sniff_delimiter("a;b;c\n1;2;3\n")
    ';'
    >>> _sniff_delimiter("single\ncolumn\n")
    ','
    """
    sample = text[:_SNIFF_CHARS]
    if len(text) > _SNIFF_CHARS and "\n" in sample:
        sample = sample[: sample.rindex("\n")]
    try:
        return csv.Sniffer().sniff(sample, delimiters=_SNIFF_DELIMITERS).delimiter
    except csv.Error:
        return ","


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
        ``.tsv``, otherwise sniffed among ``,`` ``;`` tab and ``|``.
    header : bool, optional
        The first row names the columns: it opens every chunk and is in
        each chunk's ``header``. Default: ``True``.
    rows_per_chunk : int, optional
        Most data rows in one chunk. Default: ``500``.
    encoding : str, sequence of str, callable or None, optional
        As for :class:`~scikitplot.corpus._readers.TextReader`; a UTF-8
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
        If the file is too large or binary, an option is invalid, or the
        table is malformed (a field over the :mod:`csv` field size limit,
        a NUL character).

    See Also
    --------
    scikitplot.corpus._readers.OfficeReader : ``.xlsx`` and ``.ods`` sheets.

    Notes
    -----
    **User note:** Fields are joined with tabs, as spreadsheet rows are;
    trailing empty fields and blank rows are dropped, and a line break
    inside a quoted field becomes a space, so each record is one line. Each chunk carries
    ``row_start`` and ``row_end`` (1-based record numbers, the header is
    record 1), ``delimiter``, ``encoding`` and, with a header,
    ``header``. Quoted fields may hold the delimiter, quotes (doubled) and
    line breaks; parsing is the :mod:`csv` module's lenient default, as
    spreadsheet applications read such files (an unterminated quote runs
    to the end of the file). ``source_type`` is :attr:`SourceType.SPREADSHEET`.

    **Developer note:** The :mod:`csv` module's field size limit (128 Ki
    characters) is kept: it is process-wide state, so the reader does not
    change it. The text limit in force (``max_text_chars`` or
    ``max_text_chars_per_byte``) ends a chunk early, as for spreadsheets.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = CSVReader(input_path=Path("data.csv"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".csv"
    file_types: ClassVar[list[str] | None] = [".csv", ".tsv"]

    delimiter: str | None = field(default=None)
    """Field delimiter; ``None`` uses tab for ``.tsv`` and a sniff otherwise."""

    header: bool = field(default=True)
    """Whether the first row names the columns."""

    rows_per_chunk: int = field(default=500)
    """Most data rows in one chunk."""

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """Larger files are refused. Default: 200 MB."""

    def __post_init__(self) -> None:
        """Validate the table options."""
        super().__post_init__()
        if self.delimiter is not None and (
            not isinstance(self.delimiter, str)
            or len(self.delimiter) != 1
            or self.delimiter in "\r\n\"'"
        ):
            raise ValueError(
                "CSVReader: delimiter must be one character other than a quote"
                f" or a line break; got {self.delimiter!r}."
            )
        if not isinstance(self.header, bool):
            raise TypeError(f"CSVReader: header must be a bool; got {self.header!r}.")
        rows = self.rows_per_chunk
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError(
                f"CSVReader: rows_per_chunk must be an int > 0; got {rows!r}."
            )

    def _rows(self, text: str, delimiter: str) -> Iterator[tuple[int, str]]:
        """Yield ``(record number, tab-joined line)`` for each non-blank row."""
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
        number = 0
        try:
            for record in reader:
                number += 1
                while record and not record[-1].strip():
                    record.pop()
                if record:
                    # One line per record: a line break in a field is a space.
                    yield (
                        number,
                        "\t".join(
                            " ".join(f.splitlines()) if "\n" in f else f for f in record
                        ),
                    )
        except csv.Error as exc:
            raise ValueError(
                f"CSVReader: {self.file_name}: malformed table at line"
                f" {reader.line_num}: {exc}."
            ) from exc

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Decode the table and yield chunks of ``rows_per_chunk`` rows.

        Yields
        ------
        dict
            ``text``, ``section_type`` (:attr:`SectionType.TABLE`),
            ``source_type``, ``row_start``, ``row_end``, ``delimiter``,
            ``encoding`` and, with a header, ``header``.

        Raises
        ------
        ValueError
            If the file is too large, binary or malformed.
        """
        whole = next(iter(super().get_raw_chunks()), None)
        if whole is None:
            return
        text = whole["text"]
        if text.startswith("\ufeff"):
            text = text[1:]
        delimiter = self.delimiter or (
            "\t" if self.input_path.suffix.lower() == ".tsv" else _sniff_delimiter(text)
        )
        meta = {"delimiter": delimiter, "encoding": whole["encoding"]}
        cap = self._text_limit()[0]
        rows = self._rows(text, delimiter)
        head: str | None = None
        if self.header:
            first = next(rows, None)
            if first is None:
                return
            head = first[1]
        base = len(head) if head is not None else -1
        block: list[tuple[int, str]] = []
        size = base
        for row in rows:
            block.append(row)
            size += len(row[1]) + 1
            if len(block) == self.rows_per_chunk or (cap is not None and size > cap):
                yield _chunk(block, head, meta)
                block = []
                size = base
        if block:
            yield _chunk(block, head, meta)
        elif head is not None:
            # A header with no data rows is still the table's content.
            yield _chunk([first], None, meta)


def _chunk(
    block: list[tuple[int, str]], head: str | None, meta: dict[str, str]
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
