# scikitplot/corpus/_readers/_notebook.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Reader for Jupyter notebooks (``.ipynb``).

:class:`NotebookReader` parses the notebook's JSON with the standard
library and yields one chunk per cell: markdown cells as text, code cells
as code in the kernel's language, and, when asked, the text a cell
printed. Images and other binary outputs are never read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator, Iterator, Optional  # noqa: F401

from .._base import ReadCoverage
from .._schema import SectionType, SourceType
from ._text import TextReader

logger = logging.getLogger(__name__)

__all__ = ["NotebookReader"]

_CELL_TYPES = ("markdown", "code", "raw")
#: Output data read as text, in order of preference.
_TEXT_MIMES = ("text/markdown", "text/plain")
_DEFAULT_MAX_FILE_BYTES = 100 * 1024 * 1024


def _joined(value: Any) -> str:
    """Return a notebook text field: a string or a list of line strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(v for v in value if isinstance(v, str))
    return ""


def _output_text(output: Any) -> str:
    """
    Return the text an output cell holds, or ``""``.

    ``stream`` text, an error's name and message, or the first of
    ``text/markdown`` and ``text/plain`` of a result; images, HTML and
    other data are left out.
    """
    if not isinstance(output, dict):
        return ""
    kind = output.get("output_type")
    if kind == "stream":
        return _joined(output.get("text"))
    if kind in ("pyerr", "error"):
        return f"{output.get('ename', '')}: {output.get('evalue', '')}".strip(": ")
    data = output.get("data")
    if not isinstance(data, dict):  # nbformat 3 keeps text at the top level
        return _joined(output.get("text"))
    for mime in _TEXT_MIMES:
        if mime in data:
            return _joined(data[mime])
    return ""


def _cells(notebook: dict[str, Any]) -> list[Any]:
    """Return the cells of an nbformat 4 or 3 notebook."""
    if isinstance(notebook.get("cells"), list):
        return notebook["cells"]
    cells: list[Any] = []
    for sheet in notebook.get("worksheets") or []:  # nbformat 3
        if isinstance(sheet, dict) and isinstance(sheet.get("cells"), list):
            cells.extend(sheet["cells"])
    return cells


def _language(notebook: dict[str, Any]) -> str | None:
    """Return the kernel's language from the notebook metadata, if stated."""
    meta = notebook.get("metadata")
    if not isinstance(meta, dict):
        return None
    for key, inner in (("kernelspec", "language"), ("language_info", "name")):
        block = meta.get(key)
        if isinstance(block, dict) and isinstance(block.get(inner), str):
            return block[inner].strip().lower() or None
    return None


@dataclass
class NotebookReader(TextReader):
    """
    Read a Jupyter notebook (``.ipynb``) into one chunk per cell.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the notebook.
    cell_types : sequence of str, optional
        Cells read, among ``"markdown"``, ``"code"`` and ``"raw"``.
        Default: ``("markdown", "code")``.
    outputs : bool, optional
        Also yield the text each code cell printed (stream output, errors,
        ``text/markdown`` or ``text/plain`` results), as a chunk after the
        cell. Images, HTML and other data are never read. Default:
        ``False``.
    max_file_bytes : int, optional
        Larger files are refused. Default: 100 MB.

    Attributes
    ----------
    file_types : list of str
        Class variable. ``[".ipynb"]``.
    coverage : ReadCoverage
        After a read: cells read, empty and skipped (by type).

    Raises
    ------
    ValueError
        If the file is too large, not JSON, not a notebook or nests too
        deeply.

    See Also
    --------
    scikitplot.corpus._readers.CodeReader : Source files.
    scikitplot.corpus._readers.JSONReader : JSON as path lines.

    Notes
    -----
    **User note:** Each chunk carries ``cell_index`` (0-based position
    among all cells), ``cell_type``, ``execution_count`` for code cells,
    ``language`` for code (the kernel's, from the notebook metadata, else
    ``None``) and ``is_output`` for output chunks. Markdown is
    :attr:`SectionType.TEXT`; code and outputs are :attr:`SectionType.CODE`.
    ``source_type`` is :attr:`SourceType.CODE`. Both nbformat 4 and the
    older nbformat 3 (worksheets) are read.

    **Developer note:** The notebook is parsed whole with :mod:`json`; a
    notebook with large embedded images costs memory in proportion, which
    ``max_file_bytes`` bounds. Output text counts toward the text limit
    like any other chunk.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = NotebookReader(input_path=Path("analysis.ipynb"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".ipynb"
    file_types: ClassVar[list[str] | None] = [".ipynb"]

    cell_types: tuple[str, ...] = field(default=("markdown", "code"))
    """Cells read, among ``"markdown"``, ``"code"`` and ``"raw"``."""

    outputs: bool = field(default=False)
    """Also yield the text code cells printed."""

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """Larger files are refused. Default: 100 MB."""

    def __post_init__(self) -> None:
        """Validate the options."""
        super().__post_init__()
        types = self.cell_types
        if (
            isinstance(types, str)
            or not types
            or any(t not in _CELL_TYPES for t in types)
        ):
            raise ValueError(
                f"NotebookReader: cell_types must be a non-empty sequence of"
                f" {_CELL_TYPES}; got {types!r}."
            )
        self.cell_types = tuple(types)
        if not isinstance(self.outputs, bool):
            raise TypeError(
                f"NotebookReader: outputs must be a bool; got {self.outputs!r}."
            )
        self.coverage = ReadCoverage(unit="cell")

    def _notebook(self) -> dict[str, Any] | None:
        whole = next(iter(super().get_raw_chunks()), None)
        if whole is None:
            return None
        try:
            notebook = json.loads(whole["text"].lstrip("\ufeff"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"NotebookReader: {self.file_name} is not JSON (line {exc.lineno}"
                f" column {exc.colno}: {exc.msg})."
            ) from exc
        except RecursionError as exc:
            raise ValueError(
                f"NotebookReader: {self.file_name} nests too deeply to parse."
            ) from exc
        if not isinstance(notebook, dict) or not (
            "cells" in notebook or "worksheets" in notebook
        ):
            raise ValueError(
                f"NotebookReader: {self.file_name} is JSON but not a notebook"
                " (no 'cells' or 'worksheets')."
            )
        return notebook

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield one chunk per cell (and per cell's output, with ``outputs``).

        Yields
        ------
        dict
            ``text``, ``section_type``, ``source_type``, ``cell_index``,
            ``cell_type``; code cells add ``language`` and
            ``execution_count``; outputs add ``is_output=True``.

        Raises
        ------
        ValueError
            If the file is not a readable notebook.
        """
        coverage = ReadCoverage(unit="cell")
        self.coverage = coverage
        notebook = self._notebook()
        if notebook is None:
            return
        language = _language(notebook)
        for index, cell in enumerate(_cells(notebook)):
            if not isinstance(cell, dict):
                coverage.skip("malformed")
                continue
            kind = cell.get("cell_type")
            if kind not in self.cell_types:
                coverage.skip(str(kind) if kind in _CELL_TYPES else "unknown")
                continue
            # nbformat 4 keeps code in "source", nbformat 3 in "input".
            text = _joined(cell.get("source", cell.get("input"))).rstrip()
            base: dict[str, Any] = {
                "source_type": SourceType.CODE.value,
                "cell_index": index,
                "cell_type": kind,
            }
            if kind == "code":
                count = cell.get("execution_count", cell.get("prompt_number"))
                base["language"] = language
                base["execution_count"] = count if isinstance(count, int) else None
            if not text.strip():
                coverage.empty += 1
            else:
                coverage.read += 1
                yield {
                    "text": text,
                    "section_type": (
                        SectionType.TEXT.value
                        if kind == "markdown"
                        else SectionType.CODE.value
                    ),
                    **base,
                }
            if self.outputs and kind == "code":
                printed = "\n".join(
                    t.rstrip()
                    for t in map(_output_text, cell.get("outputs") or [])
                    if t.strip()
                )
                if printed:
                    yield {
                        "text": printed,
                        "section_type": SectionType.CODE.value,
                        "is_output": True,
                        **base,
                    }
