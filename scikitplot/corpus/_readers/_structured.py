# scikitplot/corpus/_readers/_structured.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Readers for structured data: JSON, JSON Lines, YAML and TOML.

Each value becomes one line ``path: value`` (``users[0].name: Ada``), so
the text keeps what every value means. Lines are grouped by top-level item
(an array element, an object key, a JSON Lines record, a YAML document),
``items_per_chunk`` items a chunk, as spreadsheet rows are grouped.

Decoding is :class:`~scikitplot.corpus._readers.TextReader`'s. JSON uses
:mod:`json`; YAML uses PyYAML's safe loader (no tags that build objects);
TOML uses :mod:`tomllib` (Python 3.11+) or ``tomli``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator, Iterator, Optional  # noqa: F401

from .._base import ReadCoverage
from .._schema import SectionType, SourceType
from ._text import TextReader

logger = logging.getLogger(__name__)

__all__ = ["JSONReader", "TOMLReader", "YAMLReader"]

_DEFAULT_MAX_FILE_BYTES = 100 * 1024 * 1024
_DEFAULT_MAX_VALUES = 5_000_000
_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")
_VALUES = ("all", "text")


def _key_step(key: Any) -> str:
    """
    Return the path step for an object *key*: ``.name`` or ``["a b"]``.

    Examples
    --------
    >>> _key_step("name"), _key_step("a b"), _key_step(3)
    ('.name', '["a b"]', '["3"]')
    """
    text = str(key)
    if _PLAIN_KEY.match(text):
        return "." + text
    return "[" + json.dumps(text, ensure_ascii=False) + "]"


def _scalar(value: Any) -> str | None:  # noqa: PLR0911 (one return per type)
    """Return the text of a scalar, or ``None`` for a container."""
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (dt.date, dt.time)):  # datetime is a date
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        return None
    if isinstance(value, (bytes, bytearray)):  # YAML !!binary
        return f"<{len(value)} bytes>"
    return str(value)


def _lines(root: Any, prefix: str, values: str, budget: list[int]) -> Iterator[str]:
    """
    Yield ``path: value`` lines for every value under *root*, in order.

    Iterative (deep nesting cannot exhaust the stack). A container that
    contains itself (a YAML anchor) is written once as ``<cycle>``.
    *budget* is a one-item list of values still allowed; reaching zero
    raises ``ValueError``. With ``values="text"`` only strings are kept.
    """
    # Iterative walk; a marker entry leaves a container, so the set of open
    # containers is shared, not copied per node (linear in the tree).
    leave = object()
    open_ids: set[int] = set()
    stack: list[tuple[str, Any]] = [(prefix, root)]
    while stack:
        path, node = stack.pop()
        if path is leave:
            open_ids.discard(node)
            continue
        text = _scalar(node)
        if text is not None:
            budget[0] -= 1
            if budget[0] < 0:
                raise ValueError("more values than max_values")
            if values == "all" or isinstance(node, str):
                if "\n" in text or "\r" in text:
                    text = " ".join(text.split())
                yield f"{path or '$'}: {text}"
            continue
        ident = id(node)
        if ident in open_ids:
            yield f"{path or '$'}: <cycle>"
            continue
        budget[0] -= 1  # a container costs one value, so {} and [] count
        if budget[0] < 0:
            raise ValueError("more values than max_values")
        if isinstance(node, dict):
            items = [(path + _key_step(k), v) for k, v in node.items()]
        else:
            items = [(f"{path}[{i}]", v) for i, v in enumerate(node)]
        if not items:
            if values == "all":
                yield f"{path or '$'}: {'{}' if isinstance(node, dict) else '[]'}"
            continue
        open_ids.add(ident)
        stack.append((leave, ident))  # type: ignore[arg-type]
        stack.extend(reversed(items))


@dataclass
class _TreeReader(TextReader):
    """
    Shared base: decode, parse into top-level items, write path lines.

    Subclasses implement :meth:`_items`, yielding ``(record, path prefix,
    value)`` per top-level item.
    """

    # An empty list registers nothing: without it the base would inherit
    # TextReader's ".txt" and take plain text over.
    file_types: ClassVar[list[str] | None] = []
    format_name: ClassVar[str] = ""

    values: str = field(default="all")
    """``"all"`` keeps every scalar; ``"text"`` keeps strings only."""

    items_per_chunk: int = field(default=100)
    """Most top-level items in one chunk."""

    max_values: int = field(default=_DEFAULT_MAX_VALUES)
    """Most values (scalars and containers) read; more refuses."""

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """Larger files are refused. Default: 100 MB."""

    max_text_chars_per_byte: float | None = field(default=50)
    """
    Text a file may yield per byte (with a 1 Mi floor), as for archives:
    data is about as large as its text, so this only stops expansion (a
    YAML alias "bomb") early. ``None`` lifts it.
    """

    def __post_init__(self) -> None:
        """Validate the options."""
        super().__post_init__()
        name = type(self).__name__
        if self.values not in _VALUES:
            raise ValueError(
                f"{name}: values must be one of {_VALUES}; got {self.values!r}."
            )
        for option in ("items_per_chunk", "max_values"):
            value = getattr(self, option)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}: {option} must be an int > 0; got {value!r}.")
        self.coverage = ReadCoverage(unit="record")

    def _items(self, text: str) -> Iterator[tuple[int, str, Any]]:
        raise NotImplementedError

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield chunks of ``path: value`` lines, ``items_per_chunk`` items each.

        Yields
        ------
        dict
            ``text``, ``section_type``, ``source_type``
            (:attr:`SourceType.DATASET`), ``record_start``, ``record_end``,
            ``format`` and ``encoding``.

        Raises
        ------
        ValueError
            If the file is too large, binary, malformed, too deep or holds
            more than ``max_values`` values.
        """
        name = type(self).__name__
        self.coverage = ReadCoverage(unit="record")
        whole = next(iter(super().get_raw_chunks()), None)
        if whole is None:
            return
        text = whole["text"].lstrip("﻿")
        meta = {"format": self.format_name, "encoding": whole["encoding"]}
        budget = [self.max_values]
        cap = self._text_limit()[0]
        block: list[str] = []
        first: Any = None
        last: Any = None
        items = 0
        size = -1
        try:
            for record, prefix, value in self._items(text):
                wrote = False
                for line in _lines(value, prefix, self.values, budget):
                    if first is None:
                        first = record
                    last = record
                    wrote = True
                    block.append(line)
                    size += len(line) + 1
                    if cap is not None and size > cap:
                        # Flush inside an item too, so one huge item (an
                        # alias bomb) meets the text limit as it grows.
                        yield _chunk(block, first, last, meta)
                        block, first, size = [], None, -1
                if not wrote:
                    self.coverage.empty += 1
                    continue
                self.coverage.read += 1
                items += 1
                if items >= self.items_per_chunk and block:
                    yield _chunk(block, first, last, meta)
                    block, first, size, items = [], None, -1, 0
        except RecursionError as exc:
            raise ValueError(
                f"{name}: {self.file_name} nests too deeply to parse."
            ) from exc
        except ValueError as exc:
            if str(exc).startswith(name):
                raise
            raise ValueError(f"{name}: {self.file_name}: {exc}.") from exc
        if block:
            yield _chunk(block, first, last, meta)


def _chunk(
    block: list[str], first: Any, last: Any, meta: dict[str, Any]
) -> dict[str, Any]:
    return {
        "text": "\n".join(block),
        "section_type": SectionType.TEXT.value,
        "source_type": SourceType.DATASET.value,
        "record_start": first,
        "record_end": last,
        **meta,
    }


def _top_level(value: Any) -> Iterator[tuple[int, str, Any]]:
    """Split a parsed document into top-level items with their path prefix."""
    if isinstance(value, dict):
        for i, (key, inner) in enumerate(value.items()):
            yield i, _key_step(key).lstrip("."), inner
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            yield i, f"[{i}]", inner
    else:
        yield 0, "", value


@dataclass
class JSONReader(_TreeReader):
    r"""
    Read JSON (``.json``) and JSON Lines (``.jsonl``, ``.ndjson``).

    Every value becomes a line ``path: value`` (``users[0].name: Ada``);
    lines are grouped by top-level item (an array element or an object
    key, or one JSON Lines record), ``items_per_chunk`` items a chunk.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the file.
    values : {"all", "text"}, optional
        ``"all"`` (default) writes every scalar (numbers, booleans,
        ``null`` too); ``"text"`` writes strings only.
    items_per_chunk : int, optional
        Most top-level items in one chunk. Default: ``100``.
    on_error : {"refuse", "skip"}, optional
        A JSON Lines record that does not parse: ``"refuse"`` (default)
        raises ``ValueError`` naming the line; ``"skip"`` counts it in
        ``reader.coverage`` (``"malformed"``) and reads on.
    max_values : int, optional
        Most values read; more refuses. Default: 5 000 000.
    max_file_bytes : int, optional
        Larger files are refused. Default: 100 MB.

    Attributes
    ----------
    file_types : list of str
        Class variable. ``[".json", ".jsonl", ".ndjson"]``.
    coverage : ReadCoverage
        After a read: top-level items read, empty and skipped.

    Raises
    ------
    ValueError
        If the file is too large, binary, malformed or too deep.

    See Also
    --------
    YAMLReader, TOMLReader : The same output for YAML and TOML.

    Notes
    -----
    **User note:** A path step is ``.key`` for a plain key, ``["any
    key"]`` otherwise, ``[i]`` for an array index; the document itself is
    ``$``. Empty containers are written as ``{}`` or ``[]``; a line break
    in a string becomes a space. Each chunk carries ``record_start`` and
    ``record_end`` (top-level index, or JSON Lines line number), ``format``
    and ``encoding``. ``source_type`` is :attr:`SourceType.DATASET`.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = JSONReader(input_path=Path("data.json"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".json"
    file_types: ClassVar[list[str] | None] = [".json", ".jsonl", ".ndjson"]
    format_name: ClassVar[str] = "json"

    on_error: str = field(default="refuse")
    """JSON Lines: ``"refuse"`` a bad record, or ``"skip"`` and count it."""

    def __post_init__(self) -> None:
        """Validate the options."""
        super().__post_init__()
        if self.on_error not in ("refuse", "skip"):
            raise ValueError(
                f"JSONReader: on_error must be 'refuse' or 'skip'; got {self.on_error!r}."
            )

    def _items(self, text: str) -> Iterator[tuple[int, str, Any]]:
        if self.input_path.suffix.lower() in (".jsonl", ".ndjson"):
            for number, line in enumerate(text.split("\n"), start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    if self.on_error == "skip":
                        self.coverage.skip("malformed")
                        continue
                    raise ValueError(f"line {number} is not JSON ({exc.msg})") from exc
                yield number, "", value
            if self.coverage.skipped:
                logger.warning(
                    "JSONReader: %s: %s.", self.file_name, self.coverage.summary()
                )
            return
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"not JSON at line {exc.lineno} column {exc.colno} ({exc.msg})"
            ) from exc
        yield from _top_level(value)


@dataclass
class YAMLReader(_TreeReader):
    r"""
    Read YAML (``.yaml``, ``.yml``), every document of a stream.

    The output is :class:`JSONReader`'s (``path: value`` lines grouped by
    top-level item); a stream of several documents prefixes each path
    with ``[document]`` and numbers records by document.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the file.
    values : {"all", "text"}, optional
        As for :class:`JSONReader`.
    items_per_chunk : int, optional
        As for :class:`JSONReader`.
    unknown_tags : {"keep", "refuse"}, optional
        A tag the safe loader does not know: read its node as plain data
        (default), or raise ``ValueError``.
    max_values : int, optional
        Most values read, counted as written: an alias is counted each
        time it is reached, so an alias "bomb" stops here. Default:
        5 000 000.
    max_file_bytes : int, optional
        Larger files are refused. Default: 100 MB.

    Attributes
    ----------
    file_types : list of str
        Class variable. ``[".yaml", ".yml"]``.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    ValueError
        If the file is too large, binary, malformed or holds too many
        values.

    Notes
    -----
    **Security:** only PyYAML's safe loader is used (``CSafeLoader`` when
    compiled): a tag never builds a Python object or runs code. With
    ``unknown_tags="keep"`` (default) a tag it does not know
    (``!ENV``, ``!relative``, ``!!python/name:...``, as in MkDocs and
    other tool configs) is read as plain data, a scalar written as
    ``<tag> <value>``; ``"refuse"`` raises ``ValueError`` instead. A node
    that contains itself through an anchor is written as ``<cycle>``.
    Dates become ISO 8601 text; ``!!binary`` values become
    ``<n bytes>``.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = YAMLReader(input_path=Path("config.yaml"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".yaml"
    file_types: ClassVar[list[str] | None] = [".yaml", ".yml"]
    format_name: ClassVar[str] = "yaml"

    unknown_tags: str = field(default="keep")
    """``"keep"`` reads an unknown tag's node as plain data; ``"refuse"`` raises."""

    def __post_init__(self) -> None:
        """Validate the options."""
        super().__post_init__()
        if self.unknown_tags not in ("keep", "refuse"):
            raise ValueError(
                "YAMLReader: unknown_tags must be 'keep' or 'refuse';"
                f" got {self.unknown_tags!r}."
            )

    def _items(self, text: str) -> Iterator[tuple[int, str, Any]]:
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAMLReader. Install it with:\n"
                "  pip install pyyaml"
            ) from exc
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        if self.unknown_tags == "keep":
            loader = _tag_keeping_loader(yaml, loader)
        try:
            documents = list(yaml.load_all(text, Loader=loader))  # noqa: S506
        except yaml.YAMLError as exc:
            raise ValueError(f"not YAML ({' '.join(str(exc).split())})") from exc
        if len(documents) == 1:
            yield from _top_level(documents[0])
            return
        for index, document in enumerate(documents):
            if document is not None:
                yield index, f"[{index}]", document


def _tag_keeping_loader(yaml: Any, base: type) -> type:
    """
    Return a subclass of the safe loader *base* that reads unknown tags.

    A node with a tag the safe loader does not know (``!ENV``,
    ``!relative``, ``!!python/name:x``) becomes plain data: a scalar is
    written as ``"<tag> <value>"`` so the tag stays visible, a sequence a
    list, a mapping a dict. Nothing is imported, built or run: only the
    safe constructors and these three plain ones exist.
    """

    def plain(loader: Any, suffix: str, node: Any) -> Any:
        if isinstance(node, yaml.ScalarNode):
            value = loader.construct_scalar(node)
            return f"{node.tag} {value}".strip()
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_mapping(node, deep=True)

    loader = type("_TagKeepingLoader", (base,), {})
    loader.add_multi_constructor("", plain)
    return loader


@dataclass
class TOMLReader(_TreeReader):
    r"""
    Read TOML (``.toml``).

    The output is :class:`JSONReader`'s: ``path: value`` lines grouped by
    top-level table or key. Dates and times become ISO 8601 text.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the file.
    values : {"all", "text"}, optional
        As for :class:`JSONReader`.
    items_per_chunk : int, optional
        As for :class:`JSONReader`.
    max_values : int, optional
        As for :class:`JSONReader`.
    max_file_bytes : int, optional
        Larger files are refused. Default: 100 MB.

    Attributes
    ----------
    file_types : list of str
        Class variable. ``[".toml"]``.

    Raises
    ------
    ImportError
        On Python before 3.11 when ``tomli`` is not installed.
    ValueError
        If the file is too large, binary or not TOML.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = TOMLReader(input_path=Path("pyproject.toml"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".toml"
    file_types: ClassVar[list[str] | None] = [".toml"]
    format_name: ClassVar[str] = "toml"

    def _items(self, text: str) -> Iterator[tuple[int, str, Any]]:
        if sys.version_info >= (3, 11):
            import tomllib as toml  # noqa: PLC0415
        else:
            try:
                import tomli as toml  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "tomli is required for TOMLReader before Python 3.11."
                    " Install it with:\n  pip install tomli"
                ) from exc
        try:
            value = toml.loads(text)
        except toml.TOMLDecodeError as exc:
            raise ValueError(f"not TOML ({exc})") from exc
        yield from _top_level(value)
