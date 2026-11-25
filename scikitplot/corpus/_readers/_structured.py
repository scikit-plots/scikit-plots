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
from typing import (  # noqa: F401
    Any,
    ClassVar,
    Generator,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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


# ---------------------------------------------------------------------------
# Paths: one small syntax for selecting values
# ---------------------------------------------------------------------------

#: A path step: ("key", name), ("index", n), ("any_key",) or ("any_index",).
Step = Tuple[Any, ...]

_STEP_RE = re.compile(
    r"""
    \.(?P<dot>[A-Za-z_][A-Za-z0-9_-]*|\*)   # .name or .*
    | \[(?P<index>\d+|\*)\]               # [0] or [*]
    | \[(?P<quoted>"(?:[^"\\]|\\.)*")\]   # ["any key"]
    """,
    re.VERBOSE,
)


def _parse_path(expr: str) -> tuple[Step, ...]:
    r"""
    Parse a path expression into steps.

    The syntax is the one the path lines are written in: ``.key`` (a plain
    key; the first may omit the dot), ``["any key"]`` (a JSON string),
    ``[0]`` (an array index), ``[*]`` (every element) and ``.*`` or ``*``
    (every value of an object). ``$`` alone, or as a prefix, is the root.

    Parameters
    ----------
    expr : str
        The expression.

    Returns
    -------
    tuple
        The steps; ``()`` for the root.

    Raises
    ------
    ValueError
        If *expr* is not a valid path.

    Examples
    --------
    >>> _parse_path("users[*].name")
    (('key', 'users'), ('any_index',), ('key', 'name'))
    >>> _parse_path('$["odd key"].*')
    (('key', 'odd key'), ('any_key',))
    """
    if not isinstance(expr, str):
        raise ValueError(f"a path must be a str; got {expr!r}")  # noqa: TRY004
    rest = expr.strip()
    if rest.startswith("$"):
        rest = rest[1:]
    if rest and not rest.startswith((".", "[")):
        rest = "." + rest
    steps: list[Step] = []
    at = 0
    while at < len(rest):
        match = _STEP_RE.match(rest, at)
        if match is None:
            raise ValueError(f"bad path {expr!r} at character {at + 1}")
        if match.group("dot") is not None:
            name = match.group("dot")
            steps.append(("any_key",) if name == "*" else ("key", name))
        elif match.group("index") is not None:
            index = match.group("index")
            steps.append(("any_index",) if index == "*" else ("index", int(index)))
        else:
            steps.append(("key", json.loads(match.group("quoted"))))
        at = match.end()
    return tuple(steps)


def _step_matches(pattern: Step, step: Step) -> bool:
    if pattern[0] == "any_key":
        return step[0] == "key"
    if pattern[0] == "any_index":
        return step[0] == "index"
    return pattern == step


def _prefix_matches(pattern: tuple[Step, ...], head: tuple[Step, ...]) -> bool:
    """Whether *head* starts with *pattern* (the node is inside its subtree)."""
    return len(head) >= len(pattern) and all(
        _step_matches(p, s) for p, s in zip(pattern, head)
    )


def _select(value: Any, steps: tuple[Step, ...]) -> list[Any]:
    """
    Return the values *steps* reach in *value*, in document order.

    A missing key or index reaches nothing; wildcards fan out.

    Examples
    --------
    >>> _select({"a": [{"b": 1}, {"b": 2}, {}]}, _parse_path("a[*].b"))
    [1, 2]
    """
    current = [value]
    for step in steps:
        reached: list[Any] = []
        kind = step[0]
        for node in current:
            if kind == "key" and isinstance(node, dict):
                key = step[1]
                if key in node:
                    reached.append(node[key])
                else:  # YAML and TOML keys may be non-strings
                    reached.extend(v for k, v in node.items() if str(k) == key)
            elif kind == "any_key" and isinstance(node, dict):
                reached.extend(node.values())
            elif kind == "index" and isinstance(node, list) and step[1] < len(node):
                reached.append(node[step[1]])
            elif kind == "any_index" and isinstance(node, list):
                reached.extend(node)
        current = reached
    return current


class _Filter:
    """Which nodes ``include`` and ``exclude`` path patterns keep."""

    def __init__(
        self,
        include: tuple[tuple[Step, ...], ...],
        exclude: tuple[tuple[Step, ...], ...],
    ) -> None:
        self.include = include
        self.exclude = exclude
        self.depth = max((len(p) for p in include + exclude), default=0)

    def __bool__(self) -> bool:
        return bool(self.include or self.exclude)

    def decide(self, head: tuple[Step, ...]) -> str:
        """Return ``"keep"``, ``"descend"`` (an ancestor of an include) or ``"drop"``."""
        if any(_prefix_matches(p, head) for p in self.exclude):
            return "drop"
        if not self.include or any(_prefix_matches(p, head) for p in self.include):
            return "keep"
        if any(
            len(head) < len(p) and _prefix_matches(p[: len(head)], head)
            for p in self.include
        ):
            return "descend"
        return "drop"


_NO_FILTER = _Filter((), ())


def _lines(  # noqa: C901, PLR0912
    root: Any,
    prefix: str,
    values: str,
    budget: list[int],
    *,
    head: tuple[Step, ...] = (),
    keep: _Filter = _NO_FILTER,
) -> Iterator[str]:
    """
    Yield ``path: value`` lines for every value under *root*, in order.

    Iterative (deep nesting cannot exhaust the stack). A container that
    contains itself (a YAML anchor) is written once as ``<cycle>``.
    *budget* is a one-item list of values still allowed; reaching zero
    raises ``ValueError``. With ``values="text"`` only strings are kept.
    *head* is the path of *root* as steps; *keep* applies ``include`` and
    ``exclude`` patterns (only the first steps a pattern can reach are
    tracked, so the cost stays linear).
    """
    # Iterative walk; a marker entry leaves a container, so the set of open
    # containers is shared, not copied per node (linear in the tree).
    leave = object()
    depth = keep.depth
    open_ids: set[int] = set()
    stack: list[tuple[Any, Any, Any]] = [(prefix, head[:depth], root)]
    while stack:
        path, steps, node = stack.pop()
        if path is leave:
            open_ids.discard(node)
            continue
        verdict = keep.decide(steps) if keep else "keep"
        if verdict == "drop":
            continue
        text = _scalar(node)
        if text is not None:
            if verdict != "keep":
                continue
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
            if verdict == "keep":
                yield f"{path or '$'}: <cycle>"
            continue
        budget[0] -= 1  # a container costs one value, so {} and [] count
        if budget[0] < 0:
            raise ValueError("more values than max_values")
        short = len(steps) < depth
        if isinstance(node, dict):
            items = [
                (
                    path + _key_step(k),
                    (*steps, ("key", str(k))) if short else steps,
                    v,
                )
                for k, v in node.items()
            ]
        else:
            items = [
                (f"{path}[{i}]", (*steps, ("index", i)) if short else steps, v)
                for i, v in enumerate(node)
            ]
        if not items:
            if values == "all" and verdict == "keep":
                yield f"{path or '$'}: {'{}' if isinstance(node, dict) else '[]'}"
            continue
        open_ids.add(ident)
        stack.append((leave, None, ident))
        stack.extend(reversed(items))


#: Chunk keys a ``metadata_fields`` name may not take.
_RESERVED = frozenset(
    {
        "text",
        "raw_text",
        "section_type",
        "source_type",
        "format",
        "encoding",
        "record_start",
        "record_end",
        "record_index",
    }
)
_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")


def _paths(option: str, owner: str, value: Any) -> tuple[tuple[Step, ...], ...]:
    """Parse a sequence of path expressions, naming *option* on error."""
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{owner}: {option} must be a list of paths; got {value!r}.")
    try:
        return tuple(_parse_path(v) for v in value)
    except ValueError as exc:
        raise ValueError(f"{owner}: {option}: {exc}.") from exc


def _as_text(value: Any, separator: str) -> str:
    """Text of a selected value: a scalar's text, a container as compact JSON."""
    text = _scalar(value)
    if text is not None:
        return text
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


@dataclass
class _TreeReader(TextReader):
    """
    Shared base: decode, parse into documents, write text in one of two modes.

    **Paths mode** (default): every value as a ``path: value`` line, grouped
    by top-level item. **Records mode** (any of ``record_path``,
    ``text_fields``, ``template``): one chunk per record, its text chosen
    by fields or a template, chosen values carried as metadata.

    Subclasses implement :meth:`_documents`, yielding ``(number, prefix,
    head, value, split)`` per parsed document: *split* says whether the
    document's top level holds the items (JSON, TOML) or the document is
    one item (a JSON Lines record, one of several YAML documents).
    """

    # An empty list registers nothing: without it the base would inherit
    # TextReader's ".txt" and take plain text over.
    file_types: ClassVar[list[str] | None] = []
    format_name: ClassVar[str] = ""

    values: str = field(default="all")
    """``"all"`` keeps every scalar; ``"text"`` keeps strings only."""

    items_per_chunk: int = field(default=100)
    """Paths mode: most top-level items in one chunk."""

    include: Sequence[str] | None = field(default=None)
    """Path patterns whose subtrees are written; ``None`` writes all."""

    exclude: Sequence[str] | None = field(default=None)
    """Path patterns whose subtrees are left out."""

    record_path: str | None = field(default=None)
    """Records mode: where the records are in each document (``data[*]``)."""

    text_fields: Sequence[str] | None = field(default=None)
    """Records mode: paths whose values make the chunk text."""

    template: str | None = field(default=None)
    """Records mode: text with ``{path}`` placeholders; ``{{`` ``}}`` escape."""

    metadata_fields: Sequence[str] | Mapping[str, str] | None = field(default=None)
    """Records mode: paths carried as chunk metadata (a list, or name to path)."""

    field_separator: str = field(default="\n\n")
    """Records mode: what joins the values of ``text_fields``."""

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

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate the options and parse every path once."""
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
        self._filter = _Filter(
            _paths("include", name, self.include), _paths("exclude", name, self.exclude)
        )
        if self.text_fields is not None and self.template is not None:
            raise ValueError(f"{name}: give text_fields or template, not both.")
        if self.text_fields is not None and not self.text_fields:
            raise ValueError(f"{name}: text_fields must not be empty.")
        self._text_fields = _paths("text_fields", name, self.text_fields)
        self._record_steps: tuple[Step, ...] | None = None
        if self.record_path is not None:
            [self._record_steps] = _paths("record_path", name, [self.record_path])
        self._template: list[str | tuple[Step, ...]] = []
        if self.template is not None:
            if not isinstance(self.template, str):
                raise TypeError(
                    f"{name}: template must be a str; got {self.template!r}."
                )
            self._template = self._parse_template(self.template, name)
        fields = self.metadata_fields
        if fields is None:
            pairs: list[tuple[str, str]] = []
        elif isinstance(fields, Mapping):
            pairs = list(fields.items())
        elif isinstance(fields, (list, tuple)):
            pairs = [(f, f) for f in fields]
        else:
            raise TypeError(
                f"{name}: metadata_fields must be a list of paths or a mapping of"
                f" name to path; got {fields!r}."
            )
        self._metadata: list[tuple[str, tuple[Step, ...]]] = []
        for key, path in pairs:
            if not isinstance(key, str) or not key or key in _RESERVED:
                raise ValueError(
                    f"{name}: metadata name {key!r} is empty or reserved"
                    f" ({', '.join(sorted(_RESERVED))}); map it to another name."
                )
            [steps] = _paths("metadata_fields", name, [path])
            self._metadata.append((key, steps))
        if not isinstance(self.field_separator, str):
            raise TypeError(f"{name}: field_separator must be a str.")
        self._records = bool(
            self.record_path is not None
            or self.text_fields is not None
            or self.template is not None
            or self._metadata
        )
        self.coverage = ReadCoverage(unit="record")

    @staticmethod
    def _parse_template(template: str, name: str) -> list[str | tuple[Step, ...]]:
        """Split a template into literal text and placeholder paths."""
        parts: list[str | tuple[Step, ...]] = []
        at = 0
        for match in _PLACEHOLDER.finditer(template):
            parts.append(template[at : match.start()])
            token = match.group(0)
            if token in ("{{", "}}"):
                parts.append(token[0])
            else:
                try:
                    parts.append(_parse_path(match.group(1)))
                except ValueError as exc:
                    raise ValueError(f"{name}: template: {exc}.") from exc
            at = match.end()
        tail = template[at:]
        if "{" in tail or "}" in tail:
            raise ValueError(f"{name}: template has an unmatched brace: {tail!r}.")
        parts.append(tail)
        return [p for p in parts if p != ""]

    def _documents(
        self, text: str
    ) -> Iterator[tuple[int, str, tuple[Step, ...], Any, bool]]:
        raise NotImplementedError

    def _items(self, text: str) -> Iterator[tuple[Any, str, tuple[Step, ...], Any]]:
        """Yield the top-level items of every document (paths mode)."""
        for number, prefix, head, value, split in self._documents(text):
            if split:
                yield from _top_level(value)
            else:
                yield number, prefix, head, value

    def _record_text(self, record: Any, budget: list[int]) -> str:
        """Return the chunk text of one record (records mode)."""
        sep = self.field_separator
        if self._template:
            out: list[str] = []
            for part in self._template:
                if isinstance(part, str):
                    out.append(part)
                else:
                    found = _select(record, part)
                    budget[0] -= len(found)
                    out.append("\n".join(_as_text(v, sep) for v in found))
            return "".join(out)
        if self._text_fields:
            found = [v for steps in self._text_fields for v in _select(record, steps)]
            budget[0] -= len(found)
            return sep.join(t for t in (_as_text(v, sep) for v in found) if t.strip())
        return "\n".join(_lines(record, "", self.values, budget, keep=self._filter))

    def _record_chunks(
        self, text: str, meta: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Yield one chunk per record (records mode)."""
        budget = [self.max_values]
        for number, _prefix, _head, value, split in self._documents(text):
            if self._record_steps is not None:
                records = _select(value, self._record_steps)
            elif split and isinstance(value, list):
                records = value
            else:
                records = [value]
            for index, record in enumerate(records):
                body = self._record_text(record, budget)
                if budget[0] < 0:
                    raise ValueError("more values than max_values")
                if not body.strip():
                    self.coverage.empty += 1
                    continue
                self.coverage.read += 1
                chunk = {
                    "text": body,
                    "section_type": SectionType.TEXT.value,
                    "source_type": SourceType.DATASET.value,
                    "record_start": number,
                    "record_end": number,
                    "record_index": index,
                    **meta,
                }
                for key, steps in self._metadata:
                    found = _select(record, steps)
                    if found:
                        chunk[key] = found[0] if len(found) == 1 else found
                yield chunk

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:  # noqa: C901
        """
        Yield path-line chunks, or one chunk per record in records mode.

        Yields
        ------
        dict
            ``text``, ``section_type``, ``source_type``
            (:attr:`SourceType.DATASET`), ``record_start``, ``record_end``,
            ``format`` and ``encoding``; records mode adds ``record_index``
            and the ``metadata_fields``.

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
        text = whole["text"].lstrip("\ufeff")
        meta = {"format": self.format_name, "encoding": whole["encoding"]}
        try:
            if self._records:
                yield from self._record_chunks(text, meta)
            else:
                yield from self._path_chunks(text, meta)
        except RecursionError as exc:
            raise ValueError(
                f"{name}: {self.file_name} nests too deeply to parse."
            ) from exc
        except ValueError as exc:
            if str(exc).startswith(name):
                raise
            raise ValueError(f"{name}: {self.file_name}: {exc}.") from exc

    def _path_chunks(self, text: str, meta: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield ``path: value`` lines, ``items_per_chunk`` items a chunk (paths mode)."""
        budget = [self.max_values]
        cap = self._text_limit()[0]
        block: list[str] = []
        first: Any = None
        last: Any = None
        items = 0
        size = -1
        for record, prefix, head, value in self._items(text):
            wrote = False
            for line in _lines(
                value, prefix, self.values, budget, head=head, keep=self._filter
            ):
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


def _top_level(value: Any) -> Iterator[tuple[int, str, tuple[Step, ...], Any]]:
    """Split a parsed document into top-level items with their path prefix."""
    if isinstance(value, dict):
        for i, (key, inner) in enumerate(value.items()):
            yield i, _key_step(key).lstrip("."), (("key", str(key)),), inner
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            yield i, f"[{i}]", (("index", i),), inner
    else:
        yield 0, "", (), value


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
    include, exclude : list of str or None, optional
        Path patterns (see Notes): only the subtrees ``include`` reaches are
        written, and those ``exclude`` reaches are left out. Default: all.
    record_path : str or None, optional
        Records mode: where the records are in each document, for example
        ``"data.items[*]"``. ``None``: a top-level array's elements, a JSON
        Lines line, or the whole document.
    text_fields : list of str or None, optional
        Records mode: paths whose values, joined by ``field_separator``,
        make the chunk text. ``None`` writes the record's path lines.
    template : str or None, optional
        Records mode, instead of ``text_fields``: text with ``{path}``
        placeholders, for example ``"{title}\n\n{body}"``; ``{{`` and
        ``}}`` are literal braces. A placeholder that reaches several values
        joins them with line breaks; one that reaches none is empty.
    metadata_fields : list of str or mapping of str to str, optional
        Records mode: values carried as chunk metadata, as a list of paths
        (the path names the key) or a mapping of key to path (for example
        ``{"url": "meta.link"}``, which also sets the document's ``url``).
        A path reaching one value stores it; several, a list; none, nothing.
    field_separator : str, optional
        What joins ``text_fields`` values. Default: a blank line.
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
    **Paths.** One syntax names values everywhere, the one path lines are
    written in: ``.key`` (the first may omit the dot), ``["any key"]``,
    ``[i]`` an array index, ``[*]`` every element, ``.*`` or ``*`` every
    value of an object; ``$`` is the document itself. Paths are parsed when
    the reader is made, so a bad one fails at once. A template only
    substitutes values; it never evaluates anything.

    **Two modes.** *Paths mode* (default) writes every value as a line and
    groups top-level items. Giving any of ``record_path``, ``text_fields``,
    ``template`` or ``metadata_fields`` switches to *records mode*: one
    chunk per record, its text and metadata chosen by you, for example a
    chat dataset (``text_fields=["messages[*].content"]``), an export of
    articles (``template="{title}\n\n{body}"``) or an API dump
    (``record_path="data.results[*]"``). Records mode adds
    ``record_index`` (the record's position in its document).

    **User note:** Empty containers are written as ``{}`` or ``[]``; a line break
    in a string becomes a space. Each chunk carries ``record_start`` and
    ``record_end`` (top-level index, or JSON Lines line number), ``format``
    and ``encoding``. ``source_type`` is :attr:`SourceType.DATASET`.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = JSONReader(input_path=Path("data.json"))
    >>> docs = list(reader.get_documents())

    A chat dataset in JSON Lines, one document per conversation:

    >>> reader = JSONReader(
    ...     input_path=Path("chats.jsonl"),
    ...     text_fields=["messages[*].content"],
    ...     metadata_fields={"conversation_id": "id", "url": "source.link"},
    ... )

    Articles nested in an API response, with a template:

    >>> reader = JSONReader(
    ...     input_path=Path("response.json"),
    ...     record_path="data.articles[*]",
    ...     template="{title}\n\n{body}",
    ...     metadata_fields=["author", "published"],
    ... )
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

    def _documents(
        self, text: str
    ) -> Iterator[tuple[int, str, tuple[Step, ...], Any, bool]]:
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
                yield number, "", (), value, False
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
        yield 0, "", (), value, True


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

    def _documents(
        self, text: str
    ) -> Iterator[tuple[int, str, tuple[Step, ...], Any, bool]]:
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
            yield 0, "", (), documents[0], True
            return
        for index, document in enumerate(documents):
            if document is not None:
                yield index, f"[{index}]", (("index", index),), document, False


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

    def _documents(
        self, text: str
    ) -> Iterator[tuple[int, str, tuple[Step, ...], Any, bool]]:
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
        yield 0, "", (), value, True
