# scikitplot/corpus/_serialize.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
One serialisation contract for corpus reports and records.

Four text formats share one data model, a tree of ``dict`` (string keys),
``list``, ``str``, ``int``, ``float``, ``bool`` and ``None``:

``json``
    Strict RFC 8259 JSON. ``NaN`` is written as ``null`` (an undefined value);
    an infinity cannot be represented and raises. Reading refuses the
    non-standard ``NaN``/``Infinity`` tokens that :func:`json.dumps` emits by
    default.
``jsonl``
    One compact JSON object per line (JSON Lines), for row-oriented data such
    as one record per query or per document.
``yaml``
    Optional, through PyYAML's safe dumper and loader.
``toml``
    Optional, through ``tomli-w`` (write) and :mod:`tomllib` or ``tomli``
    (read). TOML has no null: a key whose value is ``None`` is omitted, and
    ``None`` inside an array raises.

Notes
-----
**User note.** :func:`write` publishes atomically: a reader never sees a
half-written file, even if the process is interrupted. Every format round-trips
every string, including the line-break characters U+0085, U+2028 and U+2029,
which are written escaped. YAML input with aliases is refused: a report never
contains them, and an untrusted file could use them to expand a small input
into gigabytes. JSON and each JSON Lines record are limited to 128 nested
objects/arrays regardless of the Python runtime's own decoder limit.

**Developer note.** :func:`to_builtin` is the only place where Python and
NumPy values become the data model. Sets are refused rather than ordered by
a guess, and mapping keys must be strings, so a JSON round trip cannot turn
``{1: ...}`` into ``{"1": ...}`` unnoticed.
"""

from __future__ import annotations

import enum
import json
import math
import numbers
import pathlib
from typing import Any, Mapping, Sequence

import numpy as np

from ._atomic import atomic_write_bytes
from ._json_safety import check_json_nesting

__all__: list[str] = []

#: Formats this module reads and writes.
FORMATS: tuple[str, ...] = ("json", "jsonl", "yaml", "toml")

#: File suffix to format. A suffix outside this table must be named explicitly.
SUFFIX_FORMATS: Mapping[str, str] = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

_ENCODING = "utf-8"

#: Line breaks of YAML and of :meth:`str.splitlines` beyond ``\n`` and ``\r``
#: that a writer could leave raw inside a string: NEL, LINE SEPARATOR and
#: PARAGRAPH SEPARATOR. Raw, a YAML reader turns them into spaces and a
#: line-splitting JSON Lines reader breaks a row in two, so they are written
#: escaped.
_LINE_BREAKS: tuple[str, ...] = ("\x85", "\u2028", "\u2029")


def resolve_format(path: Any = None, format: str | None = None) -> str:  # noqa: A002
    """
    Return the format named explicitly, or the one the path suffix maps to.

    Parameters
    ----------
    path : str or os.PathLike or None, optional
        File whose suffix decides the format when ``format`` is ``None``.
    format : {"json", "jsonl", "yaml", "toml"} or None, optional
        Explicit format. Takes precedence over the suffix.

    Returns
    -------
    str
        One of :data:`FORMATS`.

    Raises
    ------
    ValueError
        If ``format`` is unknown, or if it is ``None`` and the suffix of
        ``path`` is not in :data:`SUFFIX_FORMATS`.

    Examples
    --------
    >>> resolve_format("report.yml")
    'yaml'
    >>> resolve_format("report.txt", "json")
    'json'
    """
    if format is not None:
        if format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {format!r}.")
        return format
    if path is None:
        raise ValueError("Pass a format or a path whose suffix names one.")
    suffix = pathlib.Path(path).suffix.lower()
    try:
        return SUFFIX_FORMATS[suffix]
    except KeyError:
        raise ValueError(
            f"Cannot infer a format from the suffix {suffix!r} of {str(path)!r}; "
            f"pass format= one of {FORMATS}. Known suffixes: "
            f"{sorted(SUFFIX_FORMATS)}."
        ) from None


def to_builtin(value: Any) -> Any:
    """
    Convert ``value`` to the serialisation data model.

    Parameters
    ----------
    value : object
        ``None``, ``bool``, ``str``, a real number (Python or NumPy), an
        :class:`enum.Enum` (its value is used), a mapping with string keys,
        a ``list`` or ``tuple``, a NumPy array, or any object with a
        ``to_dict()`` method.

    Returns
    -------
    object
        A tree of ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and
        ``None``. Non-finite floats are kept; each format decides how to write
        them.

    Raises
    ------
    TypeError
        For a mapping key that is not a ``str``, for a ``set`` or
        ``frozenset`` (its order is not defined), and for any other type.

    Examples
    --------
    >>> import numpy as np
    >>> to_builtin({"a": np.float32(0.5), "b": (1, np.int64(2))})
    {'a': 0.5, 'b': [1, 2]}
    """
    value = _unwrap(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Real):
        return int(value) if isinstance(value, numbers.Integral) else float(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Mapping keys must be str to serialise; got "
                    f"{type(key).__name__} key {key!r}."
                )
            out[key] = to_builtin(item)
        return out
    if isinstance(value, (set, frozenset)):
        raise TypeError(
            "A set has no defined order; convert it to a sorted list before "
            "serialising."
        )
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    raise TypeError(f"Cannot serialise a value of type {type(value).__name__!r}.")


def _unwrap(value: Any) -> Any:
    """
    Reduce a value to the data model's types before the type switch.

    An enum becomes its value, a ``str`` subclass (``numpy.str_``) a plain
    ``str``, a ``numpy.bool_`` a ``bool``, an array its list, and any other
    object with ``to_dict()`` that mapping.
    """
    if isinstance(value, enum.Enum):
        return _unwrap(value.value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    plain = (
        Mapping,
        list,
        tuple,
        set,
        frozenset,
        str,
        bytes,
        numbers.Number,
        type(None),
    )
    if not isinstance(value, plain):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_dict()
    return value


def _json_ready(tree: Any) -> Any:
    """Replace ``NaN`` by ``None``; refuse infinities (JSON cannot hold them)."""
    if isinstance(tree, float):
        if math.isnan(tree):
            return None
        if math.isinf(tree):
            raise ValueError(
                "JSON cannot represent an infinite number; use the yaml or "
                "toml format, or replace it before serialising."
            )
        return tree
    if isinstance(tree, dict):
        return {key: _json_ready(item) for key, item in tree.items()}
    if isinstance(tree, list):
        return [_json_ready(item) for item in tree]
    return tree


def _toml_ready(tree: Any) -> Any:
    """Drop ``None`` values from tables; refuse ``None`` inside arrays."""
    if isinstance(tree, dict):
        return {
            key: _toml_ready(item) for key, item in tree.items() if item is not None
        }
    if isinstance(tree, list):
        if any(item is None for item in tree):
            raise ValueError(
                "TOML has no null value, and an array cannot skip an element; "
                "use the json or yaml format for arrays containing None."
            )
        return [_toml_ready(item) for item in tree]
    return tree


def _escape_line_breaks(text: str) -> str:
    """Escape :data:`_LINE_BREAKS` in JSON text (they only occur inside strings)."""
    for char in _LINE_BREAKS:
        text = text.replace(char, f"\\u{ord(char):04x}")
    return text


def _parse_float(text: str) -> float:
    """``json.loads`` hook: refuse a number that overflows to infinity."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(
            f"The number {text} overflows a float; strict JSON keeps numbers finite.",
        )
    return value


def _reject_constant(token: str) -> Any:
    """``json.loads`` hook: refuse the non-standard ``NaN``/``Infinity`` tokens."""
    raise ValueError(
        f"Non-standard JSON token {token!r}; strict JSON writes an undefined "
        "number as null."
    )


def _yaml() -> Any:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The yaml format requires PyYAML: pip install pyyaml"
        ) from exc
    return yaml


def _yaml_dumper(yaml: Any) -> type:
    """Return a safe dumper that double-quotes strings holding :data:`_LINE_BREAKS`."""

    class Dumper(yaml.SafeDumper):
        pass

    def represent_str(dumper: Any, data: str) -> Any:
        style = '"' if any(char in data for char in _LINE_BREAKS) else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

    Dumper.add_representer(str, represent_str)
    return Dumper


def _yaml_loader(yaml: Any) -> type:
    """Return a safe loader that refuses aliases (reports never contain them)."""

    class Loader(yaml.SafeLoader):
        def compose_node(self, parent: Any, index: Any) -> Any:
            if self.check_event(yaml.events.AliasEvent):
                raise yaml.composer.ComposerError(
                    None,
                    None,
                    "aliases are refused: a report never contains them, and an "
                    "untrusted file can use them to expand into gigabytes",
                    self.peek_event().start_mark,
                )
            return super().compose_node(parent, index)

    return Loader


def _toml_writer() -> Any:
    try:
        import tomli_w  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Writing the toml format requires tomli-w: pip install tomli-w",
        ) from exc
    return tomli_w


def _toml_reader() -> Any:
    try:
        import tomllib  # noqa: PLC0415  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Reading the toml format on Python < 3.11 requires tomli: "
                "pip install tomli"
            ) from exc
    return tomllib


def dumps(data: Any, format: str, *, indent: int | None = 2) -> str:  # noqa: A002
    r"""
    Serialise ``data`` to text in ``format``.

    Parameters
    ----------
    data : object
        Anything :func:`to_builtin` accepts. ``jsonl`` needs a sequence of
        mappings; ``toml`` needs a mapping.
    format : {"json", "jsonl", "yaml", "toml"}
        Output format.
    indent : int or None, optional
        JSON indentation; ``None`` writes compact JSON. Other formats ignore it.

    Returns
    -------
    str
        The serialised text, ending with a newline (empty for an empty
        ``jsonl`` sequence).

    Raises
    ------
    TypeError
        If ``data`` (or its shape for ``jsonl``/``toml``) is not serialisable.
    ValueError
        If ``format`` is unknown, or a value cannot be represented in it.
    ImportError
        If the optional library a format needs is not installed.

    See Also
    --------
    loads : The inverse.

    Examples
    --------
    >>> dumps({"recall@10": float("nan"), "n": 2}, "json", indent=None)
    '{"recall@10": null, "n": 2}\n'
    >>> dumps([{"q": 1}, {"q": 2}], "jsonl")
    '{"q":1}\n{"q":2}\n'
    """
    fmt = resolve_format(format=format)
    try:
        return _dumps(to_builtin(data), fmt, indent)
    except RecursionError as exc:
        raise ValueError("The data is nested too deeply to serialise.") from exc


def _dumps(tree: Any, fmt: str, indent: int | None) -> str:
    if fmt == "json":
        text = json.dumps(
            _json_ready(tree), ensure_ascii=False, allow_nan=False, indent=indent
        )
        return _escape_line_breaks(text) + "\n"
    if fmt == "jsonl":
        if not isinstance(tree, list) or not all(isinstance(row, dict) for row in tree):
            raise TypeError(
                "The jsonl format needs a sequence of mappings, one per line.",
            )
        return "".join(
            _escape_line_breaks(
                json.dumps(
                    _json_ready(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            + "\n"
            for row in tree
        )
    if fmt == "yaml":
        yaml = _yaml()
        return yaml.dump(
            tree,
            Dumper=_yaml_dumper(yaml),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    # toml
    if not isinstance(tree, dict):
        raise TypeError("The toml format needs a mapping at the top level.")
    return _toml_writer().dumps(_toml_ready(tree))


def loads(text: str, format: str) -> Any:  # noqa: A002
    r"""
    Parse ``text`` written in ``format``.

    Parameters
    ----------
    text : str
        Serialised text.
    format : {"json", "jsonl", "yaml", "toml"}
        Input format.

    Returns
    -------
    object
        The data tree. ``jsonl`` returns a list with one item per line.

    Raises
    ------
    TypeError
        If ``text`` is not a ``str``.
    ValueError
        For malformed input, including a ``NaN`` or ``Infinity`` JSON token, a
        number that overflows a float, a blank line inside JSON Lines, a YAML
        alias, and nesting too deep to parse.
    ImportError
        If the optional library a format needs is not installed.

    Examples
    --------
    >>> loads('{"a": null}', "json")
    {'a': None}
    >>> loads('{"q":1}\n{"q":2}\n', "jsonl")
    [{'q': 1}, {'q': 2}]
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__!r}.")
    fmt = resolve_format(format=format)
    try:
        return _loads(text, fmt)
    except RecursionError as exc:
        raise ValueError("The input is nested too deeply to parse.") from exc


def _loads(text: str, fmt: str) -> Any:
    if fmt == "json":
        check_json_nesting(text)
        return json.loads(
            text, parse_constant=_reject_constant, parse_float=_parse_float
        )
    if fmt == "jsonl":
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        rows = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                raise ValueError(f"JSON Lines input has a blank line at line {number}.")
            try:
                check_json_nesting(line)
                rows.append(
                    json.loads(
                        line,
                        parse_constant=_reject_constant,
                        parse_float=_parse_float,
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Invalid JSON at line {number}: {exc}") from exc
        return rows
    if fmt == "yaml":
        yaml = _yaml()
        try:
            # a SafeLoader subclass
            return yaml.load(text, Loader=_yaml_loader(yaml))  # noqa: S506
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc
    return _toml_reader().loads(text)


def write(
    path: Any,
    data: Any,
    format: str | None = None,  # noqa: A002
    *,
    indent: int | None = 2,
) -> pathlib.Path:
    """
    Serialise ``data`` and publish it atomically at ``path``.

    Parameters
    ----------
    path : str or os.PathLike
        Destination file. Parent directories are created.
    data : object
        Anything :func:`dumps` accepts for the format.
    format : {"json", "jsonl", "yaml", "toml"} or None, optional
        Explicit format; ``None`` infers it from the suffix of ``path``.
    indent : int or None, optional
        JSON indentation.

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    ValueError, TypeError, ImportError
        As :func:`dumps` and :func:`resolve_format`.
    OSError
        If the file cannot be written.
    """
    fmt = resolve_format(path, format)
    text = dumps(data, fmt, indent=indent)
    return atomic_write_bytes(pathlib.Path(path), text.encode(_ENCODING))


def read(path: Any, format: str | None = None) -> Any:  # noqa: A002
    """
    Read and parse the file at ``path``.

    Parameters
    ----------
    path : str or os.PathLike
        Source file, UTF-8 encoded.
    format : {"json", "jsonl", "yaml", "toml"} or None, optional
        Explicit format; ``None`` infers it from the suffix of ``path``.

    Returns
    -------
    object
        The data tree.

    Raises
    ------
    ValueError, ImportError
        As :func:`loads` and :func:`resolve_format`.
    OSError
        If the file cannot be read.
    """
    fmt = resolve_format(path, format)
    return loads(pathlib.Path(path).read_text(encoding=_ENCODING), fmt)


def rows_of(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """
    Return ``records`` as a list of serialisable rows.

    Parameters
    ----------
    records : sequence of mapping
        Row-oriented data.

    Returns
    -------
    list of dict

    Raises
    ------
    TypeError
        If an item is not a mapping, or is not serialisable.
    """
    rows = to_builtin(list(records))
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError("Every record must be a mapping.")
    return rows
