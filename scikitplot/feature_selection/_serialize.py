# scikitplot/feature_selection/_serialize.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Text serialisation of policies and fitted artefacts: JSON, JSON Lines, YAML, TOML.

All four formats share one data model, a tree of ``dict`` with string keys,
``list``, ``str``, ``int``, ``float``, ``bool`` and ``None``:

``json``
    Strict RFC 8259 JSON, ASCII-escaped. Non-finite numbers are refused on
    write and on read.
``jsonl``
    One JSON object per line, for streams of records (lineage edges, drift
    rows). Lines are split on ``"\n"`` only, never on other line breaks.
``yaml``
    Optional, through PyYAML's safe dumper and loader. Aliases are refused on
    read: a document written here never contains them, and an untrusted file
    can use them to expand a few bytes into gigabytes.
``toml``
    Reading uses :mod:`tomllib` (Python 3.11+) or ``tomli``; writing needs the
    optional ``tomli-w``. TOML has no null: a ``None`` mapping value is
    omitted, and ``None`` inside an array is refused.

Notes
-----
**User note.** :func:`dumps` reads its own output back and compares it with
the input before returning, so a value that a format cannot hold (a lone
surrogate in TOML, ``None`` in a TOML array) fails at write time, naming the
format, instead of producing a file that cannot be loaded (Rule 107).
:func:`write` replaces the target atomically.

**Developer note.** :func:`to_builtin` is the only place where Python, NumPy
and enumeration values enter the data model. Sets are refused rather than
ordered by a guess, and mapping keys must be strings, so a round trip cannot
turn ``{1: ...}`` into ``{"1": ...}`` unnoticed.
"""

from __future__ import annotations

import enum
import json
import math
import os
import pathlib
import tempfile
from typing import Any, Mapping

import numpy as np

__all__: list[str] = []

#: Formats this module reads and writes.
FORMATS: tuple[str, ...] = ("json", "jsonl", "yaml", "toml")

#: File suffix to format; any other suffix needs an explicit ``format``.
SUFFIX_FORMATS: Mapping[str, str] = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

#: Deepest nesting accepted when reading; policies and artefacts use fewer than 10.
MAX_DEPTH = 64

_ENCODING = "utf-8"


def resolve_format(path: Any = None, format: str | None = None) -> str:  # noqa: A002
    """
    Return the serialisation format named explicitly or by the file suffix.

    Parameters
    ----------
    path : str or os.PathLike or None, default=None
        File whose suffix names the format when ``format`` is None.
    format : {"json", "jsonl", "yaml", "toml"} or None, default=None
        Explicit format; takes precedence over the suffix.

    Returns
    -------
    str
        One of :data:`FORMATS`.

    Raises
    ------
    ValueError
        If the format is unknown or cannot be derived from the suffix.
    """
    if format is not None:
        if format not in FORMATS:
            raise ValueError(f"format={format!r} is not one of {FORMATS}")
        return format
    if path is None:
        raise ValueError("give a format, or a path whose suffix names one")
    suffix = pathlib.Path(os.fspath(path)).suffix.lower()
    try:
        return SUFFIX_FORMATS[suffix]
    except KeyError:
        raise ValueError(
            f"cannot derive a format from suffix {suffix!r}; pass format= one of"
            f" {FORMATS}"
        ) from None


def to_builtin(  # ruff: ignore[too-many-return-statements]
    value: Any,
    *,
    _path: str = "$",
) -> Any:
    """
    Convert ``value`` to the shared data model.

    Parameters
    ----------
    value : object
        Mappings with string keys, lists, tuples, strings, booleans, integers,
        finite floats, ``None``, NumPy scalars and string enumerations.

    Returns
    -------
    object
        A tree of ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and
        ``None``.

    Raises
    ------
    TypeError
        For a value of any other type (sets, bytes, arbitrary objects) or a
        non-string mapping key; the message names the path inside the tree.
    ValueError
        For a non-finite float.
    """
    if value is None or isinstance(value, (bool, str)):
        return value if not isinstance(value, enum.Enum) else value.value
    if isinstance(value, enum.Enum):
        return to_builtin(value.value, _path=_path)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite number {number!r} at {_path}")
        return number
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, enum.Enum):
                key = key.value  # noqa: PLW2901
            if not isinstance(key, str):
                raise TypeError(
                    f"mapping keys must be strings, got {type(key).__name__}"
                    f" {key!r} at {_path}"
                )
            out[key] = to_builtin(item, _path=f"{_path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [to_builtin(item, _path=f"{_path}[{i}]") for i, item in enumerate(value)]
    raise TypeError(
        f"cannot serialise {type(value).__name__} at {_path}; convert it to a"
        " list, mapping or scalar first"
    )


def _check_depth(tree: Any, limit: int = MAX_DEPTH) -> None:
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            raise ValueError(f"document nesting is deeper than {limit} levels")
        if isinstance(node, dict):
            stack.extend((child, depth + 1) for child in node.values())
        elif isinstance(node, list):
            stack.extend((child, depth + 1) for child in node)


def _check_finite(tree: Any) -> None:
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError(f"non-finite number {node!r} in document")
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _toml_ready(tree: Any) -> Any:
    """Drop ``None`` mapping values; refuse ``None`` inside arrays."""
    if isinstance(tree, dict):
        return {k: _toml_ready(v) for k, v in tree.items() if v is not None}
    if isinstance(tree, list):
        if any(item is None for item in tree):
            raise ValueError("TOML cannot hold None inside an array")
        return [_toml_ready(item) for item in tree]
    return tree


def _yaml() -> Any:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The yaml format requires PyYAML: pip install pyyaml"
        ) from exc
    return yaml


def _yaml_loader(yaml: Any) -> type:
    """Return a safe loader that refuses aliases."""

    class Loader(yaml.SafeLoader):
        def compose_node(self, parent: Any, index: Any) -> Any:
            if self.check_event(yaml.events.AliasEvent):
                raise yaml.composer.ComposerError(
                    None,
                    None,
                    "YAML aliases are refused: documents written by"
                    " scikitplot.feature_selection never contain them",
                    self.peek_event().start_mark,
                )
            return super().compose_node(parent, index)

    return Loader


def _toml_reader() -> Any:
    try:
        import tomllib  # noqa: PLC0415

        return tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        try:
            import tomli  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Reading TOML on Python < 3.11 requires tomli: pip install tomli"
            ) from exc
        return tomli


def _toml_writer() -> Any:
    try:
        import tomli_w  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Writing the toml format requires tomli-w: pip install tomli-w"
        ) from exc
    return tomli_w


def _dumps(tree: Any, fmt: str, indent: int | None) -> str:
    if fmt == "json":
        return (
            json.dumps(tree, ensure_ascii=True, allow_nan=False, indent=indent) + "\n"
        )
    if fmt == "jsonl":
        if not isinstance(tree, list) or not all(isinstance(r, dict) for r in tree):
            raise TypeError("jsonl data must be a list of mappings (one per line)")
        return "".join(
            json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            + "\n"
            for row in tree
        )
    if fmt == "yaml":
        yaml = _yaml()
        return yaml.safe_dump(
            tree, allow_unicode=False, sort_keys=False, default_flow_style=False
        )
    if not isinstance(tree, dict):
        raise TypeError("toml data must be a mapping at the top level")
    return _toml_writer().dumps(_toml_ready(tree))


def _loads(text: str, fmt: str) -> Any:
    def refuse_constant(token: str) -> Any:
        raise ValueError(f"non-standard JSON number {token!r} is refused")

    try:
        if fmt == "json":
            tree = json.loads(text, parse_constant=refuse_constant)
        elif fmt == "jsonl":
            tree = [
                json.loads(line, parse_constant=refuse_constant)
                for line in text.split("\n")
                if line.strip()
            ]
            if not all(isinstance(row, dict) for row in tree):
                raise ValueError("every jsonl line must hold a JSON object")
        elif fmt == "yaml":
            yaml = _yaml()
            try:
                tree = yaml.load(text, Loader=_yaml_loader(yaml))  # noqa: S506
            except yaml.YAMLError as exc:
                raise ValueError(f"invalid YAML: {exc}") from None
        else:
            tree = _toml_reader().loads(text)
    except RecursionError:
        raise ValueError(f"{fmt} document nesting is too deep") from None
    _check_depth(tree)
    _check_finite(tree)
    return tree


def dumps(data: Any, format: str, *, indent: int | None = 2) -> str:  # noqa: A002
    """
    Serialise ``data`` to text and verify that it reads back equal.

    Parameters
    ----------
    data : object
        Any value accepted by :func:`to_builtin`; ``jsonl`` takes a list of
        mappings and ``toml`` a mapping.
    format : {"json", "jsonl", "yaml", "toml"}
        Output format.
    indent : int or None, default=2
        JSON indentation; ignored by the other formats.

    Returns
    -------
    str
        The serialised text.

    Raises
    ------
    TypeError, ValueError
        If ``data`` is outside the data model or the format cannot represent
        it exactly.
    ImportError
        If the optional library of the format is missing.
    """
    fmt = resolve_format(format=format)
    tree = to_builtin(data)
    text = _dumps(tree, fmt, indent)
    try:
        text.encode(_ENCODING)
    except UnicodeEncodeError:
        raise ValueError(
            f"the {fmt} text holds a lone surrogate, which a UTF-8 file cannot store;"
            " use json or yaml (they escape it)"
        ) from None
    expected = _toml_ready(tree) if fmt == "toml" else tree
    if _loads(text, fmt) != expected:
        raise ValueError(
            f"the {fmt} format cannot represent this data exactly (it would not"
            " read back equal); use json"
        )
    return text


def loads(text: str, format: str) -> Any:  # noqa: A002
    """
    Parse ``text`` written in ``format``.

    Parameters
    ----------
    text : str
        The serialised document.
    format : {"json", "jsonl", "yaml", "toml"}
        Input format.

    Returns
    -------
    object
        A tree of the shared data model (``jsonl`` gives a list of mappings).
        YAML and TOML dates are returned as :mod:`datetime` objects; the
        typed loaders of this package refuse them.

    Raises
    ------
    ValueError
        For malformed text, non-finite numbers, YAML aliases or nesting deeper
        than :data:`MAX_DEPTH`.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return _loads(text, resolve_format(format=format))


def write(
    data: Any,
    path: Any,
    format: str | None = None,
) -> pathlib.Path:  # noqa: A002
    """
    Write ``data`` to ``path`` atomically.

    Parameters
    ----------
    data : object
        See :func:`dumps`.
    path : str or os.PathLike
        Target file; its parent directory must exist.
    format : str or None, default=None
        Explicit format, else derived from the suffix.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    target = pathlib.Path(os.fspath(path))
    text = dumps(data, resolve_format(target, format))
    fd, tmp = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=_ENCODING, newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return target


def read(path: Any, format: str | None = None) -> Any:  # noqa: A002
    """
    Read a document written by :func:`write` (or by hand).

    Parameters
    ----------
    path : str or os.PathLike
        Source file.
    format : str or None, default=None
        Explicit format, else derived from the suffix.

    Returns
    -------
    object
        See :func:`loads`.
    """
    source = pathlib.Path(os.fspath(path))
    fmt = resolve_format(source, format)
    return loads(source.read_text(encoding=_ENCODING), fmt)
