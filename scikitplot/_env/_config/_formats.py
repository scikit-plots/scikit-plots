# scikitplot/_env/_config/_formats.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Reading configuration documents: TOML, JSON, YAML and JSON Lines batches.

=========  ==============================  ======================================
Format     Role                            Notes
=========  ==============================  ======================================
TOML       project files people edit       comments; stdlib parser (3.11+, else
                                           ``tomli``); the recommended format
JSON       machine interchange, APIs       strict; no comments; duplicate keys
                                           are refused
YAML       CI files, Conda users           needs PyYAML (imported only for
                                           YAML); duplicate keys and aliases
                                           are refused; dates are not strings
JSONL      batches of requests             one ``{"schema", "id", "request"}``
                                           object per line; each line is
                                           checked on its own
=========  ==============================  ======================================

Notes
-----
**User notes.** All three document formats produce the same mapping, so the
same request written in each has the same digest. ``pyproject.toml`` works
too: the request is its ``[tool.scikitplot.env]`` table.

**Developer notes.** The standard JSON and YAML parsers keep the last of two
equal keys; both are given a hook here that refuses the document instead, so
a merge accident cannot silently change a request.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

from .._core._errors import ConfigError

__all__: list[str] = []

#: File suffixes and the format each implies.
SUFFIXES = {
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}

#: Largest configuration document read, in bytes.
MAX_BYTES = 4 * 1024 * 1024


def format_of(path: Any, format: str | None = None) -> str:  # noqa: A002
    """
    Return the format named explicitly or by the suffix of ``path``.

    Parameters
    ----------
    path : str or os.PathLike
        Document path.
    format : str or None, default=None
        Explicit format.

    Returns
    -------
    str
        ``"toml"``, ``"json"``, ``"yaml"`` or ``"jsonl"``.

    Raises
    ------
    ConfigError
        If the format is unknown.
    """
    if format is not None:
        if format not in set(SUFFIXES.values()):
            raise ConfigError(
                f"format={format!r} is not one of toml, json, yaml, jsonl"
            )
        return format
    suffix = Path(os.fspath(path)).suffix.lower()
    if suffix not in SUFFIXES:
        raise ConfigError(
            f"cannot tell the format of {os.fspath(path)!r}",
            hint="use a .toml, .json, .yaml or .jsonl file, or pass the format",
        )
    return SUFFIXES[suffix]


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ConfigError(f"duplicate key {key!r} in JSON document")
        out[key] = value
    return out


def _reject_constant(token: str) -> Any:
    raise ConfigError(f"non-standard JSON number {token!r}")


def _toml_loads(text: str) -> Any:
    try:
        import tomllib  # noqa: PLC0415
    except ImportError:  # pragma: no cover - Python < 3.11
        try:
            import tomli as tomllib  # noqa: PLC0415
        except ImportError:
            raise ConfigError(
                "reading TOML on Python < 3.11 needs tomli",
                hint="pip install tomli, or use a JSON configuration",
            ) from None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML: {exc}") from None


def _yaml_loads(text: str) -> Any:
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        raise ConfigError(
            "reading YAML needs PyYAML",
            hint="pip install scikit-plots[env] (PyYAML), or use a TOML or JSON configuration",
        ) from None

    class Loader(yaml.SafeLoader):
        def compose_node(self, parent: Any, index: Any) -> Any:
            if self.check_event(yaml.events.AliasEvent):
                raise ConfigError("YAML aliases are not allowed in a configuration")
            return super().compose_node(parent, index)

        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            keys = []
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in keys:
                    raise ConfigError(f"duplicate key {key!r} in YAML document")
                keys.append(key)
            return super().construct_mapping(node, deep=deep)

    try:
        return yaml.load(text, Loader=Loader)  # noqa: S506 - safe loader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from None


def loads(text: str, format: str) -> Any:  # noqa: A002
    """
    Parse one configuration document.

    Parameters
    ----------
    text : str
        Document text.
    format : {"toml", "json", "yaml"}
        Its format.

    Returns
    -------
    object
        The parsed tree (a mapping for a valid configuration).

    Raises
    ------
    ConfigError
        For malformed text, duplicate keys, YAML aliases or a missing parser.
    """
    if format == "toml":
        return _toml_loads(text)
    if format == "json":
        try:
            return json.loads(
                text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant
            )
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON: {exc}") from None
    if format == "yaml":
        return _yaml_loads(text)
    raise ConfigError(f"{format!r} is not a document format")


def read_text(path: Any) -> str:
    """
    Read a UTF-8 document, refusing anything larger than :data:`MAX_BYTES`.

    Parameters
    ----------
    path : str or os.PathLike
        File path.

    Returns
    -------
    str
        Its text.

    Raises
    ------
    ConfigError
        If the file is missing, too large or not UTF-8.
    """
    path = Path(os.fspath(path))
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from None
    if size > MAX_BYTES:
        raise ConfigError(f"{path} is larger than {MAX_BYTES} bytes")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ConfigError(f"{path} is not UTF-8 text") from None


def read_document(
    path: Any,
    format: str | None = None,
) -> tuple[dict[str, Any], str]:  # noqa: A002
    """
    Read a configuration file and return its mapping and digest.

    Parameters
    ----------
    path : str or os.PathLike
        A ``.toml``, ``.json``, ``.yaml``/``.yml`` file, or ``pyproject.toml``.
    format : str or None, default=None
        Explicit format.

    Returns
    -------
    mapping : dict
        The request table (``[tool.scikitplot.env]`` for ``pyproject.toml``).
    digest : str
        SHA-256 of the file's bytes as UTF-8.

    Raises
    ------
    ConfigError
        For an unreadable or invalid document.
    """
    fmt = format_of(path, format)
    if fmt == "jsonl":
        raise ConfigError("a .jsonl file is a batch; use load_batch")
    text = read_text(path)
    tree = loads(text, fmt)
    if Path(os.fspath(path)).name == "pyproject.toml":
        tree = ((tree or {}).get("tool") or {}).get("scikitplot", {}).get("env")
        if tree is None:
            raise ConfigError(f"{path} has no [tool.scikitplot.env] table")
    if tree is None:
        tree = {}
    if not isinstance(tree, dict):
        raise ConfigError(f"{path}: a configuration must be a table/object")
    return (
        tree,
        hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest(),
    )


def iter_jsonl(path: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    """
    Yield the records of a JSON Lines file with their line numbers.

    Parameters
    ----------
    path : str or os.PathLike
        Batch file.

    Yields
    ------
    lineno : int
        1-based line number.
    record : dict
        The parsed object.

    Raises
    ------
    ConfigError
        For a line that is not one JSON object, naming the line.
    """
    text = read_text(path)
    for lineno, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant
            )
        except (json.JSONDecodeError, ConfigError) as exc:
            raise ConfigError(f"{os.fspath(path)}:{lineno}: {exc}") from None
        if not isinstance(record, dict):
            raise ConfigError(f"{os.fspath(path)}:{lineno}: a line must hold an object")
        yield lineno, record
