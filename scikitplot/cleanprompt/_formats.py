"""
File formats: how each kind of file is read, as validated data.

Notes
-----
**User notes.** A format says which extensions it covers, which splitter reads
it, and whether a redacted file can be turned back into the original byte for
byte::

    name: env
    version: 1
    summary: Environment files - KEY=value, one per line.
    extensions: [.env]
    splitter: keyvalue
    round_trip: true
    options:
      separators: ["="]
    packs: [secrets]

``packs`` is what ``packs("auto")`` selects for a file of this format.

**Developer notes.** ``round_trip`` is a *claim the format makes*, and the
suite holds each built-in format to it: every round-trip format is encoded and
decoded in a test and compared byte for byte. A format that extracts text —
Office files, PDF — says ``false``, so nobody is told a ``.docx`` comes back.

See Also
--------
scikitplot.cleanprompt._structured : The splitters named here.
scikitplot.cleanprompt._catalog : Loads formats and maps a path to one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._packs import PackError
from ._structured import SPLITTERS

__all__ = [
    "FormatSpec",
    "format_from_document",
]

_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_EXTENSION = re.compile(r"^\.[a-z0-9][a-z0-9_.+\-]{0,15}$")
_TOP_KEYS = frozenset(
    {
        "name",
        "version",
        "summary",
        "extensions",
        "splitter",
        "round_trip",
        "options",
        "packs",
    }
)
_OPTION_KEYS = frozenset(
    {"delimiter", "separators", "comments", "header_block", "prose"}
)


@dataclass(frozen=True)
class FormatSpec:
    """
    One validated file format.

    Parameters
    ----------
    name : str
        Unique format name.
    version : int
        The definition's own version.
    summary : str
        One line.
    extensions : tuple of str
        Lower-case, with the leading dot.
    splitter : str
        One of :data:`~scikitplot.cleanprompt._structured.SPLITTERS`.
    round_trip : bool
        Whether decode restores the original bytes exactly.
    options : tuple of (str, object)
        Splitter options, sorted, so the spec stays hashable.
    packs : tuple of str
        Packs ``packs("auto")`` selects for this format.
    source : str
        Where it was loaded from.
    """

    name: str
    version: int
    summary: str
    extensions: tuple[str, ...]
    splitter: str
    round_trip: bool
    options: tuple[tuple[str, Any], ...] = ()
    packs: tuple[str, ...] = ()
    source: str = field(default="<builtin>", compare=False)

    def option_map(self) -> dict[str, Any]:
        """
        Return the splitter options as a dictionary.

        Returns
        -------
        dict
            A fresh copy; mutating it changes nothing.
        """
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.options
        }


def _freeze(value: Any) -> Any:
    """Return a hashable copy of an option value."""
    if isinstance(value, list):
        return tuple(value)
    return value


def format_from_document(  # ruff: ignore[too-many-branches]
    document: Any,
    source: str = "<builtin>",
) -> FormatSpec:
    """
    Validate a parsed format document and build its spec.

    Parameters
    ----------
    document : mapping
        The format, as parsed from YAML or JSON.
    source : str, default='<builtin>'
        Where it came from, for messages.

    Returns
    -------
    FormatSpec
        The validated format.

    Raises
    ------
    PackError
        Listing every problem found.

    Examples
    --------
    >>> spec = format_from_document(
    ...     {
    ...         "name": "env",
    ...         "version": 1,
    ...         "summary": "Env files.",
    ...         "extensions": [".ENV"],
    ...         "splitter": "keyvalue",
    ...         "round_trip": True,
    ...     }
    ... )
    >>> spec.extensions
    ('.env',)
    """
    if not isinstance(document, Mapping):
        raise PackError(source, ["the document must be a mapping"])
    problems: list[str] = [
        f"format: unknown key {key!r} (allowed: {', '.join(sorted(_TOP_KEYS))})"
        for key in sorted(set(document) - _TOP_KEYS)
    ]
    name = document.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        problems.append(f"name: {name!r} must match {_NAME.pattern}")
    version = document.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        problems.append(f"version: {version!r} must be a positive integer")
    summary = document.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append("summary: must be a non-empty string")
    extensions = []
    raw_extensions = document.get("extensions")
    if not isinstance(raw_extensions, list) or not raw_extensions:
        problems.append("extensions: must be a non-empty list")
    else:
        for index, extension in enumerate(raw_extensions):
            lowered = (
                str(extension).lower() if isinstance(extension, str) else extension
            )
            if not isinstance(lowered, str) or not _EXTENSION.match(lowered):
                problems.append(
                    f"extensions[{index}]: {extension!r} must look like '.csv'"
                )
            else:
                extensions.append(lowered)
    splitter = document.get("splitter")
    if splitter not in SPLITTERS:
        problems.append(f"splitter: {splitter!r} is not one of {', '.join(SPLITTERS)}")
    round_trip = document.get("round_trip")
    if not isinstance(round_trip, bool):
        problems.append("round_trip: must be true or false")
    options = document.get("options", {}) or {}
    if not isinstance(options, Mapping):
        problems.append("options: must be a mapping")
        options = {}
    problems.extend(
        f"options: unknown key {key!r} (allowed: {', '.join(sorted(_OPTION_KEYS))})"
        for key in sorted(set(options) - _OPTION_KEYS)
    )
    if "delimiter" in options and (
        not isinstance(options["delimiter"], str) or len(options["delimiter"]) != 1
    ):
        problems.append("options.delimiter: must be exactly one character")
    for key in ("separators", "comments"):
        if key in options:
            value = options[key]
            if not isinstance(value, list) or not all(
                isinstance(v, str) and v for v in value
            ):
                problems.append(f"options.{key}: must be a list of non-empty strings")
    problems.extend(
        f"options.{key}: must be true or false"
        for key in ("header_block", "prose")
        if key in options and not isinstance(options[key], bool)
    )
    packs = document.get("packs", []) or []
    if not isinstance(packs, list) or not all(isinstance(p, str) and p for p in packs):
        problems.append("packs: must be a list of pack names")
        packs = []
    if problems:
        raise PackError(source, problems)
    return FormatSpec(
        name=name,
        version=version,
        summary=summary.strip(),
        extensions=tuple(sorted(set(extensions))),
        splitter=splitter,
        round_trip=round_trip,
        options=tuple(sorted((key, _freeze(value)) for key, value in options.items())),
        packs=tuple(sorted(set(packs))),
        source=source,
    )
