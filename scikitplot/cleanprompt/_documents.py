"""
Split a structured artefact into regions, without re-serialising it.

Notes
-----
**User notes.** This is what lets ``cleanprompt`` take a whole notebook or
module instead of a paragraph::

    cleanprompt encode --in analysis.ipynb --out analysis.clean.ipynb

The file comes back as the same kind of file: a notebook is still a notebook,
with its cell ids, execution counts and outputs where they were.

**Developer notes — why offsets index the raw text.**

The obvious implementation of notebook support is ``json.loads``, edit, then
``json.dumps``. It is also wrong, for reasons that only show up after someone
has trusted it:

* the file comes back reformatted — different indentation, different key
  order, escapes normalised — so the diff is thousands of lines and nobody can
  see what was actually redacted;
* anything the schema does not model is dropped, and notebooks carry a great
  deal that no schema models: widget state, cell ids, attachments, per-cell
  metadata from four different tools;
* and the engine's single pass over the original text, which every invariant
  from ``I1`` to ``I9`` is stated in terms of, would become a pass over a
  *different* text.

So this module never rebuilds the document. It reports **regions**: half-open
intervals of the original string, each tagged with a role. The engine rewrites
those intervals in one pass exactly as it does for prose, and every byte
outside them survives untouched.

**Developer notes — why the JSON walk is synchronised rather than parsed with
positions.**

Finding the offset of a string that belongs to a particular cell needs two
things a single tool does not give: the document structure, and byte
positions. :func:`json.loads` has the first and discards the second; a
hand-written parser would have both at the cost of being a second JSON
implementation to maintain and get wrong.

The third option is used here. JSON string tokens appear in the raw text in a
fixed order, and a walk over the parsed structure — keys before their values,
arrays in order — visits strings in *that same order*, because Python
dictionaries preserve insertion order. Zipping the two gives every string its
position and its path, using the standard library's parser for the parsing and
a scanner for nothing but positions. The correspondence is checked rather than
assumed: a mismatch in count or content raises instead of silently
misattributing a region.

See Also
--------
scikitplot.cleanprompt._code : Reads the schema out of the code regions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterator

from ._exceptions import CleanPromptError

__all__ = [
    "FORMATS",
    "ROLES",
    "Region",
    "detect_format",
    "regions_for",
]

#: Artefact formats this module can split.
FORMATS = ("text", "python", "notebook")

#: Roles a region can carry. A detector set is chosen per role, so a base64
#: image is never scanned and a code cell is never treated as prose.
ROLES = (
    "code",  # executable source
    "prose",  # markdown, comments, docstrings
    "output",  # rendered results: tables, printed text
    "traceback",  # an error, which carries paths and values
    "metadata",  # kernel names, paths, tool state
    "binary",  # base64 payloads: never scanned
)

#: Notebook output fields that hold rendered results a reader would see.
_OUTPUT_TEXT_KEYS = frozenset({"text", "text/plain", "text/html", "text/markdown"})

#: Notebook output fields that hold encoded binary.
_BINARY_KEYS = frozenset(
    {"image/png", "image/jpeg", "image/gif", "application/pdf", "image/svg+xml"}
)

#: Matches one JSON string token, including its surrounding quotes, honouring
#: backslash escapes so that an embedded ``\"`` does not end the token.
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')

#: Suffixes that identify an artefact when the caller does not say.
_SUFFIX_FORMATS = {
    ".ipynb": "notebook",
    ".py": "python",
    ".pyi": "python",
}


@dataclass(frozen=True)
class Region:
    """
    One interval of the original document, with what it is.

    Parameters
    ----------
    start : int
        Inclusive offset into the original text.
    end : int
        Exclusive offset. Must be greater than ``start``.
    role : str
        One of :data:`ROLES`.
    path : str
        Where the region sits in the document, for reporting — for example
        ``'cells[3].outputs[0].text'``. Never parsed; only shown.

    Raises
    ------
    ValueError
        If the interval is empty or the role is unknown.

    Notes
    -----
    **Developer notes.** A region carries no text of its own. Holding a copy
    would make it possible for the copy and the document to disagree, and the
    one thing every offset in this submodule must be is an index into the text
    it came from.

    Examples
    --------
    >>> Region(0, 4, "code", "cells[0].source").length
    4
    """

    start: int
    end: int
    role: str
    path: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(
                f"Region must be non-empty: got start={self.start!r}, end={self.end!r}"
            )
        if self.start < 0:
            raise ValueError(f"Region.start must be >= 0, got {self.start!r}")
        if self.role not in ROLES:
            msg = "unknown region role {!r}; choose from {}".format(
                self.role, ", ".join(ROLES)
            )
            raise ValueError(msg)

    @property
    def length(self) -> int:
        """Return the number of characters in this region."""
        return self.end - self.start

    def slice(self, text: str) -> str:
        """
        Return this region's text out of the document it indexes.

        Parameters
        ----------
        text : str
            The original document.

        Returns
        -------
        str
            ``text[start:end]``.
        """
        return text[self.start : self.end]


def detect_format(path: str | None, text: str) -> str:
    """
    Return the artefact format, from the file name and then from the content.

    Parameters
    ----------
    path : str or None
        The file name, when there is one.
    text : str
        The document.

    Returns
    -------
    str
        One of :data:`FORMATS`.

    Notes
    -----
    **Developer notes.** The suffix is consulted first because it is what the
    user believes the file to be, and a mismatch between the suffix and the
    content is worth surfacing rather than silently overriding. Content
    sniffing is the fallback for standard input, and is deliberately narrow:
    a document is a notebook only if it parses as JSON *and* carries the two
    keys ``nbformat`` and ``cells``, which is the actual definition rather
    than a resemblance to one.

    Examples
    --------
    >>> detect_format("analysis.ipynb", "")
    'notebook'
    >>> detect_format("features.py", "")
    'python'
    >>> detect_format(None, "just a sentence")
    'text'
    """
    if path:
        lowered = str(path).lower()
        for suffix, fmt in _SUFFIX_FORMATS.items():
            if lowered.endswith(suffix):
                return fmt
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            document = json.loads(text)
        except ValueError:
            return "text"
        if (
            isinstance(document, dict)
            and "nbformat" in document
            and "cells" in document
        ):
            return "notebook"
    return "text"


def regions_for(text: str, fmt: str, path: str | None = None) -> tuple[Region, ...]:
    """
    Split a document into regions.

    Parameters
    ----------
    text : str
        The document, exactly as read.
    fmt : str
        One of :data:`FORMATS`.
    path : str, optional
        The file name, used only in region paths for reporting.

    Returns
    -------
    tuple of Region
        Disjoint regions in ascending order. Their union is a subset of the
        document: anything not covered is structure, and is left alone.

    Raises
    ------
    CleanPromptError
        If ``fmt`` is unknown, or the document does not parse as that format.

    Notes
    -----
    **User notes.** A notebook that does not parse is refused rather than
    partly processed. Half of a redacted notebook is not a safe artefact.

    Examples
    --------
    >>> [r.role for r in regions_for("x = 1", "python")]
    ['code']
    >>> regions_for("hello", "text")[0].length
    5
    """
    if fmt not in FORMATS:
        msg = "unknown artefact format {!r}; choose from {}".format(
            fmt, ", ".join(FORMATS)
        )
        raise CleanPromptError(msg)
    if not text:
        return ()
    if fmt == "text":
        return (Region(0, len(text), "prose", str(path or "")),)
    if fmt == "python":
        return (Region(0, len(text), "code", str(path or "")),)
    return _notebook_regions(text, str(path or ""))


def _json_string_tokens(text: str) -> list[tuple[int, int, str]]:
    r"""
    Return every JSON string token as ``(start, end, decoded)``.

    Notes
    -----
    **Developer notes.** The offsets returned are of the token's *contents*,
    excluding the quotes, because that is the span a rewrite may touch: a
    replacement that consumed a quote would produce invalid JSON.

    Decoding uses :func:`json.loads` on the token rather than a hand-written
    unescaper, so a ``\u`` escape, a surrogate pair and every other escape are
    handled by the standard library.
    """
    tokens = []
    for match in _JSON_STRING.finditer(text):
        raw = match.group()
        try:
            decoded = json.loads(raw)
        except ValueError:  # pragma: no cover - the pattern only matches valid tokens
            continue
        tokens.append((match.start() + 1, match.end() - 1, decoded))
    return tokens


def _walk_json_strings(node: object, path: str = "") -> Iterator[tuple[str, str]]:
    """
    Yield ``(path, value)`` for every string in a parsed document, in text order.

    Notes
    -----
    **Developer notes.** The order must match the order string tokens appear in
    the raw text: within an object a key precedes its value, and members appear
    in insertion order, which is the order :func:`json.loads` builds them.
    Within an array, elements appear in order. Those two facts are what make
    the zip in :func:`_notebook_regions` valid.
    """
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            yield f"{child}#key", key
            for found in _walk_json_strings(value, child):
                yield found
    elif isinstance(node, list):
        for index, value in enumerate(node):
            for found in _walk_json_strings(value, f"{path}[{index}]"):
                yield found


def _role_for_path(  # ruff: ignore[too-many-return-statements]
    path: str,
) -> str:
    """
    Return the region role for a notebook JSON path.

    Notes
    -----
    **Developer notes.** The mapping is by *path* rather than by content
    because content is exactly what must not be trusted here: a base64 payload
    and a printed table are both long strings, and only their position says
    which is which.

    The rule under ``outputs`` is narrow on purpose, and the first version of
    it was not. An output is a dictionary, and most of its members are
    *structure* rather than data: ``output_type`` must read ``execute_result``
    or the notebook is invalid, ``name`` is ``stdout`` or ``stderr``, and
    ``execution_count`` is a number. Only ``text``, the members of ``data``,
    and the error fields carry anything a reader would call a result. Treating
    the whole subtree as data produced ``"output_type": "[OUTPUT-1]"`` — a
    redaction that was perfectly safe and left behind a file no tool could
    open.
    """
    if path.endswith("#key"):
        return "metadata"
    leaf = path.rsplit(".", 1)[-1]
    leaf = leaf.split("[", 1)[0]
    if any(leaf == key or path.endswith(key) for key in _BINARY_KEYS):
        return "binary"
    if ".outputs" in path:
        if leaf == "traceback" or leaf in ("ename", "evalue"):
            return "traceback"
        if ".data." in path or leaf == "text":
            return "output"
        return "metadata"
    if path.startswith("cells") and leaf == "source":
        return "code"
    return "metadata"


def _notebook_regions(text: str, path: str) -> tuple[Region, ...]:
    """Split a notebook into regions without rebuilding it."""
    try:
        document = json.loads(text)
    except ValueError as exc:
        msg = (
            "{} is not valid JSON, so it cannot be a notebook: {}. "
            "A partly processed notebook is not a safe artefact, so nothing "
            "was written.".format(path or "the input", exc)
        )
        raise CleanPromptError(msg) from exc
    if not isinstance(document, dict) or "cells" not in document:
        msg_0 = (
            "{} parses as JSON but has no 'cells' key, so it is not a "
            "notebook; pass --as text to process it as plain text.".format(
                path or "the input"
            )
        )
        raise CleanPromptError(msg_0)

    tokens = _json_string_tokens(text)
    walked = list(_walk_json_strings(document))
    if len(tokens) != len(walked):
        msg_1 = (
            "could not align {} JSON string token(s) in {} with {} string(s) "
            "in the parsed notebook. This means the document could not be "
            "located reliably, so nothing was rewritten.".format(
                len(tokens), path or "the input", len(walked)
            )
        )
        raise CleanPromptError(msg_1)

    # Cell type decides how a `source` region is read: a markdown cell is prose.
    cell_types = {}
    for index, cell in enumerate(document.get("cells", []) or []):
        if isinstance(cell, dict):
            cell_types[index] = str(cell.get("cell_type", "code"))

    regions = []
    for (start, end, decoded), (json_path, value) in zip(tokens, walked):
        if decoded != value:
            raise CleanPromptError(
                f"notebook string at offset {start} did not match the parsed "
                f"document at {json_path}; nothing was rewritten."
            )
        if end <= start:
            continue
        role = _role_for_path(json_path)
        if role == "code" and json_path.startswith("cells["):
            index = int(json_path[len("cells[") : json_path.index("]")])
            if cell_types.get(index) == "markdown":
                role = "prose"
        regions.append(Region(start, end, role, json_path))
    return tuple(regions)
