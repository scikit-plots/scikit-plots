"""
Find the values of named fields in structured text, by position.

``"mrn": "00412345"`` is an eight-digit number to a pattern and a medical
record number to anyone who reads the key. This module reads the key. It turns
a CSV, a JSON document, an ``.env`` file, a shell script or a Python module
into **field regions** — intervals of the original text, each paired with the
name of the field it is the value of — so a field rule can hide a value by what
it *is* rather than by what it looks like.

Notes
-----
**User notes.** You meet this through packs. With the ``patient`` pack
selected, ``{"mrn": "00412345"}``, ``mrn,00412345`` in a CSV and
``MRN=00412345`` in an ``.env`` file are all hidden, because each carries the
key ``mrn``.

**Developer notes — every scanner returns offsets into the raw text.**

The rule of ``D-12.6`` holds here unchanged: nothing is parsed, edited and
re-serialised. Each scanner walks the text it was given and reports intervals
of *that* text, which is what lets the engine's single rewrite pass keep every
invariant and every byte outside a value.

Where a standard-library parser exists it is used as the **oracle** rather
than the implementation: the JSON scanner's tokens are decoded with
:func:`json.loads` and checked against a walk of the parsed document; the CSV
scanner's cells are what :mod:`csv` would read, which a differential test
asserts. A position-tracking scanner that disagreed with the standard parser
would misattribute a value to a field, so a disagreement raises instead.

**Developer notes — what each scanner deliberately does not do.**

The line scanners (``keyvalue`` and ``script``) read ``key = value`` shapes
one line at a time. They do not follow YAML block scalars, multi-line strings
or here-documents. A value they do not reach is still scanned by every
pattern; it is only the field rule that does not apply to it, and the format
table says so.

See Also
--------
scikitplot.cleanprompt._packs : Field rules and name normalisation.
scikitplot.cleanprompt._documents : Regions and roles for code and prose.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from ._exceptions import CleanPromptError
from ._packs import FieldSpec, normalise_field
from ._types import Span

__all__ = [
    "SPLITTERS",
    "FieldDetector",
    "FieldRegion",
    "field_regions",
    "json_tokens",
    "match_field",
    "record_starts",
]

#: Splitters this module implements. ``text``, ``python`` and ``notebook``
#: also appear in :data:`~scikitplot.cleanprompt._documents.FORMATS`.
SPLITTERS = (
    "text",
    "python",
    "notebook",
    "script",
    "json",
    "jsonl",
    "delimited",
    "keyvalue",
    "sheets",
    "office",
    "corpus",
    "archive",
)


@dataclass(frozen=True)
class FieldRegion:
    """
    The value of one named field, located in the raw text.

    Parameters
    ----------
    start : int
        Inclusive offset of the value.
    end : int
        Exclusive offset.
    field : str
        The key, exactly as written.
    token : str
        ``'string'`` for text inside quotes or a bare value, ``'number'`` or
        ``'literal'`` for a JSON scalar that has no quotes to write a
        placeholder inside, ``'prose'`` for a ``Key: value`` line of running
        text, whose value a rule may end at a clause (``FieldSpec.span``).

    Notes
    -----
    **Developer notes.** ``token`` exists for one reason. Replacing a JSON
    number with ``[MRN-1]`` produces a file that is no longer JSON, so a
    non-string scalar needs a stand-in that is itself a valid JSON value.
    The artefact layer reads this field to choose one.
    """

    start: int
    end: int
    field: str
    token: str = "string"  # ruff: ignore[hardcoded-password-string]


# --------------------------------------------------------------------------
# field matching
# --------------------------------------------------------------------------


def match_field(key: str, index: Mapping[str, FieldSpec]) -> FieldSpec | None:
    """
    Return the field rule a key belongs to, or ``None``.

    Parameters
    ----------
    key : str
        The key as written — ``DB_PASSWORD``, ``TEL;TYPE=CELL``, ``dateOfBirth``.
    index : mapping
        Normalised name to rule, as built from the selected packs.

    Returns
    -------
    FieldSpec or None
        The matching rule.

    Notes
    -----
    **Developer notes.** Three steps, each deterministic. vCard-style
    parameters after ``;`` are dropped, so ``TEL;TYPE=CELL`` is ``tel``. The
    key is normalised. Then it matches exactly, or — for a rule that allows
    it — by its trailing tokens, longest suffix first, so
    ``patient_date_of_birth`` finds ``date_of_birth`` before ``birth``.

    Examples
    --------
    >>> rule = FieldSpec(("password",), "SECRET", "id")
    >>> match_field("DB_PASSWORD", {"password": rule}).kind
    'SECRET'
    >>> exact = FieldSpec(("state",), "LOCATION", "category", suffix=False)
    >>> match_field("loading_state", {"state": exact}) is None
    True
    """
    if not key:
        return None
    base = key.split(";", 1)[0]
    name = normalise_field(base)
    if not name:
        return None
    rule = index.get(name)
    if rule is not None:
        return rule
    tokens = name.split("_")
    for start in range(1, len(tokens)):
        candidate = index.get("_".join(tokens[start:]))
        if candidate is not None and candidate.suffix:
            return candidate
    return None


#: Where a prose value ends, by the field's ``span``: a ``clause`` at the
#: first comma or semicolon, a ``token`` at the first whitespace or either.
_SPAN_END = {"clause": re.compile(r"[,;]"), "token": re.compile(r"[\s,;]")}


class FieldDetector:
    """
    Report the value of every field whose name a selected pack marks sensitive.

    Parameters
    ----------
    regions : iterable of FieldRegion
        Every field value in the document.
    index : mapping
        Normalised field name to rule.

    Notes
    -----
    **Developer notes.** A matched value is reported **whole** (invariant
    ``I13``). The key already said the whole value is the sensitive part;
    looking for a sensitive part *inside* it would reintroduce the
    value-shape reasoning this detector exists to replace.

    The priority sits above every pattern, so when an ``EMAIL`` pattern and an
    ``email`` field claim the same characters the field wins and the value is
    recorded once, under the field's kind.
    """

    __slots__ = ("_index", "_regions", "confidence", "kind", "name", "priority")

    def __init__(
        self, regions: Iterable[FieldRegion], index: Mapping[str, FieldSpec]
    ) -> None:
        self.name = "fields"
        self.kind = "FIELD"
        self.priority = 97
        self.confidence = 1.0
        self._regions = tuple(regions)
        self._index = dict(index)

    def matches(self) -> Iterator[tuple[FieldRegion, FieldSpec]]:
        """
        Yield each region together with the rule it matched.

        Yields
        ------
        tuple of (FieldRegion, FieldSpec)
            Only regions whose key matched.
        """
        # A CSV repeats its header's few names once per row; each distinct
        # name is matched once (the match is a pure function of the name).
        rules: dict[str, FieldSpec | None] = {}
        for region in self._regions:
            name = region.field
            if name not in rules:
                rules[name] = match_field(name, self._index)
            rule = rules[name]
            if rule is not None:
                yield region, rule

    def detect(self, text: str, policy: object) -> Iterator[Span]:
        """
        Yield one span per matched field value.

        Parameters
        ----------
        text : str
            The document the regions index.
        policy : RedactionPolicy
            Unused; present for protocol conformance.

        Yields
        ------
        Span
            Whole values, in document order.
        """
        del policy
        limit = len(text)
        for region, rule in self.matches():
            if region.end > limit or region.end <= region.start:
                continue
            end = region.end
            _token = region.token == "prose"  # ruff: ignore[hardcoded-password-string]
            if _token and rule.span in _SPAN_END:
                clause = _SPAN_END[rule.span].search(text, region.start, end)
                if clause is not None:
                    end = region.start + len(
                        text[region.start : clause.start()].rstrip()
                    )
                if end <= region.start:
                    continue
            yield Span(
                start=region.start,
                end=end,
                kind=rule.kind,
                text=text[region.start : end],
                detector=f"field:{normalise_field(region.field.split(';', 1)[0])}",
                priority=self.priority,
                confidence=1.0,
            )


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------

#: One JSON scalar token. Strings come first in the alternation, so a number
#: inside a string is consumed as part of the string and never seen alone.
_JSON_SCALAR = re.compile(
    r'"(?:[^"\\]|\\.)*"'
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"|\btrue\b|\bfalse\b|\bnull\b"
)


class _Pairs(list):
    """
    A JSON object kept as its ``(key, value)`` pairs, duplicates included.

    Notes
    -----
    **Developer notes.** RFC 8259 says keys *should* be unique, not must, and
    :func:`json.loads` keeps only the last of a duplicated key. The token scan
    sees every one, so a document with a duplicate key could not be aligned
    and was refused. Parsing into pairs keeps both, in text order, so the
    walk and the scan agree on every valid document (found by fuzzing).
    """


def _lines(text: str) -> list[str]:
    r"""
    Split on ``\n`` only, keeping the terminators.

    Notes
    -----
    **Developer notes.** :meth:`str.splitlines` also splits on U+2028, U+2029,
    form feeds and other separators that are ordinary characters inside a JSON
    string or a config value; a JSON Lines record holding one was cut in two
    and refused (found by fuzzing). Files here are line-oriented on ``\n``.
    """
    return re.findall(r"[^\n]*\n|[^\n]+$", text)


def _walk_scalars(node: object, key: str = "") -> Iterator[tuple[bool, str, object]]:
    """Yield ``(is_key, owning_key, value)`` for every scalar, in text order."""
    if isinstance(node, _Pairs):
        for name, value in node:
            yield True, "", name
            yield from _walk_scalars(value, str(name))
    elif isinstance(node, dict):
        for name, value in node.items():
            yield True, "", name
            yield from _walk_scalars(value, str(name))
    elif isinstance(node, list):
        for value in node:
            yield from _walk_scalars(value, key)
    else:
        yield False, key, node


def _same(decoded: object, value: object) -> bool:
    """Compare two JSON scalars without letting ``True == 1`` through."""
    return type(decoded) is type(value) and decoded == value


def _string_offsets(raw: str) -> list[int]:
    r"""
    Map each character of a decoded JSON string to its offset in ``raw``.

    Parameters
    ----------
    raw : str
        The inside of a JSON string token, without its quotes.

    Returns
    -------
    list of int
        One raw offset per decoded character, then ``len(raw)``.

    Notes
    -----
    **Developer notes.** An escape is one decoded character written as two
    raw ones (``\n``), six (``\u00e9``) or twelve (a surrogate pair,
    which :func:`json.loads` joins into one character). Region boundaries are
    only ever placed at these offsets, so a placeholder never splits an
    escape and the file stays valid JSON.
    """
    starts = []
    index = 0
    length = len(raw)
    while index < length:
        starts.append(index)
        if raw[index] != "\\":
            index += 1
        elif raw[index + 1] != "u":
            index += 2
        else:
            code = int(raw[index + 2 : index + 6], 16)
            pair = (
                raw[index + 6 : index + 8] == "\\u"  # lint
                and 0xD800 <= code < 0xDC00  # ruff: ignore[magic-value-comparison]
            )
            if (
                pair
                and 0xDC00  # ruff: ignore[magic-value-comparison]
                <= int(raw[index + 8 : index + 12], 16)
                < 0xE000  # ruff: ignore[magic-value-comparison]
            ):
                index += 12
            else:
                index += 6
    starts.append(length)
    return starts


def _prose_in_string(raw: str, start: int) -> list[FieldRegion]:
    """
    Return ``Key: value`` regions inside one JSON string's text (``CP-078``).

    Parameters
    ----------
    raw : str
        The string token's inside, as written.
    start : int
        Offset of ``raw`` in the document.

    Returns
    -------
    list of FieldRegion
        Prose regions in document offsets, never splitting an escape.
    """
    decoded = json.loads(f'"{raw}"')
    if ":" not in decoded:
        return []
    starts = _string_offsets(raw)
    if len(starts) != len(decoded) + 1:
        msg = "internal error: a JSON string's escapes could not be mapped; nothing was returned"
        raise CleanPromptError(msg)
    return [
        FieldRegion(
            start + starts[region.start],
            start + starts[region.end],
            region.field,
            "prose",
        )
        for region in _keyvalue_regions(
            decoded, separators=(":",), comments=(), prose=True
        )
        if region.end > region.start
    ]


def _json_regions(text: str, offset: int = 0) -> list[FieldRegion]:
    """
    Return a field region for every scalar that is the value of a key.

    Notes
    -----
    **Developer notes.** A string value is also read as prose: a ``note``
    holding ``password: hunter2`` or ``Diagnosis: E11.9, ...`` names its own
    field, exactly as the same line would in a text file. Without this, a
    JSON file or a tool result sent a password the same words in a prompt
    would have hidden (``CP-078``).
    """
    try:
        document = json.loads(text, object_pairs_hook=_Pairs)
    except ValueError as exc:
        msg = f"not valid JSON, so no field could be located: {exc}"
        raise CleanPromptError(msg) from exc
    tokens = list(_JSON_SCALAR.finditer(text))
    walked = list(_walk_scalars(document))
    if len(tokens) != len(walked):
        msg = (
            f"could not align {len(tokens)} JSON token(s) with {len(walked)} "
            "value(s) in the parsed document; nothing was located"
        )
        raise CleanPromptError(msg)
    regions = []
    for match, (is_key, key, value) in zip(tokens, walked):
        raw = match.group()
        decoded = json.loads(raw)
        if not _same(decoded, value):
            msg = (
                f"JSON token at offset {match.start() + offset} did not match its value"
            )
            raise CleanPromptError(msg)
        _len = len(raw) > 2  # ruff: ignore[magic-value-comparison]
        if not is_key and isinstance(value, str) and _len:
            regions.extend(_prose_in_string(raw[1:-1], match.start() + 1 + offset))
        if is_key or not key:
            continue
        if isinstance(value, str):
            if len(raw) > 2:  # ruff: ignore[magic-value-comparison]
                regions.append(
                    FieldRegion(
                        match.start() + 1 + offset, match.end() - 1 + offset, key
                    )
                )
        elif value is None:
            continue
        else:
            token = "literal" if isinstance(value, bool) else "number"
            regions.append(
                FieldRegion(match.start() + offset, match.end() + offset, key, token)
            )
    return regions


def _jsonl_regions(text: str) -> list[FieldRegion]:
    """Return field regions for JSON Lines: one document per non-blank line."""
    regions = []
    position = 0
    for line in _lines(text):
        body = line.rstrip("\r\n")
        if body.strip():
            regions.extend(_json_regions(body, offset=position))
        position += len(line)
    return regions


def json_tokens(text: str) -> list[tuple[int, int, str]]:
    """
    Return every scalar token of a JSON or JSON Lines document.

    Parameters
    ----------
    text : str
        A document that parses as JSON, or as JSON Lines.

    Returns
    -------
    list of tuple of (int, int, str)
        ``(start, end, token)`` in text order, where ``token`` is
        ``'string'`` (the span includes the quotes), ``'number'`` or
        ``'literal'``. Keys are included: they are string tokens too.

    Notes
    -----
    **Developer notes.** Only valid JSON may be passed; the caller has already
    parsed it through :func:`field_regions`. On valid JSON a left-to-right
    scan with strings first in the alternation is exact, because outside a
    string only structure, whitespace, numbers and the three literals can
    occur. That is the same scan :func:`_json_regions` aligns with the parsed
    document, so both agree on where every token is.

    Examples
    --------
    >>> json_tokens('{"a": [1, true]}')
    [(1, 4, 'string'), (7, 8, 'number'), (10, 14, 'literal')]
    """
    out = []
    for match in _JSON_SCALAR.finditer(text):
        raw = match.group()
        token = (
            "string"
            if raw.startswith('"')
            else ("number" if raw[0] in "-0123456789" else "literal")
        )
        out.append((match.start(), match.end(), token))
    return out


# --------------------------------------------------------------------------
# delimited (CSV, TSV)
# --------------------------------------------------------------------------


def _delimited_cells(  # ruff: ignore[too-many-branches]
    text: str,
    delimiter: str,
    rows: list[int] | None = None,
) -> Iterator[tuple[int, int, int, int]]:
    """
    Yield ``(row, column, start, end)`` for every cell's *content*.

    ``rows``, when given, receives the offset at which each record begins —
    the one scan decides both cells and records, so they cannot disagree.

    Notes
    -----
    **Developer notes.** RFC 4180: a quoted cell may hold the delimiter, a
    newline and ``""`` for a literal quote. The span of a quoted cell is its
    inside, so a placeholder lands between the quotes and the file stays
    valid. :mod:`csv` is the oracle, not the implementation: it reports
    contents but not positions.
    """
    row = column = 0
    index = 0
    length = len(text)
    while index < length:
        if column == 0 and rows is not None:
            rows.append(index)
        if text[index] == '"':
            start = index + 1
            index = start
            while index < length:
                if text[index] == '"':
                    if index + 1 < length and text[index + 1] == '"':
                        index += 2
                        continue
                    break
                index += 1
            end = index
            index = min(index + 1, length)
            # anything between the closing quote and the delimiter is
            # malformed; it is skipped rather than silently joined to the cell
            while index < length and text[index] not in (delimiter, "\n", "\r"):
                index += 1
        else:
            start = index
            while index < length and text[index] not in (delimiter, "\n", "\r"):
                index += 1
            end = index
        yield row, column, start, end
        if index < length and text[index] == delimiter:
            column += 1
            index += 1
            if index == length:
                yield row, column, index, index
            continue
        if index < length and text[index] == "\r":
            index += 1
        if index < length and text[index] == "\n":
            index += 1
        row += 1
        column = 0


def record_starts(
    text: str, splitter: str, options: Mapping[str, object] | None = None
) -> list[int]:
    r"""
    Return the offset at which each record of a record-oriented document begins.

    Parameters
    ----------
    text : str
        A CSV/TSV (``'delimited'``) or JSON Lines (``'jsonl'``) document.
    splitter : {'delimited', 'jsonl'}
        How records are separated.
    options : mapping, optional
        ``delimiter`` for ``'delimited'``.

    Returns
    -------
    list of int
        Ascending; the first is ``0`` when the text is not empty. For a
        delimited file the header is the first record.

    Raises
    ------
    CleanPromptError
        For any other splitter: only these two have records that no value can
        span, which is what makes cutting between them safe.

    Examples
    --------
    >>> record_starts('a,b\n"x\ny",2\n3,4\n', "delimited")
    [0, 4, 12]
    >>> record_starts('{"a": 1}\n{"a": 2}\n', "jsonl")
    [0, 9]
    """
    if splitter == "jsonl":
        return [match.start() for match in re.finditer(r"[^\n]*\n|[^\n]+$", text)]
    if splitter == "delimited":
        rows: list[int] = []
        for _ in _delimited_cells(
            text, str((options or {}).get("delimiter", ",")), rows
        ):
            pass
        return rows
    msg = f"splitter {splitter!r} has no record boundaries a document can be cut at"
    raise CleanPromptError(msg)


def _delimited_regions(text: str, delimiter: str) -> list[FieldRegion]:
    """Return a field region per data cell, named by its column header."""
    header: dict[int, str] = {}
    regions = []
    for row, column, start, end in _delimited_cells(text, delimiter):
        if row == 0:
            header[column] = text[start:end].replace('""', '"').strip()
            continue
        name = header.get(column)
        if name and end > start:
            regions.append(FieldRegion(start, end, name))
    return regions


def _sheets_regions(text: str) -> list[FieldRegion]:
    """
    Return field regions for spreadsheet text: tab-separated blocks, one per sheet.

    Notes
    -----
    **Developer notes.** This reads what :func:`~scikitplot.cleanprompt._office.extract_office_text`
    writes for a workbook: a ``## sheet N`` line, then the sheet's rows. Each
    block's first row is its header, so ``B:email`` in sheet one and ``C:email``
    in sheet three are both named ``email``.
    """
    regions: list[FieldRegion] = []
    blocks = [match.end() for match in re.finditer(r"(?m)^## sheet \d+\n", text)]
    if not blocks:
        return _delimited_regions(text, "\t")
    ends = [
        *[match.start() for match in re.finditer(r"(?m)^## sheet \d+\n", text)][1:],
        len(text),
    ]
    for start, end in zip(blocks, ends):
        regions.extend(
            FieldRegion(region.start + start, region.end + start, region.field)
            for region in _delimited_regions(text[start:end], "\t")
        )
    return regions


# --------------------------------------------------------------------------
# key = value lines
# --------------------------------------------------------------------------


def _keyvalue_regions(  # ruff: ignore[too-many-branches]
    text: str,
    separators: Iterable[str] = ("=", ":"),
    comments: Iterable[str] = ("#", ";"),
    header_block: bool = False,
    prose: bool = False,
) -> list[FieldRegion]:
    """
    Return a field region per ``key<sep>value`` line.

    Notes
    -----
    **Developer notes.** With ``prose`` a value's trailing sentence
    punctuation is left outside the region: in ``MRN: 00412345.`` the full
    stop ends the sentence, not the number. That matters beyond looks — the
    vault is keyed on the value, so ``00412345.`` and the ``00412345`` of a
    CSV would otherwise receive two different labels for one patient.

    **Developer notes.** With ``header_block`` the scan follows RFC 5322: it
    reads headers until the first blank line, extends a header over its
    folded continuation lines, and starts a new block at an mbox ``From ``
    separator. The body is left to the patterns, because ``Hi Marion: ...``
    in a message body is not a header.
    """
    seps = "".join(re.escape(sep) for sep in separators)
    line_pattern = re.compile(
        r"^(?P<lead>[ \t]*(?:-[ \t]+)?(?:export[ \t]+)?)"
        rf"(?P<key>[^\s{seps}#;\[][^{seps}\n]*?)[ \t]*(?P<sep>[{seps}])[ \t]*"
        r"(?P<value>[^\n]*?)[ \t]*$"
    )
    comment_starts = tuple(comments)
    regions: list[FieldRegion] = []
    in_headers = True
    position = 0
    last: FieldRegion | None = None
    for line in _lines(text):
        body = line.rstrip("\r\n")
        stripped = body.strip()
        if header_block:
            if body.startswith("From ") and not body.startswith("From:"):
                in_headers = True
                last = None
                position += len(line)
                continue
            if not stripped:
                in_headers = False
                last = None
                position += len(line)
                continue
            if not in_headers:
                position += len(line)
                continue
            if body[:1] in (" ", "\t") and last is not None:
                regions[-1] = FieldRegion(last.start, position + len(body), last.field)
                last = regions[-1]
                position += len(line)
                continue
        if (
            not stripped
            or stripped.startswith(comment_starts)
            or stripped.startswith("[")
        ):
            position += len(line)
            continue
        match = line_pattern.match(body)
        if match and match.group("value"):
            start = position + match.start("value")
            end = position + match.end("value")
            value = match.group("value")
            _len = len(value) >= 2  # ruff: ignore[magic-value-comparison]
            if _len and value[0] == value[-1] and value[0] in "\"'":
                start, end = start + 1, end - 1
            elif not header_block:
                # an inline comment after an unquoted value: `x = 1  # note`
                comment = re.search(r"[ \t]+[#;]", value)
                if comment:
                    end = position + match.start("value") + comment.start()
            if prose:
                end = start + len(text[start:end].rstrip(".,;!?"))
            if end > start:
                last = FieldRegion(
                    start,
                    end,
                    match.group("key").strip(),
                    "prose" if prose else "string",
                )
                regions.append(last)
        position += len(line)
    return regions


# --------------------------------------------------------------------------
# scripts: shell, JavaScript, SQL, R, and notebook code as raw JSON
# --------------------------------------------------------------------------

_SCRIPT_QUOTED = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_.\-]*)[\"']?[ \t]*(?:=>|:=|<-|=|:)[ \t]*"
    r"(?P<q>\\?[\"'`])(?P<value>(?:\\[^\"'`\n]|(?!(?P=q))[^\n\\])*)(?P=q)"
)
_SCRIPT_BARE = re.compile(
    r"(?m)(?:^|(?<=[\s;]))(?:export[ \t]+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>[^\s\"'`;#|&$(][^\s;|&]*)"
)
_SCRIPT_FLAG = re.compile(
    r"(?P<key>--[A-Za-z][A-Za-z0-9\-]*)(?:=|[ \t]+)(?P<value>[^\s\-\"'][^\s]*)"
)


def _script_regions(text: str) -> list[FieldRegion]:
    """
    Return field regions for assignments and long options in a script.

    Notes
    -----
    **Developer notes.** Three shapes, each unambiguous on its own line:
    ``key = "value"`` in any common spelling of assignment (``=``, ``:``,
    ``=>``, ``:=``, ``<-``); ``KEY=value`` as a shell writes it; and
    ``--flag value``. The quoted form accepts a quote escaped with a
    backslash, which is how notebook code appears inside its JSON — so the
    same scanner reads a script and a notebook's raw cell source.
    Overlapping hits keep the first, in text order.
    """
    found: list[FieldRegion] = []
    for pattern in (_SCRIPT_QUOTED, _SCRIPT_BARE, _SCRIPT_FLAG):
        for match in pattern.finditer(text):
            start, end = match.start("value"), match.end("value")
            if end > start:
                found.append(FieldRegion(start, end, match.group("key").lstrip("-")))
    found.sort(key=lambda region: (region.start, -region.end))
    kept: list[FieldRegion] = []
    for region in found:
        if kept and region.start < kept[-1].end:
            continue
        kept.append(region)
    return kept


# --------------------------------------------------------------------------
# Python source
# --------------------------------------------------------------------------


def _line_starts(text: str) -> list[int]:
    """Return the character offset at which each line begins."""
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _char_offset(text: str, starts: list[int], line: int, byte_col: int) -> int:
    """Convert an AST ``(lineno, col_offset)`` — a UTF-8 byte column — to a character offset."""
    begin = starts[line - 1]
    line_text = text[begin : starts[line] if line < len(starts) else len(text)]
    return begin + len(
        line_text.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore")
    )


def _constant_region(  # ruff: ignore[too-many-return-statements]
    text: str,
    starts: list[int],
    node: ast.AST,
    key: str,
) -> FieldRegion | None:
    """Return the value region of a literal constant, or ``None``."""
    if not isinstance(node, ast.Constant) or getattr(node, "end_lineno", None) is None:
        return None
    value = node.value
    start = _char_offset(text, starts, node.lineno, node.col_offset)
    end = _char_offset(text, starts, node.end_lineno, node.end_col_offset)
    source = text[start:end]
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return FieldRegion(start, end, key, "number")
    if not isinstance(value, str):
        return None
    prefix = re.match(r"(?i)[rbuf]*", source).group()
    body = source[len(prefix) :]
    for quote in ('"""', "'''", '"', "'"):
        if (
            body.startswith(quote)
            and body.endswith(quote)
            and len(body) >= 2 * len(quote)
        ):
            inner_start = start + len(prefix) + len(quote)
            inner_end = end - len(quote)
            # An implicitly concatenated literal ("a" "b") is one constant
            # spanning two tokens; its inside is not one value. Verified by
            # re-reading the reconstructed literal, and skipped if it differs.
            # Parsed, never evaluated: the reconstructed literal must be one
            # string constant equal to the value the tree holds.
            try:
                rebuilt = ast.parse(source, mode="eval").body
            except SyntaxError:
                rebuilt = None
            same = (
                isinstance(rebuilt, ast.Constant)
                and rebuilt.value == value
                and quote not in body[len(quote) : -len(quote)]
            )
            if same and inner_end > inner_start:
                return FieldRegion(inner_start, inner_end, key)
            return None
    return None


def _python_regions(  # ruff: ignore[too-many-branches]
    text: str,
) -> list[FieldRegion]:
    """
    Return field regions for literal values bound to names in Python source.

    Notes
    -----
    **Developer notes.** Syntax, not text: ``password = "x"``,
    ``self.api_key = "x"``, ``{"mrn": 412345}`` and ``connect(password="x")``
    are each a literal in a position the language says is bound to that name.
    A file that does not parse yields nothing here and is still scanned by
    every pattern; the artefact report already names unparsable code.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    starts = _line_starts(text)
    regions: list[FieldRegion] = []

    def target_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    for node in ast.walk(tree):
        pairs: list[tuple[str, ast.AST]] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = target_name(target)
                if name:
                    pairs.append((name, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            name = target_name(node.target)
            if name:
                pairs.append((name, node.value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    pairs.append((key.value, value))
        elif isinstance(node, ast.Call):
            pairs.extend(
                (keyword.arg, keyword.value) for keyword in node.keywords if keyword.arg
            )
        for name, value in pairs:
            region = _constant_region(text, starts, value, name)
            if region is not None:
                regions.append(region)
    regions.sort(key=lambda region: (region.start, region.end))
    return regions


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def field_regions(
    text: str,
    splitter: str,
    options: Mapping[str, object] | None = None,
) -> tuple[FieldRegion, ...]:
    r"""
    Return every field value in a document, for the splitter that reads it.

    Parameters
    ----------
    text : str
        The document, exactly as read.
    splitter : str
        One of :data:`SPLITTERS`.
    options : mapping, optional
        Splitter options from the format definition: ``delimiter``,
        ``separators``, ``comments``, ``header_block``, ``prose``.

    Returns
    -------
    tuple of FieldRegion
        In ascending order. Empty for splitters that carry no field names.

    Raises
    ------
    CleanPromptError
        If ``splitter`` is unknown, or a JSON document cannot be aligned.

    Examples
    --------
    >>> [
    ...     text[r.start : r.end]
    ...     for text in ['{"mrn": "0041", "n": 7}']
    ...     for r in field_regions(text, "json")
    ... ]
    ['0041', '7']
    >>> [
    ...     (r.field, "a,b\n1,2"[r.start : r.end])
    ...     for r in field_regions("a,b\n1,2", "delimited")
    ... ]
    [('a', '1'), ('b', '2')]
    >>> [r.field for r in field_regions("export DB_PASSWORD=hunter2", "script")]
    ['DB_PASSWORD']
    """
    if splitter not in SPLITTERS:
        msg = f"unknown splitter {splitter!r}; choose from {', '.join(SPLITTERS)}"
        raise CleanPromptError(msg)
    opts = dict(options or {})
    if splitter == "json":
        found = _json_regions(text)
    elif splitter == "jsonl":
        found = _jsonl_regions(text)
    elif splitter == "delimited":
        found = _delimited_regions(text, str(opts.get("delimiter", ",")))
    elif splitter == "keyvalue":
        found = _keyvalue_regions(
            text,
            separators=tuple(opts.get("separators", ("=", ":"))),
            comments=tuple(opts.get("comments", ("#", ";"))),
            header_block=bool(opts.get("header_block", False)),
            prose=bool(opts.get("prose", False)),
        )
    elif splitter in ("script", "notebook"):
        found = _script_regions(text)
    elif splitter == "python":
        found = _python_regions(text)
    elif splitter == "sheets":
        found = _sheets_regions(text)
    elif splitter == "text":
        found = _keyvalue_regions(text, separators=(":",), comments=(), prose=True)
    else:
        found = []
    return tuple(sorted(found, key=lambda region: (region.start, region.end)))
