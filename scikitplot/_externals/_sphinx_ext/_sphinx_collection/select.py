"""
Domain-agnostic selection over collections of mappings.

Every collection directive in this project -- a gallery of downstream
projects, a list of videos, a table of datasets, a wall of contributors --
wants the same four operations over its records: *filter*, *sort*, *group*,
*paginate*. Implementing them once, over plain mappings, means a new
collection directive inherits all four for free and cannot drift from the
others in behaviour.

The engine knows nothing about videos, images or links. A record is a
mapping; a field is a key; that is the whole data model.

Filtering without executing anything
------------------------------------
The obvious way to offer flexible filtering is to evaluate an expression.
That is also a remote code execution hole in a documentation build: a
``:filter:`` string reaching ``eval`` turns any pull request that edits a
``.md`` file into arbitrary code running in CI. So this module defines a
small, closed grammar (:func:`parse_filter`) that is *parsed*, never
executed. It supports the comparisons authors actually need and nothing
else, and an unrecognised operator is an error rather than a fallback.

Notes
-----
**User-focused.** Filters are written as ``field OP value``, joined with
commas, and every term must hold::

    :filter: category=tutorial, level~beginner, stars>100
    :filter: tags:python|julia, !deprecated
    :sort: category,-published
    :group-by: category

**Developer-focused.** Sorting is total and stable, and comparison is
type-aware without guessing: a key whose values are all numbers sorts
numerically, all dates chronologically, and anything else
case-insensitively as text. Records missing the sort key are never dropped
and never interleave unpredictably -- they sort last in *both* directions,
because "unknown" does not belong at either end of a ranking. Ties fall
back to source order, so the same input always renders byte-identically.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "OPERATORS",
    "FilterError",
    "FilterTerm",
    "Selection",
    "apply_selection",
    "get_field",
    "group_records",
    "has_field",
    "parse_filter",
    "parse_sort",
]


class FilterError(ValueError):
    """
    Raised when a filter, sort or group specification is malformed.

    Messages name the offending term and list the valid alternatives, so a
    directive can surface them to a page author unchanged.
    """


#: Label used for records whose grouping key is missing or empty. Chosen
#: over dropping them: a record with no category is still a record the
#: reader asked to see.
UNGROUPED_LABEL = "Ungrouped"

#: Separator for the "any of" operator's alternatives.
_ANY_SEPARATOR = "|"

#: A well-formed field name or dotted path. Enforced so that a term with a
#: typo'd operator -- ``category ??? tutorial`` -- is reported as an error
#: instead of quietly becoming a presence test on a field of that literal
#: name, which no record has and which therefore renders an empty gallery
#: with no indication that the filter was nonsense.
_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$")


# -- field access -------------------------------------------------------------


def has_field(record: Mapping[str, Any], path: str) -> bool:
    """
    Return whether a dotted field path exists, even when its value is null.

    This differs from :func:`get_field`, where both a missing path and a
    present ``None`` value return ``None``. Configuration validation needs the
    structural distinction so heterogeneous records may legitimately omit a
    field while a typo that exists nowhere can be reported early.
    """
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def get_field(record: Mapping[str, Any], path: str) -> Any:
    """
    Read a possibly nested field from a record.

    Parameters
    ----------
    record : mapping
        The record to read from.
    path : str
        A field name, or a dotted path into nested mappings
        (``"author.name"``). Dotted access exists because real YAML
        collections nest, and forcing authors to flatten their data to make
        it sortable would be the tool dictating the data model.

    Returns
    -------
    Any
        The value, or ``None`` if any step of the path is missing. Missing
        is deliberately not an error: a heterogeneous collection where only
        some records carry a field is normal, and filtering on that field
        should exclude the others rather than fail the build.

    Examples
    --------
    >>> get_field({"a": {"b": 1}}, "a.b")
    1
    >>> get_field({"a": 1}, "missing") is None
    True
    """
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


# -- value coercion -----------------------------------------------------------

#: ISO-8601 date or datetime, the only date shapes accepted for comparison.
#: Deliberately narrow: guessing between ``03/04/2024`` as March 4th and
#: April 3rd would silently reorder a gallery differently per locale.
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?")


def _as_date(value: Any) -> _dt.datetime | None:
    """
    Interpret a value as a datetime, if it unambiguously is one.

    Parameters
    ----------
    value : Any
        A ``date``, ``datetime``, or ISO-8601 string.

    Returns
    -------
    datetime.datetime or None
        A naive-UTC-normalised datetime, or ``None`` if the value is not an
        unambiguous date.
    """
    if isinstance(value, _dt.datetime):
        return (
            value.replace(tzinfo=None)
            if value.tzinfo is None
            else (value.astimezone(_dt.timezone.utc).replace(tzinfo=None))
        )
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day)
    if isinstance(value, str) and _ISO_RE.match(value.strip()):
        text = value.strip().replace(" ", "T")
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = _dt.datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return parsed
    return None


def _as_number(value: Any) -> float | None:
    """
    Interpret a value as a number, if it unambiguously is one.

    Parameters
    ----------
    value : Any
        An int, float, or numeric string.

    Returns
    -------
    float or None
        The number, or ``None``. Booleans are excluded: ``True`` comparing
        equal to ``1`` in a sort would be a surprise, not a feature.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _comparable(values: Sequence[Any]) -> Callable[[Any], Any]:
    """
    Choose one comparison key function for a whole column of values.

    Parameters
    ----------
    values : sequence
        Every non-missing value the sort key takes across the collection.

    Returns
    -------
    callable
        A key function mapping a value to something orderable.

    Notes
    -----
    The decision is made **once per column, not per value**. Deciding per
    value produces a comparison that is not a total order -- ``"10" < "9"``
    as text but ``10 > 9`` as numbers -- and Python's sort on a
    non-total order silently returns an arbitrary arrangement that changes
    with input order. Choosing one interpretation for the column makes the
    result deterministic, which is what lets a rebuilt page be
    byte-identical.
    """
    if values and all(_as_number(value) is not None for value in values):
        return lambda value: _as_number(value)  # ruff: ignore[unnecessary-lambda]
    if values and all(_as_date(value) is not None for value in values):
        return lambda value: _as_date(value)  # ruff: ignore[unnecessary-lambda]
    return lambda value: str(value).casefold()


def _as_text(value: Any) -> str:
    """
    Render a value as text for substring and prefix matching.

    Parameters
    ----------
    value : Any
        Any field value.

    Returns
    -------
    str
        A casefolded string. Sequences are joined so that a substring
        filter over a list field behaves the way an author expects.
    """
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value).casefold()
    return str(value).casefold()


def _values_of(value: Any) -> list[Any]:
    """
    Normalise a field value to the list of atoms it should match against.

    Parameters
    ----------
    value : Any
        A scalar or a sequence.

    Returns
    -------
    list
        The atoms. A list-valued field (``tags``) matches a term if *any*
        element matches, which is the only reading that makes
        ``tags=python`` mean what an author intends.
    """
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


# -- filter grammar -----------------------------------------------------------


def _op_equal(value: Any, operand: str) -> bool:
    """Test case-insensitive equality against any atom of ``value``."""
    return any(str(atom).casefold() == operand.casefold() for atom in _values_of(value))


def _op_contains(value: Any, operand: str) -> bool:
    """Test case-insensitive substring containment."""
    return operand.casefold() in _as_text(value)


def _op_prefix(value: Any, operand: str) -> bool:
    """Test whether any atom starts with ``operand``."""
    return any(
        str(atom).casefold().startswith(operand.casefold())
        for atom in _values_of(value)
    )


def _op_suffix(value: Any, operand: str) -> bool:
    """Test whether any atom ends with ``operand``."""
    return any(
        str(atom).casefold().endswith(operand.casefold()) for atom in _values_of(value)
    )


def _op_any(value: Any, operand: str) -> bool:
    """Test membership in a ``|``-separated set of alternatives."""
    wanted = {part.strip().casefold() for part in operand.split(_ANY_SEPARATOR)}
    return any(str(atom).casefold() in wanted for atom in _values_of(value))


def _compare(value: Any, operand: str) -> int | None:
    """
    Three-way compare a field value against a filter operand.

    Parameters
    ----------
    value : Any
        The field value.
    operand : str
        The literal written in the filter.

    Returns
    -------
    int or None
        ``-1``, ``0`` or ``1``, or ``None`` when the two are not comparable
        (which makes any ordered comparison against them false rather than
        an error, so one odd record cannot fail a build).
    """
    for coerce in (_as_number, _as_date):
        left, right = coerce(value), coerce(operand)
        if left is not None and right is not None:
            return (left > right) - (left < right)
    left_text, right_text = str(value).casefold(), operand.casefold()
    return (left_text > right_text) - (left_text < right_text)


def _ordered(predicate: Callable[[int], bool]) -> Callable[[Any, str], bool]:
    """
    Build an ordered-comparison operator from a predicate on ``cmp``.

    Parameters
    ----------
    predicate : callable
        Maps a three-way comparison result to a boolean.

    Returns
    -------
    callable
        An operator function.
    """

    def operator(value: Any, operand: str) -> bool:
        result = _compare(value, operand)
        return False if result is None else predicate(result)

    return operator


#: The complete filter grammar. Longer operators are listed first so that
#: ``>=`` is not mis-split as ``>``. Extending the grammar is one entry;
#: nothing here can execute author-supplied code.
OPERATORS: tuple[tuple[str, Callable[[Any, str], bool]], ...] = (
    ("!~", lambda value, operand: not _op_contains(value, operand)),
    ("!=", lambda value, operand: not _op_equal(value, operand)),
    (">=", _ordered(lambda result: result >= 0)),
    ("<=", _ordered(lambda result: result <= 0)),
    ("~", _op_contains),
    ("^", _op_prefix),
    ("$", _op_suffix),
    (":", _op_any),
    ("=", _op_equal),
    (">", _ordered(lambda result: result > 0)),
    ("<", _ordered(lambda result: result < 0)),
)


@dataclass(frozen=True)
class FilterTerm:
    """
    One parsed filter condition.

    Attributes
    ----------
    field : str
        Field name or dotted path.
    operator : str
        An operator token from :data:`OPERATORS`, or ``""`` for a bare
        presence test.
    operand : str
        The literal to compare against, or ``""`` for a presence test.
    negated : bool
        Whether the term was written with a leading ``!`` (presence tests
        only).
    """

    field: str
    operator: str = ""
    operand: str = ""
    negated: bool = False

    def matches(self, record: Mapping[str, Any]) -> bool:
        """
        Test one record against this term.

        Parameters
        ----------
        record : mapping
            The record to test.

        Returns
        -------
        bool
            Whether the record satisfies the term. A missing field satisfies
            only a negated presence test: no other comparison can be
            truthfully asserted about a value that is not there.
        """
        value = get_field(record, self.field)
        if not self.operator:
            present = value not in (None, "", [], {}, False)
            return present != self.negated
        if value is None:
            return False
        for token, function in OPERATORS:
            if token == self.operator:
                return function(value, self.operand)
        raise FilterError(f"unknown operator {self.operator!r}")


def _split_terms(text: str) -> list[str]:
    """
    Split a filter expression on commas that are not inside quotes.

    Parameters
    ----------
    text : str
        The raw filter expression.

    Returns
    -------
    list of str
        Trimmed, non-empty terms.

    Notes
    -----
    Quote-aware so that a literal containing a comma -- ``title~"a, b"`` --
    stays one term. Splitting naively would silently produce two nonsense
    conditions instead of one correct one.
    """
    terms: list[str] = []
    current: list[str] = []
    quote = ""
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
            current.append(char)
            continue
        if char == ",":
            terms.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if quote:
        raise FilterError(f"unbalanced {quote!r} quote in filter {text!r}")
    terms.append("".join(current).strip())
    return [term for term in terms if term]


def parse_filter(text: str) -> list[FilterTerm]:
    """
    Parse a filter expression into terms combined with AND.

    Parameters
    ----------
    text : str
        Comma-separated terms. Each is ``field``, ``!field``, or
        ``field OP value`` where ``OP`` is one of ``=``, ``!=``, ``~``
        (contains), ``!~``, ``^`` (starts with), ``$`` (ends with), ``:``
        (any of, ``|``-separated), ``>``, ``>=``, ``<``, ``<=``.

    Returns
    -------
    list of FilterTerm
        The parsed terms. An empty expression yields an empty list, which
        selects everything.

    Raises
    ------
    FilterError
        If a term names no field or uses no recognised operator.

    Examples
    --------
    >>> [
    ...     (t.field, t.operator, t.operand)
    ...     for t in parse_filter("category=tutorial, stars>100")
    ... ]
    [('category', '=', 'tutorial'), ('stars', '>', '100')]
    >>> term = parse_filter("!deprecated")[0]
    >>> term.field, term.negated
    ('deprecated', True)
    >>> [t.operand for t in parse_filter('title~"a, b"')]
    ['a, b']
    """
    terms: list[FilterTerm] = []
    operator_tokens = "|".join(re.escape(token) for token, _ in OPERATORS)
    expression = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*)"
        r"\s*(" + operator_tokens + r")(.*)$"
    )
    for raw in _split_terms(text or ""):
        match = expression.fullmatch(raw)
        if match:
            field_name, token, operand = match.groups()
            operand = operand.strip()
            _len = len(operand) >= 2  # ruff: ignore[magic-value-comparison]
            quoted = _len and operand[0] in "\"'" and operand[-1] == operand[0]
            if quoted:
                operand = operand[1:-1]
            elif not operand or operand[0] in "=<>~^$:!":
                raise FilterError(
                    f"filter term {raw!r}: expected a value after {token!r}; "
                    "quote empty values or values beginning with operator characters"
                )
            terms.append(FilterTerm(field_name, token, operand))
            continue
        negated = raw.startswith("!")
        field_name = raw[1:].strip() if negated else raw
        if not _FIELD_RE.fullmatch(field_name):
            raise FilterError(
                f"cannot parse filter term {raw!r}; expected field OP value"
            )
        terms.append(FilterTerm(field_name, negated=negated))
    return terms


def parse_sort(text: str) -> list[tuple[str, bool]]:
    """
    Parse a sort specification into ordered (field, descending) pairs.

    Parameters
    ----------
    text : str
        Comma-separated field names, each optionally prefixed with ``-``
        for descending order.

    Returns
    -------
    list of (str, bool)
        Sort keys in priority order.

    Raises
    ------
    FilterError
        If a key is empty.

    Examples
    --------
    >>> parse_sort("category,-published")
    [('category', False), ('published', True)]
    >>> parse_sort("")
    []
    """
    keys: list[tuple[str, bool]] = []
    for raw in (text or "").split(","):
        token = raw.strip()
        if not token:
            continue
        descending = token.startswith("-")
        name = token[1:].strip() if descending else token
        if not _FIELD_RE.match(name):
            raise FilterError(
                f"sort key {raw!r} is not a valid field name (letters, "
                f"digits, '_', '-', and '.' for nested fields)"
            )
        keys.append((name, descending))
    return keys


# -- selection ----------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """
    A declarative selection over a collection of records.

    All fields are optional; an empty :class:`Selection` returns every
    record in source order.

    Attributes
    ----------
    terms : sequence of FilterTerm
        Conditions combined with AND.
    sort_keys : sequence of (str, bool)
        Sort fields with their descending flags, in priority order.
    group_by : str
        Field to partition on, or ``""`` for a single ungrouped section.
    limit : int or None
        Maximum records to return, applied after sorting.
    offset : int
        Records to skip, applied after sorting and before ``limit``.
    """

    terms: Sequence[FilterTerm] = ()
    sort_keys: Sequence[tuple[str, bool]] = ()
    group_by: str = ""
    limit: int | None = None
    offset: int = 0

    def __post_init__(self) -> None:
        """
        Validate pagination bounds eagerly.

        Raises
        ------
        FilterError
            If ``limit`` or ``offset`` is negative.
        """
        if self.limit is not None and self.limit < 0:
            raise FilterError(f"limit must not be negative, got {self.limit}")
        if self.offset < 0:
            raise FilterError(f"offset must not be negative, got {self.offset}")

    @classmethod
    def from_text(
        cls,
        *,
        filter_text: str = "",
        sort_text: str = "",
        group_by: str = "",
        limit: int | None = None,
        offset: int = 0,
    ) -> Selection:
        """
        Build a selection from directive-option strings.

        Parameters
        ----------
        filter_text : str, optional
            Value of a ``:filter:`` option.
        sort_text : str, optional
            Value of a ``:sort:`` option.
        group_by : str, optional
            Value of a ``:group-by:`` option.
        limit, offset : int, optional
            Pagination bounds.

        Returns
        -------
        Selection
            The validated selection.

        Raises
        ------
        FilterError
            If any specification is malformed.
        """
        return cls(
            terms=parse_filter(filter_text),
            sort_keys=parse_sort(sort_text),
            group_by=(group_by or "").strip(),
            limit=limit,
            offset=offset,
        )


def _sorted(records: list[Mapping[str, Any]], selection: Selection) -> list:
    """
    Sort records totally and stably.

    Parameters
    ----------
    records : list of mapping
        Records in source order.
    selection : Selection
        Supplies the sort keys.

    Returns
    -------
    list of mapping
        A new, sorted list. Sort keys are applied by successive stable
        sorts in reverse priority order -- the standard technique for
        multi-key ordering -- and records missing a key are held out and
        appended, so they land last regardless of direction.
    """
    ordered = list(records)
    for name, descending in reversed(list(selection.sort_keys)):
        present = [r for r in ordered if get_field(r, name) is not None]
        absent = [r for r in ordered if get_field(r, name) is None]
        key = _comparable([get_field(r, name) for r in present])
        present.sort(key=lambda r: key(get_field(r, name)), reverse=descending)
        ordered = present + absent
    return ordered


def apply_selection(
    records: Iterable[Mapping[str, Any]], selection: Selection
) -> tuple[list[Mapping[str, Any]], int]:
    """
    Filter, sort and paginate a collection.

    Parameters
    ----------
    records : iterable of mapping
        The full collection.
    selection : Selection
        The selection to apply.

    Returns
    -------
    selected : list of mapping
        Records after filtering, sorting, ``offset`` and ``limit``.
    total : int
        How many records matched the filters *before* pagination, so a
        caller can honestly report "showing 24 of 1043" rather than
        silently truncating.

    Examples
    --------
    >>> data = [
    ...     {"title": "Beta", "stars": 30, "kind": "demo"},
    ...     {"title": "alpha", "stars": 200, "kind": "tutorial"},
    ...     {"title": "Gamma", "kind": "tutorial"},
    ... ]
    >>> chosen, total = apply_selection(data, Selection.from_text(sort_text="title"))
    >>> [r["title"] for r in chosen]
    ['alpha', 'Beta', 'Gamma']
    >>> chosen, total = apply_selection(
    ...     data, Selection.from_text(filter_text="kind=tutorial, stars>100")
    ... )
    >>> [r["title"] for r in chosen], total
    (['alpha'], 1)
    >>> chosen, _ = apply_selection(data, Selection.from_text(sort_text="-stars"))
    >>> [r["title"] for r in chosen]
    ['alpha', 'Beta', 'Gamma']
    """
    matched = [
        record
        for record in records
        if all(term.matches(record) for term in selection.terms)
    ]
    total = len(matched)
    window = _sorted(matched, selection)[selection.offset :]
    if selection.limit is not None:
        window = window[: selection.limit]
    return window, total


def group_records(
    records: Sequence[Mapping[str, Any]], selection: Selection
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """
    Partition records into labelled sections.

    Parameters
    ----------
    records : sequence of mapping
        Already filtered, sorted and paginated records.
    selection : Selection
        Supplies ``group_by``.

    Returns
    -------
    list of (str, list of mapping)
        Section label paired with its records. With no ``group_by`` this is
        a single ``("", records)`` pair, so callers have exactly one code
        path for grouped and ungrouped rendering.

        Section order follows first appearance, so it is governed by the
        same ``sort`` the reader asked for rather than by a second, hidden
        rule. :data:`UNGROUPED_LABEL` is always placed last when present,
        because a catch-all section reads as a footnote, not a heading.

        A record whose grouping field is a *list* appears in one section per
        element. That is deliberate: grouping a collection by ``tags`` where
        an item carries three tags has no other sensible reading, and
        silently picking the first tag would hide items from two of the
        three sections a reader is looking at.

    Examples
    --------
    >>> data = [
    ...     {"n": 1, "kind": "demo"},
    ...     {"n": 2},
    ...     {"n": 3, "kind": "demo"},
    ... ]
    >>> [
    ...     (label, len(rs))
    ...     for label, rs in group_records(data, Selection.from_text(group_by="kind"))
    ... ]
    [('demo', 2), ('Ungrouped', 1)]
    >>> tagged = [{"n": 1, "tags": ["a", "b"]}, {"n": 2, "tags": ["b"]}]
    >>> [
    ...     (label, len(rs))
    ...     for label, rs in group_records(tagged, Selection.from_text(group_by="tags"))
    ... ]
    [('a', 1), ('b', 2)]
    """
    if not selection.group_by:
        return [("", list(records))]

    sections: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        value = get_field(record, selection.group_by)
        labels = [
            str(atom).strip() if atom is not None else "" for atom in _values_of(value)
        ]
        # A repeated tag must not duplicate its card. Zero/False are valid groups.
        labels = list(dict.fromkeys(label or UNGROUPED_LABEL for label in labels))
        for label in labels or [UNGROUPED_LABEL]:
            sections.setdefault(label, []).append(record)

    ordered = [(k, v) for k, v in sections.items() if k != UNGROUPED_LABEL]
    if UNGROUPED_LABEL in sections:
        ordered.append((UNGROUPED_LABEL, sections[UNGROUPED_LABEL]))
    return ordered
