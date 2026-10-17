# scikitplot/corpus/_filters.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A backend-neutral filter expression tree.

Nine composable operators -- ``Eq``, ``NotEq``, ``In``, ``NotIn``, ``Range``,
``Exists``, ``And``, ``Or``, ``Not`` -- with a **mandatory per-operator
capability response** from every backend that evaluates them.

Notes
-----
**User-focused.**

.. code-block:: python

    from scikitplot.corpus import And, Eq, Exists, Not, Or, Range

    expr = And(
        Eq("language", "en"),
        Or(Eq("source_type", "book"), Eq("source_type", "paper")),
        Range("chunk_index", lo=0, hi=100),
        Not(Exists("parent_doc_id")),
    )
    expr.matches(document)  # evaluate locally
    expr.operators()  # {'And', 'Eq', 'Exists', 'Not', 'Or', 'Range'}

**Developer-focused.**  ``StorageQuery`` was a fixed-field filter: eight fields,
six of them implicit-``AND`` equality checks.  Of the nine operators only ``Eq``
was expressible, and only on the six fields someone had anticipated; ``And`` was
implicit and never explicit; disjunction and negation were unexpressible at any
price (finding F-R07-02).  Adding a filterable attribute meant changing the
dataclass *and every backend*.

Two rules carry over from the retrofit in :mod:`scikitplot.corpus._storage`:

*Never silently ignore a filter.*  A backend answers ``SUPPORTED``,
``EMULATED`` or ``REJECTED`` per operator, and the answer is part of the result
rather than an optional courtesy.  An unsupported filter that quietly returns
everything *over*-reports, which looks like success -- worse than an empty
result, which a caller tends to notice (finding F-R07-01).

*``EMULATED`` is first-class.*  A backend that can answer an operator by
scanning genuinely can answer it; the honest report is "answered, but not
natively", because the two differ in cost and sometimes in ranking.

See Also
--------
scikitplot.corpus._storage._storage.FilterSupport : the three-way response.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Iterable

__all__: list[str] = [
    "And",
    "Eq",
    "Exists",
    "Filter",
    "FilterCapability",
    "In",
    "Not",
    "NotEq",
    "NotIn",
    "Or",
    "Range",
    "SecurityFilter",
    "UnsupportedOperatorError",
    "filter_from_dict",
    "filter_to_dict",
    "filterable_fields",
    "validate_filter",
]

_MISSING = object()


class UnsupportedOperatorError(ValueError):
    """Raised when a backend is asked to evaluate an operator it rejects."""


#: Prefix of a field that addresses one key of ``CorpusDocument.metadata``.
METADATA_PREFIX = "metadata."


def _read(document: Any, field: str) -> Any:
    """Read ``field`` from a document or mapping, or ``_MISSING``.

    ``metadata.<key>`` reads one key of the document's ``metadata`` mapping.
    """
    if isinstance(document, dict) and field in document:
        return document[field]
    if field.startswith(METADATA_PREFIX):
        meta = (
            document.get("metadata")
            if isinstance(document, dict)
            else getattr(document, "metadata", None)
        )
        if not isinstance(meta, dict):
            return _MISSING
        return meta.get(field[len(METADATA_PREFIX) :], _MISSING)
    if isinstance(document, dict):
        return _MISSING
    return getattr(document, field, _MISSING)


def _plain(value: Any) -> Any:
    """Reduce enums to their values so comparisons are shape-independent."""
    return getattr(value, "value", value)


def _key(value: Any) -> tuple[str, Any]:
    """Return a comparison key that keeps ``True`` apart from ``1``."""
    value = _plain(value)
    return ("bool", value) if isinstance(value, bool) else ("value", value)


def _equal(actual: Any, expected: Any) -> bool:
    """Whether two values are equal; a value that cannot be compared is not.

    Notes
    -----
    A field holding an array, a mapping or another object whose ``==`` does not
    return a ``bool`` makes the leaf not match instead of raising: a filter is
    a predicate over heterogeneous documents (the rule ``Range`` already had).
    """
    try:
        result = _key(actual) == _key(expected)
    except (TypeError, ValueError):
        return False
    return result is True


def _lookup(values: tuple[Any, ...]) -> frozenset | None:
    """Return the comparison keys of ``values`` once, or ``None`` if unhashable."""
    try:
        return frozenset(_key(v) for v in values)
    except TypeError:
        return None


def _member(actual: Any, values: tuple[Any, ...], lookup: frozenset | None) -> bool:
    """Whether ``actual`` equals one of ``values`` (precomputed when possible)."""
    if lookup is not None:
        try:
            return _key(actual) in lookup
        except TypeError:
            return False
    return any(_equal(actual, v) for v in values)


class Filter:
    """Base class for filter expressions.

    Notes
    -----
    **Developer.**  Subclasses implement :meth:`matches` and :meth:`operators`.
    Composition uses the explicit :class:`And` / :class:`Or` / :class:`Not`
    nodes rather than operator overloading: ``&`` and ``|`` read as bitwise
    operations and their precedence relative to comparisons is a classic source
    of silently wrong predicates.
    """

    def matches(self, document: Any) -> bool:
        """Whether ``document`` satisfies this expression."""
        raise NotImplementedError

    def operators(self) -> set[str]:
        """Names of every operator appearing in this expression."""
        raise NotImplementedError

    def describe(self) -> str:
        """Return a readable rendering of the expression."""
        return repr(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form; see :func:`filter_to_dict`."""
        return filter_to_dict(self)


@dataclasses.dataclass(frozen=True)
class Eq(Filter):
    """``field == value``."""

    field: str
    value: Any

    def matches(self, document: Any) -> bool:
        """Whether the field equals :attr:`value`."""
        actual = _read(document, self.field)
        return actual is not _MISSING and _equal(actual, self.value)

    def operators(self) -> set[str]:
        """Return ``{'Eq'}``."""
        return {"Eq"}


@dataclasses.dataclass(frozen=True)
class NotEq(Filter):
    """``field != value``.

    Notes
    -----
    A **missing** field does not satisfy ``NotEq``.  Treating absence as
    inequality would make ``NotEq`` match documents that have no such field at
    all, which is almost never what a caller means; use ``Not(Exists(field))``
    to ask that question deliberately.
    """

    field: str
    value: Any

    def matches(self, document: Any) -> bool:
        """Whether the field is present and differs from :attr:`value`."""
        actual = _read(document, self.field)
        if actual is _MISSING:
            return False
        try:
            different = _key(actual) != _key(self.value)
        except (TypeError, ValueError):
            return False
        return different is True

    def operators(self) -> set[str]:
        """Return ``{'NotEq'}``."""
        return {"NotEq"}


@dataclasses.dataclass(frozen=True)
class In(Filter):
    """``field in values``."""

    field: str
    values: tuple[Any, ...]

    def __init__(self, field: str, values: Iterable[Any]) -> None:
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "_lookup", _lookup(self.values))

    def matches(self, document: Any) -> bool:
        """Whether the field's value is among :attr:`values`."""
        actual = _read(document, self.field)
        if actual is _MISSING:
            return False
        return _member(actual, self.values, self._lookup)  # type: ignore[attr-defined]

    def operators(self) -> set[str]:
        """Return ``{'In'}``."""
        return {"In"}


@dataclasses.dataclass(frozen=True)
class NotIn(Filter):
    """``field not in values``; a missing field does not match."""

    field: str
    values: tuple[Any, ...]

    def __init__(self, field: str, values: Iterable[Any]) -> None:
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "_lookup", _lookup(self.values))

    def matches(self, document: Any) -> bool:
        """Whether the field is present, comparable and not among :attr:`values`."""
        actual = _read(document, self.field)
        if actual is _MISSING:
            return False
        try:
            hash(_key(actual))
        except TypeError:
            return False
        return not _member(actual, self.values, self._lookup)  # type: ignore[attr-defined]

    def operators(self) -> set[str]:
        """Return ``{'NotIn'}``."""
        return {"NotIn"}


@dataclasses.dataclass(frozen=True)
class Range(Filter):
    """``lo <= field <= hi``, with either bound optional.

    Parameters
    ----------
    field : str
    lo : Any, optional
        Inclusive lower bound. ``None`` leaves the range open below.
    hi : Any, optional
        Inclusive upper bound. ``None`` leaves the range open above.
    """

    field: str
    lo: Any = None
    hi: Any = None

    def matches(self, document: Any) -> bool:
        """Whether the field lies within the bounds.

        Notes
        -----
        A value that cannot be compared with a bound does **not** match, rather
        than raising: a filter is a predicate over heterogeneous documents, and
        one badly-typed field should exclude that document, not the query.
        """
        actual = _read(document, self.field)
        if actual is _MISSING or actual is None:
            return False
        actual = _plain(actual)
        try:
            if self.lo is not None and actual < _plain(self.lo):
                return False
            if self.hi is not None and actual > _plain(self.hi):
                return False
        except TypeError:
            return False
        return True

    def operators(self) -> set[str]:
        """Return ``{'Range'}``."""
        return {"Range"}


@dataclasses.dataclass(frozen=True)
class Exists(Filter):
    """The field is present and not ``None``."""

    field: str

    def matches(self, document: Any) -> bool:
        """Whether the field is present and non-``None``."""
        actual = _read(document, self.field)
        return actual is not _MISSING and actual is not None

    def operators(self) -> set[str]:
        """Return ``{'Exists'}``."""
        return {"Exists"}


@dataclasses.dataclass(frozen=True)
class And(Filter):
    """All sub-expressions must match.  Empty ``And`` matches everything."""

    clauses: tuple[Filter, ...]

    def __init__(self, *clauses: Filter) -> None:
        object.__setattr__(self, "clauses", tuple(clauses))

    def matches(self, document: Any) -> bool:
        """Whether every clause matches."""
        return all(clause.matches(document) for clause in self.clauses)

    def operators(self) -> set[str]:
        """Return ``{'And'}`` plus every nested operator."""
        found = {"And"}
        for clause in self.clauses:
            found |= clause.operators()
        return found


@dataclasses.dataclass(frozen=True)
class Or(Filter):
    """At least one sub-expression must match.  Empty ``Or`` matches nothing."""

    clauses: tuple[Filter, ...]

    def __init__(self, *clauses: Filter) -> None:
        object.__setattr__(self, "clauses", tuple(clauses))

    def matches(self, document: Any) -> bool:
        """Whether any clause matches."""
        return any(clause.matches(document) for clause in self.clauses)

    def operators(self) -> set[str]:
        """Return ``{'Or'}`` plus every nested operator."""
        found = {"Or"}
        for clause in self.clauses:
            found |= clause.operators()
        return found


@dataclasses.dataclass(frozen=True)
class Not(Filter):
    """Negates a sub-expression."""

    clause: Filter

    def matches(self, document: Any) -> bool:
        """Whether the inner clause does not match."""
        return not self.clause.matches(document)

    def operators(self) -> set[str]:
        """Return ``{'Not'}`` plus every nested operator."""
        return {"Not"} | self.clause.operators()


#: Every operator the AST defines.
ALL_OPERATORS = frozenset(
    {"Eq", "NotEq", "In", "NotIn", "Range", "Exists", "And", "Or", "Not"}
)


@dataclasses.dataclass(frozen=True)
class FilterCapability:
    """A backend's declared answer for the operators in one expression.

    Parameters
    ----------
    supported : frozenset of str
        Executed natively.
    emulated : frozenset of str
        Executed by a fallback path with equivalent semantics.
    rejected : frozenset of str
        Refused; the query does not run.

    Notes
    -----
    **Developer.**  Every operator in an expression must appear in exactly one
    set.  :meth:`for_expression` enforces that, so "silently ignored" is not a
    reachable state rather than merely a discouraged one.
    """

    supported: frozenset[str] = frozenset()
    emulated: frozenset[str] = frozenset()
    rejected: frozenset[str] = frozenset()

    @classmethod
    def for_expression(
        cls,
        expression: Filter,
        *,
        native: Iterable[str] = (),
        emulated: Iterable[str] = (),
    ) -> FilterCapability:
        """Classify every operator in ``expression``.

        Parameters
        ----------
        expression : Filter
            Filter
        native : iterable of str, optional
            Operators this backend executes natively.
        emulated : iterable of str, optional
            Operators it answers by a fallback path.

        Returns
        -------
        FilterCapability
            Anything neither native nor emulated is ``rejected`` -- the default
            is refusal, never silent omission.
        """
        used = expression.operators()
        native_set = used & set(native)
        emulated_set = (used & set(emulated)) - native_set
        return cls(
            supported=frozenset(native_set),
            emulated=frozenset(emulated_set),
            rejected=frozenset(used - native_set - emulated_set),
        )

    @property
    def ok(self) -> bool:
        """Whether every operator can be answered."""
        return not self.rejected

    def raise_if_rejected(self) -> None:
        """Raise if any operator was refused.

        Raises
        ------
        UnsupportedOperatorError
            Naming the refused operators.
        """
        if self.rejected:
            raise UnsupportedOperatorError(
                f"backend cannot evaluate operator(s) {sorted(self.rejected)}; "
                "refusing to run a query whose filter would be only partly "
                "applied. A partly-applied filter returns more than it should, "
                "which looks like success."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "supported": sorted(self.supported),
            "emulated": sorted(self.emulated),
            "rejected": sorted(self.rejected),
        }


# ---------------------------------------------------------------------------
# JSON form (R2-2): so MCP tools and agent tool calls can carry a filter
# ---------------------------------------------------------------------------

#: Limits on a filter read from JSON, which may come from an untrusted caller.
#: Evaluation costs O(nodes x documents), so the node budget bounds query cost.
MAX_FILTER_DEPTH = 32
MAX_FILTER_NODES = 256
MAX_FILTER_VALUES = 4096
MAX_FILTER_STRING = 4096
MAX_FILTER_INT = 2**63

#: Document fields a filter may name: the scalar fields of ``CorpusDocument``.
#: Arrays and raw payloads have no meaningful equality; ``metadata`` is reached
#: one key at a time as ``metadata.<key>``.
_UNFILTERABLE = frozenset({"embedding", "raw_tensor", "raw_bytes", "metadata"})


def filterable_fields() -> frozenset[str]:
    """
    Return the document fields a filter may name.

    Returns
    -------
    frozenset of str
        The fields of :class:`~scikitplot.corpus.CorpusDocument` except
        ``embedding``, ``raw_tensor``, ``raw_bytes`` and ``metadata``; one
        metadata key is addressed as ``"metadata.<key>"``.
    """
    from ._schema import CorpusDocument  # noqa: PLC0415

    return frozenset(f.name for f in dataclasses.fields(CorpusDocument)) - _UNFILTERABLE


def _check_field(field: Any, where: str) -> str:
    """Return ``field`` if a filter may name it; raise naming the path."""
    if not isinstance(field, str) or not field:
        raise ValueError(f"{where}: must be a non-empty str")
    if len(field) > MAX_FILTER_STRING:
        raise ValueError(f"{where}: longer than {MAX_FILTER_STRING} characters")
    if field.startswith(METADATA_PREFIX):
        key = field[len(METADATA_PREFIX) :]
        if not key:
            raise ValueError(f"{where}: 'metadata.' needs a key")
        return field
    if field not in filterable_fields():
        raise ValueError(
            f"{where}: unknown field {field!r}; a filter may name a scalar "
            "CorpusDocument field (filterable_fields()) or 'metadata.<key>'"
        )
    return field


_LEAF_KEYS = {
    "eq": ("field", "value"),
    "not_eq": ("field", "value"),
    "in": ("field", "values"),
    "not_in": ("field", "values"),
    "range": ("field", "lo", "hi"),
    "exists": ("field",),
}
_OPS = {**dict.fromkeys(_LEAF_KEYS), "and": None, "or": None, "not": None}


def _json_scalar(value: Any, where: str) -> Any:
    """Return ``value`` as a JSON scalar, refusing anything else."""
    value = _plain(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > MAX_FILTER_STRING:
            raise ValueError(f"{where}: longer than {MAX_FILTER_STRING} characters")
        return value
    if isinstance(value, int):
        if abs(value) > MAX_FILTER_INT:
            raise ValueError(f"{where}: integer beyond +/-2**63")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: a filter value must be finite, got {value!r}")
        return value
    raise ValueError(
        f"{where}: a filter value must be a JSON scalar (str, int, float, bool or "
        f"null), got {type(value).__name__}"
    )


def filter_to_dict(expression: Filter) -> dict[str, Any]:
    """
    Return the JSON form of a filter expression.

    Parameters
    ----------
    expression : Filter
        Any expression of the nine operators.

    Returns
    -------
    dict
        ``{"op": "eq", "field": ..., "value": ...}``, ``{"op": "in", "field":
        ..., "values": [...]}``, ``{"op": "range", "field": ..., "lo": ...,
        "hi": ...}``, ``{"op": "exists", "field": ...}``, ``{"op": "and",
        "args": [...]}``, ``{"op": "or", "args": [...]}``, ``{"op": "not",
        "arg": {...}}``; ``not_eq`` and ``not_in`` like ``eq`` and ``in``.
        Enum values are written as their values.

    Raises
    ------
    ValueError
        If a value is not a JSON scalar (or is not finite), naming its path.
    TypeError
        If ``expression`` is not one of the nine operators.

    Examples
    --------
    >>> filter_to_dict(And(Eq("language", "en"), Not(Exists("parent_doc_id"))))
    {'op': 'and', 'args': [{'op': 'eq', 'field': 'language', 'value': 'en'}, {'op': 'not', 'arg': {'op': 'exists', 'field': 'parent_doc_id'}}]}
    """

    def encode(node: Any, where: str) -> dict[str, Any]:  # noqa: PLR0911 - one per op
        if isinstance(node, Eq):
            return {
                "op": "eq",
                "field": node.field,
                "value": _json_scalar(node.value, where),
            }
        if isinstance(node, NotEq):
            return {
                "op": "not_eq",
                "field": node.field,
                "value": _json_scalar(node.value, where),
            }
        if isinstance(node, (In, NotIn)):
            return {
                "op": "in" if isinstance(node, In) else "not_in",
                "field": node.field,
                "values": [
                    _json_scalar(v, f"{where}.values[{i}]")
                    for i, v in enumerate(node.values)
                ],
            }
        if isinstance(node, Range):
            return {
                "op": "range",
                "field": node.field,
                "lo": _json_scalar(node.lo, f"{where}.lo"),
                "hi": _json_scalar(node.hi, f"{where}.hi"),
            }
        if isinstance(node, Exists):
            return {"op": "exists", "field": node.field}
        if isinstance(node, (And, Or)):
            return {
                "op": "and" if isinstance(node, And) else "or",
                "args": [
                    encode(c, f"{where}.args[{i}]") for i, c in enumerate(node.clauses)
                ],
            }
        if isinstance(node, Not):
            return {"op": "not", "arg": encode(node.clause, f"{where}.arg")}
        raise TypeError(f"{where}: not a filter expression: {type(node).__name__}")

    return encode(expression, "filter")


def filter_from_dict(data: Any) -> Filter:
    """
    Rebuild a filter expression from its JSON form, refusing anything off contract.

    Parameters
    ----------
    data : dict
        As written by :func:`filter_to_dict`, possibly from an untrusted caller.

    Returns
    -------
    Filter
        The expression; ``filter_to_dict(filter_from_dict(d)) == d`` for every
        accepted ``d``.

    Raises
    ------
    ValueError
        For an unknown ``op``, a missing or unknown key, a field a filter may
        not name (see :func:`filterable_fields`; ``metadata.<key>`` is
        allowed), a value that is not a finite JSON scalar, a list where none
        belongs (or none where one does), a string longer than
        :data:`MAX_FILTER_STRING`, an integer beyond ``+/-2**63``, or a tree of
        more than :data:`MAX_FILTER_DEPTH` levels, :data:`MAX_FILTER_NODES`
        nodes or :data:`MAX_FILTER_VALUES` values. The message names the path.

    Examples
    --------
    >>> filter_from_dict({"op": "in", "field": "language", "values": ["en", "de"]})
    In(field='language', values=('en', 'de'))
    """
    budget = {"nodes": 0, "values": 0}

    def scalar(value: Any, where: str) -> Any:
        if isinstance(value, (list, dict, tuple)):
            # Untrusted JSON of the wrong shape is a bad value (documented).
            msg = f"{where}: a filter value must be a JSON scalar"
            raise ValueError(msg)  # noqa: TRY004
        return _json_scalar(value, where)

    def decode(node: Any, where: str, depth: int) -> Filter:
        if depth >= MAX_FILTER_DEPTH:
            raise ValueError(
                f"{where}: filter nested deeper than {MAX_FILTER_DEPTH} levels"
            )
        budget["nodes"] += 1
        if budget["nodes"] > MAX_FILTER_NODES:
            raise ValueError(f"{where}: filter has more than {MAX_FILTER_NODES} nodes")
        if not isinstance(node, dict):
            msg = f"{where}: a filter node must be an object"
            raise ValueError(msg)  # noqa: TRY004
        op = node.get("op")
        if not isinstance(op, str) or op not in _OPS:
            raise ValueError(
                f"{where}: unknown op {op!r}; expected one of {sorted(_OPS)}"
            )
        if op in ("and", "or"):
            expected = ("op", "args")
        elif op == "not":
            expected = ("op", "arg")
        else:
            expected = ("op", *_LEAF_KEYS[op])
        if set(node) != set(expected):
            raise ValueError(
                f"{where}: op {op!r} takes exactly the keys {sorted(expected)}, got "
                f"{sorted(map(str, node))}"
            )
        if op in ("and", "or"):
            args = node["args"]
            if not isinstance(args, list):
                raise ValueError(f"{where}.args: must be a list")
            clauses = [
                decode(a, f"{where}.args[{i}]", depth + 1) for i, a in enumerate(args)
            ]
            return And(*clauses) if op == "and" else Or(*clauses)
        if op == "not":
            return Not(decode(node["arg"], f"{where}.arg", depth + 1))
        return leaf(node, op, where)

    def leaf(node: dict[str, Any], op: str, where: str) -> Filter:
        field = _check_field(node["field"], f"{where}.field")
        if op in ("in", "not_in"):
            values = node["values"]
            if not isinstance(values, list):
                raise ValueError(f"{where}.values: must be a list")
            budget["values"] += len(values)
            if budget["values"] > MAX_FILTER_VALUES:
                raise ValueError(
                    f"{where}: filter has more than {MAX_FILTER_VALUES} values"
                )
            items = [scalar(v, f"{where}.values[{i}]") for i, v in enumerate(values)]
            return In(field, items) if op == "in" else NotIn(field, items)
        if op == "range":
            return Range(
                field,
                lo=scalar(node["lo"], f"{where}.lo"),
                hi=scalar(node["hi"], f"{where}.hi"),
            )
        if op == "exists":
            return Exists(field)
        value = scalar(node["value"], f"{where}.value")
        return Eq(field, value) if op == "eq" else NotEq(field, value)

    return decode(data, "filter", 0)


# ---------------------------------------------------------------------------
# Security filter (R2-2): a scope a caller can narrow but never widen
# ---------------------------------------------------------------------------


def validate_filter(expression: Any) -> Filter:
    """
    Return ``expression`` checked and frozen to the JSON-able form.

    Parameters
    ----------
    expression : Filter or mapping
        A filter expression, or its JSON form.

    Returns
    -------
    Filter
        A new tree of the nine built-in operators holding only JSON scalars
        and naming only filterable fields: ``filter_from_dict(filter_to_dict(
        expression))``. Nothing reachable from it can be mutated afterwards.

    Raises
    ------
    TypeError
        If ``expression`` is neither a :class:`Filter` nor a mapping, or holds
        a node that is not one of the nine operators.
    ValueError
        As :func:`filter_from_dict` and :func:`filter_to_dict`.
    """
    if isinstance(expression, Filter):
        data = filter_to_dict(expression)
    elif isinstance(expression, dict):
        data = expression
    else:
        raise TypeError(
            "a filter must be a Filter expression or its JSON form (a mapping), "
            f"got {type(expression).__name__}"
        )
    return filter_from_dict(data)


@dataclasses.dataclass(frozen=True)
class SecurityFilter:
    """
    A scope bound to an index and AND-ed with every query's own filter.

    Parameters
    ----------
    expression : Filter
        The documents a caller of this index may see.
    label : str, optional
        Name for reports. Default ``"security"``.

    Notes
    -----
    **User.** Bind it where the index is built (``RetrievalIndex(...,
    security_filter=...)``), not per query: a query's ``filter`` is AND-ed with
    it, so a caller (an MCP tool, an agent's tool call) can narrow the result
    but never widen it, and there is no argument through which to drop it.

    **Developer.** The expression is replaced by :func:`validate_filter`'s copy
    at construction: the nine built-in operators, JSON scalars only, filterable
    fields only. A misspelled field is refused here instead of reading as
    missing (which would make a deny-list scope ``Not(Eq(...))`` admit
    everything), and nothing reachable from the stored expression can be
    mutated by whoever built it.

    Examples
    --------
    >>> scope = SecurityFilter(Eq("collection_id", "public"))
    >>> scope.combine(Eq("language", "en"))
    And(clauses=(Eq(field='collection_id', value='public'), Eq(field='language', value='en')))
    """

    expression: Filter
    label: str = "security"

    def __post_init__(self) -> None:
        """Validate and freeze the expression; refuse anything else."""
        if not isinstance(self.expression, Filter):
            raise TypeError(
                f"SecurityFilter.expression must be a Filter, got "
                f"{type(self.expression).__name__}"
            )
        if not isinstance(self.label, str) or not self.label:
            raise TypeError("SecurityFilter.label must be a non-empty str")
        object.__setattr__(self, "expression", validate_filter(self.expression))

    def combine(self, expression: Filter | None) -> Filter:
        """Return the security expression AND-ed with ``expression``."""
        if expression is None:
            return self.expression
        return And(self.expression, expression)
