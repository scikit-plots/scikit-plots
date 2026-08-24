# scikitplot/corpus/_filters.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""A backend-neutral filter expression tree.

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
    "UnsupportedOperatorError",
]

_MISSING = object()


class UnsupportedOperatorError(ValueError):
    """Raised when a backend is asked to evaluate an operator it rejects."""


def _read(document: Any, field: str) -> Any:
    """Read ``field`` from a document or mapping, or ``_MISSING``."""
    if isinstance(document, dict):
        return document.get(field, _MISSING)
    return getattr(document, field, _MISSING)


def _plain(value: Any) -> Any:
    """Reduce enums to their values so comparisons are shape-independent."""
    return getattr(value, "value", value)


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


@dataclasses.dataclass(frozen=True)
class Eq(Filter):
    """``field == value``."""

    field: str
    value: Any

    def matches(self, document: Any) -> bool:
        """Whether the field equals :attr:`value`."""
        actual = _read(document, self.field)
        return actual is not _MISSING and _plain(actual) == _plain(self.value)

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
        return actual is not _MISSING and _plain(actual) != _plain(self.value)

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

    def matches(self, document: Any) -> bool:
        """Whether the field's value is among :attr:`values`."""
        actual = _read(document, self.field)
        if actual is _MISSING:
            return False
        return _plain(actual) in {_plain(v) for v in self.values}

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

    def matches(self, document: Any) -> bool:
        """Whether the field is present and not among :attr:`values`."""
        actual = _read(document, self.field)
        if actual is _MISSING:
            return False
        return _plain(actual) not in {_plain(v) for v in self.values}

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
