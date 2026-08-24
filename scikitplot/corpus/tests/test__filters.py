# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`scikitplot.corpus._filters` — the filter AST (F-R07-02)."""

from __future__ import annotations

import json

import pytest

from .._filters import (
    ALL_OPERATORS,
    And,
    Eq,
    Exists,
    FilterCapability,
    In,
    Not,
    NotEq,
    NotIn,
    Or,
    Range,
    UnsupportedOperatorError,
)
from .._schema import CorpusDocument

__all__: "list[str]" = [
    "TestOperators",
    "TestMissingFieldSemantics",
    "TestComposition",
    "TestCapabilityResponse",
]


@pytest.fixture(name="doc")
def _doc() -> CorpusDocument:
    return CorpusDocument.create(
        input_path="a.txt", chunk_index=5, text="hello", language="en"
    )


class TestOperators:
    """Each of the nine operators."""

    def test_eq(self, doc) -> None:
        assert Eq("language", "en").matches(doc)
        assert not Eq("language", "fr").matches(doc)

    def test_eq_compares_enums_by_value(self, doc) -> None:
        """Filters must be shape-independent about enum vs raw value."""
        assert Eq("source_type", "unknown").matches(doc)

    def test_not_eq(self, doc) -> None:
        assert NotEq("language", "fr").matches(doc)
        assert not NotEq("language", "en").matches(doc)

    def test_in_and_not_in(self, doc) -> None:
        assert In("chunk_index", [1, 5, 9]).matches(doc)
        assert not In("chunk_index", [1, 9]).matches(doc)
        assert NotIn("chunk_index", [1, 9]).matches(doc)
        assert not NotIn("chunk_index", [5]).matches(doc)

    @pytest.mark.parametrize(
        ("lo", "hi", "expected"),
        [(0, 100, True), (10, None, False), (None, 3, False), (5, 5, True)],
    )
    def test_range_bounds_are_inclusive(self, doc, lo, hi, expected) -> None:
        assert Range("chunk_index", lo=lo, hi=hi).matches(doc) is expected

    def test_range_on_incomparable_type_excludes_rather_than_raises(self, doc) -> None:
        """
        One badly-typed field should exclude that document, not the query.

        A filter is a predicate over heterogeneous documents; raising would make
        a single odd record fail an entire corpus scan.
        """
        assert Range("language", lo=0).matches(doc) is False

    def test_exists(self, doc) -> None:
        assert Exists("language").matches(doc)
        assert not Exists("parent_doc_id").matches(doc)  # present but None
        assert not Exists("no_such_field").matches(doc)


class TestMissingFieldSemantics:
    """Absence is not inequality."""

    def test_not_eq_does_not_match_a_missing_field(self, doc) -> None:
        """
        ``NotEq`` on an absent field must be False.

        Treating absence as inequality would make ``NotEq`` match documents
        that have no such field at all -- almost never what a caller means.
        ``Not(Exists(field))`` asks that question deliberately.
        """
        assert not NotEq("no_such_field", "x").matches(doc)
        assert Not(Exists("no_such_field")).matches(doc)

    def test_not_in_does_not_match_a_missing_field(self, doc) -> None:
        assert not NotIn("no_such_field", ["x"]).matches(doc)


class TestComposition:
    """The operators F-R07-02 recorded as unexpressible at any price."""

    def test_explicit_and(self, doc) -> None:
        assert And(Eq("language", "en"), Range("chunk_index", lo=0)).matches(doc)

    def test_disjunction(self, doc) -> None:
        assert Or(Eq("language", "fr"), Eq("language", "en")).matches(doc)
        assert not Or(Eq("language", "fr"), Eq("language", "de")).matches(doc)

    def test_negation(self, doc) -> None:
        assert Not(Eq("language", "fr")).matches(doc)

    def test_empty_and_matches_everything(self, doc) -> None:
        assert And().matches(doc)

    def test_empty_or_matches_nothing(self, doc) -> None:
        assert not Or().matches(doc)

    def test_deep_nesting(self, doc) -> None:
        expr = And(
            Eq("language", "en"),
            Or(Eq("source_type", "book"), Eq("source_type", "unknown")),
            Range("chunk_index", lo=0, hi=100),
            Not(Exists("parent_doc_id")),
        )
        assert expr.matches(doc)

    def test_operators_reports_the_whole_tree(self) -> None:
        expr = And(Eq("a", 1), Or(NotIn("b", [2]), Not(Exists("c"))))
        assert expr.operators() == {"And", "Eq", "Or", "NotIn", "Not", "Exists"}

    def test_every_operator_is_reachable(self) -> None:
        expr = And(
            Eq("a", 1),
            NotEq("b", 2),
            In("c", [3]),
            NotIn("d", [4]),
            Range("e", lo=0),
            Or(Exists("f")),
            Not(Exists("g")),
        )
        assert expr.operators() == set(ALL_OPERATORS)


class TestCapabilityResponse:
    """ADR-R07-002 — never silently ignore a filter."""

    def test_operators_are_partitioned_exactly(self) -> None:
        expr = And(Eq("a", 1), Or(Exists("b")), Not(Exists("c")))
        cap = FilterCapability.for_expression(expr, native=["Eq", "And"])
        union = cap.supported | cap.emulated | cap.rejected
        assert union == expr.operators()
        assert not (cap.supported & cap.emulated)
        assert not (cap.supported & cap.rejected)

    def test_unknown_operators_default_to_rejected(self) -> None:
        """Refusal is the default; silent omission is unreachable."""
        expr = And(Eq("a", 1), Or(Exists("b")))
        cap = FilterCapability.for_expression(expr, native=["Eq"])
        assert "Or" in cap.rejected and "Exists" in cap.rejected
        assert not cap.ok

    def test_emulated_is_first_class(self) -> None:
        expr = Range("a", lo=0)
        cap = FilterCapability.for_expression(expr, native=[], emulated=["Range"])
        assert cap.emulated == frozenset({"Range"})
        assert cap.ok  # emulated still answers the filter

    def test_native_wins_over_emulated(self) -> None:
        cap = FilterCapability.for_expression(
            Eq("a", 1), native=["Eq"], emulated=["Eq"]
        )
        assert cap.supported == frozenset({"Eq"})
        assert cap.emulated == frozenset()

    def test_raise_if_rejected_names_the_operators(self) -> None:
        cap = FilterCapability.for_expression(Or(Exists("b")), native=[])
        with pytest.raises(UnsupportedOperatorError, match="Exists"):
            cap.raise_if_rejected()

    def test_full_support_does_not_raise(self) -> None:
        expr = And(Eq("a", 1))
        FilterCapability.for_expression(
            expr, native=sorted(expr.operators())
        ).raise_if_rejected()

    def test_capability_is_serialisable(self) -> None:
        cap = FilterCapability.for_expression(Eq("a", 1), native=["Eq"])
        assert json.dumps(cap.to_dict())
