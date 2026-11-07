# scikitplot/corpus/tests/test__routing.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`scikitplot.corpus._routing` and the shapes :func:`route` adds.

The labelled routing set (``data/routing_set.json``) is the acceptance test of
the routing extension: every case names the query, what the index has, and the
shapes, legs and unavailable legs it must route to.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from .._agentic import route
from .._retrieval import LegKind
from .._routing import (
    DEFAULT_ROUTING_POLICY,
    QueryShape,
    RoutingPolicy,
    RoutingRule,
    query_tokens,
)

ROUTING_SET = pathlib.Path(__file__).parent / "data" / "routing_set.json"
CASES = json.loads(ROUTING_SET.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["query"] or "<empty>" for c in CASES])
def test_the_labelled_routing_set(case):
    plan = route(
        case["query"],
        has_dense=case["has_dense"],
        has_graph=case["has_graph"],
        has_metadata=case["has_metadata"],
    )
    assert list(plan.shapes) == case["shapes"]
    assert [leg.value for leg in plan.legs] == case["legs"]
    assert [leg.value for leg in plan.unavailable] == case["unavailable"]


def test_the_set_covers_every_default_shape_and_plain_queries():
    seen = {shape for case in CASES for shape in case["shapes"]}
    assert seen == {shape.value for shape in QueryShape}
    assert sum(1 for case in CASES if not case["shapes"]) >= 10  # noqa: PLR2004


class TestRoute:
    def test_a_plain_query_routes_exactly_as_before(self):
        plan = route("quantum")
        assert plan.legs == (LegKind.LEXICAL, LegKind.DENSE)
        assert plan.reason == "hybrid"
        assert plan.shapes == ()
        assert plan.unavailable == ()

    def test_the_reason_names_shapes_and_missing_paths(self):
        plan = route("how many sections", has_metadata=False)
        assert plan.reason == (
            "hybrid; shapes: aggregation, hierarchy; unavailable: metadata, graph"
        )

    def test_an_empty_policy_detects_nothing(self):
        plan = route("how many", has_metadata=True, policy=RoutingPolicy())
        assert plan.legs == (LegKind.LEXICAL, LegKind.DENSE)
        assert plan.shapes == ()

    def test_a_custom_rule_can_add_any_leg_but_only_when_available(self):
        semantic = RoutingPolicy(
            (RoutingRule("paraphrase", (LegKind.DENSE,), phrases=("similar to",)),)
        )
        assert route("similar to x", has_dense=False, policy=semantic).unavailable == (
            LegKind.DENSE,
        )
        assert route("similar to x", policy=semantic).legs == (
            LegKind.LEXICAL,
            LegKind.DENSE,
        )

    def test_route_is_deterministic(self):
        query = "how many chapters since 2020 are not related to fusion"
        assert route(query, has_graph=True, has_metadata=True) == route(
            query, has_graph=True, has_metadata=True
        )

    @pytest.mark.parametrize(
        ("query", "policy"), [(None, None), (b"x", None), ("x", "default")]
    )
    def test_wrong_types_are_refused(self, query, policy):
        with pytest.raises(TypeError):
            route(query, policy=policy)


class TestTokens:
    def test_tokens_are_lowercase_words_with_apostrophes(self):
        assert query_tokens("Isn\u2019t it in Section 2?") == (
            "isn't",
            "it",
            "in",
            "section",
            "2",
        )

    def test_phrases_match_whole_consecutive_words_only(self):
        rule = RoutingRule("x", (LegKind.GRAPH,), phrases=("part of",))
        assert rule.matches(query_tokens("is it part of that"))
        assert not rule.matches(query_tokens("part and parcel of"))
        assert not rule.matches(query_tokens("partial offset"))

    def test_a_long_query_is_scanned_linearly(self):
        query = "not " * 20000 + "x"
        assert DEFAULT_ROUTING_POLICY.detect(query) == ("negation",)


class TestPolicy:
    def test_shapes_are_reported_in_rule_order(self):
        assert DEFAULT_ROUTING_POLICY.shapes == tuple(s.value for s in QueryShape)

    def test_another_language_is_a_policy(self):
        german = RoutingPolicy(
            (
                RoutingRule(
                    QueryShape.NEGATION, (LegKind.METADATA,), phrases=("nicht", "ohne")
                ),
                RoutingRule(
                    QueryShape.AGGREGATION, (LegKind.METADATA,), phrases=("wie viele",)
                ),
            )
        )
        assert german.detect("Wie viele Seiten ohne Bilder") == (
            "negation",
            "aggregation",
        )

    def test_a_matcher_detects_what_phrases_cannot(self):
        version = RoutingRule(
            "version",
            (LegKind.METADATA,),
            matcher=lambda tokens: any(
                t.startswith("v") and t[1:].isdigit() for t in tokens
            ),
        )
        assert RoutingPolicy((version,)).detect("changes in v2") == ("version",)

    def test_a_matcher_must_return_a_bool(self):
        rule = RoutingRule("odd", (LegKind.METADATA,), matcher=lambda tokens: 1)
        with pytest.raises(TypeError, match="not bool"):
            RoutingPolicy((rule,)).detect("x")

    def test_with_rules_adds_and_replaces(self):
        extra = RoutingRule("version", (LegKind.METADATA,), phrases=("release",))
        extended = DEFAULT_ROUTING_POLICY.with_rules(extra)
        assert extended.shapes[-1] == "version"
        narrow = RoutingRule(
            QueryShape.NEGATION, (LegKind.METADATA,), phrases=("except",)
        )
        with pytest.raises(ValueError, match="already in the policy"):
            DEFAULT_ROUTING_POLICY.with_rules(narrow)
        replaced = DEFAULT_ROUTING_POLICY.with_rules(narrow, replace=True)
        assert replaced.shapes == DEFAULT_ROUTING_POLICY.shapes
        assert replaced.detect("not this") == ()
        assert DEFAULT_ROUTING_POLICY.detect("not this") == ("negation",)

    def test_legs_for_lists_each_leg_once_in_rule_order(self):
        assert DEFAULT_ROUTING_POLICY.legs_for(
            ["hierarchy", "negation", "multi_hop"]
        ) == (
            LegKind.GRAPH,
            LegKind.METADATA,
        )

    @pytest.mark.parametrize(
        ("build", "error", "match"),
        [
            (
                lambda: RoutingRule("Bad", (LegKind.GRAPH,), phrases=("a",)),
                ValueError,
                "shape",
            ),
            (
                lambda: RoutingRule("x", (LegKind.GRAPH,)),
                ValueError,
                "phrases or a matcher",
            ),
            (lambda: RoutingRule("x", "graph", phrases=("a",)), TypeError, "legs"),
            (lambda: RoutingRule("x", ("hops",), phrases=("a",)), ValueError, "hops"),
            (
                lambda: RoutingRule("x", (LegKind.GRAPH,), phrases="a"),
                TypeError,
                "phrases",
            ),
            (
                lambda: RoutingRule("x", (LegKind.GRAPH,), phrases=("Two  Words",)),
                ValueError,
                "lowercase",
            ),
            (
                lambda: RoutingRule("x", (LegKind.GRAPH,), phrases=("a.b",)),
                ValueError,
                "lowercase",
            ),
            (
                lambda: RoutingRule("x", (LegKind.GRAPH,), matcher="yes"),
                TypeError,
                "matcher",
            ),
            (
                lambda: RoutingPolicy(
                    (
                        RoutingRule("x", (LegKind.GRAPH,), phrases=("a",)),
                        RoutingRule("x", (LegKind.GRAPH,), phrases=("b",)),
                    )
                ),
                ValueError,
                "twice",
            ),
            (lambda: RoutingPolicy(("not a rule",)), TypeError, "RoutingRule"),
        ],
    )
    def test_malformed_rules_and_policies_are_refused(self, build, error, match):
        with pytest.raises(error, match=match):
            build()
