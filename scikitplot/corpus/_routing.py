# scikitplot/corpus/_routing.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Query-shape detection for :func:`~scikitplot.corpus.route`, by declared rules.

Notes
-----
**User-focused.** Some questions need more than keyword and vector matching:
"how many", "not", "before 2020", "the relationship between", "the sections
under". A :class:`RoutingPolicy` detects these *shapes* from the words of the
query and names the extra evidence paths each one needs; :func:`route` adds a
path only when the index has it. :data:`DEFAULT_ROUTING_POLICY` covers five
shapes in English:

=================  ===============================  ==============
Shape              Typical words                    Adds
=================  ===============================  ==============
``multi_hop``      relationship between, whose      graph
``aggregation``    how many, total, average         metadata
``negation``       not, without, except             metadata
``temporal``       before, since, latest, a year    metadata
``hierarchy``      parent, section, part of         graph
=================  ===============================  ==============

A policy is data: build your own from :class:`RoutingRule` objects (another
language, a domain vocabulary, a custom matcher), or extend the default with
:meth:`RoutingPolicy.with_rules`. Rules never call a model.

**Developer-focused.** Matching is on whole words, never on regular
expressions supplied by a caller: the query is lowercased, curly apostrophes
are made straight, and it is split into ``[a-z0-9]+('[a-z]+)?`` tokens; a
phrase matches where its tokens occur consecutively. That is linear in the
query length for any rule set, so a policy cannot be made into a
denial-of-service by an unlucky pattern, and its result is a pure function of
the query. Shapes are reported in rule order, each once.
"""

from __future__ import annotations

import dataclasses
import re
from enum import unique
from typing import Callable, Iterable, Sequence

from ._retrieval import LegKind
from ._schema import _StrEnumBase

__all__: list[str] = [
    "DEFAULT_ROUTING_POLICY",
    "QueryShape",
    "RoutingPolicy",
    "RoutingRule",
    "query_tokens",
]

#: U+2019, written as an escape so this file stays ASCII.
_CURLY_APOSTROPHE = "\u2019"
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SHAPE = re.compile(r"[a-z][a-z0-9_]{0,63}")
#: Bounds of a token read as a year by the default temporal rule.
_YEAR_MIN, _YEAR_MAX = 1000, 2999
_YEAR_DIGITS = 4


@unique
class QueryShape(_StrEnumBase):
    """The built-in shapes of :data:`DEFAULT_ROUTING_POLICY`."""

    MULTI_HOP = "multi_hop"
    """The answer joins facts across documents or entities."""

    AGGREGATION = "aggregation"
    """The answer counts, sums, averages or lists everything of a kind."""

    NEGATION = "negation"
    """The answer excludes something."""

    TEMPORAL = "temporal"
    """The answer depends on a date, a period or an order in time."""

    HIERARCHY = "hierarchy"
    """The answer depends on what contains what."""


def query_tokens(query: str) -> tuple[str, ...]:
    """
    Return the words a routing rule matches against.

    Parameters
    ----------
    query : str
        The query text.

    Returns
    -------
    tuple of str
        Lowercase ``[a-z0-9]+('[a-z]+)?`` tokens, in order.

    Raises
    ------
    TypeError
        If ``query`` is not a string.

    Examples
    --------
    >>> query_tokens("Isn't it in Section 2?")
    ("isn't", 'it', 'in', 'section', '2')
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    return tuple(_TOKEN.findall(query.lower().replace(_CURLY_APOSTROPHE, "'")))


def _phrase_tokens(phrase: str) -> tuple[str, ...]:
    tokens = query_tokens(phrase)
    if not tokens or " ".join(tokens) != phrase:
        raise ValueError(
            f"routing phrase {phrase!r} must be lowercase words of [a-z0-9] "
            "(with an optional 'x suffix), separated by single spaces"
        )
    return tokens


def _contains(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    width = len(phrase)
    first = phrase[0]
    return any(
        tokens[i] == first and tuple(tokens[i : i + width]) == tuple(phrase)
        for i in range(len(tokens) - width + 1)
    )


@dataclasses.dataclass(frozen=True)
class RoutingRule:
    """
    One shape: the words that reveal it and the evidence paths it needs.

    Parameters
    ----------
    shape : str or QueryShape
        Shape name, ``[a-z][a-z0-9_]{0,63}``.
    legs : iterable of LegKind
        Paths the shape needs, added by :func:`route` when available.
    phrases : iterable of str, default=()
        Words or phrases (lowercase, single spaces) any one of which reveals
        the shape.
    matcher : callable or None, default=None
        ``matcher(tokens) -> bool``, for what phrases cannot say (a year, a
        number). The shape is detected if a phrase or the matcher matches.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If the shape name or a phrase is malformed, or the rule has neither
        phrases nor a matcher.
    """

    shape: str
    legs: tuple[LegKind, ...]
    phrases: tuple[str, ...] = ()
    matcher: Callable[[tuple[str, ...]], bool] | None = None

    def __post_init__(self) -> None:
        shape = str(
            self.shape.value if isinstance(self.shape, QueryShape) else self.shape
        )
        if not _SHAPE.fullmatch(shape):
            raise ValueError(
                f"RoutingRule.shape must match {_SHAPE.pattern}, got {shape!r}"
            )
        object.__setattr__(self, "shape", shape)
        if isinstance(self.legs, (str, LegKind)) or not isinstance(self.legs, Iterable):
            raise TypeError(f"rule {shape!r}: legs must be an iterable of LegKind")
        object.__setattr__(self, "legs", tuple(LegKind(leg) for leg in self.legs))
        if isinstance(self.phrases, str) or not isinstance(self.phrases, Iterable):
            raise TypeError(f"rule {shape!r}: phrases must be an iterable of str")
        phrases = tuple(self.phrases)
        for phrase in phrases:
            if not isinstance(phrase, str):
                raise TypeError(f"rule {shape!r}: every phrase must be a string")
            _phrase_tokens(phrase)
        object.__setattr__(self, "phrases", phrases)
        if self.matcher is not None and not callable(self.matcher):
            raise TypeError(f"rule {shape!r}: matcher must be callable or None")
        if not phrases and self.matcher is None:
            raise ValueError(f"rule {shape!r}: needs phrases or a matcher")
        object.__setattr__(self, "_compiled", tuple(_phrase_tokens(p) for p in phrases))

    def matches(self, tokens: tuple[str, ...]) -> bool:
        """
        Return whether the query, as :func:`query_tokens`, has this shape.

        Raises
        ------
        TypeError
            If ``matcher`` returns something other than a bool.
        """
        if any(_contains(tokens, phrase) for phrase in self._compiled):
            return True
        if self.matcher is None:
            return False
        verdict = self.matcher(tokens)
        if not isinstance(verdict, bool):
            raise TypeError(
                f"rule {self.shape!r}: matcher returned {type(verdict).__name__}, "
                "not bool"
            )
        return verdict


@dataclasses.dataclass(frozen=True)
class RoutingPolicy:
    """
    An ordered set of :class:`RoutingRule`, one per shape.

    Parameters
    ----------
    rules : iterable of RoutingRule
        Rules with unique shapes; shapes are reported in this order.

    Raises
    ------
    TypeError
        If a rule is not a :class:`RoutingRule`.
    ValueError
        If two rules name one shape.

    See Also
    --------
    route : Uses a policy to choose evidence paths.

    Examples
    --------
    >>> from scikitplot.corpus import DEFAULT_ROUTING_POLICY
    >>> DEFAULT_ROUTING_POLICY.detect("How many sections are not about X?")
    ('aggregation', 'negation', 'hierarchy')
    """

    rules: tuple[RoutingRule, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.rules, RoutingRule) or not isinstance(self.rules, Iterable):
            raise TypeError("RoutingPolicy.rules must be an iterable of RoutingRule")
        rules = tuple(self.rules)
        for rule in rules:
            if not isinstance(rule, RoutingRule):
                raise TypeError(f"not a RoutingRule: {type(rule).__name__}")
        shapes = [rule.shape for rule in rules]
        if len(set(shapes)) != len(shapes):
            raise ValueError(f"RoutingPolicy: a shape appears twice in {shapes}")
        object.__setattr__(self, "rules", rules)

    @property
    def shapes(self) -> tuple[str, ...]:
        """Every shape this policy can report, in order."""
        return tuple(rule.shape for rule in self.rules)

    def detect(self, query: str) -> tuple[str, ...]:
        """
        Return the shapes of ``query``, in rule order.

        Parameters
        ----------
        query : str
            The query text.

        Returns
        -------
        tuple of str
            Each detected shape once; empty for a plain query.
        """
        tokens = query_tokens(query)
        return tuple(rule.shape for rule in self.rules if rule.matches(tokens))

    def legs_for(self, shapes: Iterable[str]) -> tuple[LegKind, ...]:
        """Return the legs the given shapes need, each once, in rule order."""
        wanted = set(shapes)
        legs: list[LegKind] = []
        for rule in self.rules:
            if rule.shape in wanted:
                legs.extend(leg for leg in rule.legs if leg not in legs)
        return tuple(legs)

    def with_rules(self, *rules: RoutingRule, replace: bool = False) -> RoutingPolicy:
        """
        Return a policy with ``rules`` added.

        Parameters
        ----------
        *rules : RoutingRule
            New rules, appended in order.
        replace : bool, default=False
            Replace an existing rule of the same shape (keeping its position)
            instead of refusing it.

        Returns
        -------
        RoutingPolicy

        Raises
        ------
        ValueError
            If a shape is already present and ``replace`` is false.
        """
        by_shape = {rule.shape: rule for rule in rules}
        kept = []
        for rule in self.rules:
            if rule.shape in by_shape:
                if not replace:
                    raise ValueError(f"shape {rule.shape!r} is already in the policy")
                kept.append(by_shape.pop(rule.shape))
            else:
                kept.append(rule)
        return RoutingPolicy((*kept, *(r for r in rules if r.shape in by_shape)))


def _has_year(tokens: tuple[str, ...]) -> bool:
    return any(
        len(token) == _YEAR_DIGITS
        and token.isdigit()
        and _YEAR_MIN <= int(token) <= _YEAR_MAX
        for token in tokens
    )


#: English rules for the five :class:`QueryShape` shapes.
DEFAULT_ROUTING_POLICY = RoutingPolicy(
    (
        RoutingRule(
            QueryShape.MULTI_HOP,
            (LegKind.GRAPH,),
            phrases=(
                "relationship between",
                "relation between",
                "related to",
                "connected to",
                "linked to",
                "in common",
                "difference between",
                "compared to",
                "compare",
                "versus",
                "vs",
                "whose",
                "path from",
            ),
        ),
        RoutingRule(
            QueryShape.AGGREGATION,
            (LegKind.METADATA,),
            phrases=(
                "how many",
                "how much",
                "number of",
                "count",
                "total",
                "sum of",
                "average",
                "list all",
                "all the",
                "maximum",
                "minimum",
                "most common",
            ),
        ),
        RoutingRule(
            QueryShape.NEGATION,
            (LegKind.METADATA,),
            phrases=(
                "not",
                "no",
                "without",
                "except",
                "excluding",
                "other than",
                "never",
                "none",
                "neither",
                "nor",
                "isn't",
                "aren't",
                "wasn't",
                "weren't",
                "doesn't",
                "don't",
                "didn't",
                "can't",
                "cannot",
                "won't",
            ),
        ),
        RoutingRule(
            QueryShape.TEMPORAL,
            (LegKind.METADATA,),
            phrases=(
                "before",
                "after",
                "since",
                "until",
                "during",
                "latest",
                "earliest",
                "recent",
                "recently",
                "newest",
                "oldest",
                "when",
                "ago",
                "today",
                "yesterday",
            ),
            matcher=_has_year,
        ),
        RoutingRule(
            QueryShape.HIERARCHY,
            (LegKind.GRAPH,),
            phrases=(
                "parent",
                "child",
                "children",
                "section",
                "sections",
                "subsection",
                "chapter",
                "chapters",
                "part of",
                "belongs to",
                "contained in",
                "contains",
                "nested",
                "ancestor",
                "descendant",
                "hierarchy",
                "subclass",
                "superclass",
                "top level",
            ),
        ),
    )
)
