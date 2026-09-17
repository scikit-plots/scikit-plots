"""
Parameter-domain regressions for :mod:`scikitplot.rank_bm25` (slice S-13).

``k1``, ``b``, ``epsilon`` and ``delta`` previously reached the scoring
arithmetic unchecked. ``k1=0`` made both numerator and denominator zero for any
document lacking the query term and produced ``NaN`` scores; ``b`` outside
``[0, 1]`` and a negative ``epsilon`` produced rankings with no defined
meaning. These checks pin the accepted domain at construction, where the
mistake is still attributable.

See Also
--------
scikitplot.rank_bm25._rank_bm25.BM25Okapi
scikitplot.rank_bm25._rank_bm25.BM25L
scikitplot.rank_bm25._rank_bm25.BM25Plus
"""

import math

import pytest

from .._rank_bm25 import BM25L, BM25Okapi, BM25Plus

CORPUS = [["a", "b"], ["b"], ["c"]]
CLASSES = (BM25Okapi, BM25L, BM25Plus)


@pytest.mark.parametrize("scorer", CLASSES)
@pytest.mark.parametrize("k1", [0, -1.5], ids=["zero", "negative"])
def test_k1_outside_its_domain_is_refused(scorer, k1):
    """``k1`` must be positive: zero divides zero by zero for an absent term."""
    with pytest.raises(ValueError) as excinfo:
        scorer(CORPUS, k1=k1)
    assert "k1" in str(excinfo.value)


@pytest.mark.parametrize("scorer", CLASSES)
@pytest.mark.parametrize("b", [2.0, -1.0], ids=["above-one", "negative"])
def test_b_outside_zero_to_one_is_refused(scorer, b):
    """``b`` is a length-normalisation weight and is only defined on [0, 1]."""
    with pytest.raises(ValueError) as excinfo:
        scorer(CORPUS, b=b)
    assert "b" in str(excinfo.value)


def test_negative_epsilon_is_refused():
    """``epsilon`` scales a floor; a negative value inverts what the floor means."""
    with pytest.raises(ValueError) as excinfo:
        BM25Okapi(CORPUS, epsilon=-5.0)
    assert "epsilon" in str(excinfo.value)


@pytest.mark.parametrize("scorer", [BM25L, BM25Plus])
def test_negative_delta_is_refused(scorer):
    """``delta`` is an additive floor and is not defined below zero."""
    with pytest.raises(ValueError) as excinfo:
        scorer(CORPUS, delta=-1.0)
    assert "delta" in str(excinfo.value)


@pytest.mark.parametrize("scorer", CLASSES)
def test_boolean_parameters_are_refused(scorer):
    """``True`` is an ``int`` in Python; it is not a scoring parameter."""
    with pytest.raises(TypeError):
        scorer(CORPUS, k1=True)


@pytest.mark.parametrize("scorer", CLASSES)
def test_non_numeric_parameters_are_refused(scorer):
    """A string that looks like a number is still not a number."""
    with pytest.raises(TypeError):
        scorer(CORPUS, b="0.75")


@pytest.mark.parametrize("scorer", CLASSES)
def test_no_score_is_nan_after_validation(scorer):
    """Every score over the accepted domain is a real number."""
    index = scorer(CORPUS)
    assert all(not math.isnan(value) for value in index.get_scores(["a"]))


@pytest.mark.parametrize("scorer", CLASSES)
def test_boundary_values_are_accepted(scorer):
    """The documented endpoints are inside the domain, not outside it."""
    for kwargs in ({"b": 0.0}, {"b": 1.0}, {"k1": 0.01}):
        index = scorer(CORPUS, **kwargs)
        assert len(index.get_scores(["a"])) == 3


def test_defaults_are_unchanged():
    """Validation does not move the defaults callers already rely on."""
    okapi = BM25Okapi(CORPUS)
    assert (okapi.k1, okapi.b, okapi.epsilon) == (1.5, 0.75, 0.25)
    assert (BM25L(CORPUS).delta, BM25Plus(CORPUS).delta) == (0.5, 1)
