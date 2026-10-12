# scikitplot/corpus/_metrics/tests/test__sklearn_compat.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The vendored scorer base: its contract, and parity with scikit-learn's."""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import pytest

from .. import _sklearn_compat
from .._ranking import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from .._sklearn_compat import CorpusScorer as Vendored
from .._text import token_f1_score


class Index:
    def __init__(self, rankings):
        self.rankings = rankings

    def search(self, query):
        return self.rankings[query]

    def predict(self, X):
        return [self.rankings[q] for q in X]


def _vendored(func, method="predict", **kwargs):
    return Vendored(func, 1, kwargs, method, id_getter=None, on_degraded="raise")


class TestVendoredBase:
    def test_call_and_sample_weight(self):
        index = Index({"a": ["a"], "b": ["x"]})
        scorer = _vendored(recall_at_k, k=1)
        assert scorer(index, ["a", "b"], [["a"], ["b"]]) == 0.5
        assert scorer(index, ["a", "b"], [["a"], ["b"]], sample_weight=[3, 1]) == 0.75

    def test_positional_sample_weight_is_deprecated_not_refused(self):
        """Review F13: scikit-learn 1.3-1.8 accept it; 1.9 warns (removal in 1.11)."""
        index = Index({"a": ["a"], "b": ["x"]})
        scorer = _vendored(recall_at_k, k=1)
        with pytest.warns(FutureWarning, match="keyword"):
            assert scorer(index, ["a", "b"], [["a"], ["b"]], [3, 1]) == 0.75
        with pytest.raises(TypeError, match="at most one extra positional"):
            scorer(index, ["a", "b"], [["a"], ["b"]], [3, 1], sample_weight=[3, 1])
        params = inspect.signature(Vendored.__call__).parameters
        assert params["sample_weight"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_extra_keywords_need_routing(self):
        with pytest.raises(ValueError, match="metadata routing"):
            _vendored(recall_at_k)(Index({}), [], [], k=3)

    def test_set_score_request_needs_scikit_learn(self, monkeypatch):
        """Review F15: the advice depends on whether scikit-learn is installed."""
        import sys

        try:
            import sklearn  # noqa: F401
        except ImportError:
            installed = False
        else:
            installed = True
        expected = "this scorer uses the vendored base" if installed else "install scikit-learn"
        with pytest.raises(RuntimeError, match=expected):
            _vendored(recall_at_k).set_score_request(sample_weight=True)
        monkeypatch.setitem(sys.modules, "sklearn", None)
        with pytest.raises(RuntimeError, match="install scikit-learn"):
            _vendored(recall_at_k).set_score_request(sample_weight=True)

    def test_overlap_warning(self):
        scorer = _vendored(recall_at_k, k=1)
        with pytest.warns(UserWarning, match=r"Overlapping parameters are: \{'k'\}"):
            scorer._warn_overlap("msg", {"k": 2})

    def test_accept_sample_weight(self):
        assert _vendored(recall_at_k)._accept_sample_weight()
        assert not _vendored(lambda y_true, y_pred: 0.0)._accept_sample_weight()


def test_attribution_is_retained():
    """BSD-3-Clause condition 1: the notice travels with the adapted source."""
    text = Path(_sklearn_compat.__file__).read_text(encoding="utf-8")
    for line in (
        "afe5bcd80ce17fd8ca88c33beb17534bd36423bc",
        "Copyright (c) 2007-2026 The scikit-learn developers.",
        "Redistributions of source code must retain the above copyright notice",
        'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
    ):
        assert line in text, line


def test_parity_with_scikit_learns_base():
    """Both bases give the same score for the same scorer, input and weights."""
    pytest.importorskip("sklearn")
    from .._sklearn_scorer import CorpusScorer as Native

    rng = random.Random(41)  # noqa: S311 - seeded test data
    pool = [f"d{i}" for i in range(8)]
    queries = [f"q{i}" for i in range(20)]
    rankings = {q: rng.sample(pool, rng.randint(0, 6)) for q in queries}
    y_true = [{d: rng.choice([0, 1, 2]) for d in rng.sample(pool, rng.randint(1, 4))} for _ in queries]
    weights = [rng.choice([0.5, 1.0, 2.0]) for _ in queries]
    index = Index(rankings)
    cases = [
        (recall_at_k, {"k": 3, "zero_division": 0.0}),
        (ndcg_at_k, {"k": 5, "zero_division": 0.0}),
        (mean_reciprocal_rank, {}),
    ]
    for func, kwargs in cases:
        for method in ("predict", "search"):
            for sign in (1, -1):
                native = Native(func, sign, dict(kwargs), method)
                vendored = Vendored(func, sign, dict(kwargs), method)
                assert native(index, queries, y_true) == vendored(index, queries, y_true)
                assert native(index, queries, y_true, sample_weight=weights) == vendored(
                    index, queries, y_true, sample_weight=weights
                )
                assert repr(native) == repr(vendored)


def test_answer_metric_on_the_vendored_base():
    class Answerer:
        def predict(self, X):
            return ["paris" for _ in X]

    assert _vendored(token_f1_score)(Answerer(), ["q"], [["Paris", "City of Paris"]]) == 1.0
