"""
The precomputed keyword path returns exactly what the pure-Python loop returns.

Equality is exact (``==`` on the ``(row, score)`` lists): the precomputed path
promises the same floating-point additions in the same order, not an
approximation of them.
"""

from __future__ import annotations

import random

import pytest

from .. import _similarity
from .._similarity import _BM25Index

pytest.importorskip("numpy")
if _similarity._scoring_engine() is None:  # pragma: no cover - partial install
    pytest.skip("scikitplot.rank_bm25 is not installed", allow_module_level=True)


def _build(seed, n_docs, vocab):
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    docs = [
        [words[min(int(rng.expovariate(0.35)), vocab - 1)] for _ in range(rng.randint(0, 15))]
        for _ in range(n_docs)
    ]
    index = _BM25Index(k1=rng.choice([0.5, 1.2, 1.5, 2.0]), b=rng.choice([0.0, 0.75, 1.0]))
    index.build(docs)
    queries = [[rng.choice(words + ["absent"]) for _ in range(rng.randint(1, 6))] for _ in range(15)]
    queries += [["absent"], ["w0", "w0"], ["w1", "absent", "w2", "w1"]]
    return index, queries, rng


def _pure(index, *args, **kwargs):
    original = _similarity._scoring_engine
    _similarity._scoring_engine = lambda: None
    try:
        return index.query(*args, **kwargs)
    finally:
        _similarity._scoring_engine = original


@pytest.mark.parametrize("seed", range(25))
def test_precomputed_equals_pure(seed):
    index, queries, rng = _build(seed, n_docs=rng_size(seed), vocab=12)
    for query in queries:
        for top_k in (1, 3, 10, 500):
            assert index.query(query, top_k) == _pure(index, query, top_k)
            allowed = frozenset(i for i in range(index._n_docs) if rng.random() < 0.5)
            assert index.query(query, top_k, allowed=allowed) == _pure(
                index, query, top_k, allowed=allowed
            )


def rng_size(seed):
    return 5 + (seed * 37) % 120


def test_scoped_query_keeps_pure_path():
    index, queries, _ = _build(1, n_docs=50, vocab=10)
    scope = frozenset(range(0, 50, 2))
    for query in queries:
        assert index.query(query, 10, scope=scope) == _pure(index, query, 10, scope=scope)


def test_rebuild_discards_precomputed_columns():
    index = _BM25Index()
    index.build([["a", "b"], ["b"]])
    first = index.query(["a"], 5)
    index.build([["c"], ["a", "a"], ["a"]])
    assert index.query(["a"], 5) == _pure(index, ["a"], 5)
    assert index.query(["a"], 5) != first


def test_empty_index_and_nonpositive_top_k():
    index = _BM25Index()
    index.build([])
    assert index.query(["a"], 5) == []
    index.build([["a"]])
    assert index.query(["a"], 0) == []
