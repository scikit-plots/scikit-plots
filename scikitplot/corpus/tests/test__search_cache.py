"""
CachedSearch: equal answers, never stale, never across scopes, never degraded.
"""

from __future__ import annotations

import random
import threading

import pytest

from scikitplot.corpus import (
    CachedSearch,
    CorpusDocument,
    Eq,
    LogOddsFusion,
    ProbabilityFusion,
    RetrievalConfig,
    RetrievalIndex,
    SecurityFilter,
)

np = pytest.importorskip("numpy")
_WORDS = ["alpha", "beta", "gamma", "delta"]


def _docs(seed, n=30):
    rng = random.Random(seed)  # noqa: S311
    return [
        CorpusDocument.create(
            "c.txt", i, " ".join(rng.choice(_WORDS) for _ in range(5)) + f" {i}",
            collection_id=rng.choice(["public", "secret"]),
        ).replace(embedding=np.asarray([rng.random() for _ in range(4)], dtype=np.float32))
        for i in range(n)
    ]


def _index(docs, security=None):
    index = RetrievalIndex(RetrievalConfig(backend="bruteforce"), security_filter=security)
    index.build(docs)
    return index


def _ids(response):
    return [(h.doc.doc_id, h.score) for h in response]


KW = RetrievalConfig(match_mode="keyword", top_k=5, backend="bruteforce")


def test_a_hit_equals_the_index_answer():
    index = _index(_docs(1))
    cache = CachedSearch(index)
    for query in ("alpha", "beta gamma", "alpha"):
        assert _ids(cache.search(query, config=KW)) == _ids(index.search(query, config=KW))
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 2


def test_every_argument_is_part_of_the_key():
    index = _index(_docs(2))
    cache = CachedSearch(index)
    qe = np.ones(4, dtype=np.float32)
    base = cache.key("alpha", KW, qe, None)
    assert base == cache.key("alpha", KW, qe.copy(), None)
    others = {
        cache.key("beta", KW, qe, None),
        cache.key("alpha", RetrievalConfig(match_mode="keyword", top_k=6), qe, None),
        cache.key("alpha", KW, qe * 2, None),
        cache.key("alpha", KW, qe.astype(np.float64), None),
        cache.key("alpha", KW, qe, Eq("collection_id", "public")),
        cache.key("alpha", KW, None, None),
    }
    assert base not in others and len(others) == 6


def test_a_rebuild_is_never_served_stale():
    index = _index(_docs(3))
    cache = CachedSearch(index)
    before = _ids(cache.search("alpha", config=KW))
    index.build(_docs(4))
    after = cache.search("alpha", config=KW)
    assert _ids(after) == _ids(index.search("alpha", config=KW))
    assert _ids(after) != before
    assert cache.stats()["hits"] == 0


def test_scopes_never_share_entries():
    docs = _docs(5)
    public = CachedSearch(_index(docs, SecurityFilter(Eq("collection_id", "public"), label="s")))
    secret = CachedSearch(_index(docs, SecurityFilter(Eq("collection_id", "secret"), label="s")))
    assert public.key("alpha", KW) != secret.key("alpha", KW)  # same label, other scope


def test_editing_a_result_does_not_change_the_cache():
    cache = CachedSearch(_index(_docs(6)))
    first = cache.search("alpha", config=KW)
    first.hits.clear()
    assert len(cache.search("alpha", config=KW)) > 0


def test_degraded_responses_are_not_kept():
    index = _index(_docs(7))
    cache = CachedSearch(index)
    hybrid = RetrievalConfig(match_mode="hybrid", top_k=5, backend="bruteforce")
    response = cache.search("alpha", config=hybrid)  # no query embedding: dense leg degraded
    assert response.status.value == "degraded"
    assert cache.stats()["stored"] == 0


def test_probability_fusion_is_uncacheable():
    class Fitted:
        fitted, prevalence_ = True, 0.5

        def predict_proba(self, scores):
            return [0.5 for _ in scores]

    cfg = RetrievalConfig(
        match_mode="hybrid", top_k=3, backend="bruteforce",
        probability_fusion=ProbabilityFusion(
            policy=LogOddsFusion(base_rate=0.5, weights={"dense": 0.0}),
            calibrators={"lexical": Fitted()},
        ),
    )
    cache = CachedSearch(_index(_docs(8)))
    cache.search("alpha", config=cfg)
    cache.search("alpha", config=cfg)
    assert cache.stats()["uncacheable"] == 2 and cache.stats()["size"] == 0


def test_lru_bound_and_clear():
    cache = CachedSearch(_index(_docs(9)), max_entries=2)
    for query in ("alpha", "beta", "gamma"):
        cache.search(query, config=KW)
    assert cache.stats()["size"] == 2 and cache.stats()["evicted"] == 1
    cache.search("gamma", config=KW)
    assert cache.stats()["hits"] == 1
    cache.clear()
    assert cache.stats()["size"] == 0


def test_threads_get_the_same_answers():
    index = _index(_docs(10))
    cache = CachedSearch(index, max_entries=3)
    want = {q: _ids(index.search(q, config=KW)) for q in _WORDS}
    errors = []

    def work():
        for _ in range(50):
            for q in _WORDS:
                if _ids(cache.search(q, config=KW)) != want[q]:
                    errors.append(q)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_refusals():
    with pytest.raises(ValueError):
        CachedSearch(object(), max_entries=0)
    unbuilt = CachedSearch(RetrievalIndex(RetrievalConfig(backend="bruteforce")))
    assert unbuilt.key("alpha") is None
