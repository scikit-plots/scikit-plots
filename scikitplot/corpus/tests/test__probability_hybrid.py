"""
Opt-in probability fusion in hybrid search: equivalence, outcomes, refusals.
"""

from __future__ import annotations

import math
import random

import pytest

from scikitplot.corpus import (
    CorpusDocument,
    Eq,
    LogOddsFusion,
    ProbabilityFusion,
    RetrievalConfig,
    RetrievalIndex,
    SecurityFilter,
)

np = pytest.importorskip("numpy")

_WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]


class _Logistic:
    """A fitted calibrator: sigmoid(a * score + b)."""

    fitted = True

    def __init__(self, a, b, prevalence):
        self.a, self.b, self.prevalence_ = a, b, prevalence

    def predict_proba(self, scores):
        return [1.0 / (1.0 + math.exp(-(self.a * s + self.b))) for s in scores]


def _corpus(rng, n=40):
    return [
        CorpusDocument.create(
            "c.txt",
            i,
            " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 7))) + f" {i}",
            collection_id=rng.choice(["public", "secret"]),
        ).replace(embedding=np.asarray([rng.random() for _ in range(4)], dtype=np.float32))
        for i in range(n)
    ]


def _index(docs, security=None):
    index = RetrievalIndex(RetrievalConfig(backend="bruteforce"), security_filter=security)
    index.build(docs)
    return index


def _fusion(**weights):
    return ProbabilityFusion(
        policy=LogOddsFusion(base_rate=0.2, weights=weights),
        calibrators={"lexical": _Logistic(0.8, -2.0, 0.3), "dense": _Logistic(5.0, -3.0, 0.25)},
    )


def _config(top_k, fusion):
    return RetrievalConfig(
        match_mode="hybrid", top_k=top_k, backend="bruteforce", probability_fusion=fusion
    )


def test_equals_fusing_the_legs_by_hand():
    rng = random.Random(906)  # noqa: S311 - a seeded test sample
    for _ in range(30):
        docs = _corpus(rng)
        index = _index(docs)
        query = " ".join(rng.sample(_WORDS, 2))
        qe = np.asarray([rng.random() for _ in range(4)], dtype=np.float32)
        top_k = rng.randint(1, 6)
        fusion = _fusion(dense=rng.choice([0.5, 1.0]))
        fetch_k = min(top_k * 3, len(docs))
        legs = []
        for name, mode in (("lexical", "keyword"), ("dense", "semantic")):
            hits = index.search(
                query,
                config=RetrievalConfig(match_mode=mode, top_k=fetch_k, backend="bruteforce"),
                query_embedding=qe,
            )
            legs.append(
                (name, [(h.doc.doc_id, h.native_score) for h in hits], fusion.calibrators[name])
            )
        expected = LogOddsFusion(
            base_rate=0.2, weights=dict(fusion.policy.weights), top_k=top_k
        ).fuse(legs)
        got = index.search(query, config=_config(top_k, fusion), query_embedding=qe)
        assert [h.doc.doc_id for h in got] == [i.key for i in expected.items]
        assert [h.score for h in got] == [i.probability for i in expected.items]


def test_hits_carry_probabilities_and_leg_provenance():
    docs = _corpus(random.Random(1))  # noqa: S311
    qe = np.ones(4, dtype=np.float32)
    response = _index(docs).search("alpha beta", config=_config(5, _fusion()), query_embedding=qe)
    assert response.status.value == "success"
    for hit in response:
        assert 0.0 < hit.score < 1.0
        assert hit.native_metric == "logodds_probability"
        assert hit.native_score is None
        assert {c.native_metric for c in hit.contributions} <= {"bm25", "cosine_similarity"}
    scores = [h.score for h in response]
    assert scores == sorted(scores, reverse=True)


def test_a_zero_weight_leg_is_skipped_and_needs_no_calibrator():
    docs = _corpus(random.Random(2))  # noqa: S311
    fusion = ProbabilityFusion(
        policy=LogOddsFusion(base_rate=0.2, weights={"dense": 0.0}),
        calibrators={"lexical": _Logistic(0.8, -2.0, 0.3)},
    )
    response = _index(docs).search("alpha", config=_config(5, fusion))
    legs = {leg.leg.value: leg.status.value for leg in response.legs}
    assert legs["dense"] == "skipped"
    assert all(len(h.contributions) == 1 for h in response)


def test_default_config_still_uses_rank_fusion():
    docs = _corpus(random.Random(3))  # noqa: S311
    hits = _index(docs).search(
        "alpha",
        config=RetrievalConfig(match_mode="hybrid", top_k=5, backend="bruteforce"),
        query_embedding=np.ones(4, dtype=np.float32),
    )
    assert {h.native_metric for h in hits} == {"reciprocal_rank_fusion"}


def _scoped_fusion(scope, **weights):
    base = _fusion(**weights)
    return ProbabilityFusion(policy=base.policy, calibrators=base.calibrators, scope=scope)


def test_a_calibrator_fitted_under_the_index_scope_fuses_by_hand():
    rng = random.Random(4)  # noqa: S311 - a seeded test sample
    public = SecurityFilter(Eq("collection_id", "public"))
    for _ in range(10):
        docs = _corpus(rng)
        index = _index(docs, public)
        query = " ".join(rng.sample(_WORDS, 2))
        qe = np.asarray([rng.random() for _ in range(4)], dtype=np.float32)
        top_k = rng.randint(1, 6)
        fusion = _scoped_fusion(public, dense=0.5)
        fetch_k = min(top_k * 3, len(docs))
        legs = []
        for name, mode in (("lexical", "keyword"), ("dense", "semantic")):
            hits = index.search(
                query,
                config=RetrievalConfig(match_mode=mode, top_k=fetch_k, backend="bruteforce"),
                query_embedding=qe,
            )
            legs.append(
                (name, [(h.doc.doc_id, h.native_score) for h in hits], fusion.calibrators[name])
            )
        expected = LogOddsFusion(
            base_rate=0.2, weights=dict(fusion.policy.weights), top_k=top_k
        ).fuse(legs)
        got = index.search(query, config=_config(top_k, fusion), query_embedding=qe)
        assert [h.doc.doc_id for h in got] == [i.key for i in expected.items]
        assert [h.score for h in got] == [i.probability for i in expected.items]
        assert all(h.doc.collection_id == "public" for h in got)


def test_an_unscoped_calibrator_is_refused_on_a_scoped_index():
    public = SecurityFilter(Eq("collection_id", "public"))
    index = _index(_corpus(random.Random(4)), public)  # noqa: S311
    with pytest.raises(ValueError, match="fitted under scope unscoped") as info:
        index.search("alpha", config=_config(5, _fusion()), query_embedding=np.ones(4))
    assert public.digest in str(info.value)


def test_a_scoped_calibrator_is_refused_on_an_unscoped_index():
    public = SecurityFilter(Eq("collection_id", "public"))
    index = _index(_corpus(random.Random(4)))  # noqa: S311
    with pytest.raises(ValueError, match="bound to unscoped") as info:
        index.search(
            "alpha", config=_config(5, _scoped_fusion(public)), query_embedding=np.ones(4)
        )
    assert public.digest in str(info.value)


def test_one_label_two_filters_are_two_scopes():
    public = SecurityFilter(Eq("collection_id", "public"), label="tenant")
    secret = SecurityFilter(Eq("collection_id", "secret"), label="tenant")
    index = _index(_corpus(random.Random(4)), secret)  # noqa: S311
    with pytest.raises(ValueError, match="fitted under scope sha256:") as info:
        index.search(
            "alpha", config=_config(5, _scoped_fusion(public)), query_embedding=np.ones(4)
        )
    assert public.digest in str(info.value)
    assert secret.digest in str(info.value)


def test_the_label_is_not_part_of_the_scope():
    fitted = SecurityFilter(Eq("collection_id", "public"), label="fit")
    bound = SecurityFilter(Eq("collection_id", "public"), label="serve")
    index = _index(_corpus(random.Random(4)), bound)  # noqa: S311
    response = index.search(
        "alpha", config=_config(5, _scoped_fusion(fitted)), query_embedding=np.ones(4)
    )
    assert response.status.value == "success"


def test_dense_only_fusion_runs_under_any_scope():
    docs = _corpus(random.Random(6))  # noqa: S311
    dense_only = ProbabilityFusion(
        policy=LogOddsFusion(base_rate=0.2, weights={"lexical": 0.0}),
        calibrators={"dense": _Logistic(5.0, -3.0, 0.25)},
    )
    index = _index(docs, SecurityFilter(Eq("collection_id", "public")))
    response = index.search(
        "alpha", config=_config(5, dense_only), query_embedding=np.ones(4, dtype=np.float32)
    )
    assert response.status.value == "success"
    assert response
    assert all(h.doc.collection_id == "public" for h in response)
    assert {leg.leg.value: leg.status.value for leg in response.legs}["lexical"] == "skipped"


def test_scope_must_be_a_security_filter():
    with pytest.raises(TypeError, match="scope must be a SecurityFilter"):
        _scoped_fusion("public")
    assert _fusion().scope_digest is None


def test_config_refusals():
    with pytest.raises(ValueError, match="hybrid' only"):
        RetrievalConfig(match_mode="keyword", probability_fusion=_fusion())
    with pytest.raises(ValueError, match="hybrid_alpha is a rank-fusion weight"):
        RetrievalConfig(match_mode="hybrid", hybrid_alpha=0.3, probability_fusion=_fusion())
    with pytest.raises(TypeError, match="ProbabilityFusion"):
        RetrievalConfig(match_mode="hybrid", probability_fusion=LogOddsFusion(base_rate=0.2))


def test_probability_fusion_refusals():
    policy = LogOddsFusion(base_rate=0.2)

    class Unfitted:
        fitted = False

    with pytest.raises(ValueError, match="no calibrator"):
        ProbabilityFusion(policy=policy, calibrators={"lexical": _Logistic(1, 0, 0.5)})
    with pytest.raises(ValueError, match="not fitted"):
        ProbabilityFusion(
            policy=policy,
            calibrators={"lexical": Unfitted(), "dense": _Logistic(1, 0, 0.5)},
        )
    with pytest.raises(ValueError, match="unknown leg names"):
        ProbabilityFusion(policy=policy, calibrators={"graph": _Logistic(1, 0, 0.5)})
    with pytest.raises(ValueError, match="weight 0"):
        ProbabilityFusion(
            policy=LogOddsFusion(base_rate=0.2, default_weight=0.0), calibrators={}
        )
    with pytest.raises(TypeError, match="LogOddsFusion"):
        ProbabilityFusion(policy=object(), calibrators={})


def test_with_fitted_bm25_calibrators():
    rank_bm25 = pytest.importorskip("scikitplot.rank_bm25")
    rng = random.Random(5)  # noqa: S311
    docs = _corpus(rng, n=60)
    index = _index(docs)
    scores, labels, qids = [], [], []
    for q in range(30):
        word = rng.choice(_WORDS)
        for hit in index.search(word, config=RetrievalConfig(match_mode="keyword", top_k=15)):
            scores.append(hit.native_score)
            labels.append(int(word in hit.doc.text.split()[:2]))
            qids.append(str(q))
    lexical = rank_bm25.ScoreCalibrator("isotonic").fit(
        scores, labels, query_ids=qids, population="hybrid-lexical-top15"
    )
    fusion = ProbabilityFusion(
        policy=LogOddsFusion(base_rate=float(np.mean(labels)), weights={"dense": 0.0}),
        calibrators={"lexical": lexical},
    )
    hits = index.search("gamma", config=_config(5, fusion))
    assert hits and all(0.0 <= h.score <= 1.0 for h in hits)
