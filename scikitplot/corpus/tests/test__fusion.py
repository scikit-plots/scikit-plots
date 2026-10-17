# corpus/tests/test__fusion.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
One rank fusion (R2-1, decision D2): the contract and every corpus path that uses it.

The contract tests are the same in ``scikitplot/mcp/tests/test__fusion.py``,
each run against its own submodule's copy of ``_fusion.py``; the copies are
held byte-identical by ``scikitplot/tests/test_submodule_independence.py``.
"""

from __future__ import annotations

import math
import random
import types
from fractions import Fraction

import pytest

from scikitplot.corpus import FusionPolicy, LegKind, LegResult, fuse_by_rank
from scikitplot.corpus._retrieval import LegOutcome, LegStatus
from scikitplot.corpus._similarity._similarity import (
    RetrievalConfig,
    RetrievalHit,
    RetrievalIndex,
)

# ---------------------------------------------------------------------------
# The contract (R2-1, decision D2), tested on this submodule's own copy
# ---------------------------------------------------------------------------


def _reference(legs, k, weights, default_weight=1.0, top_k=None):
    """Brute-force fusion written from the module docstring.

    Order by the exact rational score, then the key; report ``math.fsum`` of the
    correctly rounded per-leg terms.
    """
    totals, terms = {}, {}
    for name, keys in legs:
        weight = float(weights.get(name, default_weight))
        if weight == 0:
            continue
        best = {}
        for rank, key in enumerate(keys, start=1):
            best.setdefault(key, rank)
        for key, rank in best.items():
            totals[key] = totals.get(key, Fraction(0)) + Fraction(weight) / (k + rank)
            terms.setdefault(key, []).append(weight / (k + rank))
    ordered = sorted(totals, key=lambda key: (-totals[key], key))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [(key, math.fsum(terms[key])) for key in ordered]


class TestRule:
    def test_the_worked_example(self):
        fused = FusionPolicy(k=60).fuse(
            [("lexical", ["A", "B", "C"]), ("dense", ["C", "A", "D"])]
        )
        assert [item.key for item in fused] == ["A", "C", "B", "D"]
        assert fused[0].score == math.fsum([1 / 61, 1 / 62])
        assert fused[1].score == math.fsum([1 / 61, 1 / 63])
        assert [(c.leg, c.position, c.rank) for c in fused[1].contributions] == [
            ("lexical", 0, 3),
            ("dense", 1, 1),
        ]

    def test_a_tie_falls_to_the_lower_key(self):
        fused = FusionPolicy().fuse([("a", ["b", "a"]), ("b", ["a", "b"])])
        assert [item.key for item in fused] == ["a", "b"]
        assert fused[0].score == fused[1].score

    def test_an_exact_tie_is_a_tie_whatever_the_rounding(self):
        # Equal weights and permuted ranks: equal exact sums, which float
        # addition in different orders need not reproduce bit for bit.
        legs = [("x", ["d", "e", "f"]), ("y", ["e", "f", "d"]), ("z", ["f", "d", "e"])]
        fused = FusionPolicy(k=1).fuse(legs)
        assert [item.key for item in fused] == ["d", "e", "f"]
        assert len({item.score for item in fused}) == 1

    def test_a_repeat_within_a_leg_counts_once_at_its_best_rank(self):
        fused = FusionPolicy().fuse([("lex", ["a", "a", "b"]), ("dense", ["b", "c"])])
        assert [item.key for item in fused] == ["b", "a", "c"]
        assert [c.rank for c in fused[1].contributions] == [1]

    def test_a_zero_weight_leg_adds_nothing(self):
        policy = FusionPolicy(weights={"dense": 0.0})
        fused = policy.fuse([("lexical", ["a"]), ("dense", ["z", "a"])])
        assert [item.key for item in fused] == ["a"]
        assert [c.leg for c in fused[0].contributions] == ["lexical"]

    @pytest.mark.parametrize(("alpha", "leg"), [(0.0, 0), (1.0, 1)])
    def test_alpha_0_and_1_return_one_leg_exactly(self, alpha, leg):
        legs = [("lexical", ["a", "b", "c", "b"]), ("dense", ["z", "c", "y"])]
        fused = FusionPolicy.from_alpha(alpha).fuse(legs)
        expected = list(dict.fromkeys(legs[leg][1]))
        assert [item.key for item in fused] == expected
        assert [item.score for item in fused] == [
            1.0 / (60 + rank) for rank in range(1, len(expected) + 1)
        ]

    def test_legs_sharing_a_name_share_its_weight(self):
        policy = FusionPolicy(weights={"dense": 2.0})
        fused = policy.fuse([("dense", ["a"]), ("dense", ["b"]), ("lexical", ["c"])])
        assert [(i.key, i.contributions[0].position) for i in fused] == [
            ("a", 0),
            ("b", 1),
            ("c", 2),
        ]

    def test_top_k_and_empty_input(self):
        legs = [("lexical", ["a", "b", "c"])]
        assert [i.key for i in FusionPolicy(top_k=2).fuse(legs)] == ["a", "b"]
        assert FusionPolicy(top_k=0).fuse(legs) == []
        assert FusionPolicy().fuse([]) == []
        assert FusionPolicy().fuse([("lexical", [])]) == []

    def test_the_weights_are_a_read_only_copy(self):
        weights = {"dense": 1.0}
        policy = FusionPolicy(weights=weights)
        weights["dense"] = 0.0
        assert policy.weight("dense") == 1.0
        with pytest.raises(TypeError):
            policy.weights["dense"] = 0.0  # type: ignore[index]

    def test_a_random_sample_agrees_with_the_reference(self):
        rng = random.Random(20261013)  # noqa: S311 - a seeded test sample
        alphabet = [f"d{i:02d}" for i in range(30)]
        for _ in range(1500):
            k = rng.randint(1, 80)
            names = [
                rng.choice(["lexical", "dense", "graph"])
                for _ in range(rng.randint(0, 4))
            ]
            legs = [
                (name, [rng.choice(alphabet) for _ in range(rng.randint(0, 25))])
                for name in names
            ]
            weights = {
                name: rng.choice([0.0, 0.25, 0.5, 1.0, rng.random() * 3])
                for name in set(names)
                if rng.random() < 0.7
            }
            default = rng.choice([0.0, 1.0, 0.3])
            top_k = rng.choice([None, 0, 1, 5, 50])
            policy = FusionPolicy(
                k=k, weights=weights, default_weight=default, top_k=top_k
            )
            got = [(item.key, item.score) for item in policy.fuse(legs)]
            assert got == _reference(legs, k, weights, default, top_k)


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"k": 0}, ValueError),
            ({"k": True}, TypeError),
            ({"k": 60.0}, TypeError),
            ({"top_k": -1}, ValueError),
            ({"top_k": False}, TypeError),
            ({"default_weight": -0.5}, ValueError),
            ({"default_weight": float("nan")}, ValueError),
            ({"weights": {"a": float("inf")}}, ValueError),
            ({"weights": {"a": True}}, TypeError),
            ({"weights": {"a": "1"}}, TypeError),
            ({"weights": {1: 1.0}}, TypeError),
            ({"weights": [("a", 1.0)]}, TypeError),
        ],
    )
    def test_a_bad_policy_is_refused(self, kwargs, error):
        with pytest.raises(error):
            FusionPolicy(**kwargs)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), True, "0.5"])
    def test_a_bad_alpha_is_refused(self, alpha):
        with pytest.raises((TypeError, ValueError)):
            FusionPolicy.from_alpha(alpha)

    @pytest.mark.parametrize(
        ("legs", "error", "match"),
        [
            ([["lexical", ["a"]]], TypeError, "pair"),
            ([("lexical", ["a"], 1.0)], TypeError, "pair"),
            ([(1, ["a"])], TypeError, "name"),
            ([("lexical", [1])], TypeError, "key"),
            ([("lexical", ["a", ""])], ValueError, "empty"),
        ],
    )
    def test_a_bad_leg_is_refused(self, legs, error, match):
        with pytest.raises(error, match=match):
            FusionPolicy().fuse(legs)

    def test_numpy_numbers_are_accepted(self):
        np = pytest.importorskip("numpy")
        policy = FusionPolicy(k=np.int64(60), weights={"a": np.float32(0.5)})
        assert policy.k == 60
        assert isinstance(policy.k, int)
        assert policy.weight("a") == 0.5


# ---------------------------------------------------------------------------
# Every corpus path fuses by the contract
# ---------------------------------------------------------------------------


def _hits(ids, metric):
    return [
        RetrievalHit(
            doc=types.SimpleNamespace(doc_id=i),
            score=1.0,
            match_mode="x",
            native_score=float(n),
            native_metric=metric,
        )
        for n, i in enumerate(ids)
    ]


def _hybrid(lexical, dense, alpha, top_k=50, rrf_k=60):
    index = types.SimpleNamespace(backend_name="probe", _generation=None)
    config = RetrievalConfig(hybrid_alpha=alpha, top_k=top_k, rrf_k=rrf_k)
    return RetrievalIndex._fuse_hybrid(
        index, "q", _hits(lexical, "bm25"), _hits(dense, "cosine_similarity"), config
    )


def _by_rank(lexical, dense, weights, top_k=50, rrf_k=60):
    legs = [
        LegResult(
            hits=tuple(_hits(ids, metric)),
            outcome=LegOutcome(leg=kind, status=LegStatus.SUCCESS, hit_count=len(ids)),
        )
        for kind, ids, metric in (
            (LegKind.LEXICAL, lexical, "bm25"),
            (LegKind.DENSE, dense, "cosine_similarity"),
        )
    ]
    return fuse_by_rank(legs, rrf_k=rrf_k, weights=weights, top_k=top_k)


class TestCorpusPaths:
    @pytest.mark.parametrize(("alpha", "leg"), [(0.0, 0), (1.0, 1)])
    def test_hybrid_alpha_0_and_1_return_one_leg_exactly(self, alpha, leg):
        legs = (["a", "b", "c"], ["z", "c", "y"])
        hits = _hybrid(*legs, alpha)
        assert [h.doc.doc_id for h in hits] == legs[leg]

    def test_hybrid_hits_carry_their_contributions(self):
        hits = _hybrid(["a", "b"], ["b", "c"], 0.5)
        assert [h.doc.doc_id for h in hits] == ["b", "a", "c"]
        assert [(c.leg, c.rank, c.native_metric) for c in hits[0].contributions] == [
            (LegKind.LEXICAL, 1, "bm25"),
            (LegKind.DENSE, 0, "cosine_similarity"),
        ]
        assert hits[0].native_score is None
        assert hits[0].native_metric == "reciprocal_rank_fusion"
        assert [h.rank for h in hits] == [0, 1, 2]

    def test_the_two_corpus_paths_and_the_contract_agree(self):
        rng = random.Random(865)  # noqa: S311 - a seeded test sample
        alphabet = [f"d{i:02d}" for i in range(25)]
        for _ in range(400):
            lexical = [rng.choice(alphabet) for _ in range(rng.randint(0, 20))]
            dense = [rng.choice(alphabet) for _ in range(rng.randint(0, 20))]
            alpha = rng.choice([0.0, 0.25, 0.5, 1.0, rng.random()])
            rrf_k = rng.randint(1, 80)
            top_k = rng.randint(1, 30)
            hybrid = [
                (h.doc.doc_id, h.score)
                for h in _hybrid(lexical, dense, alpha, top_k, rrf_k)
            ]
            ranked = [
                (h.doc.doc_id, s)
                for h, s, _ in _by_rank(
                    lexical,
                    dense,
                    {LegKind.LEXICAL: 1.0 - alpha, LegKind.DENSE: alpha},
                    top_k,
                    rrf_k,
                )
            ]
            reference = _reference(
                [("lexical", lexical), ("dense", dense)],
                rrf_k,
                {"lexical": 1.0 - alpha, "dense": alpha},
                0.0,
                top_k,
            )
            assert hybrid == ranked == reference

    def test_fuse_by_rank_refuses_a_hit_without_identity(self):
        legs = [
            LegResult(
                hits=(types.SimpleNamespace(),),
                outcome=LegOutcome(
                    leg=LegKind.LEXICAL, status=LegStatus.SUCCESS, hit_count=1
                ),
            )
        ]
        with pytest.raises(ValueError, match="no doc_id or node_id"):
            fuse_by_rank(legs)

    @pytest.mark.parametrize("rrf_k", [0, -1, True, 2.5])
    def test_retrieval_config_refuses_a_bad_rrf_k(self, rrf_k):
        with pytest.raises((TypeError, ValueError), match="RetrievalConfig"):
            RetrievalConfig(rrf_k=rrf_k)


# ---------------------------------------------------------------------------
# Review findings (2e)
# ---------------------------------------------------------------------------


class TestReviewFindings:
    def test_a_policy_is_hashable_picklable_and_copyable(self):
        import copy  # noqa: PLC0415
        import dataclasses  # noqa: PLC0415
        import pickle  # noqa: PLC0415

        policy = FusionPolicy(k=7, weights={"dense": 2.0, "lexical": 0.5}, top_k=3)
        assert policy.weights == (("dense", 2.0), ("lexical", 0.5))
        assert dict(policy.weights) == {"dense": 2.0, "lexical": 0.5}
        assert hash(policy) == hash(
            FusionPolicy(k=7, weights={"lexical": 0.5, "dense": 2.0}, top_k=3)
        )
        assert pickle.loads(pickle.dumps(policy)) == policy  # noqa: S301 - own object
        assert copy.deepcopy(policy) == policy
        assert dataclasses.replace(policy, k=8).weights == policy.weights
        assert dataclasses.asdict(policy)["weights"] == policy.weights

    def test_a_leg_given_two_weights_is_refused(self):
        with pytest.raises(ValueError, match="two weights"):
            FusionPolicy(weights=(("a", 1.0), ("a", 2.0)))

    def test_decimal_weights_are_accepted_and_huge_ones_refused(self):
        import decimal  # noqa: PLC0415

        assert FusionPolicy(weights={"a": decimal.Decimal("0.5")}).weight("a") == 0.5
        with pytest.raises(ValueError, match="too large"):
            FusionPolicy(weights={"a": 10**400})

    def test_float_close_scores_are_ordered_exactly(self):
        # Many legs whose float sums can round differently from the exact sums:
        # the order must still follow the exact rational score, then the key.
        rng = random.Random(4865)  # noqa: S311 - a seeded test sample
        for _ in range(200):
            keys = [f"k{i}" for i in range(6)]
            legs = [(f"leg{j}", rng.sample(keys, len(keys))) for j in range(7)]
            weights = {f"leg{j}": rng.choice([0.1, 0.2, 0.3, 0.7]) for j in range(7)}
            got = [i.key for i in FusionPolicy(k=1, weights=weights).fuse(legs)]
            assert got == [key for key, _ in _reference(legs, 1, weights)]

    def test_hybrid_refuses_a_document_without_a_str_doc_id(self):
        index = types.SimpleNamespace(backend_name="probe", _generation=None)
        bad = [
            RetrievalHit(doc=types.SimpleNamespace(doc_id=7), score=1.0, match_mode="x")
        ]
        with pytest.raises(ValueError, match="non-empty str doc_id"):
            RetrievalIndex._fuse_hybrid(index, "q", bad, [], RetrievalConfig())

    def test_fuse_by_rank_does_not_ask_a_weight_zero_leg_for_identities(self):
        legs = [
            LegResult(
                hits=tuple(_hits(["a"], "bm25")),
                outcome=LegOutcome(
                    leg=LegKind.LEXICAL, status=LegStatus.SUCCESS, hit_count=1
                ),
            ),
            LegResult(
                hits=(types.SimpleNamespace(),),
                outcome=LegOutcome(
                    leg=LegKind.GRAPH, status=LegStatus.SUCCESS, hit_count=1
                ),
            ),
        ]
        fused = fuse_by_rank(legs, weights={LegKind.GRAPH: 0.0})
        assert [hit.doc.doc_id for hit, _s, _c in fused] == ["a"]


class TestHybridSkipsWeightZeroLegs:
    @staticmethod
    def _index():
        from scikitplot.corpus import CorpusDocument  # noqa: PLC0415

        np = pytest.importorskip("numpy")
        docs = [
            CorpusDocument.create("a.txt", i, text).replace(
                embedding=np.asarray(vector, dtype=np.float32)
            )
            for i, (text, vector) in enumerate(
                [
                    ("alpha beta", [1.0, 0.0]),
                    ("beta gamma", [0.0, 1.0]),
                    ("gamma", [1.0, 1.0]),
                ]
            )
        ]
        index = RetrievalIndex(config=RetrievalConfig(backend="bruteforce"))
        index.build(docs)
        return index, np

    def test_alpha_0_never_runs_the_dense_leg(self):
        index, _np = self._index()
        response = index.search(
            "beta",
            config=RetrievalConfig(
                match_mode="hybrid", hybrid_alpha=0.0, backend="bruteforce"
            ),
        )
        legs = {leg.leg: leg.status for leg in response.legs}
        assert legs[LegKind.DENSE] == LegStatus.SKIPPED
        assert response.status.value == "success"
        assert len(response.hits) == 2

    def test_alpha_1_without_an_embedding_fails_instead_of_degrading(self):
        index, _np = self._index()
        response = index.search(
            "beta",
            config=RetrievalConfig(
                match_mode="hybrid", hybrid_alpha=1.0, backend="bruteforce"
            ),
        )
        legs = {leg.leg: leg.status for leg in response.legs}
        assert legs[LegKind.LEXICAL] == LegStatus.SKIPPED
        assert legs[LegKind.DENSE] == LegStatus.FAILED
        assert response.status.value == "failed"
        assert list(response.hits) == []
