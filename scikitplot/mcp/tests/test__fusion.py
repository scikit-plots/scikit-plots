# mcp/tests/test__fusion.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
One rank fusion (R2-1, decision D2): the contract and the MCP paths that use it.

The contract tests are the same in ``scikitplot/corpus/tests/test__fusion.py``,
each run against its own submodule's copy of ``_fusion.py``.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from scikitplot.mcp._core import RetrievedChunk
from scikitplot.mcp._fusion import FusionPolicy
from scikitplot.mcp._hybrid import HybridRetriever, reciprocal_rank_fusion

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
# The MCP paths fuse by the contract
# ---------------------------------------------------------------------------


class _Fixed:
    """A leg returning fixed chunks."""

    def __init__(self, ids):
        self._ids = ids

    def search(self, query, k=5):
        return [
            RetrievedChunk(text=f"text {i}", source_uri="", doc_id=i) for i in self._ids
        ][:k]


class TestMcpPaths:
    def test_reciprocal_rank_fusion_is_the_contract(self):
        rng = random.Random(1865)  # noqa: S311 - a seeded test sample
        alphabet = [f"d{i:02d}" for i in range(25)]
        for _ in range(400):
            legs = [
                (
                    rng.choice([0.0, 0.5, 1.0, rng.random() * 2]),
                    [rng.choice(alphabet) for _ in range(rng.randint(0, 20))],
                )
                for _ in range(rng.randint(0, 3))
            ]
            k = rng.randint(1, 80)
            got = list(reciprocal_rank_fusion(legs, k=k, key=str).items())
            reference = _reference(
                [(str(i), keys) for i, (_w, keys) in enumerate(legs)],
                k,
                {str(i): w for i, (w, _keys) in enumerate(legs)},
                0.0,
            )
            assert got == reference

    def test_hybrid_retriever_orders_by_the_contract(self):
        retriever = HybridRetriever(
            [_Fixed(["b", "a", "c"]), _Fixed(["c", "a"])], weights=[1.0, 2.0]
        )
        got = [chunk.doc_id for chunk in retriever.search("q", k=3)]
        reference = _reference(
            [("0", ["b", "a", "c"]), ("1", ["c", "a"])], 60, {"0": 1.0, "1": 2.0}, 0.0
        )
        assert got == [key for key, _score in reference]

    def test_mcp_and_corpus_fuse_alike(self):
        corpus = pytest.importorskip(
            "scikitplot.corpus",
            reason="the scikit-plots-mcp distribution can be installed without corpus",
        )
        rng = random.Random(2865)  # noqa: S311 - a seeded test sample
        alphabet = [f"d{i:02d}" for i in range(20)]
        for _ in range(300):
            lexical = [rng.choice(alphabet) for _ in range(rng.randint(0, 15))]
            dense = [rng.choice(alphabet) for _ in range(rng.randint(0, 15))]
            alpha = rng.choice([0.0, 0.5, 1.0, rng.random()])
            mcp = list(
                reciprocal_rank_fusion(
                    [(1.0 - alpha, lexical), (alpha, dense)], key=str
                ).items()
            )
            policy = corpus.FusionPolicy.from_alpha(alpha)
            ours = [
                (item.key, item.score)
                for item in policy.fuse([("lexical", lexical), ("dense", dense)])
            ]
            assert mcp == ours


class TestReviewFindings:
    def test_an_int_doc_id_leg_still_fuses(self):
        class _IntIds:
            def search(self, query, k=5):
                return [RetrievedChunk(text="t", source_uri="", doc_id=101)]

        retriever = HybridRetriever([_Fixed(["x1"]), _IntIds()])
        assert [str(c.doc_id) for c in retriever.search("q", k=3)] == ["101", "x1"]

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"weights": ["1.0"]}, TypeError, "weights"),
            ({"weights": [float("nan")]}, ValueError, "weights"),
            ({"rrf_k": 60.0}, TypeError, "rrf_k"),
            ({"rrf_k": 0}, ValueError, "rrf_k"),
        ],
    )
    def test_the_retriever_validates_through_the_contract(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            HybridRetriever([_Fixed(["a"])], **kwargs)
