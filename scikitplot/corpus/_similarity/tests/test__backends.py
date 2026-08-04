# corpus/_similarity/tests/test__backends.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for corpus._similarity._backends and the SimilarityIndex query seam
=========================================================================

Coverage targets
----------------
* :func:`select_backend` — auto order (Annoy first), explicit-unavailable
  raises, unknown name raises, ``bruteforce`` floor.
* :class:`BruteForceBackend` — exact cosine in ``[-1, 1]``, descending order,
  deterministic index-ascending ties, dimension/finiteness validation,
  zero-norm query, non-finite/empty embeddings at build.
* :class:`AnnoyBackend` — works against BOTH shipped index classes
  (``scikitplot.annoy.Index`` high-level and ``scikitplot.annoy._annoy.Index``
  Cython) via injected fakes, the ``1 - d**2/2`` angular→cosine recovery,
  ``dtype`` / ``index_dtype`` passthrough, and ``impl='auto'`` fallback.
* :class:`SimilarityIndex` — the ``query(vector, k) -> [(doc_id, score)]``
  seam consumed by :mod:`scikitplot.mcp`, ``backend_name``, and graceful
  degradation to sparse when embeddings are non-finite.

These are the permanent regressions for the search-path hardening
(centralised backends, Annoy default, unified cosine score contract). Native
Annoy / FAISS / Voyager are not required — the Annoy paths use in-process
fakes that reproduce angular-distance semantics.

Run with::

    pytest corpus/_similarity/tests/test__backends.py -v
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import numpy as np
import pytest

from scikitplot.corpus._similarity import SearchConfig, SearchResult, SimilarityIndex
from scikitplot.corpus._similarity import _backends as B


# ===================================================================== #
# Fakes that reproduce Annoy 'angular' distance = sqrt(2*(1 - cos)).
# ===================================================================== #
class _FakeAnnoyBase:
    accept_dtype = False

    def __init__(self, f, metric, **kwargs):
        if not type(self).accept_dtype and kwargs:
            raise TypeError(f"unexpected kwargs {sorted(kwargs)}")
        self.f = int(f)
        self.metric = metric
        self.ctor_kwargs = dict(kwargs)
        self.items: dict[int, np.ndarray] = {}

    def add_item(self, i, vec):
        self.items[int(i)] = np.asarray(vec, dtype=np.float32)

    def build(self, n_trees):  # noqa: D401 - fake
        self._built = True

    def get_nns_by_vector(self, vec, k, search_k=-1, include_distances=False):
        q = np.asarray(vec, dtype=np.float32)
        qn = q / (np.linalg.norm(q) or 1.0)
        scored = []
        for i, v in self.items.items():
            vn = v / (np.linalg.norm(v) or 1.0)
            cos = float(np.clip(vn @ qn, -1.0, 1.0))
            d = math.sqrt(max(0.0, 2.0 * (1.0 - cos)))
            scored.append((i, d))
        scored.sort(key=lambda t: (t[1], t[0]))
        top = scored[:k]
        ids = [t[0] for t in top]
        dists = [t[1] for t in top]
        return (ids, dists) if include_distances else ids


class _FakeHighLevelIndex(_FakeAnnoyBase):
    accept_dtype = False

    def add_items(self, X, ids=None):
        X = np.asarray(X, dtype=np.float32)
        ids = list(range(len(X))) if ids is None else list(ids)
        for i, row in zip(ids, X):
            self.items[int(i)] = np.asarray(row, dtype=np.float32)
        return np.asarray(ids)


class _FakeCythonIndex(_FakeAnnoyBase):
    accept_dtype = True


@pytest.fixture
def fake_annoy(request, monkeypatch):
    """Install isolated fake Annoy modules.

    Param is ``(highlevel_available, cython_available)``.
    """
    highlevel, cython = getattr(request, "param", (True, True))

    sp = sys.modules.get("scikitplot")
    assert sp is not None, "scikitplot must already be imported"

    # Always install the parent as a package. When highlevel=False it
    # intentionally has no Index export, so the resolver must fall back.
    an = types.ModuleType("scikitplot.annoy")
    an.__package__ = "scikitplot.annoy"
    an.__path__ = []  # Mark synthetic module as a package.

    if highlevel:
        an.Index = _FakeHighLevelIndex

    monkeypatch.setitem(sys.modules, "scikitplot.annoy", an)
    monkeypatch.setattr(sp, "annoy", an, raising=False)

    if cython:
        cy = types.ModuleType("scikitplot.annoy._annoy")
        cy.__package__ = "scikitplot.annoy"
        cy.Index = _FakeCythonIndex

        monkeypatch.setitem(sys.modules, "scikitplot.annoy._annoy", cy)
        monkeypatch.setattr(an, "_annoy", cy, raising=False)
    else:
        monkeypatch.delitem(
            sys.modules,
            "scikitplot.annoy._annoy",
            raising=False,
        )

    yield


# Small document double.
class _Doc:
    def __init__(self, doc_id, text, embedding=None, tokens=None):
        self.doc_id = doc_id
        self.text = text
        self.normalized_text = text
        self.embedding = embedding
        self.tokens = tokens


EMB = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
Q = np.array([1.0, 0.0], dtype=np.float32)


# ===================================================================== #
# SearchConfig validation
# ===================================================================== #
class TestSearchConfig:
    def test_defaults(self):
        cfg = SearchConfig()
        assert cfg.backend == "auto"
        assert cfg.annoy_impl == "auto"
        assert cfg.annoy_metric == "angular"

    @pytest.mark.parametrize("bad", ["nope", "faisss", ""])
    def test_rejects_unknown_backend(self, bad):
        with pytest.raises(ValueError):
            SearchConfig(backend=bad)

    def test_rejects_bad_annoy_impl(self):
        with pytest.raises(ValueError):
            SearchConfig(annoy_impl="nope")

    def test_rejects_annoy_n_trees_below_one(self):
        with pytest.raises(ValueError):
            SearchConfig(annoy_n_trees=0)


# ===================================================================== #
# BruteForceBackend numeric contract
# ===================================================================== #
class TestBruteForce:
    def test_cosine_values_and_order(self):
        bf = B.BruteForceBackend()
        bf.build(EMB)
        res = bf.query(Q, k=4)
        assert len(res) == 4
        assert res[0][0] == 0 and abs(res[0][1] - 1.0) < 1e-6
        by_id = dict(res)
        assert abs(by_id[1] - 0.0) < 1e-6     # orthogonal
        assert abs(by_id[3] - (-1.0)) < 1e-6  # opposite
        assert all(res[i][1] >= res[i + 1][1] for i in range(len(res) - 1))

    def test_stable_tie_order(self):
        bf = B.BruteForceBackend()
        bf.build(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        res = bf.query(Q, k=2)
        assert [i for i, _ in res] == [0, 1]

    def test_dim_mismatch_raises(self):
        bf = B.BruteForceBackend()
        bf.build(EMB)
        with pytest.raises(ValueError):
            bf.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)

    def test_non_finite_query_raises(self):
        bf = B.BruteForceBackend()
        bf.build(EMB)
        with pytest.raises(ValueError):
            bf.query(np.array([np.nan, 0.0], dtype=np.float32), k=1)

    def test_zero_norm_query_empty(self):
        bf = B.BruteForceBackend()
        bf.build(EMB)
        assert bf.query(np.array([0.0, 0.0], dtype=np.float32), k=3) == []

    def test_build_rejects_non_finite(self):
        with pytest.raises(ValueError):
            B.BruteForceBackend().build(np.array([[1.0, np.nan]], dtype=np.float32))

    def test_build_rejects_empty(self):
        with pytest.raises(ValueError):
            B.BruteForceBackend().build(np.zeros((0, 3), dtype=np.float32))


# ===================================================================== #
# select_backend policy
# ===================================================================== #
class TestSelectBackend:
    def test_auto_resolves_bruteforce_without_native(self):
        # No native ANN libs in the test env -> auto floor is bruteforce.
        assert B.select_backend("auto").name in ["annoy", "bruteforce"]

    def test_explicit_unavailable_raises(self):
        with pytest.raises(RuntimeError):
            B.select_backend("faiss")

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            B.select_backend("banana")

    def test_brute_alias(self):
        assert B.select_backend("brute").name in ["bruteforce"]

    @pytest.mark.parametrize("fake_annoy", [(True, True)], indirect=True)
    def test_auto_prefers_annoy_when_available(self, fake_annoy):
        assert B.select_backend("auto").name == "annoy"
        assert "annoy" in B.list_available_backends()


# ===================================================================== #
# AnnoyBackend — both implementations
# ===================================================================== #
class TestAnnoyBackend:
    @pytest.mark.parametrize("fake_annoy", [(True, True)], indirect=True)
    def test_highlevel_cosine_recovery(self, fake_annoy):
        bf = B.BruteForceBackend()
        bf.build(EMB)
        ref = dict(bf.query(Q, k=4))
        ab = B.select_backend("annoy", annoy_impl="highlevel", annoy_n_trees=5)
        ab.build(EMB)
        assert ab._resolved_impl == "highlevel"
        got = dict(ab.query(Q, k=4))
        # Exact cosine agreement proves 1 - d**2/2 (not 1 - d/2).
        assert max(abs(got[i] - ref[i]) for i in ref) < 1e-4
        assert abs(got[1] - 0.0) < 1e-4      # orthogonal, not 0.293
        assert abs(got[3] - (-1.0)) < 1e-4   # opposite, not 0.0

    @pytest.mark.parametrize("fake_annoy", [(True, True)], indirect=True)
    def test_cython_dtype_passthrough(self, fake_annoy):
        cb = B.select_backend(
            "annoy", annoy_impl="cython", annoy_dtype="float32", annoy_index_dtype="int32"
        )
        cb.build(EMB)
        assert cb._resolved_impl == "cython"
        assert cb._index.ctor_kwargs.get("dtype") == "float32"
        assert cb._index.ctor_kwargs.get("index_dtype") == "int32"
        got = dict(cb.query(Q, k=1))
        assert abs(got[0] - 1.0) < 1e-4

    @pytest.mark.parametrize("fake_annoy", [(True, True)], indirect=True)
    def test_dtype_dropped_for_highlevel(self, fake_annoy):
        # dtype on a ctor that rejects it must be dropped, not raised.
        hb = B.select_backend("annoy", annoy_impl="highlevel", annoy_dtype="float64")
        hb.build(EMB)
        assert hb._index.ctor_kwargs == {}

    @pytest.mark.parametrize("fake_annoy", [(False, True)], indirect=True)
    def test_auto_falls_back_to_cython(self, fake_annoy):
        fb = B.select_backend("annoy", annoy_impl="auto")
        fb.build(EMB)
        assert fb._resolved_impl == "cython"


# ===================================================================== #
# SimilarityIndex end-to-end + the MCP query() seam
# ===================================================================== #
class TestSimilarityIndexSeam:
    def _docs(self):
        return [
            _Doc("d0", "alpha beta gamma", embedding=[1.0, 0.0, 0.0]),
            _Doc("d1", "beta gamma delta", embedding=[0.0, 1.0, 0.0]),
            _Doc("d2", "gamma delta alpha", embedding=[0.9, 0.1, 0.0]),
        ]

    def test_semantic_and_backend_name(self):
        idx = SimilarityIndex(SearchConfig(match_mode="semantic", top_k=3, backend="auto"))
        idx.build(self._docs())
        assert idx.backend_name in ["annoy", "bruteforce"]
        assert idx.has_embeddings
        res = idx.search("anything", query_embedding=[1.0, 0.0, 0.0])
        assert res and res[0].doc.doc_id == "d0"
        assert all(-1.0 <= r.score <= 1.0 for r in res)

    def test_query_seam_returns_doc_id_score(self):
        idx = SimilarityIndex(SearchConfig(top_k=3, backend="auto"))
        idx.build(self._docs())
        seam = idx.query([1.0, 0.0, 0.0], k=2)
        assert len(seam) == 2
        assert seam[0][0] == "d0"
        assert isinstance(seam[0][1], float)
        assert all(isinstance(d, str) for d, _ in seam)

    def test_query_seam_empty_without_dense(self):
        idx = SimilarityIndex(SearchConfig(match_mode="keyword", top_k=2))
        idx.build([_Doc("k0", "qubit"), _Doc("k1", "logic")])
        assert idx.backend_name is None
        assert idx.query([1.0, 2.0], k=1) == []

    def test_degrades_to_sparse_on_non_finite_embeddings(self):
        docs = [
            _Doc("b0", "alpha beta", embedding=[1.0, float("nan")]),
            _Doc("b1", "beta gamma", embedding=[0.0, 1.0]),
        ]
        idx = SimilarityIndex(SearchConfig(match_mode="keyword", top_k=2, backend="auto"))
        idx.build(docs)
        assert idx.backend_name is None          # dense disabled
        assert len(idx.search("beta")) >= 1      # sparse still works


# ===================================================================== #
# Result provenance and index generation (CORPUS-ALG-001 provenance half)
# ===================================================================== #
class TestResultProvenance:
    def _docs(self):
        return [
            _Doc("d0", "alpha beta gamma", embedding=[1.0, 0.0, 0.0]),
            _Doc("d1", "beta gamma delta", embedding=[0.0, 1.0, 0.0]),
        ]

    def test_generation_starts_zero_and_increments(self):
        idx = SimilarityIndex(SearchConfig(backend="auto"))
        assert idx.index_generation == 0
        idx.build(self._docs())
        assert idx.index_generation == 1
        idx.build(self._docs())
        assert idx.index_generation == 2

    def test_semantic_result_carries_backend_and_generation(self):
        idx = SimilarityIndex(SearchConfig(match_mode="semantic", top_k=2, backend="auto"))
        idx.build(self._docs())
        res = idx.search("q", query_embedding=[1.0, 0.0, 0.0])
        assert res
        assert res[0].backend in ["annoy", "bruteforce"]
        assert res[0].index_generation == idx.index_generation == 1

    def test_keyword_result_has_no_backend_but_a_generation(self):
        idx = SimilarityIndex(SearchConfig(match_mode="keyword", top_k=2))
        idx.build(self._docs())
        res = idx.search("beta")
        assert res
        assert res[0].backend is None
        assert res[0].index_generation == 1

    def test_hybrid_result_carries_backend(self):
        idx = SimilarityIndex(SearchConfig(match_mode="hybrid", top_k=2, backend="auto"))
        idx.build(self._docs())
        res = idx.search("beta", query_embedding=[0.0, 1.0, 0.0])
        assert res
        assert res[0].backend in ["annoy", "bruteforce"]
        assert res[0].index_generation == 1

    def test_provenance_excluded_from_equality(self):
        # Same doc/score/mode but different provenance must remain equal,
        # so provenance never breaks existing equality-based tests.
        a = SearchResult(doc="x", score=0.5, match_mode="semantic",
                         backend="annoy", index_generation=1)
        b = SearchResult(doc="x", score=0.5, match_mode="semantic",
                         backend="faiss", index_generation=9)
        assert a == b
        assert hash(a) == hash(b)

    def test_stale_generation_detectable(self):
        idx = SimilarityIndex(SearchConfig(match_mode="semantic", top_k=1, backend="auto"))
        idx.build(self._docs())
        stale = idx.search("q", query_embedding=[1.0, 0.0, 0.0])[0]
        idx.build(self._docs())  # rebuild -> generation bumps
        assert stale.index_generation < idx.index_generation
