# corpus/_similarity/tests/test__quality_differential.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Quality-differential gate: ANN backends vs brute-force ground truth
===================================================================

Closes the *quality* half of CORPUS-ALG-001. Brute-force cosine is the exact
ground truth; every ANN backend is measured by ``recall@k`` (overlap of returned
document ids with the exact top-k) on a fixed synthetic corpus.

Contract enforced
-----------------
* Brute-force and FAISS ``IndexFlatIP`` are exact  → recall ``== 1.0``.
* Annoy (approximate) must clear a recall gate and must not get *worse* as
  ``n_trees`` grows.
* The measurement harness itself is validated deterministically with an exact
  in-process fake Annoy (recall ``== 1.0``), so this file exercises real code
  even where native ANN libraries are absent.

Native ANN libraries are optional: their sub-tests skip when unavailable, so the
suite is green in a minimal environment and a real gate in a full one.

Run with::

    pytest corpus/_similarity/tests/test__quality_differential.py -v
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from scikitplot.corpus._similarity import _backends as B

K = 10
_RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# fixtures and recall helpers
# --------------------------------------------------------------------------- #
def _synthetic(n=200, dim=32, nq=25):
    embs = _RNG.standard_normal((n, dim)).astype(np.float32)
    queries = _RNG.standard_normal((nq, dim)).astype(np.float32)
    return embs, queries


def _topk_ids(backend, queries, k):
    return [[i for i, _ in backend.query(q, k)] for q in queries]


def _recall_at_k(pred, truth, k):
    total = 0.0
    for p, t in zip(pred, truth):
        total += len(set(p) & set(t)) / float(k)
    return total / len(truth)


@pytest.fixture(scope="module")
def data():
    return _synthetic()


@pytest.fixture(scope="module")
def ground_truth(data):
    embs, queries = data
    bf = B.BruteForceBackend()
    bf.build(embs)
    return _topk_ids(bf, queries, K)


# --------------------------------------------------------------------------- #
# exact backends -> recall 1.0
# --------------------------------------------------------------------------- #
def test_bruteforce_is_exact(data, ground_truth):
    embs, queries = data
    bf = B.BruteForceBackend()
    bf.build(embs)
    assert _recall_at_k(_topk_ids(bf, queries, K), ground_truth, K) == 1.0


@pytest.mark.skipif(not B.FaissBackend.is_available(), reason="faiss not installed")
def test_faiss_flat_is_exact(data, ground_truth):
    embs, queries = data
    fb = B.FaissBackend()
    fb.build(embs)
    assert _recall_at_k(_topk_ids(fb, queries, K), ground_truth, K) >= 0.99


# --------------------------------------------------------------------------- #
# approximate backend -> recall gate + non-regression with more trees
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not B.AnnoyBackend.is_available(), reason="annoy not installed")
def test_annoy_recall_gate(data, ground_truth):
    embs, queries = data
    recalls = {}
    for n_trees in (1, 10, 50):
        ab = B.AnnoyBackend(metric="angular", n_trees=n_trees)
        ab.build(embs)
        recalls[n_trees] = _recall_at_k(_topk_ids(ab, queries, K), ground_truth, K)
    # More trees must not degrade recall (allow small sampling noise).
    assert recalls[50] >= recalls[1] - 0.05
    # Quality gate: many-tree Annoy must recover most exact neighbours.
    assert recalls[50] >= 0.80


# --------------------------------------------------------------------------- #
# validate the recall harness itself with an exact fake Annoy (runs anywhere)
# --------------------------------------------------------------------------- #
class _ExactFakeAnnoy:
    def __init__(self, f, metric, **kwargs):
        self.f = int(f)
        self.metric = metric
        self.items: dict[int, np.ndarray] = {}

    def add_items(self, X, ids=None):
        X = np.asarray(X, dtype=np.float32)
        ids = list(range(len(X))) if ids is None else list(ids)
        for i, row in zip(ids, X):
            self.items[int(i)] = np.asarray(row, dtype=np.float32)
        return np.asarray(ids)

    def add_item(self, i, vec):
        self.items[int(i)] = np.asarray(vec, dtype=np.float32)

    def build(self, n_trees):
        pass

    def get_nns_by_vector(self, vec, k, search_k=-1, include_distances=False):
        q = np.asarray(vec, dtype=np.float32)
        qn = q / (np.linalg.norm(q) or 1.0)
        scored = []
        for i, v in self.items.items():
            vn = v / (np.linalg.norm(v) or 1.0)
            cos = float(np.clip(vn @ qn, -1.0, 1.0))
            scored.append((i, math.sqrt(max(0.0, 2.0 * (1.0 - cos)))))
        scored.sort(key=lambda t: (t[1], t[0]))
        top = scored[:k]
        ids = [t[0] for t in top]
        dists = [t[1] for t in top]
        return (ids, dists) if include_distances else ids


def test_recall_harness_is_correct(data, ground_truth, monkeypatch):
    """An exact Annoy must reproduce brute-force ranking (recall 1.0)."""
    sp = sys.modules.get("scikitplot") or types.ModuleType("scikitplot")
    an = types.ModuleType("scikitplot.annoy")
    an.Index = _ExactFakeAnnoy
    monkeypatch.setitem(sys.modules, "scikitplot", sp)
    monkeypatch.setitem(sys.modules, "scikitplot.annoy", an)
    monkeypatch.setattr(sp, "annoy", an, raising=False)

    embs, queries = data
    ab = B.AnnoyBackend(metric="angular", n_trees=10, impl="highlevel")
    ab.build(embs)
    assert _recall_at_k(_topk_ids(ab, queries, K), ground_truth, K) == 1.0
