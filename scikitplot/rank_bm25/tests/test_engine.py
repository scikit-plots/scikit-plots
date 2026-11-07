"""
Exactness and contract tests for the precomputed scoring engine.

Every engine result is compared with the reference evaluation
(``backend="reference"``) with ``numpy.array_equal`` -- bit equality, NaN
included -- never with a tolerance: the engine's contract is that only the
time changes.
"""

from __future__ import annotations

import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from .._engine import SparseScoreEngine, numba_available, top_positions
from .._rank_bm25 import BM25L, BM25Okapi, BM25Plus

RECIPES = (BM25Okapi, BM25L, BM25Plus)
ENGINE_BACKENDS = ["numpy"] + (["numba"] if numba_available() else [])


def _corpus(seed, n_docs, vocab, empty_every=0):
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    docs = []
    for row in range(n_docs):
        if empty_every and row % empty_every == 0:
            docs.append([])
            continue
        length = rng.randint(1, 12)
        # Skewed draw, so some terms sit in over half the documents
        # (negative Okapi IDF, the epsilon floor) and some in one.
        docs.append([words[min(int(rng.expovariate(0.4)), vocab - 1)] for _ in range(length)])
    if not any(docs):
        docs[0] = ["w0"]
    return docs


def _queries(seed, vocab):
    rng = random.Random(seed + 1)
    words = [f"w{i}" for i in range(vocab)] + ["unknown-term"]
    queries = [[rng.choice(words) for _ in range(rng.randint(1, 5))] for _ in range(12)]
    queries += [[], ["unknown-term"], ["w0", "w0", "w0"], ["w1", "unknown-term", "w1"]]
    return queries


def _pair(cls, corpus, backend, **params):
    ids = [f"d{i}" for i in range(len(corpus))]
    engine = cls(corpus, doc_ids=ids, backend=backend, **params)
    reference = cls(corpus, doc_ids=ids, backend="reference", **params)
    return engine, reference


PARAMS = [
    {},
    {"b": 0.0},
    {"b": 1.0},
    {"k1": 0.01, "b": 0.3},
    {"k1": 3.0, "b": 0.9},
]


@pytest.mark.parametrize("backend", ENGINE_BACKENDS)
@pytest.mark.parametrize("cls", RECIPES)
@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("empty_every", [0, 5])
def test_scores_are_bit_identical_to_reference(backend, cls, params, seed, empty_every):
    corpus = _corpus(seed, n_docs=40, vocab=15, empty_every=empty_every)
    engine, reference = _pair(cls, corpus, backend, **params)
    for query in _queries(seed, 15):
        with np.errstate(divide="ignore", invalid="ignore"):
            got = engine.get_scores(query)
            want = reference.get_scores(query)
        assert got.dtype == want.dtype
        assert np.array_equal(got, want, equal_nan=True), (query, got, want)


@pytest.mark.parametrize("backend", ENGINE_BACKENDS)
@pytest.mark.parametrize("cls", RECIPES)
@pytest.mark.parametrize("seed", range(6))
def test_rankings_match_reference(backend, cls, seed):
    corpus = _corpus(seed, n_docs=60, vocab=8)  # small vocab: many ties
    engine, reference = _pair(cls, corpus, backend)
    for query in _queries(seed, 8):
        for n in (1, 3, 7, 60, 100):
            assert engine.get_top_ids(query, n=n) == reference.get_top_ids(query, n=n)
            assert engine.get_top_n(query, corpus, n=n) == reference.get_top_n(
                query, corpus, n=n
            )


def test_tiny_corpus_negative_idf_floor():
    # Every term in over half the documents: Okapi's epsilon floor applies and
    # a matching document can score below a non-matching one.
    corpus = [["a", "b"], ["a", "b"], ["a"], ["c"]]
    for backend in ENGINE_BACKENDS:
        engine, reference = _pair(BM25Okapi, corpus, backend)
        for query in (["a"], ["a", "b"], ["c", "a"]):
            assert np.array_equal(engine.get_scores(query), reference.get_scores(query))
            assert engine.get_top_ids(query, n=4) == reference.get_top_ids(query, n=4)


def test_single_document_corpus():
    for backend in ENGINE_BACKENDS:
        for cls in RECIPES:
            engine, reference = _pair(cls, [["x", "y", "x"]], backend)
            assert np.array_equal(engine.get_scores(["x"]), reference.get_scores(["x"]))


def test_reload_scores_identically(tmp_path):
    corpus = _corpus(3, n_docs=30, vocab=10)
    for cls in RECIPES:
        built = cls(corpus, doc_ids=[f"d{i}" for i in range(30)])
        built.save(tmp_path / cls.__name__)
        loaded = cls.load(tmp_path / cls.__name__)
        for query in _queries(3, 10):
            assert np.array_equal(built.get_scores(query), loaded.get_scores(query))


class TestBackendContract:
    def test_unknown_backend_is_refused(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            BM25Okapi([["a"]], backend="gpu")

    def test_backend_is_not_part_of_identity(self):
        corpus = [["a", "b"], ["c"]]
        a = BM25Okapi(corpus, backend="numpy")
        b = BM25Okapi(corpus, backend="reference")
        assert a.identity == b.identity

    def test_reported_backend(self):
        assert BM25Okapi([["a"]], backend="numpy").backend == "numpy"
        assert BM25Okapi([["a"]], backend="reference").backend == "reference"
        expected = "numba" if numba_available() else "numpy"
        assert BM25Okapi([["a"]]).backend == expected

    def test_overridden_formula_uses_reference(self):
        class Custom(BM25Okapi):
            def _term_contribution(self, idf, frequency, length):
                return idf * frequency * 2.0

        index = Custom([["a", "b"], ["b"]], backend="numpy")
        assert index.backend == "reference"
        assert index.get_scores(["b"]).tolist() == index._reference_scores(["b"]).tolist()

    def test_prepare_returns_self_and_builds_engine(self):
        index = BM25Okapi([["a", "b"], ["c"]], backend="numpy")
        assert index.prepare() is index
        assert index._engine_cache is not None
        assert index._engine.nbytes() > 0

    def test_disabled_numba_refuses_explicit_request(self, monkeypatch):
        monkeypatch.setenv("SCIKITPLOT_RANK_BM25_DISABLE_NUMBA", "1")
        with pytest.raises(ImportError, match="Numba is not importable"):
            BM25Okapi([["a"]], backend="numba")
        assert BM25Okapi([["a"]]).backend == "numpy"


def test_importing_the_package_does_not_import_numba():
    code = textwrap.dedent(
        """
        import sys
        from scikitplot.rank_bm25 import _rank_bm25
        index = _rank_bm25.BM25Okapi([["a", "b"], ["c"]], backend="numpy")
        index.get_scores(["a"])
        print("numba" in sys.modules)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.skipif(not numba_available(), reason="Numba not installed")
def test_concurrent_numba_queries_are_safe_under_every_threading_layer():
    # Under the "workqueue" layer a parallel=True kernel called from two
    # threads at once terminates the process; run in a child so a regression
    # fails this test instead of killing the test run.
    code = textwrap.dedent(
        """
        import threading
        from scikitplot.rank_bm25 import BM25Okapi
        corpus = [[f"t{(i * 7 + j) % 50}" for j in range(20)] for i in range(500)]
        ids = [str(i) for i in range(500)]
        index = BM25Okapi(corpus, doc_ids=ids, backend="numba").prepare()
        reference = BM25Okapi(corpus, doc_ids=ids, backend="reference")
        queries = [[f"t{k}", f"t{(k * 3) % 50}"] for k in range(30)]
        want = [reference.get_top_ids(q, n=5) for q in queries]
        failures = []
        def work():
            for _ in range(10):
                if [index.get_top_ids(q, n=5) for q in queries] != want:
                    failures.append(1)
        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        print("ok" if not failures else "mismatch")
        """
    )
    import os

    env = {**os.environ, "NUMBA_THREADING_LAYER": "workqueue"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "ok"


@pytest.mark.parametrize("seed", range(20))
def test_top_positions_equals_stable_full_sort(seed):
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 4, size=50).astype(float)  # heavy ties
    if seed % 3 == 0:
        scores[rng.integers(0, 50, size=3)] = np.nan
    mask = rng.random(50) < 0.7
    for n in (0, 1, 5, 17, 50, 80):
        rows = np.flatnonzero(mask)
        want = rows[np.argsort(-scores[rows], kind="stable")][:n]
        assert top_positions(scores, n, mask).tolist() == want.tolist()
        want_all = np.argsort(-scores, kind="stable")[:n]
        assert top_positions(scores, n).tolist() == want_all.tolist()


# -- the absent-term argument is checked, not assumed ---------------------------


@pytest.mark.parametrize("cls", RECIPES)
@pytest.mark.parametrize("b", [0.0, 1.0])
def test_empty_documents_need_no_special_rows(cls, b):
    """With the empty-document case defined, every row shares one absent value."""
    index = cls([["a"], [], ["a", "b", "b"]], b=b, backend="numpy")
    index.prepare()
    engine = index._engine
    assert not hasattr(engine, "irregular_rows")
    assert np.array_equal(
        index.get_scores(["a", "zz"]),
        cls([["a"], [], ["a", "b", "b"]], b=b, backend="reference").get_scores(
            ["a", "zz"]
        ),
    )


def test_an_engine_refuses_a_length_dependent_absent_contribution():
    """A formula breaking the exactness argument fails at build, not silently."""
    class LengthDependent(BM25Plus):
        def _term_contribution(self, idf, frequency, length):
            return idf * (frequency + length)

    index = LengthDependent([["a"], ["a", "b"]], backend="reference")
    with pytest.raises(RuntimeError, match="absent-term contribution"):
        SparseScoreEngine(index)
