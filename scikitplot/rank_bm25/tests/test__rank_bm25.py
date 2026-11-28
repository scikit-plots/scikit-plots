import json
import math
import re
import warnings

import numpy as np
import pytest

from .._rank_bm25 import BM25Adpt, BM25L, BM25Okapi, BM25Plus, BM25T


corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]
tokenized_corpus = [doc.split(" ") for doc in corpus]

algs = [
    BM25Okapi(tokenized_corpus),
    BM25L(tokenized_corpus),
    BM25Plus(tokenized_corpus)
]


def test_corpus_loading():
    for alg in algs:
        assert alg.corpus_size == 3
        assert alg.avgdl == 5
        assert alg.doc_len == [4, 6, 5]


def tokenizer(doc):
    return doc.split(" ")


def test_tokenizer():
    bm25 = BM25Okapi(corpus, tokenizer=tokenizer)
    assert bm25.corpus_size == 3
    assert bm25.avgdl == 5
    assert bm25.doc_len == [4, 6, 5]


# -- empty documents: the zero length normaliser (b == 1) ----------------------

#: Row 1 is empty; row 3 is non-empty and holds none of the query terms.
EMPTY_CORPUS = [["a", "b"], [], ["b", "c", "c"], ["d"]]
EMPTY_QUERY = ["b", "zz", "c", "b"]
RECIPE_CLASSES = (BM25Okapi, BM25L, BM25Plus, BM25Adpt, BM25T)


def _historic_contribution(index, idf, frequency, length, term=None):
    """The pre-fix formulas, written inline, for a non-empty document."""
    norm = 1 - index.b + index.b * length / index.avgdl
    if isinstance(index, (BM25Adpt, BM25T)):
        k1 = index.term_k1.get(term, 1.0)
        return idf * (frequency * (k1 + 1) / (frequency + k1 * norm))
    if isinstance(index, BM25Okapi):
        return idf * (frequency * (index.k1 + 1) / (frequency + index.k1 * norm))
    if isinstance(index, BM25L):
        ctd = frequency / norm
        return idf * (index.k1 + 1) * (ctd + index.delta) / (index.k1 + ctd + index.delta)
    return idf * (
        index.delta + (frequency * (index.k1 + 1)) / (index.k1 * norm + frequency)
    )


@pytest.mark.parametrize("backend", ["reference", "numpy"])
@pytest.mark.parametrize("b", [0.0, 0.5, 1.0, 1])
@pytest.mark.parametrize("cls", RECIPE_CLASSES)
def test_an_empty_document_scores_the_absent_term_value(cls, b, backend):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        index = cls(EMPTY_CORPUS, b=b, backend=backend)
        scores = index.get_scores(EMPTY_QUERY)
        batch = index.get_batch_scores(EMPTY_QUERY, [1])
    assert np.isfinite(scores).all()
    absent = 0.0
    for term in EMPTY_QUERY:
        absent += index._contribution(
            index.idf.get(term) or 0, 0, index.avgdl, index._k1_for(term)
        )
    assert scores[1] == absent == batch[0]
    # An empty document and a non-empty one lacking every query term tie: the
    # absent value does not depend on length.
    assert index.get_scores(["zz", "a"])[1] == index.get_scores(["zz", "a"])[2]


@pytest.mark.parametrize("b", [0.0, 0.3, 0.75, 1.0])
@pytest.mark.parametrize("cls", RECIPE_CLASSES)
def test_non_empty_documents_keep_their_historic_scores_bit_for_bit(cls, b):
    index = cls(EMPTY_CORPUS, b=b, backend="reference")
    scores = index.get_scores(EMPTY_QUERY)
    for row, document in enumerate(EMPTY_CORPUS):
        if not document:
            continue
        expected = 0.0
        for term in EMPTY_QUERY:
            expected += _historic_contribution(
                index,
                index.idf.get(term) or 0,
                index.doc_freqs[row].get(term) or 0,
                index.doc_len[row],
                term,
            )
        assert scores[row].tobytes() == np.float64(expected).tobytes()


@pytest.mark.parametrize("cls", RECIPE_CLASSES)
def test_the_length_normaliser_is_never_zero(cls):
    index = cls(EMPTY_CORPUS, b=1.0)
    assert index._length_norm(0) == 1.0
    lengths = np.array(index.doc_len)
    norm = index._length_norm(lengths)
    assert norm[1] == 1.0
    assert all(
        norm[row] == 1 - index.b + index.b * length / index.avgdl
        for row, length in enumerate(index.doc_len)
        if length
    )
    assert not math.isnan(float(index._length_norm(0)))


@pytest.mark.parametrize("cls", RECIPE_CLASSES)
def test_an_empty_document_still_scores_finite_after_reload(cls, tmp_path):
    built = cls(EMPTY_CORPUS, b=1.0)
    built.save(tmp_path)
    loaded = cls.load(tmp_path)
    assert np.isfinite(loaded.get_scores(EMPTY_QUERY)).all()
    assert np.array_equal(loaded.get_scores(EMPTY_QUERY), built.get_scores(EMPTY_QUERY))


# -- term-specific k1: BM25Adpt and BM25T --------------------------------------

ADAPTIVE_CLASSES = (BM25Adpt, BM25T)


def _zipf_corpus(seed=3, n_docs=300, vocabulary=200):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(vocabulary)]
    weights = 1.0 / np.arange(1, vocabulary + 1)
    weights /= weights.sum()
    return [
        [str(w) for w in rng.choice(words, size=int(rng.integers(3, 90)), p=weights)]
        for _ in range(n_docs)
    ]


@pytest.mark.parametrize("cls", ADAPTIVE_CLASSES)
def test_adaptive_engine_matches_the_reference_bit_for_bit(cls):
    corpus = _zipf_corpus()
    fast = cls(corpus, backend="numpy")
    reference = cls(corpus, backend="reference")
    rng = np.random.default_rng(0)
    for _ in range(25):
        query = [f"w{int(i)}" for i in rng.integers(0, 220, size=4)]
        expected = reference.get_scores(query)
        assert fast.get_scores(query).tobytes() == expected.tobytes()
        assert fast.get_batch_scores(query, [0, 7, 3]) == expected[[0, 7, 3]].tolist()


@pytest.mark.parametrize("cls", ADAPTIVE_CLASSES)
def test_adaptive_k1_is_per_term_read_only_and_has_no_k1_parameter(cls):
    index = cls(_zipf_corpus())
    assert set(index.term_k1) == set(index.idf)
    assert all(k > 0 for k in index.term_k1.values())
    assert "k1" not in index.parameters
    with pytest.raises(TypeError):
        index.term_k1["w0"] = 1.0
    with pytest.raises(TypeError):
        cls(_zipf_corpus(), k1=1.2)


def test_bm25t_weight_is_log_n_over_df():
    corpus = [["a", "b"], ["a"], ["a", "c", "c"]]
    index = BM25T(corpus)
    assert index.idf == {
        "a": 0.0,
        "b": math.log(3 / 1),
        "c": math.log(3 / 1),
    }


@pytest.mark.parametrize(
    "kwargs",
    [{"k1_min": 2.0, "k1_max": 1.0}, {"k1_min": 1.0, "k1_max": 1.0}],
)
def test_bm25adpt_refuses_an_empty_k1_interval(kwargs):
    with pytest.raises(ValueError, match="k1_min must be less than k1_max"):
        BM25Adpt(_zipf_corpus(), **kwargs)


@pytest.mark.parametrize("name", ["k1_min", "k1_max", "k1_fallback"])
def test_bm25adpt_refuses_a_non_positive_k1_bound(name):
    with pytest.raises(ValueError, match=name):
        BM25Adpt(_zipf_corpus(), **{name: 0.0})


def test_bm25adpt_parameters_change_the_identity():
    corpus = _zipf_corpus()
    assert BM25Adpt(corpus).identity != BM25Adpt(corpus, k1_fallback=0.5).identity


@pytest.mark.parametrize("cls", ADAPTIVE_CLASSES)
def test_adaptive_reload_recomputes_k1_bit_for_bit(cls, tmp_path):
    built = cls(_zipf_corpus(), doc_ids=[str(i) for i in range(300)])
    built.save(tmp_path)
    loaded = cls.load(tmp_path)
    assert dict(loaded.term_k1) == dict(built.term_k1)
    query = ["w0", "w5", "w80", "unseen"]
    assert loaded.get_scores(query).tobytes() == built.get_scores(query).tobytes()
    assert loaded.get_top_ids(query, 5) == built.get_top_ids(query, 5)


def _edit_saved_state(root, edit):
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    path = root / pointer["generation"] / "index.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    edit(state)
    path.write_text(json.dumps(state), encoding="utf-8")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda state: state.pop("term_k1"), "term_k1 is missing"),
        (lambda state: state["term_k1"].pop("w0"), "term_k1 is missing"),
        (lambda state: state["term_k1"].update(extra=1.0), "term_k1 is missing"),
        (lambda state: state["term_k1"].update(w0=0.0), "positive finite"),
        (lambda state: state["term_k1"].update(w0=-1.0), "positive finite"),
        (lambda state: state["term_k1"].update(w0="1.0"), "positive finite"),
        (lambda state: state["term_k1"].update(w0=True), "positive finite"),
    ],
)
@pytest.mark.parametrize("cls", ADAPTIVE_CLASSES)
def test_adaptive_reload_refuses_a_malformed_term_k1(cls, edit, message, tmp_path):
    cls(_zipf_corpus()).save(tmp_path)
    _edit_saved_state(tmp_path, edit)
    with pytest.raises(ValueError, match=message):
        cls.load(tmp_path)


@pytest.mark.parametrize("value", [123.0, 0.0005, 0.002 + 10.0])
def test_bm25adpt_reload_refuses_a_k1_its_parameters_cannot_produce(value, tmp_path):
    BM25Adpt(_zipf_corpus()).save(tmp_path)
    _edit_saved_state(tmp_path, lambda state: state["term_k1"].update(w0=value))
    with pytest.raises(ValueError, match="cannot produce"):
        BM25Adpt.load(tmp_path)


def test_bm25adpt_fitted_k1_lie_in_the_interval_or_take_the_fallback():
    index = BM25Adpt(_zipf_corpus(), k1_min=0.05, k1_max=3.0, k1_fallback=0.01)
    values = set(index.term_k1.values())
    assert 0.01 in values
    assert all(v == 0.01 or 0.05 <= v <= 3.0 for v in values)


def test_a_constant_k1_index_persists_no_term_k1(tmp_path):
    BM25Okapi(_zipf_corpus()).save(tmp_path)
    pointer = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    state = json.loads(
        (tmp_path / pointer["generation"] / "index.json").read_text(encoding="utf-8")
    )
    assert "term_k1" not in state


@pytest.mark.parametrize("cls", ADAPTIVE_CLASSES)
def test_adaptive_indexes_are_refused_by_another_scorer(cls, tmp_path):
    cls(_zipf_corpus()).save(tmp_path)
    other = BM25T if cls is BM25Adpt else BM25Adpt
    with pytest.raises(ValueError, match=f"load it with {cls.__name__}"):
        other.load(tmp_path)


@pytest.mark.parametrize("cls", ADAPTIVE_CLASSES)
def test_an_unseen_query_term_contributes_nothing(cls):
    index = cls(_zipf_corpus())
    with_unseen = index.get_scores(["w1", "never-seen"])
    assert with_unseen.tobytes() == (index.get_scores(["w1"]) + 0.0).tobytes()
