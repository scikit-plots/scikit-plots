import math
import re
import warnings

import numpy as np
import pytest

from .._rank_bm25 import BM25Okapi, BM25L, BM25Plus


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
RECIPE_CLASSES = (BM25Okapi, BM25L, BM25Plus)


def _historic_contribution(index, idf, frequency, length):
    """The pre-fix formulas, written inline, for a non-empty document."""
    norm = 1 - index.b + index.b * length / index.avgdl
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
        absent += index._term_contribution(index.idf.get(term) or 0, 0, index.avgdl)
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
