"""
Input-contract regressions for :mod:`scikitplot.rank_bm25` (slice S-11).

A scorer is built from a sequence of *token sequences*. Passing raw strings is
the most likely mistake, and it previously succeeded: each document was
iterated element-wise, so a string indexed as characters and produced a
plausible but meaningless ranking. Empty inputs previously surfaced as
``ZeroDivisionError`` from the averaging arithmetic, which names neither the
problem nor the fix.

See Also
--------
scikitplot.rank_bm25._rank_bm25.BM25
"""

import pytest

from .._rank_bm25 import BM25L, BM25Okapi, BM25Plus

CLASSES = (BM25Okapi, BM25L, BM25Plus)
GOOD = [["hello", "there"], ["quite", "windy"]]


@pytest.mark.parametrize("scorer", CLASSES)
def test_raw_strings_are_refused(scorer):
    """A corpus of strings is refused rather than indexed as characters."""
    with pytest.raises(TypeError) as excinfo:
        scorer(["hello world", "quite windy"])
    message = str(excinfo.value)
    assert "tokenizer" in message
    assert "0" in message


@pytest.mark.parametrize("scorer", CLASSES)
def test_bytes_documents_are_refused(scorer):
    """Bytes are iterable too, and are refused for the same reason."""
    with pytest.raises(TypeError):
        scorer([b"hello world"])


@pytest.mark.parametrize("scorer", CLASSES)
def test_empty_corpus_names_the_problem(scorer):
    """An empty corpus raises a named input error, not arithmetic."""
    with pytest.raises(ValueError) as excinfo:
        scorer([])
    assert "at least one document" in str(excinfo.value)


@pytest.mark.parametrize("scorer", CLASSES)
@pytest.mark.parametrize("corpus", [[[]], [[], []]], ids=["one-empty", "all-empty"])
def test_corpus_without_tokens_names_the_problem(scorer, corpus):
    """A corpus whose documents hold no tokens is a distinct, named error."""
    with pytest.raises(ValueError) as excinfo:
        scorer(corpus)
    message = str(excinfo.value)
    assert "token" in message
    assert "at least one document" not in message


@pytest.mark.parametrize("scorer", CLASSES)
def test_empty_documents_alongside_real_ones_are_still_allowed(scorer):
    """An empty document is legitimate as long as the corpus holds tokens."""
    index = scorer([[], ["hello"]])
    assert index.corpus_size == 2
    assert index.doc_len == [0, 1]


@pytest.mark.parametrize("scorer", CLASSES)
def test_tokenized_corpus_is_unaffected(scorer):
    """The ordinary path is unchanged."""
    index = scorer(GOOD)
    assert index.corpus_size == 2
    assert index.avgdl == 2
    assert len(index.get_scores(["hello"])) == 2


def test_tokenizer_path_still_accepts_strings():
    """With a tokenizer the caller supplies strings by design; that still works."""
    index = BM25Okapi(["hello there", "quite windy"], tokenizer=str.split)
    assert index.corpus_size == 2
    assert index.doc_len == [2, 2]
