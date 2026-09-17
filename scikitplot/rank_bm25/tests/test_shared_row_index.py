"""
Shared row-index contract in the lexical index (slice S-14).

S-12 made the guards durable without changing what they accept. This slice
changes what they accept, to the one rule in
:mod:`scikitplot._utils._indexing`: negative positions and booleans are refused,
and the result count is validated.

See Also
--------
scikitplot.rank_bm25._validation.require_count
scikitplot.rank_bm25._validation.require_index
"""

import numpy as np
import pytest

from .._validation import require_index
from .._rank_bm25 import BM25L, BM25Okapi, BM25Plus

CORPUS = [["a", "b"], ["b"], ["c"]]
DOCUMENTS = ["first", "second", "third"]
CLASSES = (BM25Okapi, BM25L, BM25Plus)


@pytest.mark.parametrize("scorer", CLASSES)
@pytest.mark.parametrize("position", [-1, -3])
def test_negative_batch_positions_are_refused(scorer, position):
    """A negative position no longer selects a row from the end."""
    with pytest.raises(IndexError):
        scorer(CORPUS).get_batch_scores(["a"], [position])


@pytest.mark.parametrize("scorer", CLASSES)
def test_boolean_batch_positions_are_refused(scorer):
    """Booleans never reach NumPy fancy indexing as a mask."""
    with pytest.raises(TypeError):
        scorer(CORPUS).get_batch_scores(["a"], [True, False])


@pytest.mark.parametrize("scorer", CLASSES)
def test_numpy_integer_positions_are_accepted(scorer):
    """Positions arriving as NumPy integers still work."""
    assert len(scorer(CORPUS).get_batch_scores(["a"], [np.int64(0)])) == 1


@pytest.mark.parametrize("scorer", CLASSES)
def test_valid_positions_are_unaffected(scorer):
    """The ordinary path is unchanged."""
    assert len(scorer(CORPUS).get_batch_scores(["a"], [0, 2])) == 2


@pytest.mark.parametrize("n", [-1, -5])
def test_negative_result_counts_are_refused(n):
    """A negative count silently truncated the ranking; it is refused."""
    with pytest.raises(ValueError):
        BM25Okapi(CORPUS).get_top_n(["a"], DOCUMENTS, n=n)


def test_zero_and_oversized_counts_still_work():
    """Zero means nothing, and asking for more than exists returns what exists."""
    index = BM25Okapi(CORPUS)
    assert index.get_top_n(["a"], DOCUMENTS, n=0) == []
    assert len(index.get_top_n(["a"], DOCUMENTS, n=99)) == 3


def test_boolean_result_count_is_refused():
    """``True`` is not a number of results."""
    with pytest.raises(TypeError):
        BM25Okapi(CORPUS).get_top_n(["a"], DOCUMENTS, n=True)


def test_both_modules_answer_the_same_question():
    """The lexical index and the shared primitive agree on every candidate."""
    index = BM25Okapi(CORPUS)
    for candidate in (-1, 0, 2, 3, True):
        try:
            require_index(candidate, 3)
        except Exception as exc:  # noqa: BLE001
            with pytest.raises(type(exc)):
                index.get_batch_scores(["a"], [candidate])
        else:
            assert len(index.get_batch_scores(["a"], [candidate])) == 1
