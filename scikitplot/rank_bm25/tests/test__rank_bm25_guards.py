"""
Guard durability regressions for :mod:`scikitplot.rank_bm25` (slice S-12).

Every check here runs the guarded call twice: once in this interpreter and once
in a child started with ``-O``, which removes ``assert`` statements. A guard
that only holds in the first case is not a guard, because ``-O`` is a normal
way to run production code.

See Also
--------
scikitplot.rank_bm25._rank_bm25.BM25.get_top_n
scikitplot.rank_bm25._rank_bm25.BM25Okapi.get_batch_scores
"""

import json
import subprocess
import sys
import textwrap

import pytest

from .._rank_bm25 import BM25L, BM25Okapi, BM25Plus

CORPUS = [["hello", "there"], ["quite", "windy"], ["the", "weather"]]
CLASSES = (BM25Okapi, BM25L, BM25Plus)


def _run_child(body, optimised):
    """
    Execute ``body`` in a fresh interpreter and return its JSON result.

    Parameters
    ----------
    body : str
        Source executed after ``scikitplot.rank_bm25`` has been imported as
        ``rb``. It must print one JSON object.
    optimised : bool
        Whether to pass ``-O``, which removes ``assert`` statements.

    Returns
    -------
    dict
        The parsed object printed by the child.
    """
    script = textwrap.dedent(
        """
        import json
        from scikitplot.rank_bm25 import _rank_bm25 as rb
        """
    ) + textwrap.dedent(body)
    flags = ["-O"] if optimised else []
    proc = subprocess.run(
        [sys.executable, *flags, "-c", script],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("optimised", [False, True], ids=["default", "-O"])
def test_get_top_n_rejects_mismatched_documents(optimised):
    """A document sequence that does not match the index is refused under ``-O`` too."""
    result = _run_child(
        """
        index = rb.BM25Okapi([["a", "b"], ["b"], ["c"]])
        try:
            index.get_top_n(["a"], ["only-one-document"], n=1)
            print(json.dumps({"raised": None}))
        except Exception as exc:
            print(json.dumps({"raised": type(exc).__name__, "message": str(exc)}))
        """,
        optimised,
    )
    assert result["raised"] == "ValueError"
    assert "3" in result["message"] and "1" in result["message"]


@pytest.mark.parametrize("optimised", [False, True], ids=["default", "-O"])
def test_get_batch_scores_rejects_out_of_range_position(optimised):
    """An out-of-range position is refused in every scorer, under ``-O`` too."""
    result = _run_child(
        """
        out = {}
        for name in ("BM25Okapi", "BM25L", "BM25Plus"):
            index = getattr(rb, name)([["a", "b"], ["b"], ["c"]])
            try:
                index.get_batch_scores(["a"], [99])
                out[name] = None
            except Exception as exc:
                out[name] = type(exc).__name__
        print(json.dumps(out))
        """,
        optimised,
    )
    assert result == {"BM25Okapi": "IndexError", "BM25L": "IndexError",
                      "BM25Plus": "IndexError"}


@pytest.mark.parametrize("scorer", CLASSES)
def test_batch_scores_message_names_the_offending_position(scorer):
    """The refusal says which position was out of range and what the bound is."""
    index = scorer(CORPUS)
    with pytest.raises(IndexError) as excinfo:
        index.get_batch_scores(["hello"], [0, 7])
    message = str(excinfo.value)
    assert "7" in message
    assert "3" in message


@pytest.mark.parametrize("scorer", CLASSES)
def test_valid_positions_still_score(scorer):
    """The guard does not change the result for positions that were always valid."""
    index = scorer(CORPUS)
    scores = index.get_batch_scores(["hello"], [0, 2])
    assert len(scores) == 2
    assert scores[0] != 0.0


def test_get_top_n_still_ranks_matching_documents():
    """A correctly sized document sequence is unaffected by the guard."""
    documents = ["hello there", "quite windy", "the weather"]
    index = BM25Okapi(CORPUS)
    assert index.get_top_n(["quite"], documents, n=1) == ["quite windy"]
