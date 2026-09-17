"""
Executable-example regressions (slice S-20).

The packaged examples imported a top-level ``rank_bm25`` this distribution does
not provide, and wrote their outputs as comments after ``>>>`` lines, so they
could not run and nothing would have noticed when they stopped being true.

See Also
--------
scikitplot.rank_bm25
"""

import doctest

import pytest

from ... import rank_bm25


def test_the_module_examples_execute():
    """Every ``>>>`` line in the package docstring runs and matches its output."""
    results = doctest.testmod(rank_bm25, verbose=False)
    assert results.attempted > 0
    assert results.failed == 0


def test_the_examples_use_this_distribution_s_import_path():
    """An example that imports a package we do not ship cannot be followed."""
    source = rank_bm25.__doc__ or ""
    assert "from scikitplot.rank_bm25 import" in source
    assert "from rank_bm25 import" not in source


def test_no_example_output_is_written_as_a_comment():
    """Output after ``>>>`` must be an expected value, not prose beside it."""
    lines = (rank_bm25.__doc__ or "").splitlines()
    commented = [line for line in lines
                 if line.strip().startswith(">>> #")
                 or line.strip().startswith("... #")]
    assert commented == []


@pytest.mark.parametrize("scorer_name", ["BM25Okapi", "BM25L", "BM25Plus"])
def test_every_exported_scorer_is_reachable_from_the_example_path(scorer_name):
    """What the examples import is what the package exports."""
    assert hasattr(rank_bm25, scorer_name)
    assert scorer_name in rank_bm25.__all__
