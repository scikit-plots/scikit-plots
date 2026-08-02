# scikitplot/annoy/_annoy/tests/test_metric_aliases.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the metric alias / error-message honesty follow-up.

``parse_metric`` accepts documented aliases (cosine, l2, lstsq, l1, cityblock,
taxicab, @, ., dotproduct, inner, innerproduct) in addition to the five canonical
names, but the "Invalid metric" errors previously listed only the canonical names
(and the two error sites disagreed). The messages now consistently list the
canonical names and point to the documented aliases. These tests lock both the
alias acceptance and the honest error text.
"""
import pytest

from scikitplot.annoy._annoy import annoylib as A

CANONICAL = ["angular", "euclidean", "manhattan", "dot", "hamming"]
ALIASES = ["cosine", "l2", "lstsq", "l1", "cityblock", "taxicab",
           "@", ".", "dotproduct", "inner", "innerproduct"]


@pytest.mark.parametrize("metric", CANONICAL + ALIASES)
def test_all_documented_metrics_and_aliases_accepted(metric):
    idx = A.Index(4, metric)
    idx.add_item(0, [1.0] * 4)
    idx.build(2)
    assert idx.get_n_items() == 1


def test_invalid_metric_error_lists_canonical_and_points_to_aliases():
    with pytest.raises(ValueError) as ei:
        A.Index(4, "not_a_metric")
    msg = str(ei.value)
    for name in CANONICAL:
        assert name in msg, f"error message omits canonical metric {name!r}"
    # honest about aliases, with a pointer to the full list
    assert "cosine" in msg
    assert "alias" in msg.lower()
    assert "docstring" in msg.lower()


def test_alias_maps_to_same_metric_as_canonical():
    # an alias must behave identically to its canonical metric
    def nns(metric):
        idx = A.Index(4, metric)
        for i in range(8):
            idx.add_item(i, [float(i), float(i), 0.0, 0.0])
        idx.build(5)
        return idx.get_nns_by_item(0, 4)

    assert nns("cosine") == nns("angular")
    assert nns("l2") == nns("euclidean")
    assert nns("taxicab") == nns("manhattan")
