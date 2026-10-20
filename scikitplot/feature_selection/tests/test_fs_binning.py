# scikitplot/feature_selection/tests/test_fs_binning.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Bin recommendations: Knuth's rule against its formula, policies and edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import special

from scikitplot.feature_selection import (
    knuth_bin_count,
    make_kbins_discretizer,
    profile_features,
)


def _knuth_reference(data, m_max):
    """Negative log posterior of Knuth (2006), as in astropy's ``_KnuthF.eval``."""

    def neg_logp(m):
        nk, _ = np.histogram(data, np.linspace(data.min(), data.max(), m + 1))
        n = data.size
        return -(
            n * np.log(m)
            + special.gammaln(0.5 * m)
            - m * special.gammaln(0.5)
            - special.gammaln(n + 0.5 * m)
            + np.sum(special.gammaln(nk + 0.5))
        )

    return min(range(1, m_max + 1), key=lambda m: (neg_logp(m), m))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("size", [40, 1000, 5000])
@pytest.mark.parametrize("dist", ["normal", "exponential", "uniform"])
def test_knuth_count_is_the_exhaustive_argmax(seed, size, dist):
    data = getattr(np.random.default_rng(seed), dist)(size=size)
    assert knuth_bin_count(data, 1, 80) == _knuth_reference(data, 80)


def test_knuth_count_contract():
    assert knuth_bin_count([3.0, 3.0, 3.0], 2, 9) == 2
    with pytest.raises(ValueError):
        knuth_bin_count([], 1, 5)
    with pytest.raises(ValueError):
        knuth_bin_count([1.0, np.inf], 1, 5)
    with pytest.raises(ValueError):
        knuth_bin_count([1.0, 2.0], 5, 2)


def _rec(values, **binning):
    X = pd.DataFrame({"x": values})
    return profile_features(X, {"binning": binning} if binning else None)["x"].binning


def test_suitability_states():
    rng = np.random.default_rng(0)
    assert _rec(rng.normal(size=500)).state == "true"
    assert _rec(list("ab") * 50).state == "not_applicable"
    assert _rec([1.0, 2.0, 3.0]).state == "unknown"
    assert _rec([0, 1] * 50).reasons == ("two_or_fewer_values",)
    assert _rec([1, 2, 3, 4] * 25, rule="fixed", n_bins=10).reasons == (
        "no_more_distinct_values_than_bins",
    )


def test_declared_categorical_is_not_binned():
    X = pd.DataFrame({"x": np.random.default_rng(0).normal(size=200)})
    rec = profile_features(X, {"kinds": {"x": "categorical_nominal"}})["x"].binning
    assert rec.state == "not_applicable"


def test_fixed_rule_and_uniform_edges():
    rec = _rec(np.linspace(0.0, 10.0, 200), rule="fixed", n_bins=5)
    assert rec.n_bins == 5 and rec.edges == tuple(np.linspace(0.0, 10.0, 6))


def test_counts_are_clipped_and_reported():
    rec = _rec(np.random.default_rng(0).normal(size=5000), rule="sqrt", max_bins=10)
    assert rec.n_bins == 10 and rec.rule_n_bins > 10
    assert "clipped_to_max_bins" in rec.reasons


def test_tied_quantiles_merge_bins():
    values = [0.0] * 150 + list(np.linspace(1.0, 2.0, 50))
    rec = _rec(values, rule="fixed", n_bins=10, strategy="quantile")
    assert rec.n_bins < 10 and "tied_quantiles_merged" in rec.reasons
    assert len(rec.edges) == rec.n_bins + 1


@pytest.mark.parametrize(
    "rule",
    ["auto", "fd", "scott", "sturges", "doane", "rice", "sqrt", "stone", "knuth"],
)
def test_every_rule_gives_a_valid_recommendation(rule):
    rec = _rec(np.random.default_rng(1).gamma(2.0, size=800), rule=rule)
    assert rec.state == "true" and 2 <= rec.n_bins <= 64
    assert np.all(np.diff(rec.edges) > 0)


def test_make_kbins_discretizer_uses_the_recommended_counts():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=400),
            "b": rng.gamma(2.0, size=400),
            "c": list("xy") * 200,
        }
    )
    schema = profile_features(X)
    columns, kb = make_kbins_discretizer(schema, encode="ordinal")
    assert kb.strategy == "uniform"
    assert columns == ["a", "b"]
    assert list(kb.n_bins) == [schema["a"].binning.n_bins, schema["b"].binning.n_bins]
    kb.fit(X[columns])
    with pytest.raises(ValueError, match="no suitable"):
        make_kbins_discretizer(schema, ["c"])
