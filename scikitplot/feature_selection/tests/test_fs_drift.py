# scikitplot/feature_selection/tests/test_fs_drift.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""ReferenceProfile: frozen buckets, statuses, metrics and persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scikitplot.feature_selection import (
    IncompatibleSchemaError,
    ReferenceProfile,
    profile_features,
)


def _train(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "c": rng.choice(list("abc"), n, p=[0.5, 0.3, 0.2]),
            "k": rng.integers(0, 5, n),
            "t": pd.date_range("2020", periods=n, freq="h"),
        }
    )


@pytest.fixture(scope="module")
def reference():
    return ReferenceProfile.fit(_train())


def _status(report):
    return report.set_index("feature")["status"].to_dict()


def test_the_same_distribution_is_stable(reference):
    report = reference.drift(_train(seed=1))
    assert _status(report) == {
        "x": "stable",
        "c": "stable",
        "k": "stable",
        "t": "not_applicable",
    }
    assert (report["psi"].dropna() < 0.1).all()


def test_the_training_data_itself_has_zero_psi(reference):
    report = reference.drift(_train()).set_index("feature")
    assert report.loc["x", "psi"] == pytest.approx(0.0, abs=1e-12)
    assert report.loc["c", "js_distance"] == pytest.approx(0.0, abs=1e-12)


def test_a_shift_is_drift(reference):
    later = _train(seed=2).assign(x=lambda d: d["x"] + 1.5)
    row = reference.drift(later).set_index("feature").loc["x"]
    assert row["status"] == "drift" and 0.0 < row["js_distance"] <= 1.0


def test_missingness_is_part_of_the_distribution(reference):
    later = _train(seed=3)
    later.loc[later.index[:400], "c"] = None
    row = reference.drift(later).set_index("feature").loc["c"]
    assert row["missing_delta"] == pytest.approx(0.4)
    assert row["status"] == "drift"


def test_unseen_categories_are_reported(reference):
    later = _train(seed=4)
    later.loc[later.index[:100], "c"] = "zzz"
    row = reference.drift(later).set_index("feature").loc["c"]
    assert row["unseen_ratio"] == pytest.approx(0.1)


def test_truncated_references_cannot_know_unseen_values():
    train = pd.DataFrame({"c": [f"v{i % 80}" for i in range(4000)]})
    ref = ReferenceProfile.fit(train, policy={"drift": {"max_categories": 10}})
    assert ref.features[0].truncated
    assert ref.drift(train).loc[0, "unseen_ratio"] is None


def test_integer_categories_survive_a_float_cast():
    train = pd.DataFrame({"k": np.arange(1000) % 4})
    later = pd.DataFrame({"k": (np.arange(1000) % 4).astype(float)})
    later.loc[0, "k"] = np.nan
    row = ReferenceProfile.fit(train).drift(later).loc[0]
    assert row["unseen_ratio"] == 0.0 and row["status"] == "stable"


def test_mixed_types_sharing_a_key_are_refused():
    train = pd.DataFrame({"c": pd.Series([1, "1", "a", "b"] * 20, dtype=object)})
    schema = profile_features(train, {"kinds": {"c": "categorical_nominal"}})
    with pytest.raises(ValueError, match="share the category key"):
        ReferenceProfile.fit(train, schema)


@pytest.mark.parametrize(
    ("later", "status"),
    [
        (lambda d: d.drop(columns=["x"]), "missing_column"),
        (lambda d: d.assign(x=d["x"].astype(str) + "!"), "type_changed"),
        (lambda d: d.head(10), "insufficient"),
    ],
)
def test_structural_statuses(reference, later, status):
    report = reference.drift(later(_train(seed=5)))
    assert _status(report)["x"] == status


def test_infinities_fall_in_the_end_buckets(reference):
    later = _train(seed=6)
    later.loc[later.index[:5], "x"] = np.inf
    report = reference.drift(later).set_index("feature")
    assert report.loc["x", "status"] in ("stable", "warn")


def test_thresholds_come_from_the_policy():
    strict = ReferenceProfile.fit(
        _train(), policy={"drift": {"psi_warn": 0.0, "psi_alert": 0.001}}
    )
    assert _status(strict.drift(_train(seed=7)))["x"] == "drift"


@pytest.mark.parametrize("fmt", ["json", "yaml", "toml"])
def test_reference_round_trips(reference, fmt, tmp_path):
    pytest.importorskip("yaml") if fmt == "yaml" else None
    pytest.importorskip("tomli_w") if fmt == "toml" else None
    assert ReferenceProfile.loads(reference.dumps(fmt), fmt) == reference
    path = reference.save(tmp_path / f"ref.{fmt}")
    later = _train(seed=8)
    pd.testing.assert_frame_equal(
        ReferenceProfile.load(path).drift(later), reference.drift(later)
    )


def test_reference_documents_are_checked(reference):
    data = reference.to_dict()
    with pytest.raises(IncompatibleSchemaError):
        ReferenceProfile.from_dict({**data, "schema_version": 2})
    with pytest.raises(ValueError):
        ReferenceProfile.from_dict({**data, "colour": 1})
    broken = dict(data, features=[dict(data["features"][0], method="magic")])
    with pytest.raises(ValueError, match="method"):
        ReferenceProfile.from_dict(broken)


def test_explicit_columns_and_missing_names():
    train = _train()
    ref = ReferenceProfile.fit(train, columns=["k"])
    assert ref.names == ["k"]
    with pytest.raises(KeyError):
        ReferenceProfile.fit(train, columns=["nope"])


def test_identifiers_are_not_compared():
    train = _train().assign(code=[f"id{i}" for i in range(1000)])
    report = ReferenceProfile.fit(train).drift(train.assign(code="x"))
    assert _status(report)["code"] == "not_applicable"
