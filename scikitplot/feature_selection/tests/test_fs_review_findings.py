# scikitplot/feature_selection/tests/test_fs_review_findings.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regressions for the independent review of FS-1 (Rule 92); each failed before its fix."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from scikitplot.feature_selection import (
    FeatureFilter,
    FeatureLineage,
    FeaturePolicy,
    FeatureReference,
    FeatureSchema,
    LineageError,
    ReferenceProfile,
    profile_features,
    trace_pipeline,
)


def _changed_outputs(est, X, column):
    """Output positions that change when only ``column`` is perturbed (ground truth)."""

    def dense(m):
        return np.asarray(m.toarray() if hasattr(m, "toarray") else m, dtype=float)

    base = dense(est.transform(X))
    moved = X.copy()
    moved[column] = moved[column] * 1.7 + 3.0
    after = dense(est.transform(moved))
    return {
        j
        for j in range(base.shape[1])
        if not np.allclose(base[:, j], after[:, j], equal_nan=True)
    }


# 1 --------------------------------------------------------------------------


def test_unused_categories_are_not_values():
    one = profile_features(
        pd.DataFrame({"x": pd.Categorical(["a"] * 40, categories=list("abc"))})
    )
    assert one["x"].stats.n_unique == 1 and one["x"].is_("constant")
    two = profile_features(
        pd.DataFrame({"x": pd.Categorical(["a", "b"] * 20, categories=list("abcd"))})
    )
    assert two["x"].is_("binary") and two["x"].kind == "binary"
    X = pd.DataFrame(
        {"c": pd.Categorical(["a"] * 40, categories=list("abc")), "y": np.arange(40.0)}
    )
    assert list(FeatureFilter().fit(X).get_feature_names_out()) == ["y"]


# 2 --------------------------------------------------------------------------


def test_duplicate_fallback_names_are_refused():
    X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) ** 2})
    # ColumnTransformer has verbose_feature_names_out on every supported
    # scikit-learn; FeatureUnion only from 1.5.
    ct = ColumnTransformer(
        [("s1", StandardScaler(), ["a"]), ("s2", MinMaxScaler(), ["a"])],
        verbose_feature_names_out=False,
    ).fit(X)
    with pytest.raises(LineageError, match="duplicate names"):
        trace_pipeline(ct, X=X)
    if "verbose_feature_names_out" in FeatureUnion.__init__.__code__.co_varnames:
        fu = FeatureUnion(
            [("s1", StandardScaler()), ("s2", MinMaxScaler())],
            verbose_feature_names_out=False,
        ).fit(X)
        with pytest.raises(LineageError, match="duplicate names"):
            trace_pipeline(fu, X=X)


def test_values_by_name_are_refused_when_names_repeat():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    class Twice(StandardScaler):
        def transform(self, X, copy=None):
            out = super().transform(X)
            return np.c_[out, out]

        def get_feature_names_out(self, input_features=None):
            names = list(super().get_feature_names_out(input_features))
            return np.asarray(names + names, dtype=object)

        def get_feature_lineage(self, input_features=None):
            from scikitplot.feature_selection import FeatureMapping

            n = len(input_features)
            return FeatureMapping(tuple((i,) for i in range(n)) * 2, "expand")

    lin = trace_pipeline(Twice().fit(X), X=X)
    with pytest.raises(ValueError, match="duplicate names"):
        lin.attribute(pd.Series([1.0, 2.0, 30.0, 40.0], index=lin.output_names))
    assert lin.attribute([1.0, 2.0, 30.0, 40.0]).by_input.to_dict() == {
        "a": 31.0,
        "b": 42.0,
    }


# 3 --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "est",
    [
        Normalizer(),
        TfidfTransformer(),
        KNNImputer(n_neighbors=2),
        IterativeImputer(random_state=0, max_iter=3),
    ],
    ids=lambda e: type(e).__name__,
)
def test_steps_that_combine_columns_are_not_exact(est):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((30, 3)) + 0.1, columns=list("abc"))
    X.loc[::4, "b"] = np.nan if "Imputer" in type(est).__name__ else X.loc[::4, "b"]
    est.fit(X)
    lin = trace_pipeline(est, X=X)
    truth = _changed_outputs(est, X, "a")
    claimed = {j for j, name in enumerate(lin.output_names) if "a" in lin.origins(name)}
    assert truth <= claimed  # never fewer parents than the real dependency
    affected = [lin.output_names[j] for j in truth if lin.output_names[j] != "a"]
    assert affected  # the data really mixes columns ...
    for name in affected:  # ... so nothing it touches is attributed to one column
        assert lin.quality(name) == "mixed"


def test_imputer_indicators_stay_exact():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, np.nan, 4.0]})
    est = KNNImputer(n_neighbors=1, add_indicator=True).fit(X)
    lin = trace_pipeline(est, X=X)
    assert lin.quality("missingindicator_a") == "exact"
    assert lin.origins("missingindicator_a") == ("a",)
    assert lin.quality("a") == "mixed"


# 4 and 5 --------------------------------------------------------------------


@pytest.fixture(scope="module")
def lineage():
    X = pd.DataFrame({"a": np.arange(8.0), "c": list("xyzx" * 2)})
    from sklearn.preprocessing import OneHotEncoder

    ct = ColumnTransformer(
        [
            ("n", StandardScaler(), ["a"]),
            ("cat", OneHotEncoder(sparse_output=False), ["c"]),
        ]
    ).fit(X)
    return trace_pipeline(ct, X=X)


def test_lineage_round_trips_through_toml(lineage):
    pytest.importorskip("tomli_w")
    assert FeatureLineage.loads(lineage.dumps("toml"), "toml") == lineage


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(output_ids=[-1, -2]),
        lambda d: d.update(output_ids=[999]),
        lambda d: d.update(input_ids=[d["output_ids"][0]]),
        lambda d: d["nodes"][0].update(name=123),
        lambda d: d["stages"][0].update(n_in="lots"),
        lambda d: d.update(final_estimator={"x": 1}),
        lambda d: d["stages"][0].pop("relation"),
        lambda d: d["edges"][0].update(parent=True),
        lambda d: d["edges"][0].update(relation="sideways"),
        lambda d: d["stage_inputs"].reverse(),
    ],
)
def test_lineage_loader_refuses_off_contract_records(lineage, mutate):
    data = copy.deepcopy(lineage.to_dict())
    mutate(data)
    with pytest.raises(ValueError):
        FeatureLineage.from_dict(data)


# 6 --------------------------------------------------------------------------


def _reference_dict():
    X = pd.DataFrame({"x": np.random.default_rng(0).normal(size=200)})
    return ReferenceProfile.fit(X).to_dict()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f.update(proportions=[1.0]),
        lambda f: f.update(cuts=list(reversed(f["cuts"]))),
        lambda f: f.update(kind="nonsense"),
        lambda f: f.update(name=7),
        lambda f: f.update(n_reference="12"),
        lambda f: f.update(proportions=[p * 2 for p in f["proportions"]]),
    ],
)
def test_reference_loader_refuses_off_contract_records(mutate):
    data = _reference_dict()
    mutate(data["features"][0])
    with pytest.raises(ValueError):
        ReferenceProfile.from_dict(data)


def test_reference_loader_refuses_duplicate_features():
    data = _reference_dict()
    data["features"].append(copy.deepcopy(data["features"][0]))
    with pytest.raises(ValueError, match="distinct"):
        ReferenceProfile.from_dict(data)


def test_schema_records_are_checked_for_consistency():
    schema = profile_features(
        pd.DataFrame({"x": np.arange(40.0), "y": np.arange(40.0)})
    )
    cases = [
        lambda d: d["features"][0]["stats"].update(name="other"),
        lambda d: d["features"][0]["stats"].update(n_missing=-5),
        lambda d: d["features"][1].update(position=0),
        lambda d: d["features"][0]["assessments"][0].update(params={"x": {"n": [1]}}),
        lambda d: d["features"][0]["binning"].update(reasons="abc"),
        lambda d: d["features"][0]["binning"].update(strategy=["weird"]),
        lambda d: d["features"][0]["binning"].update(feature=42),
    ]
    for mutate in cases:
        data = copy.deepcopy(schema.to_dict())
        mutate(data)
        with pytest.raises(ValueError):
            FeatureSchema.from_dict(data)


# 7 --------------------------------------------------------------------------


def test_numeric_strings_are_a_type_change_not_a_number():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "x": rng.normal(size=300),
            "flag": rng.integers(0, 2, 300),
            "b": rng.choice([True, False], 300),
        }
    )
    later = train.assign(
        x=train["x"].round(3).astype(str),
        flag=train["flag"].astype(str),
        b=train["b"].astype(str),
    )
    report = ReferenceProfile.fit(train).drift(later).set_index("feature")
    assert report["status"].to_dict() == {
        "x": "type_changed",
        "flag": "type_changed",
        "b": "type_changed",
    }
    ints_to_floats = train.assign(flag=train["flag"].astype(float))
    assert (
        ReferenceProfile.fit(train)
        .drift(ints_to_floats)
        .set_index("feature")
        .loc["flag", "status"]
        == "stable"
    )


# 8 --------------------------------------------------------------------------


def test_a_final_model_with_transform_is_the_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
    y = (X["a"] > 0).astype(int)
    pipe = Pipeline([("sc", StandardScaler()), ("lda", LinearDiscriminantAnalysis())])
    pipe.fit(X, y)
    lin = trace_pipeline(pipe, X=X)
    assert lin.final_estimator == "LinearDiscriminantAnalysis"
    assert lin.output_names == list("abc")
    assert set(lin.attribute(pipe[-1].coef_[0]).by_input.index) == set("abc")
    traced = trace_pipeline(pipe, X=X, trace_final=True)
    assert traced.final_estimator is None and len(traced.output_names) == 1


# 9 --------------------------------------------------------------------------


def test_a_manual_not_binary_changes_a_boolean_kind():
    X = pd.DataFrame({"flag": [True, False] * 10})
    schema = profile_features(X, {"overrides": {"flag": {"binary": False}}})
    assert schema["flag"].state("binary") == "false"
    assert schema["flag"].kind != "binary"


# 10 -------------------------------------------------------------------------


def test_numpy_integers_in_a_policy_are_stored_as_int():
    policy = FeaturePolicy(
        drift={"n_bins": np.int64(5)},
        binning={"max_bins": np.int32(9)},
        assessments={
            "skewed": {"provider": "skewed", "params": {"min_count": np.int64(9)}}
        },
    )
    assert type(policy.drift.n_bins) is int and type(policy.binning.max_bins) is int
    assert len(policy.digest) == 64
    assert FeaturePolicy.loads(policy.dumps("json"), "json") == policy


def test_feature_reference_field_order_is_backwards_compatible():
    ref = FeatureReference("x", "binary", "categorical")
    assert ref.family == "" and ref.note == ""


# Re-review --------------------------------------------------------------------


def test_a_final_pipeline_is_traced_down_to_its_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=40), "c": list("xyz" * 13) + ["x"]})
    y = (X["a"] > 0).astype(int)
    inner = Pipeline(
        [
            (
                "ohe",
                ColumnTransformer(
                    [("o", OneHotEncoder(sparse_output=False), ["c"])],
                    remainder="passthrough",
                ),
            ),
            ("lr", LogisticRegression()),
        ]
    )
    pre = ColumnTransformer(
        [("p", "passthrough", ["a", "c"])], verbose_feature_names_out=False
    )
    pipe = Pipeline([("pre", pre), ("model", inner)]).set_output(transform="pandas")
    pipe.fit(X, y)
    for trace_final in (False, True):
        lin = trace_pipeline(pipe, X=X, trace_final=trace_final)
        assert lin.final_estimator == "LogisticRegression"
        assert len(lin.output_names) == inner[-1].coef_.shape[1] == 4
    by_input = lin.attribute(np.abs(inner[-1].coef_[0])).by_input
    assert set(by_input.index) == {"a", "c"}


def test_edges_narrower_than_float_spacing_are_merged():
    eps = np.finfo(float).eps
    values = np.concatenate(
        [1 - np.arange(1, 201) * eps / 2, 1 + np.arange(0, 40) * eps]
    )
    X = pd.DataFrame({"x": np.tile(values, 2)})
    schema = profile_features(
        X, {"binning": {"rule": "fixed", "n_bins": 200, "max_bins": 200}}
    )
    rec = schema["x"].binning
    assert rec.is_suitable and "tied_edges_merged" in rec.reasons
    assert len(rec.edges) == rec.n_bins + 1 and np.all(np.diff(rec.edges) > 0)
    assert FeatureSchema.loads(schema.dumps("json")) == schema


def test_references_without_a_family_still_load():
    data = _reference_dict()
    for feature in data["features"]:
        feature.pop("family")
    assert ReferenceProfile.from_dict(data).features[0].family == ""


def test_all_missing_and_object_numbers_are_not_type_changes():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"f": rng.normal(size=300), "i": rng.integers(0, 5, 300)})
    ref = ReferenceProfile.fit(train)
    empty = train.assign(f=pd.Series([None] * 300, dtype=object))
    row = ref.drift(empty).set_index("feature").loc["f"]
    assert row["status"] == "insufficient" and row["missing_current"] == 1.0
    objects = train.assign(i=pd.Series([int(v) for v in train["i"]], dtype=object))
    assert ref.drift(objects).set_index("feature").loc["i", "status"] == "stable"
