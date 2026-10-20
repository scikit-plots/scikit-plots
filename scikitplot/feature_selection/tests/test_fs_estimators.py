# scikitplot/feature_selection/tests/test_fs_estimators.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""scikit-learn contract of FeatureProfiler, FeatureFilter, FeatureColumnSelector, FeatureRenamer."""

from __future__ import annotations

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from scikitplot.feature_selection import (
    FeatureColumnSelector,
    FeatureFilter,
    FeatureProfiler,
    FeatureRenamer,
    SchemaMismatchError,
    SchemaMismatchWarning,
    profile_features,
)

from ._helpers import mixed_frame, target_policy


def _model_frame(n=240, seed=0):
    X = mixed_frame(n, seed).drop(columns=["joined", "grade", "empty"])
    y = X.pop("churn")
    return X, y


def _upper(name):
    return name.upper()


# --------------------------------------------------------------------------- #
# FeatureProfiler
# --------------------------------------------------------------------------- #


def test_profiler_passes_the_same_frame_through():
    X, _ = _model_frame()
    prof = FeatureProfiler().fit(X)
    assert prof.transform(X) is X
    assert prof.schema_.n_rows == len(X)
    assert list(prof.get_feature_names_out()) == list(X.columns)


def test_profiler_output_features_drops_other_roles():
    X = mixed_frame()
    prof = FeatureProfiler(policy=target_policy(), output="features").fit(X)
    out = prof.transform(X)
    assert "churn" not in out.columns and "customer_id" not in out.columns
    assert list(prof.get_feature_names_out()) == list(out.columns)


def test_profiler_never_refits_on_later_data():
    X, _ = _model_frame()
    prof = FeatureProfiler().fit(X.iloc[:100])
    before = prof.schema_
    prof.transform(X.iloc[100:])
    assert prof.schema_ is before and before.n_rows == 100


def test_missing_columns_always_raise():
    X, _ = _model_frame()
    prof = FeatureProfiler(on_mismatch="ignore").fit(X)
    with pytest.raises(SchemaMismatchError, match="missing"):
        prof.transform(X.drop(columns=["kids"]))


@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
def test_extra_and_reordered_columns_follow_on_mismatch(mode):
    X, _ = _model_frame()
    prof = FeatureProfiler(on_mismatch=mode).fit(X)
    later = X[list(reversed(X.columns))].assign(extra=1)
    if mode == "raise":
        with pytest.raises(SchemaMismatchError, match="extra_column"):
            prof.transform(later)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = prof.transform(later)
    assert list(out.columns) == list(X.columns)
    assert any(issubclass(w.category, SchemaMismatchWarning) for w in caught) == (
        mode == "warn"
    )


def test_parameter_values_are_validated_in_fit():
    X, _ = _model_frame()
    with pytest.raises(ValueError, match="output"):
        FeatureProfiler(output="some").fit(X)
    with pytest.raises(ValueError, match="on_mismatch"):
        FeatureProfiler(on_mismatch="maybe").fit(X)


def test_ndarray_input_is_refused_with_advice():
    with pytest.raises(TypeError, match="pd.DataFrame"):
        FeatureProfiler().fit(np.zeros((3, 2)))


def test_profiler_clone_pickle_and_params():
    X, _ = _model_frame()
    prof = FeatureProfiler(policy=target_policy(), output="features")
    assert clone(prof).get_params() == prof.get_params()
    fitted = prof.fit(X.assign(churn=0))
    again = pickle.loads(pickle.dumps(fitted))
    assert again.schema_ == fitted.schema_


def test_profiler_audit():
    X, _ = _model_frame()
    prof = FeatureProfiler().fit(X)
    assert prof.audit(X.drop(columns=["kids"]))["issue"].tolist() == ["missing_column"]


# --------------------------------------------------------------------------- #
# FeatureFilter
# --------------------------------------------------------------------------- #


def test_filter_drops_constant_identifiers_and_target():
    X = mixed_frame()
    f = FeatureFilter(policy={"roles": {"target": ["churn"]}}).fit(X)
    report = f.report_
    assert report.loc["const", "reasons"] == ("constant",)
    assert report.loc["empty", "reasons"] == ("constant",)
    assert report.loc["customer_id", "reasons"] == ("identifier_like",)
    assert report.loc["churn", "reasons"] == ("role=target",)
    out = f.transform(X)
    assert list(out.columns) == list(f.get_feature_names_out())
    assert f.get_support().sum() == out.shape[1]
    assert list(f.get_support(indices=True)) == [
        i for i, k in enumerate(f.support_) if k
    ]


def test_filter_keep_overrides_assessments_not_roles():
    X = mixed_frame()
    f = FeatureFilter(
        policy={"roles": {"target": ["churn"]}}, keep=("const", "churn")
    ).fit(X)
    assert f.report_.loc["const", "kept"]
    assert not f.report_.loc["churn", "kept"]


def test_filter_refuses_unknown_names_and_empty_results():
    X = pd.DataFrame({"c": [1] * 10})
    with pytest.raises(ValueError, match="every column"):
        FeatureFilter().fit(X)
    with pytest.raises(ValueError, match="unknown assessment"):
        FeatureFilter(drop=("constnt",)).fit(mixed_frame())
    with pytest.raises(ValueError, match="keep names"):
        FeatureFilter(keep=("nope",)).fit(mixed_frame())
    with pytest.raises(ValueError, match="keep_roles"):
        FeatureFilter(keep_roles=("label",)).fit(mixed_frame())


# --------------------------------------------------------------------------- #
# FeatureColumnSelector
# --------------------------------------------------------------------------- #


def _routing_ct():
    return ColumnTransformer(
        [
            (
                "skewed",
                RobustScaler(),
                FeatureColumnSelector(kind="numeric_continuous", is_=("skewed",)),
            ),
            (
                "normal",
                StandardScaler(),
                FeatureColumnSelector(
                    kind=("numeric_continuous", "numeric_discrete"),
                    is_not=("skewed", "constant"),
                ),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                FeatureColumnSelector(
                    kind="categorical_nominal", is_not=("identifier_like",)
                ),
            ),
        ],
        remainder="drop",
    )


def test_selector_routes_columns_by_assessment():
    X, _ = _model_frame()
    X = X.assign(age=X["age"].fillna(X["age"].median()))
    ct = _routing_ct().fit(X)
    routes = {name: cols for name, _, cols in ct.transformers_ if name != "remainder"}
    assert routes == {
        "skewed": ["income"],
        "normal": ["height", "age", "kids"],
        "cat": ["city"],
    }


def test_selector_is_resolved_on_each_training_fold():
    X = pd.DataFrame(
        {"x": [0.0] * 10 + list(np.linspace(1, 2, 10)), "y": np.arange(20.0)}
    )
    sel = FeatureColumnSelector(is_not=("constant",))
    assert sel(X.iloc[:10]) == ["y"]
    assert sel(X) == ["x", "y"]


def test_selector_value_contract_and_fixed_schema():
    X, _ = _model_frame()
    sel = FeatureColumnSelector(kind=["numeric_continuous"], is_=["skewed"])
    assert sel.kind == ("numeric_continuous",) and sel.is_ == ("skewed",)
    assert pickle.loads(pickle.dumps(sel)) == sel and copy.deepcopy(sel) == sel
    fixed = FeatureColumnSelector(kind="numeric_continuous", schema=profile_features(X))
    assert fixed(X) == ["income", "height", "age"]
    with pytest.raises(SchemaMismatchError):
        fixed(X.drop(columns=["age"]))


def test_full_pipeline_under_cross_validation_and_grid_search():
    X, y = _model_frame()
    X = X.assign(age=X["age"].fillna(X["age"].median()))
    pipe = Pipeline(
        [
            ("profile", FeatureProfiler()),
            ("filter", FeatureFilter()),
            ("ct", _routing_ct()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=KFold(3, shuffle=True, random_state=0))
    assert scores.shape == (3,)
    grid = GridSearchCV(pipe, {"model__C": [0.1, 1.0]}, cv=3).fit(X, y)
    assert grid.best_estimator_.named_steps["profile"].schema_.n_rows == len(X)
    restored = pickle.loads(pickle.dumps(grid.best_estimator_))
    np.testing.assert_allclose(restored.predict_proba(X), grid.predict_proba(X))


def test_set_output_pandas_keeps_names():
    X, _ = _model_frame()
    X = X.assign(age=X["age"].fillna(0))
    pipe = Pipeline([("filter", FeatureFilter()), ("ct", _routing_ct())]).set_output(
        transform="pandas"
    )
    out = pipe.fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(pipe.get_feature_names_out())


# --------------------------------------------------------------------------- #
# FeatureRenamer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ({"a": "alpha"}, ["alpha", "b", "c"]),
        (["x", "y", "z"], ["x", "y", "z"]),
        (_upper, ["A", "B", "C"]),
        ("col-{index}", ["col-1", "col-2", "col-3"]),
        ("{name}_raw", ["a_raw", "b_raw", "c_raw"]),
        (None, ["a", "b", "c"]),
    ],
)
def test_renamer_rules(rule, expected):
    X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    r = FeatureRenamer(rule, start=1).fit(X)
    assert list(r.transform(X).columns) == expected
    assert list(r.get_feature_names_out()) == expected


def test_renamer_runs_a_callable_once():
    calls = []

    def rule(name):
        calls.append(name)
        return f"{name}_{len(calls)}"

    X = pd.DataFrame({"a": [1], "b": [2]})
    r = FeatureRenamer(rule).fit(X)
    r.transform(X)
    r.transform(X)
    assert calls == ["a", "b"] and r.plan_.names_out == ["a_1", "b_2"]


def test_renamer_numpy_input_and_mismatch():
    r = FeatureRenamer("f{index}").fit(np.zeros((2, 3)))
    assert not hasattr(r, "feature_names_in_")
    assert list(r.transform(np.ones((1, 3))).columns) == ["f0", "f1", "f2"]
    assert list(r.get_feature_names_out(["p", "q", "s"])) == ["f0", "f1", "f2"]
    with pytest.raises(SchemaMismatchError):
        r.transform(np.ones((1, 2)))
    framed = FeatureRenamer({"a": "b2"}).fit(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(SchemaMismatchError, match="fitted order"):
        framed.transform(pd.DataFrame({"b": [2], "a": [1]}))


def test_renamer_pickles_with_a_named_function():
    r = FeatureRenamer(_upper).fit(pd.DataFrame({"a": [1]}))
    assert pickle.loads(pickle.dumps(r)).plan_ == r.plan_
    assert clone(FeatureRenamer(_upper)).rule is _upper


def test_estimators_advertise_nan_and_string_support():
    est = FeatureProfiler()
    try:
        tags = est.__sklearn_tags__()
    except AttributeError:  # scikit-learn < 1.6 reads _more_tags through _get_tags
        tags = est._get_tags()
        assert tags["allow_nan"] and "string" in tags["X_types"]
        return
    assert tags.input_tags.allow_nan and tags.input_tags.string
