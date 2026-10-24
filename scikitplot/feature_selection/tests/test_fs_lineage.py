# scikitplot/feature_selection/tests/test_fs_lineage.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""trace_pipeline: exact adapters, width reconciliation, navigation and attribution."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

from scikitplot.feature_selection import (
    FeatureColumnSelector,
    FeatureFilter,
    FeatureLineage,
    FeatureMapping,
    FeatureRenamer,
    LineageError,
    trace_pipeline,
)


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "income": rng.lognormal(10, 1, n),
            "city": rng.choice(["a", "b", "c", "rare"], n, p=[0.4, 0.3, 0.27, 0.03]),
            "kids": rng.integers(0, 4, n),
            "id": [f"r{i}" for i in range(n)],
        }
    )
    X.loc[::5, "age"] = np.nan
    y = rng.integers(0, 2, n)
    return X, y


def _pipeline():
    num = Pipeline(
        [("imp", SimpleImputer(add_indicator=True)), ("sc", StandardScaler())]
    )
    ct = ColumnTransformer(
        [
            ("num", num, ["age", "income"]),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    min_frequency=10,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
                ["city"],
            ),
        ],
        remainder="passthrough",
    )
    return Pipeline(
        [
            ("filter", FeatureFilter()),
            ("ct", ct),
            ("sel", SelectKBest(f_classif, k=4)),
            ("rf", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]
    )


@pytest.fixture(scope="module")
def fitted():
    X, y = _frame()
    pipe = _pipeline().fit(X, y)
    return X, y, pipe, trace_pipeline(pipe, X=X.head(40))


def test_widths_reconcile_at_every_stage(fitted):
    X, _, pipe, lin = fitted
    summary = lin.stage_summary().set_index("path")
    observed = summary["observed_width"].dropna()
    assert (observed == summary.loc[observed.index, "n_out"]).all()
    assert lin.output_names == list(pipe[:-1].get_feature_names_out())
    assert lin.final_estimator == "RandomForestClassifier"
    assert summary.loc["ct/cat", "n_out"] == len(
        pipe.named_steps["ct"].named_transformers_["cat"].get_feature_names_out()
    )


def test_raw_columns_are_navigable(fitted):
    _, _, pipe, lin = fitted
    unreached = {n for n in lin.input_names if not lin.descendants(n)}
    expected = {n: "filter" if n == "id" else "sel" for n in unreached}
    assert "id" in unreached and lin.dropped() == expected
    ct_names = list(pipe.named_steps["ct"].get_feature_names_out())
    for name in ct_names:
        origin = lin.origins(name, stage="ct")
        assert len(origin) == 1
    assert set(lin.descendants("city", to="ct")) == {
        n for n in ct_names if n.startswith("cat__")
    }
    assert lin.origins("num__missingindicator_age", stage="ct") == ("age",)
    assert lin.parents("num__age", stage="ct") == [("ct/num/sc", "age")]


def test_treatments_list_the_chain_per_raw_column(fitted):
    _, _, _, lin = fitted
    chain = lin.treatments().loc["age", "steps"]
    assert chain[:3] == (
        "filter:FeatureFilter",
        "ct/num/imp:SimpleImputer",
        "ct/num/sc:StandardScaler",
    )
    assert lin.treatments().loc["id", "status"] == "dropped"
    history = lin.history("city")
    assert list(history["stage"])[:3] == ["filter", "ct/cat", "ct"]


def test_onehot_importances_sum_back_to_the_raw_column(fitted):
    _, _, pipe, lin = fitted
    importances = pipe[-1].feature_importances_
    result = lin.attribute(importances)
    expected = {}
    for name, value in zip(lin.output_names, importances):
        (origin,) = lin.origins(name)
        expected[origin] = expected.get(origin, 0.0) + value
    assert result.by_input.to_dict() == pytest.approx(expected)
    assert result.unattributed.empty
    assert result.by_input.sum() == pytest.approx(importances.sum())


def test_two_dimensional_values_are_summed_per_sample(fitted):
    _, _, _, lin = fitted
    values = np.random.default_rng(0).normal(size=(5, len(lin.output_names)))
    result = lin.attribute(values)
    np.testing.assert_allclose(result.by_input.sum(axis=1), values.sum(axis=1))
    frame = pd.DataFrame(values, columns=lin.output_names, index=list("abcde"))
    assert list(lin.attribute(frame).by_input.index) == list("abcde")


def test_values_must_align(fitted):
    _, _, _, lin = fitted
    with pytest.raises(ValueError, match="columns"):
        lin.attribute(np.ones(len(lin.output_names) + 1))
    with pytest.raises(ValueError, match="keys"):
        lin.attribute(pd.Series([1.0], index=["nope"]))
    with pytest.raises(ValueError, match="how"):
        lin.attribute(np.ones(len(lin.output_names)), how="median")
    by_name = pd.Series(
        np.arange(len(lin.output_names), dtype=float), index=lin.output_names
    )[::-1]
    assert lin.attribute(by_name).by_input.sum() == pytest.approx(by_name.sum())


def test_lineage_persists(fitted):
    _, _, _, lin = fitted
    assert FeatureLineage.loads(lin.dumps("json")) == lin
    assert pickle.loads(pickle.dumps(lin)) == lin
    rows = lin.dumps("jsonl").splitlines()
    assert len(rows) == len(lin.edges)
    with pytest.raises(ValueError):
        FeatureLineage.loads(lin.dumps("jsonl"), "jsonl")


def test_poly_and_pca_are_not_attributed_by_a_guess():
    X, y = _frame()
    X = X[["age", "income"]].fillna(0)
    pipe = Pipeline(
        [
            (
                "fu",
                FeatureUnion(
                    [
                        ("pca", PCA(1)),
                        ("poly", PolynomialFeatures(2, include_bias=True)),
                    ]
                ),
            ),
        ]
    ).fit(X)
    lin = trace_pipeline(pipe, X=X)
    result = lin.attribute(np.ones(len(lin.output_names)))
    reasons = result.unattributed.set_index("output")["reason"].to_dict()
    assert reasons["pca__pca0"] == "mixed_lineage"
    assert reasons["poly__age income"] == "multiple_origins"
    assert reasons["poly__1"] == "no_origin"
    assert result.by_input.to_dict() == {"age": 2.0, "income": 2.0}
    assert lin.quality("pca__pca0") == "mixed"


@pytest.mark.parametrize(
    "transformer",
    [
        KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="uniform"),
        KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform"),
        SplineTransformer(n_knots=4),
        MissingIndicator(features="all"),
        VarianceThreshold(),
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        FeatureAgglomeration(n_clusters=1),
        SimpleImputer(),
    ],
    ids=lambda t: type(t).__name__,
)
def test_known_adapters_are_exact_and_reconcile(transformer):
    X = pd.DataFrame(
        {"u": np.linspace(0.0, 1.0, 30), "v": np.linspace(1.0, 3.0, 30) ** 2}
    )
    X.loc[3, "u"] = np.nan
    if not isinstance(transformer, (MissingIndicator, SimpleImputer)):
        X = X.fillna(0.5)
    fitted = transformer.fit(X)
    lin = trace_pipeline(fitted, X=X)
    assert lin.output_names == list(fitted.get_feature_names_out())
    expected = "mixed" if isinstance(transformer, FeatureAgglomeration) else "exact"
    if isinstance(transformer, FeatureAgglomeration):
        assert lin.origins(lin.output_names[0]) == ("u", "v")
        assert lin.stage_summary().loc[1, "quality"] == "exact"
    else:
        assert {lin.quality(n) for n in lin.output_names} == {expected}


@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
def test_simple_imputer_dropping_an_empty_column():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan] * 3, "c": [1.0, 2.0, 3.0]})
    imp = SimpleImputer(add_indicator=True).fit(X)
    lin = trace_pipeline(imp, X=X)
    assert lin.output_names == list(imp.get_feature_names_out())
    assert "b" not in lin.output_names  # the values are dropped ...
    assert lin.origins("missingindicator_b") == ("b",)  # ... the indicator survives
    assert lin.dropped() == {}
    plain = SimpleImputer().fit(X)
    assert trace_pipeline(plain, X=X).dropped() == {"b": "(root)"}


def test_target_encoder_multiclass_blocks():
    from sklearn.preprocessing import TargetEncoder

    X = pd.DataFrame({"c": list("abcab") * 12, "d": list("xyz") * 20})
    y = np.arange(60) % 3
    encoder = TargetEncoder(random_state=0).fit(X, y)
    lin = trace_pipeline(encoder, X=X)
    assert [lin.origins(n)[0] for n in lin.output_names] == ["c"] * 3 + ["d"] * 3


def test_remainder_drop_and_numpy_input():
    arr = np.random.default_rng(0).normal(size=(20, 3))
    ct = ColumnTransformer([("s", StandardScaler(), [0, 2])], remainder="drop").fit(arr)
    lin = trace_pipeline(ct, X=arr)
    assert lin.input_names == ["x0", "x1", "x2"]
    assert lin.output_names == ["s__x0", "s__x2"]
    assert lin.dropped() == {"x1": "(root)"}


def test_callable_selector_and_renamer_are_traced():
    X, _ = _frame()
    ct = ColumnTransformer(
        [("n", StandardScaler(), FeatureColumnSelector(kind="numeric_continuous"))],
        verbose_feature_names_out=True,
    )
    pipe = Pipeline([("ct", ct), ("rn", FeatureRenamer(lambda n: n.split("__")[1]))])
    pipe.set_output(transform="pandas")
    data = X[["age", "income", "city"]].fillna(0)
    pipe.fit(data)
    lin = trace_pipeline(pipe, X=data)
    assert lin.output_names == ["age", "income"]
    assert lin.origins("income") == ("income",)
    assert lin.dropped() == {"city": "ct"}


class _Opaque(TransformerMixin, BaseEstimator):
    """A transformer without names or lineage: width known only from data."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.c_[np.asarray(X), np.asarray(X)[:, :1]]


class _Liar(_Opaque):
    def get_feature_names_out(self, input_features=None):
        return np.asarray(["only_one"], dtype=object)


class _Exact(_Opaque):
    def get_feature_names_out(self, input_features=None):
        return np.asarray([*input_features, f"{input_features[0]}_copy"], dtype=object)

    def get_feature_lineage(self, input_features=None):
        n = len(input_features)
        return FeatureMapping(tuple((i,) for i in range(n)) + ((0,),), "expand")


def test_unknown_steps_need_a_sample_and_are_never_attributed():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    opaque = _Opaque().fit(X)
    with pytest.raises(LineageError, match="cannot tell how many"):
        trace_pipeline(opaque, ["a", "b"])
    lin = trace_pipeline(opaque, ["a", "b"], X=X)
    assert len(lin.output_names) == 3
    result = lin.attribute([1.0, 1.0, 1.0])
    assert result.by_input.empty
    assert set(result.unattributed["reason"]) == {"unknown_lineage"}


def test_disagreeing_metadata_fails_loudly():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(LineageError, match="transform returned"):
        trace_pipeline(_Liar().fit(X), ["a", "b"], X=X)


def test_the_lineage_protocol_makes_custom_steps_exact():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    lin = trace_pipeline(_Exact().fit(X), ["a", "b"], X=X)
    assert lin.output_names == ["a", "b", "a_copy"]
    assert lin.attribute([1.0, 2.0, 3.0]).by_input.to_dict() == {"a": 4.0, "b": 2.0}


def test_input_names_are_checked():
    X, y = _frame()
    pipe = _pipeline().fit(X, y)
    with pytest.raises(LineageError, match="differ"):
        trace_pipeline(pipe, list("abcde"))
    with pytest.raises(LineageError, match="columns"):
        trace_pipeline(pipe, X=X.iloc[:, :2])
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        trace_pipeline(_pipeline())


def test_lineage_documents_are_checked(fitted):
    _, _, _, lin = fitted
    data = lin.to_dict()
    from scikitplot.feature_selection import IncompatibleSchemaError

    with pytest.raises(IncompatibleSchemaError):
        FeatureLineage.from_dict({**data, "schema_version": 9})
    bad = dict(
        data,
        edges=[
            {
                "parent": 0,
                "child": 10**6,
                "stage": "x",
                "relation": "subset",
                "quality": "exact",
            }
        ],
    )
    with pytest.raises(ValueError, match="outside"):
        FeatureLineage.from_dict(bad)
