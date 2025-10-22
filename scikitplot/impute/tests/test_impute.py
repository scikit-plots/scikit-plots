# flake8: noqa: D213

import io
import re
import warnings
from itertools import product

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest

from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning

# make IterativeImputer available
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import (
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    LIL_CONTAINERS,
)

# pytest.ini equivalent inside file
pytestmark = pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning",
    "ignore:Skipping features without any observed values",
)


def _assert_array_equal_and_same_dtype(x, y):
    assert_array_equal(x, y)
    assert x.dtype == y.dtype


def _assert_allclose_and_same_dtype(x, y):
    assert_allclose(x, y)
    assert x.dtype == y.dtype


def _check_statistics(
    X, X_true, strategy, statistics, missing_values, sparse_container
):
    """Utility function for testing imputation for a given strategy.

    Test with dense and sparse arrays

    Check that:
        - the statistics (mean, median, mode) are correct
        - the missing values are imputed correctly"""

    err_msg = "Parameters: strategy = %s, missing_values = %s, sparse = {0}" % (
        strategy,
        missing_values,
    )

    assert_ae = assert_array_equal

    if X.dtype.kind == "f" or X_true.dtype.kind == "f":
        assert_ae = assert_array_almost_equal

    # Normal matrix
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    X_trans = imputer.fit(X).transform(X.copy())
    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(False))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(False))

    # Sparse matrix
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputer.fit(sparse_container(X))
    X_trans = imputer.transform(sparse_container(X.copy()))

    if sparse.issparse(X_trans):
        X_trans = X_trans.toarray()

    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(True))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(True))


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_imputation_shape(strategy, csr_container):
    # Verify the shapes of the imputed matrix for different strategies.
    X = np.random.randn(10, 2)
    X[::2] = np.nan

    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(csr_container(X))
    assert X_imputed.shape == (10, 2)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == (10, 2)

    iterative_imputer = IterativeImputer(initial_strategy=strategy)
    X_imputed = iterative_imputer.fit_transform(X)
    assert X_imputed.shape == (10, 2)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_imputation_deletion_warning(strategy):
    X = np.ones((3, 5))
    X[:, 0] = np.nan
    imputer = SimpleImputer(strategy=strategy).fit(X)

    with pytest.warns(UserWarning, match="Skipping"):
        imputer.transform(X)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_imputation_deletion_warning_feature_names(strategy):
    pd = pytest.importorskip("pandas")

    missing_values = np.nan
    feature_names = np.array(["a", "b", "c", "d"], dtype=object)
    X = pd.DataFrame(
        [
            [missing_values, missing_values, 1, missing_values],
            [4, missing_values, 2, 10],
        ],
        columns=feature_names,
    )

    imputer = SimpleImputer(strategy=strategy).fit(X)

    # check SimpleImputer returning feature name attribute correctly
    assert_array_equal(imputer.feature_names_in_, feature_names)

    # ensure that skipped feature warning includes feature name
    with pytest.warns(
        UserWarning, match=r"Skipping features without any observed values: \['b'\]"
    ):
        imputer.transform(X)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_error_sparse_0(strategy, csc_container):
    # check that error are raised when missing_values = 0 and input is sparse
    X = np.ones((3, 5))
    X[0] = 0
    X = csc_container(X)

    imputer = SimpleImputer(strategy=strategy, missing_values=0)
    with pytest.raises(ValueError, match="Provide a dense array"):
        imputer.fit(X)

    imputer.fit(X.toarray())
    with pytest.raises(ValueError, match="Provide a dense array"):
        imputer.transform(X)


def safe_median(arr, *args, **kwargs):
    # np.median([]) raises a TypeError for numpy >= 1.10.1
    length = arr.size if hasattr(arr, "size") else len(arr)
    return np.nan if length == 0 else np.median(arr, *args, **kwargs)


def safe_mean(arr, *args, **kwargs):
    # np.mean([]) raises a RuntimeWarning for numpy >= 1.10.1
    length = arr.size if hasattr(arr, "size") else len(arr)
    return np.nan if length == 0 else np.mean(arr, *args, **kwargs)


# @pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_mean_median(csc_container):
    # Test imputation using the mean and median strategies, when
    # missing_values != 0.
    rng = np.random.RandomState(0)

    dim = 10
    dec = 10
    shape = (dim * dim, dim + dec)

    zeros = np.zeros(shape[0])
    values = np.arange(1, shape[0] + 1)
    values[4::2] = -values[4::2]

    tests = [
        ("mean", np.nan, lambda z, v, p: safe_mean(np.hstack((z, v)))),
        ("median", np.nan, lambda z, v, p: safe_median(np.hstack((z, v)))),
    ]

    for strategy, test_missing_values, true_value_fun in tests:
        X = np.empty(shape)
        X_true = np.empty(shape)
        true_statistics = np.empty(shape[1])

        # Create a matrix X with columns
        #    - with only zeros,
        #    - with only missing values
        #    - with zeros, missing values and values
        # And a matrix X_true containing all true values
        for j in range(shape[1]):
            nb_zeros = (j - dec + 1 > 0) * (j - dec + 1) * (j - dec + 1)
            nb_missing_values = max(shape[0] + dec * dec - (j + dec) * (j + dec), 0)
            nb_values = shape[0] - nb_zeros - nb_missing_values

            z = zeros[:nb_zeros]
            p = np.repeat(test_missing_values, nb_missing_values)
            v = values[rng.permutation(len(values))[:nb_values]]

            true_statistics[j] = true_value_fun(z, v, p)

            # Create the columns
            X[:, j] = np.hstack((v, z, p))

            if 0 == test_missing_values:
                # XXX unreached code as of v0.22
                X_true[:, j] = np.hstack(
                    (v, np.repeat(true_statistics[j], nb_missing_values + nb_zeros))
                )
            else:
                X_true[:, j] = np.hstack(
                    (v, z, np.repeat(true_statistics[j], nb_missing_values))
                )

            # Shuffle them the same way
            np.random.RandomState(j).shuffle(X[:, j])
            np.random.RandomState(j).shuffle(X_true[:, j])

        # Mean doesn't support columns containing NaNs, median does
        if strategy == "median":
            cols_to_keep = ~np.isnan(X_true).any(axis=0)
        else:
            cols_to_keep = ~np.isnan(X_true).all(axis=0)

        X_true = X_true[:, cols_to_keep]

        _check_statistics(
            X, X_true, strategy, true_statistics, test_missing_values, csc_container
        )


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_median_special_cases(csc_container):
    # Test median imputation with sparse boundary cases
    X = np.array(
        [
            [0, np.nan, np.nan],  # odd: implicit zero
            [5, np.nan, np.nan],  # odd: explicit nonzero
            [0, 0, np.nan],  # even: average two zeros
            [-5, 0, np.nan],  # even: avg zero and neg
            [0, 5, np.nan],  # even: avg zero and pos
            [4, 5, np.nan],  # even: avg nonzeros
            [-4, -5, np.nan],  # even: avg negatives
            [-1, 2, np.nan],  # even: crossing neg and pos
        ]
    ).transpose()

    X_imputed_median = np.array(
        [
            [0, 0, 0],
            [5, 5, 5],
            [0, 0, 0],
            [-5, 0, -2.5],
            [0, 5, 2.5],
            [4, 5, 4.5],
            [-4, -5, -4.5],
            [-1, 2, 0.5],
        ]
    ).transpose()
    statistics_median = [0, 5, 0, -2.5, 2.5, 4.5, -4.5, 0.5]

    _check_statistics(
        X, X_imputed_median, "median", statistics_median, np.nan, csc_container
    )


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("dtype", [None, object, str])
def test_imputation_mean_median_error_invalid_type(strategy, dtype):
    X = np.array([["a", "b", 3], [4, "e", 6], ["g", "h", 9]], dtype=dtype)
    msg = "non-numeric data:\ncould not convert string to float:"
    with pytest.raises(ValueError, match=msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("type", ["list", "dataframe"])
def test_imputation_mean_median_error_invalid_type_list_pandas(strategy, type):
    X = [["a", "b", 3], [4, "e", 6], ["g", "h", 9]]
    if type == "dataframe":
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X)
    msg = "non-numeric data:\ncould not convert string to float:"
    with pytest.raises(ValueError, match=msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)


@pytest.mark.parametrize("strategy", ["constant", "most_frequent"])
@pytest.mark.parametrize("dtype", [str, np.dtype("U"), np.dtype("S")])
def test_imputation_const_mostf_error_invalid_types(strategy, dtype):
    # Test imputation on non-numeric data using "most_frequent" and "constant"
    # strategy
    X = np.array(
        [
            [np.nan, np.nan, "a", "f"],
            [np.nan, "c", np.nan, "d"],
            [np.nan, "b", "d", np.nan],
            [np.nan, "c", "d", "h"],
        ],
        dtype=dtype,
    )

    err_msg = "SimpleImputer does not support data"
    with pytest.raises(ValueError, match=err_msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(X).transform(X)


# @pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_imputation_most_frequent(csc_container):
    # Test imputation using the most-frequent strategy.
    X = np.array(
        [
            [-1, -1, 0, 5],
            [-1, 2, -1, 3],
            [-1, 1, 3, -1],
            [-1, 2, 3, 7],
        ]
    )

    X_true = np.array(
        [
            [2, 0, 5],
            [2, 3, 3],
            [1, 3, 3],
            [2, 3, 7],
        ]
    )

    # scipy.stats.mode, used in SimpleImputer, doesn't return the first most
    # frequent as promised in the doc but the lowest most frequent. When this
    # test will fail after an update of scipy, SimpleImputer will need to be
    # updated to be consistent with the new (correct) behaviour
    _check_statistics(X, X_true, "most_frequent", [np.nan, 2, 3, 3], -1, csc_container)


# @pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
@pytest.mark.parametrize("marker", [None, np.nan, "NAN", "", 0])
def test_imputation_most_frequent_objects(marker):
    # Test imputation using the most-frequent strategy.
    X = np.array(
        [
            [marker, marker, "a", "f"],
            [marker, "c", marker, "d"],
            [marker, "b", "d", marker],
            [marker, "c", "d", "h"],
        ],
        dtype=object,
    )

    X_true = np.array(
        [
            ["c", "a", "f"],
            ["c", "d", "d"],
            ["b", "d", "d"],
            ["c", "d", "h"],
        ],
        dtype=object,
    )

    imputer = SimpleImputer(missing_values=marker, strategy="most_frequent")
    X_trans = imputer.fit(X).transform(X)

    assert_array_equal(X_trans, X_true)


# @pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
@pytest.mark.parametrize("dtype", [object, "category"])
def test_imputation_most_frequent_pandas(dtype):
    # Test imputation using the most frequent strategy on pandas df
    pd = pytest.importorskip("pandas")

    f = io.StringIO("Cat1,Cat2,Cat3,Cat4\n,i,x,\na,,y,\na,j,,\nb,j,x,")

    df = pd.read_csv(f, dtype=dtype)

    X_true = np.array(
        [["a", "i", "x"], ["a", "j", "y"], ["a", "j", "x"], ["b", "j", "x"]],
        dtype=object,
    )

    imputer = SimpleImputer(strategy="most_frequent")
    X_trans = imputer.fit_transform(df)

    assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("X_data, missing_value", [(1, 0), (1.0, np.nan)])
def test_imputation_constant_error_invalid_type(X_data, missing_value):
    # Verify that exceptions are raised on invalid fill_value type
    X = np.full((3, 5), X_data, dtype=float)
    X[0, 0] = missing_value

    fill_value = "x"
    err_msg = f"fill_value={fill_value!r} (of type {type(fill_value)!r}) cannot be cast"
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer = SimpleImputer(
            missing_values=missing_value, strategy="constant", fill_value=fill_value
        )
        imputer.fit_transform(X)


# @pytest.mark.parametrize("keep_empty_features", [True, False])
# def test_imputation_constant_integer(keep_empty_features):
#     # Test imputation using the constant strategy on integers
#     X = np.array([[-1, 2, 3, -1], [4, -1, 5, -1], [6, 7, -1, -1], [8, 9, 0, -1]])

#     X_true = np.array([[0, 2, 3, 0], [4, 0, 5, 0], [6, 7, 0, 0], [8, 9, 0, 0]])
#     if not keep_empty_features:
#         X_true = X_true[:, :-1]

#     imputer = SimpleImputer(
#         keep_empty_features=bool(keep_empty_features),
#         missing_values=-1,
#         strategy="constant",
#         fill_value=0,
#     )
#     X_trans = imputer.fit_transform(X)

#     assert_array_equal(X_trans, X_true)


# @pytest.mark.parametrize("array_constructor", CSR_CONTAINERS + [np.asarray])
# @pytest.mark.parametrize("keep_empty_features", [True, False])
# def test_imputation_constant_float(array_constructor, keep_empty_features):
#     # Test imputation using the constant strategy on floats
#     X = np.array(
#         [
#             [np.nan, 1.1, 0, np.nan],
#             [1.2, np.nan, 1.3, np.nan],
#             [0, 0, np.nan, np.nan],
#             [1.4, 1.5, 0, np.nan],
#         ]
#     )

#     X_true = np.array(
#         [[-1, 1.1, 0, -1], [1.2, -1, 1.3, -1], [0, 0, -1, -1], [1.4, 1.5, 0, -1]]
#     )
#     if not keep_empty_features:
#         X_true = X_true[:, :-1]

#     X = array_constructor(X)

#     X_true = array_constructor(X_true)

#     imputer = SimpleImputer(
#         strategy="constant", fill_value=-1, keep_empty_features=keep_empty_features
#     )
#     X_trans = imputer.fit_transform(X)

#     assert_allclose_dense_sparse(X_trans, X_true)


# @pytest.mark.parametrize("marker", [None, np.nan, "NAN", "", 0])
# @pytest.mark.parametrize("keep_empty_features", [True, False])
# def test_imputation_constant_object(marker, keep_empty_features):
#     # Test imputation using the constant strategy on objects
#     X = np.array(
#         [
#             [marker, "a", "b", marker],
#             ["c", marker, "d", marker],
#             ["e", "f", marker, marker],
#             ["g", "h", "i", marker],
#         ],
#         dtype=object,
#     )

#     X_true = np.array(
#         [
#             ["missing", "a", "b", "missing"],
#             ["c", "missing", "d", "missing"],
#             ["e", "f", "missing", "missing"],
#             ["g", "h", "i", "missing"],
#         ],
#         dtype=object,
#     )
#     if not keep_empty_features:
#         X_true = X_true[:, :-1]

#     imputer = SimpleImputer(
#         missing_values=marker,
#         strategy="constant",
#         fill_value="missing",
#         keep_empty_features=keep_empty_features,
#     )
#     X_trans = imputer.fit_transform(X)

#     assert_array_equal(X_trans, X_true)


# @pytest.mark.parametrize("dtype", [object, "category"])
# @pytest.mark.parametrize("keep_empty_features", [True, False])
# def test_imputation_constant_pandas(dtype, keep_empty_features):
#     # Test imputation using the constant strategy on pandas df
#     pd = pytest.importorskip("pandas")

#     f = io.StringIO("Cat1,Cat2,Cat3,Cat4\n,i,x,\na,,y,\na,j,,\nb,j,x,")

#     df = pd.read_csv(f, dtype=dtype)

#     X_true = np.array(
#         [
#             ["missing_value", "i", "x", "missing_value"],
#             ["a", "missing_value", "y", "missing_value"],
#             ["a", "j", "missing_value", "missing_value"],
#             ["b", "j", "x", "missing_value"],
#         ],
#         dtype=object,
#     )
#     if not keep_empty_features:
#         X_true = X_true[:, :-1]

#     imputer = SimpleImputer(
#         strategy="constant", keep_empty_features=keep_empty_features
#     )
#     X_trans = imputer.fit_transform(df)

#     assert_array_equal(X_trans, X_true)


@pytest.mark.parametrize("X", [[[1], [2]], [[1], [np.nan]]])
def test_iterative_imputer_one_feature(X):
    # check we exit early when there is a single feature
    imputer = IterativeImputer().fit(X)
    assert imputer.n_iter_ == 0
    imputer = IterativeImputer()
    imputer.fit([[1], [2]])
    assert imputer.n_iter_ == 0
    imputer.fit([[1], [np.nan]])
    assert imputer.n_iter_ == 0


def test_imputation_pipeline_grid_search():
    # Test imputation within a pipeline + gridsearch.
    X = _sparse_random_matrix(100, 100, density=0.10)
    missing_values = X.data[0]

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=missing_values)),
            ("tree", tree.DecisionTreeRegressor(random_state=0)),
        ]
    )

    parameters = {"imputer__strategy": ["mean", "median", "most_frequent"]}

    Y = _sparse_random_matrix(100, 1, density=0.10).toarray()
    gs = GridSearchCV(pipeline, parameters)
    gs.fit(X, Y)


def test_imputation_copy():
    # Test imputation with copy
    X_orig = _sparse_random_matrix(5, 5, density=0.75, random_state=0)

    # copy=True, dense => copy
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert not np.all(X == Xt)

    # copy=True, sparse csr => copy
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=True)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert not np.all(X.data == Xt.data)

    # copy=False, dense => no copy
    X = X_orig.copy().toarray()
    imputer = SimpleImputer(missing_values=0, strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt[0, 0] = -1
    assert_array_almost_equal(X, Xt)

    # copy=False, sparse csc => no copy
    X = X_orig.copy().tocsc()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert_array_almost_equal(X.data, Xt.data)

    # copy=False, sparse csr => copy
    X = X_orig.copy()
    imputer = SimpleImputer(missing_values=X.data[0], strategy="mean", copy=False)
    Xt = imputer.fit(X).transform(X)
    Xt.data[0] = -1
    assert not np.all(X.data == Xt.data)

    # Note: If X is sparse and if missing_values=0, then a (dense) copy of X is
    # made, even if copy=False.
